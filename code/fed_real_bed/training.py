from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import WindowDataset, build_window_cache, load_window_cache
from .evaluation import (
    bootstrap_eer_ci,
    build_prototypes,
    per_group_metrics,
    roc_metrics,
    save_roc,
    temporal_fusion,
    verification_trials,
)
from .federated import run_federated_training
from .models import EEGAuthenticator, count_parameters
from .protocols import assert_split_valid, build_protocol_split
from .utils import Timer, ensure_dir, get_device, save_json, seed_everything


def _maybe_cache(cfg: ExperimentConfig) -> Path:
    cache_dir = ensure_dir(cfg.cache_dir)
    stimulus = str(cfg.raw["data"].get("stimulus", "all")).lower().replace("*", "all")
    existing = sorted(cache_dir.glob(f"bed_windows_{stimulus}_{cfg.raw['data']['target_sampling_rate']}hz_*.npz"))
    if existing:
        return existing[0]
    if stimulus not in {"all", ""}:
        all_existing = sorted(cache_dir.glob(f"bed_windows_all_{cfg.raw['data']['target_sampling_rate']}hz_*.npz"))
        if all_existing:
            return all_existing[0]
    return build_window_cache(cfg)


def extract_embeddings(model: EEGAuthenticator, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    embs, labels, qualities, indices = [], [], [], []
    with torch.no_grad():
        for x, y, q, idx in loader:
            x = x.to(device, non_blocking=True)
            emb, _ = model(x)
            embs.append(emb.cpu().numpy())
            labels.append(y.numpy())
            qualities.append(q.numpy())
            indices.append(idx.numpy())
    return np.concatenate(embs), np.concatenate(labels), np.concatenate(qualities), np.concatenate(indices)


def train_epoch(
    model: EEGAuthenticator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: dict,
    prox_state: dict[str, torch.Tensor] | None = None,
    prox_mu: float = 0.0,
) -> dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    sup_w = float(cfg["loss"].get("supcon_weight", 0.0))
    for x, y, q, _ in tqdm(loader, desc="train", leave=False):
        x, y, q = x.to(device, non_blocking=True), y.to(device, non_blocking=True), q.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"):
            emb, logits = model(x, y, q)
            loss = ce(logits, y)
            if sup_w > 0:
                loss = loss + sup_w * model.supcon(emb, y)
            if prox_state is not None and prox_mu > 0.0:
                prox = torch.zeros((), device=device)
                for name, param in model.named_parameters():
                    if param.requires_grad and name in prox_state:
                        prox = prox + torch.sum((param - prox_state[name].to(device)) ** 2)
                loss = loss + 0.5 * float(prox_mu) * prox
        scaler.scale(loss).backward()
        if cfg["train"].get("grad_clip"):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach()) * len(x)
        total += len(x)
        correct += int((logits.argmax(dim=1) == y).sum())
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}


def eval_classifier(model: EEGAuthenticator, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y, q, _ in loader:
            x, y, q = x.to(device), y.to(device), q.to(device)
            _, logits = model(x, y, q)
            loss = ce(logits, y)
            loss_sum += float(loss) * len(x)
            total += len(x)
            correct += int((logits.argmax(dim=1) == y).sum())
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}


def eval_verification_epoch(
    model: EEGAuthenticator,
    enroll_loader: DataLoader,
    probe_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    data: dict[str, np.ndarray],
) -> dict[str, float]:
    E_enroll, Y_enroll, _, enroll_indices = extract_embeddings(model, enroll_loader, device)
    E_probe, Y_probe, _, probe_indices = extract_embeddings(model, probe_loader, device)
    condition_on = str(cfg["verification"].get("condition_on", "none")).lower()
    enroll_context = None
    probe_context = None
    if condition_on == "stimulus":
        stimuli = data.get("stimulus")
        if stimuli is None:
            raise ValueError("verification.condition_on=stimulus requires stimulus labels in the cache.")
        enroll_context = stimuli[enroll_indices]
        probe_context = stimuli[probe_indices]
    bank = build_prototypes(
        E_enroll,
        Y_enroll,
        method=str(cfg["verification"]["prototype"]),
        k=int(cfg["verification"].get("kmeans_k", 3)),
        seed=int(cfg["experiment"]["seed"]),
        context=enroll_context,
    )
    scores = verification_trials(
        E_probe,
        Y_probe,
        bank,
        score_norm=str(cfg["verification"].get("score_norm", "none")),
        cohort_size=int(cfg["verification"].get("cohort_size", 100)),
        probe_context=probe_context,
    )
    return roc_metrics(scores["label"].to_numpy(), scores["score"].to_numpy(), cfg["verification"].get("target_far"))


def _loader(data: dict[str, np.ndarray], idx: np.ndarray, cfg: dict, shuffle: bool) -> DataLoader:
    ds = WindowDataset(data["X"], data["y"], data.get("quality"), idx)
    return DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        drop_last=shuffle and len(ds) >= int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def run_training(cfg: ExperimentConfig) -> dict[str, Any]:
    seed_everything(int(cfg.raw["experiment"]["seed"]), bool(cfg.raw["experiment"].get("deterministic", False)))
    out_dir = ensure_dir(cfg.results_dir)
    ensure_dir(out_dir / "checkpoints")
    cfg.save_resolved(out_dir / "config.resolved.yaml")

    cache_path = _maybe_cache(cfg)
    data = load_window_cache(cache_path)
    cfg.raw["model"]["num_subjects"] = int(len(np.unique(data["y"])))
    split = build_protocol_split(data, cfg.raw)
    assert_split_valid(split, data["y"])

    device = get_device(str(cfg.raw["experiment"].get("device", "auto")))
    samples = int(data["X"].shape[-1])
    model = EEGAuthenticator(cfg.raw, samples=samples).to(device)

    fed_result = None
    if bool(cfg.raw.get("federated", {}).get("enabled", False)):
        fed_dataset = WindowDataset(data["X"], data["y"], data.get("quality"), split.train_idx)
        fed_client_ids = data["y"][split.train_idx]
        fed_result = run_federated_training(model, fed_dataset, cfg.raw, device, fed_client_ids)
        pd.DataFrame(fed_result.history).to_csv(out_dir / "federated_history.csv", index=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.raw["train"]["lr"]), weight_decay=float(cfg.raw["train"]["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.raw["train"]["amp"]) and device.type == "cuda")

    train_loader = _loader(data, split.train_idx, cfg.raw, True)
    val_loader = _loader(data, split.validation_idx if len(split.validation_idx) else split.train_idx, cfg.raw, False)
    enroll_eval_loader = _loader(data, split.enroll_idx, cfg.raw, False)
    log_rows: list[dict[str, float]] = []
    selection_metric = str(cfg.raw["train"].get("selection_metric", "val_eer")).lower()
    best_score = float("inf")
    patience_left = int(cfg.raw["train"]["patience"])
    best_path = out_dir / "checkpoints" / "best.pt"

    with Timer() as train_timer:
        for epoch in range(1, int(cfg.raw["train"]["epochs"]) + 1):
            train_m = train_epoch(model, train_loader, optimizer, scaler, device, cfg.raw)
            val_m = eval_classifier(model, val_loader, device)
            val_bio = eval_verification_epoch(model, enroll_eval_loader, val_loader, device, cfg.raw, data)
            row = {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_acc": train_m["acc"],
                "val_loss": val_m["loss"],
                "val_acc": val_m["acc"],
                "val_eer_percent": val_bio["eer_percent"],
                "val_auc": val_bio["auc"],
                "val_threshold": val_bio["threshold"],
            }
            log_rows.append(row)
            current_score = val_bio["eer_percent"] if selection_metric == "val_eer" else val_m["loss"]
            if current_score < best_score:
                best_score = current_score
                patience_left = int(cfg.raw["train"]["patience"])
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg.raw,
                        "epoch": epoch,
                        "selection_metric": selection_metric,
                        "selection_score": best_score,
                        "val_loss": val_m["loss"],
                        "val_eer_percent": val_bio["eer_percent"],
                        "val_auc": val_bio["auc"],
                    },
                    best_path,
                )
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if not log_rows:
            val_m = eval_classifier(model, val_loader, device)
            val_bio = eval_verification_epoch(model, enroll_eval_loader, val_loader, device, cfg.raw, data)
            log_rows.append(
                {
                    "epoch": 0,
                    "train_loss": float("nan"),
                    "train_acc": float("nan"),
                    "val_loss": val_m["loss"],
                    "val_acc": val_m["acc"],
                    "val_eer_percent": val_bio["eer_percent"],
                    "val_auc": val_bio["auc"],
                    "val_threshold": val_bio["threshold"],
                }
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.raw,
                    "epoch": 0,
                    "selection_metric": selection_metric,
                    "selection_score": val_bio["eer_percent"] if selection_metric == "val_eer" else val_m["loss"],
                    "val_loss": val_m["loss"],
                    "val_eer_percent": val_bio["eer_percent"],
                    "val_auc": val_bio["auc"],
                },
                best_path,
            )

    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    enroll_loader = _loader(data, split.enroll_idx, cfg.raw, False)
    val_probe_loader = _loader(data, split.validation_idx if len(split.validation_idx) else split.test_idx, cfg.raw, False)
    test_loader = _loader(data, split.test_idx, cfg.raw, False)

    E_enroll, Y_enroll, _, enroll_indices = extract_embeddings(model, enroll_loader, device)
    E_val, Y_val, _, val_indices = extract_embeddings(model, val_probe_loader, device)
    E_test, Y_test, _, test_indices = extract_embeddings(model, test_loader, device)
    condition_on = str(cfg.raw["verification"].get("condition_on", "none")).lower()
    enroll_context = None
    val_context = None
    test_context = None
    if condition_on == "stimulus":
        stimuli = data.get("stimulus")
        if stimuli is None:
            raise ValueError("verification.condition_on=stimulus requires stimulus labels in the cache.")
        enroll_context = stimuli[enroll_indices]
        val_context = stimuli[val_indices]
        test_context = stimuli[test_indices]

    bank = build_prototypes(
        E_enroll,
        Y_enroll,
        method=str(cfg.raw["verification"]["prototype"]),
        k=int(cfg.raw["verification"].get("kmeans_k", 3)),
        seed=int(cfg.raw["experiment"]["seed"]),
        context=enroll_context,
    )
    bank.save(out_dir / "prototypes.npz")

    score_norm = str(cfg.raw["verification"].get("score_norm", "none"))
    cohort_size = int(cfg.raw["verification"].get("cohort_size", 100))
    val_scores = verification_trials(E_val, Y_val, bank, score_norm=score_norm, cohort_size=cohort_size, probe_context=val_context)
    val_metrics = roc_metrics(val_scores["label"].to_numpy(), val_scores["score"].to_numpy(), cfg.raw["verification"].get("target_far"))
    threshold = float(val_metrics["threshold"])

    adaptation_cfg = cfg.raw.get("adaptation", {})
    adapted_bank = bank
    if bool(adaptation_cfg.get("supervised_session2_prototypes", False)) and len(split.validation_idx):
        alpha = float(adaptation_cfg.get("session2_weight", 0.5))
        E_adapt = np.concatenate([E_enroll, E_val], axis=0)
        Y_adapt = np.concatenate([Y_enroll, Y_val], axis=0)
        if enroll_context is not None and val_context is not None:
            C_adapt = np.concatenate([enroll_context, val_context], axis=0)
        else:
            C_adapt = None
        # Repeat enrollment samples so alpha<0.5 keeps Session-1 prototypes dominant.
        if alpha < 0.5 and len(E_enroll):
            repeats = max(1, int(round((1.0 - alpha) / max(alpha, 1e-6))))
            E_adapt = np.concatenate([np.repeat(E_enroll, repeats, axis=0), E_val], axis=0)
            Y_adapt = np.concatenate([np.repeat(Y_enroll, repeats, axis=0), Y_val], axis=0)
            if C_adapt is not None:
                C_adapt = np.concatenate([np.repeat(enroll_context, repeats, axis=0), val_context], axis=0)
        adapted_bank = build_prototypes(
            E_adapt,
            Y_adapt,
            method=str(cfg.raw["verification"]["prototype"]),
            k=int(cfg.raw["verification"].get("kmeans_k", 3)),
            seed=int(cfg.raw["experiment"]["seed"]),
            context=C_adapt,
        )
        adapted_bank.save(out_dir / "prototypes_session2_personalized.npz")

    test_scores = verification_trials(E_test, Y_test, adapted_bank, score_norm=score_norm, cohort_size=cohort_size, probe_context=test_context)
    test_meta = pd.DataFrame(
        {
            "probe_index": np.arange(len(test_indices)),
            "source_index": test_indices,
            "subject": data["y"][test_indices],
            "session": data["session"][test_indices],
            "stimulus": data.get("stimulus", np.full(len(data["y"]), "UNKNOWN", dtype=object))[test_indices],
            "quality": data.get("quality", np.ones(len(data["y"])))[test_indices],
        }
    )
    test_scores_for_fusion = test_scores.merge(
        test_meta[["probe_index", "quality"]],
        on="probe_index",
        how="left",
    )

    fusion_results: dict[str, Any] = {}
    for method in cfg.raw["verification"]["fusion"]["methods"]:
        fused = temporal_fusion(test_scores_for_fusion, str(method), threshold=threshold)
        labels = fused["label"].to_numpy()
        scores = fused["score"].to_numpy()
        metrics = roc_metrics(labels, scores, cfg.raw["verification"].get("target_far"))
        ci_low, ci_high = bootstrap_eer_ci(labels, scores, seed=int(cfg.raw["experiment"]["seed"]))
        metrics["eer_ci95_low"] = ci_low
        metrics["eer_ci95_high"] = ci_high
        fusion_results[str(method)] = metrics
        fused.to_csv(out_dir / f"scores_{method}.csv", index=False)

    primary_scores = temporal_fusion(test_scores_for_fusion, "single", threshold=threshold)
    primary_scores.to_csv(out_dir / "scores.csv", index=False)
    val_scores.to_csv(out_dir / "validation_scores.csv", index=False)
    test_meta.to_csv(out_dir / "probe_metadata.csv", index=False)
    save_roc(primary_scores["label"].to_numpy(), primary_scores["score"].to_numpy(), out_dir)
    per_group_metrics(primary_scores, test_meta, "subject").to_csv(out_dir / "per_subject_metrics.csv", index=False)
    per_group_metrics(primary_scores, test_meta, "session").to_csv(out_dir / "per_session_metrics.csv", index=False)
    per_group_metrics(primary_scores, test_meta, "stimulus").to_csv(out_dir / "per_stimulus_metrics.csv", index=False)

    summary = {
        "experiment": cfg.raw["experiment"]["name"],
        "protocol": split.name,
        "cache_path": str(cache_path),
        "device": str(device),
        "params": count_parameters(model),
        "best_epoch": int(ckpt["epoch"]),
        "selection_metric": ckpt.get("selection_metric", "val_loss"),
        "selection_score": float(ckpt.get("selection_score", ckpt["val_loss"])),
        "best_val_loss": float(ckpt["val_loss"]),
        "best_val_eer_percent": float(ckpt.get("val_eer_percent", float("nan"))),
        "best_val_auc": float(ckpt.get("val_auc", float("nan"))),
        "train_seconds": train_timer.elapsed,
        "federated": None if fed_result is None else {
            "rounds": fed_result.rounds,
            "clients_per_round": fed_result.clients_per_round,
            "history_csv": "federated_history.csv",
        },
        "session2_prototype_personalization": bool(adaptation_cfg.get("supervised_session2_prototypes", False)),
        "validation_metrics": val_metrics,
        "test_metrics": fusion_results,
        "threshold_selected_on": cfg.raw["protocol"].get("threshold_source", "validation"),
    }
    save_json(summary, out_dir / "metrics.json")
    return summary


def run_preprocess(cfg: ExperimentConfig) -> Path:
    return build_window_cache(cfg)
