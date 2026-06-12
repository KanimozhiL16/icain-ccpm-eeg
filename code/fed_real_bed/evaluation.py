from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import auc, roc_curve

from .utils import l2_normalize


@dataclass(frozen=True)
class PrototypeBank:
    vectors: np.ndarray
    owners: np.ndarray
    contexts: np.ndarray | None = None

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {"vectors": self.vectors, "owners": self.owners}
        if self.contexts is not None:
            payload["contexts"] = self.contexts
        np.savez_compressed(p, **payload)

    @staticmethod
    def load(path: str | Path) -> "PrototypeBank":
        data = np.load(path, allow_pickle=True)
        contexts = data["contexts"] if "contexts" in data.files else None
        return PrototypeBank(vectors=data["vectors"], owners=data["owners"], contexts=contexts)


def build_prototypes(
    emb: np.ndarray,
    y: np.ndarray,
    method: str = "mean",
    k: int = 3,
    seed: int = 2026,
    context: np.ndarray | None = None,
) -> PrototypeBank:
    emb = l2_normalize(emb.astype(np.float32))
    vectors: list[np.ndarray] = []
    owners: list[int] = []
    contexts: list[str] = []
    if context is None:
        group_keys = [(sid, None) for sid in sorted(np.unique(y))]
    else:
        context = np.asarray([str(c).upper() for c in context], dtype=object)
        group_keys = []
        for sid in sorted(np.unique(y)):
            for ctx in sorted(np.unique(context[y == sid])):
                group_keys.append((sid, str(ctx)))

    for sid, ctx in group_keys:
        mask = y == sid if ctx is None else ((y == sid) & (context == ctx))
        Ei = emb[mask]
        if len(Ei) == 0:
            continue
        if method in {"mean", "ema"} or len(Ei) < 2:
            centers = [Ei.mean(axis=0)]
        elif method in {"kmeans", "subcenter"}:
            k_use = min(int(k), len(Ei))
            km = KMeans(n_clusters=k_use, random_state=seed, n_init=10)
            km.fit(Ei)
            centers = list(km.cluster_centers_)
        else:
            raise ValueError(f"Unsupported prototype method: {method}")
        for center in centers:
            vectors.append(center.astype(np.float32))
            owners.append(int(sid))
            if ctx is not None:
                contexts.append(ctx)
    if not vectors:
        raise ValueError("No enrollment prototypes were built.")
    ctx_array = np.asarray(contexts, dtype=object) if contexts else None
    return PrototypeBank(vectors=l2_normalize(np.stack(vectors)), owners=np.asarray(owners, dtype=np.int64), contexts=ctx_array)


def adaptive_snorm(scores_by_proto: np.ndarray, cohort_size: int = 100) -> np.ndarray:
    k = min(cohort_size, scores_by_proto.shape[1])
    top = np.sort(scores_by_proto, axis=1)[:, -k:]
    mu = top.mean(axis=1, keepdims=True)
    sigma = top.std(axis=1, keepdims=True) + 1e-6
    return (scores_by_proto - mu) / sigma


def _normalize_row(scores: np.ndarray, cohort_size: int) -> np.ndarray:
    k = min(cohort_size, len(scores))
    top = np.sort(scores)[-k:]
    return (scores - top.mean()) / (top.std() + 1e-6)


def verification_trials(
    probe_emb: np.ndarray,
    probe_y: np.ndarray,
    bank: PrototypeBank,
    score_norm: str = "none",
    cohort_size: int = 100,
    probe_context: np.ndarray | None = None,
) -> pd.DataFrame:
    E = l2_normalize(probe_emb.astype(np.float32))
    P = l2_normalize(bank.vectors.astype(np.float32))
    sim = E @ P.T
    score_norm = str(score_norm).lower()
    contexts = None
    if bank.contexts is not None and probe_context is not None:
        contexts = np.asarray([str(c).upper() for c in probe_context], dtype=object)
    if score_norm == "adaptive_snorm" and contexts is None:
        sim = adaptive_snorm(sim, cohort_size)

    rows: list[dict[str, Any]] = []
    subjects = sorted(np.unique(bank.owners))
    for i, true_id in enumerate(probe_y):
        ctx = contexts[i] if contexts is not None else None
        row_sim = sim[i]
        if score_norm == "adaptive_snorm" and ctx is not None:
            eligible = bank.contexts == ctx
            cohort = row_sim[eligible] if np.any(eligible) else row_sim
            top = np.sort(cohort)[-min(cohort_size, len(cohort)) :]
            row_sim = (row_sim - top.mean()) / (top.std() + 1e-6)
        for sid in subjects:
            mask = bank.owners == sid
            if ctx is not None:
                ctx_mask = mask & (bank.contexts == ctx)
                if np.any(ctx_mask):
                    mask = ctx_mask
            score = float(row_sim[mask].max())
            if score_norm in {"cohort_margin", "ccpm", "adaptive_snorm_margin"}:
                other_mask = bank.owners != sid
                if ctx is not None and bank.contexts is not None:
                    other_ctx_mask = other_mask & (bank.contexts == ctx)
                    if np.any(other_ctx_mask):
                        other_mask = other_ctx_mask
                impostor_reference = float(row_sim[other_mask].max()) if np.any(other_mask) else 0.0
                score = score - impostor_reference
            row = {"probe_index": i, "claimed_id": int(sid), "true_id": int(true_id), "score": score, "label": int(sid == true_id)}
            if ctx is not None:
                row["context"] = str(ctx)
            rows.append(row)
    return pd.DataFrame(rows)


def roc_metrics(labels: np.ndarray, scores: np.ndarray, target_far: list[float] | None = None) -> dict[str, Any]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    out: dict[str, Any] = {
        "eer": eer,
        "eer_percent": eer * 100.0,
        "auc": roc_auc,
        "threshold": threshold,
        "far_at_threshold": float(fpr[idx]),
        "frr_at_threshold": float(fnr[idx]),
    }
    for far in target_far or [0.01, 0.05, 0.10]:
        valid = np.where(fpr <= far)[0]
        tar = float(tpr[valid[-1]]) if len(valid) else 0.0
        out[f"tar_at_far_{far:.3f}"] = tar
    return out


def bootstrap_eer_ci(labels: np.ndarray, scores: np.ndarray, seed: int = 2026, n: int = 200) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    idx = np.arange(len(labels))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(labels[b])) < 2:
            continue
        values.append(roc_metrics(labels[b], scores[b])["eer_percent"])
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def temporal_fusion(scores_df: pd.DataFrame, method: str, threshold: float | None = None) -> pd.DataFrame:
    if method == "single":
        return scores_df.copy()
    if method.startswith("mean_"):
        win = int(method.split("_")[1])
        out = scores_df.copy()
        group_cols = ["claimed_id"]
        if "context" in out.columns:
            group_cols.append("context")
        out["score"] = (
            out.sort_values("probe_index")
            .groupby(group_cols)["score"]
            .transform(lambda s: s.rolling(win, min_periods=1).mean())
        )
        return out
    if method.startswith("qmean_") or method.startswith("quality_mean_"):
        win = int(method.split("_")[-1])
        if "quality" not in scores_df.columns:
            return temporal_fusion(scores_df, f"mean_{win}", threshold)
        out = scores_df.copy()
        group_cols = ["claimed_id"]
        if "context" in out.columns:
            group_cols.append("context")
        pieces = []
        for _, grp in out.sort_values("probe_index").groupby(group_cols, sort=False):
            quality = grp["quality"].astype(float).clip(lower=1e-3)
            weighted = grp["score"].astype(float) * quality
            numerator = weighted.rolling(win, min_periods=1).sum()
            denominator = quality.rolling(win, min_periods=1).sum().clip(lower=1e-6)
            pieces.append(pd.Series((numerator / denominator).to_numpy(), index=grp.index))
        fused = pd.concat(pieces).sort_index()
        out.loc[fused.index, "score"] = fused
        return out
    if method.startswith("majority_"):
        if threshold is None:
            raise ValueError("majority fusion requires a threshold")
        win = int(method.split("_")[1])
        out = scores_df.copy()
        accept = (out["score"].to_numpy() >= threshold).astype(float)
        out["_accept"] = accept
        group_cols = ["claimed_id"]
        if "context" in out.columns:
            group_cols.append("context")
        out["score"] = (
            out.sort_values("probe_index")
            .groupby(group_cols)["_accept"]
            .transform(lambda s: s.rolling(win, min_periods=1).mean())
        )
        out = out.drop(columns=["_accept"])
        return out
    raise ValueError(f"Unsupported fusion method: {method}")


def per_group_metrics(scores: pd.DataFrame, meta: pd.DataFrame, group_col: str) -> pd.DataFrame:
    merged = scores.merge(meta[["probe_index", group_col]], on="probe_index", how="left")
    rows = []
    for value, grp in merged.groupby(group_col):
        if grp["label"].nunique() < 2:
            continue
        m = roc_metrics(grp["label"].to_numpy(), grp["score"].to_numpy())
        rows.append({group_col: value, **m})
    return pd.DataFrame(rows)


def save_roc(labels: np.ndarray, scores: np.ndarray, out_dir: Path) -> None:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(out_dir / "roc.csv", index=False)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Accept Rate")
    plt.ylabel("True Accept Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc.png", dpi=200)
    plt.close()
