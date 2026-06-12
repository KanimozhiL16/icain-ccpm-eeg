from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import WindowDataset
from .models import EEGAuthenticator


@dataclass
class FederatedResult:
    rounds: int
    clients_per_round: int
    history: list[dict[str, float]]


def _state_average(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    out = {}
    total = float(sum(weights))
    for key in states[0]:
        first = states[0][key].detach().cpu()
        if not torch.is_floating_point(first):
            out[key] = first.clone()
            continue
        out[key] = sum(state[key].detach().cpu() * (w / total) for state, w in zip(states, weights))
    return out


def run_federated_training(
    global_model: EEGAuthenticator,
    dataset: WindowDataset,
    cfg: dict,
    device: torch.device,
    client_ids: np.ndarray,
) -> FederatedResult:
    """Subject-as-client FedAvg/FedProx skeleton for privacy-preserving BED experiments."""
    from .training import train_epoch

    fed_cfg = cfg["federated"]
    if not fed_cfg.get("enabled", False):
        return FederatedResult(rounds=0, clients_per_round=0, history=[])

    rng = np.random.default_rng(int(cfg["experiment"]["seed"]))
    unique_clients = np.unique(client_ids)
    frac = float(fed_cfg.get("client_fraction", 1.0))
    clients_per_round = max(1, int(round(len(unique_clients) * frac)))
    history: list[dict[str, float]] = []

    for rnd in range(1, int(fed_cfg["rounds"]) + 1):
        selected = rng.choice(unique_clients, size=clients_per_round, replace=False)
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        prox_state = {k: v.detach().cpu().clone() for k, v in global_model.named_parameters()}
        prox_mu = float(fed_cfg.get("fedprox_mu", 0.0)) if str(fed_cfg.get("method", "fedavg")).lower() == "fedprox" else 0.0
        states, weights = [], []
        for cid in selected:
            idx = np.where(client_ids == cid)[0]
            local_model = deepcopy(global_model).to(device)
            opt = torch.optim.AdamW(local_model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
            scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")
            loader = DataLoader(
                Subset(dataset, idx.tolist()),
                batch_size=int(cfg["train"]["batch_size"]),
                shuffle=True,
                drop_last=False,
                num_workers=int(cfg["train"].get("num_workers", 0)),
                pin_memory=torch.cuda.is_available(),
            )
            metrics = {}
            for _ in range(int(fed_cfg["local_epochs"])):
                metrics = train_epoch(local_model, loader, opt, scaler, device, cfg, prox_state=prox_state, prox_mu=prox_mu)
            states.append({k: v.detach().cpu() for k, v in local_model.state_dict().items()})
            weights.append(float(len(idx)))
            history.append({"round": float(rnd), "client": float(cid), **metrics})
        averaged = _state_average(states, weights)
        for key, value in global_state.items():
            if key not in averaged:
                averaged[key] = value
        global_model.load_state_dict(averaged)
    return FederatedResult(rounds=int(fed_cfg["rounds"]), clients_per_round=clients_per_round, history=history)
