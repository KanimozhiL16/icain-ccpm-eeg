from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluation import PrototypeBank


def recalibrate_batch_norm(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 20) -> None:
    """Update BN running statistics on calibration windows without changing weights."""
    was_training = model.training
    model.train()
    with torch.no_grad():
        for i, (x, *_rest) in enumerate(loader):
            if i >= max_batches:
                break
            model(x.to(device))
    model.train(was_training)


def prototype_ema_update(bank: PrototypeBank, emb: np.ndarray, claimed_ids: np.ndarray, alpha: float = 0.05) -> PrototypeBank:
    vectors = bank.vectors.copy()
    for e, sid in zip(emb, claimed_ids):
        mask = bank.owners == sid
        if not np.any(mask):
            continue
        best_local = np.argmax(e @ vectors[mask].T)
        global_idx = np.where(mask)[0][best_local]
        vectors[global_idx] = (1.0 - alpha) * vectors[global_idx] + alpha * e
        vectors[global_idx] /= np.linalg.norm(vectors[global_idx]) + 1e-12
    return PrototypeBank(vectors=vectors.astype(np.float32), owners=bank.owners.copy())

