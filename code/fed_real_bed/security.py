from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CancellableTemplate:
    digest: str
    salt: str
    dimension: int


def cancellable_hash(embedding: np.ndarray, salt: str, bits: int = 128) -> CancellableTemplate:
    """Quantize the sign pattern and hash it for cancellable prototype storage."""
    emb = embedding.astype(np.float32)
    signs = (emb[:bits] >= 0).astype(np.uint8)
    packed = np.packbits(signs).tobytes()
    digest = hmac.new(salt.encode("utf-8"), packed, hashlib.sha256).hexdigest()
    return CancellableTemplate(digest=digest, salt=salt, dimension=int(len(emb)))


def additive_noise_attack(x: np.ndarray, snr_db: float, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    power = np.mean(x**2)
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape)
    return (x + noise).astype(np.float32)


def replay_attack_shift(x: np.ndarray, shift_samples: int) -> np.ndarray:
    return np.roll(x, shift=shift_samples, axis=-1).astype(np.float32)

