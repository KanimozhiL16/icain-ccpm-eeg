from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_config
from .evaluation import PrototypeBank
from .models import EEGAuthenticator


class VerifyRequest(BaseModel):
    eeg: list[list[float]]
    claimed_id: int


app = FastAPI(title="FED-REAL-BED Real-Time EEG Authentication API")
_MODEL: EEGAuthenticator | None = None
_BANK: PrototypeBank | None = None
_CFG: dict | None = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_runtime() -> None:
    global _MODEL, _BANK, _CFG
    cfg_path = os.environ.get("FED_REAL_BED_CONFIG")
    run_dir = os.environ.get("FED_REAL_BED_RUN_DIR")
    if not cfg_path or not run_dir:
        return
    cfg = load_config(cfg_path).raw
    ckpt_path = Path(run_dir) / "checkpoints" / "best.pt"
    proto_path = Path(run_dir) / "prototypes.npz"
    if not ckpt_path.exists() or not proto_path.exists():
        return
    samples = int(cfg["data"]["target_sampling_rate"] * cfg["data"]["window_sec"])
    model = EEGAuthenticator(cfg, samples=samples).to(_DEVICE)
    ckpt = torch.load(ckpt_path, map_location=_DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _MODEL = model
    _BANK = PrototypeBank.load(proto_path)
    _CFG = cfg


@app.on_event("startup")
def startup() -> None:
    load_runtime()


@app.post("/verify")
def verify(req: VerifyRequest) -> dict[str, float | bool | int]:
    if _MODEL is None or _BANK is None:
        raise HTTPException(status_code=503, detail="Model runtime not loaded. Set FED_REAL_BED_CONFIG and FED_REAL_BED_RUN_DIR.")
    x = np.asarray(req.eeg, dtype=np.float32)
    if x.ndim != 2:
        raise HTTPException(status_code=400, detail="eeg must be [channels, samples].")
    with torch.no_grad():
        emb, _ = _MODEL(torch.tensor(x[None], dtype=torch.float32, device=_DEVICE))
    e = emb.cpu().numpy()[0]
    mask = _BANK.owners == int(req.claimed_id)
    if not np.any(mask):
        raise HTTPException(status_code=404, detail="claimed_id not enrolled.")
    score = float((e @ _BANK.vectors[mask].T).max())
    threshold = float(os.environ.get("FED_REAL_BED_THRESHOLD", "0.0"))
    return {"claimed_id": int(req.claimed_id), "score": score, "threshold": threshold, "accept": bool(score >= threshold)}

