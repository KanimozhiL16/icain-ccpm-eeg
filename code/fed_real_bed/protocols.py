from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .data import normalize_session, session_set


@dataclass(frozen=True)
class ProtocolSplit:
    train_idx: np.ndarray
    enroll_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray
    name: str


def _as_upper(values: np.ndarray) -> np.ndarray:
    return np.asarray([str(v).upper() for v in values], dtype=object)


def build_protocol_split(data: dict[str, np.ndarray], cfg: dict) -> ProtocolSplit:
    name = str(cfg["protocol"]["name"]).upper()
    y = data["y"]
    sessions = np.asarray([normalize_session(s) for s in data["session"]], dtype=object)
    stimuli = _as_upper(data.get("stimulus", np.full(len(y), "UNKNOWN", dtype=object)))
    target_stimulus = str(cfg["data"].get("stimulus", "ALL")).upper()

    base_mask = np.ones(len(y), dtype=bool)
    if target_stimulus not in {"ALL", "*", ""}:
        base_mask &= stimuli == target_stimulus

    if name == "P0":
        train_sessions = {"r01", "r02"}
        test_sessions = {"r03"}
        train_idx = np.where(base_mask & np.isin(sessions, list(train_sessions)))[0]
        test_idx = np.where(base_mask & np.isin(sessions, list(test_sessions)))[0]
        tr, val = train_test_split(train_idx, test_size=0.2, stratify=y[train_idx], random_state=cfg["experiment"]["seed"])
        return ProtocolSplit(train_idx=tr, enroll_idx=train_idx, validation_idx=val, test_idx=test_idx, name=name)

    enroll_sessions = session_set(cfg["protocol"].get("enroll_sessions", ["r01"]))
    val_sessions = session_set(cfg["protocol"].get("validation_sessions", ["r02"]))
    test_sessions = session_set(cfg["protocol"].get("test_sessions", ["r03"]))

    enroll_idx = np.where(base_mask & np.isin(sessions, list(enroll_sessions)))[0]
    validation_idx = np.where(base_mask & np.isin(sessions, list(val_sessions)))[0]
    test_idx = np.where(base_mask & np.isin(sessions, list(test_sessions)))[0]

    if name == "P1":
        train_idx = enroll_idx
        both_later = np.concatenate([validation_idx, test_idx])
        return ProtocolSplit(train_idx=train_idx, enroll_idx=enroll_idx, validation_idx=validation_idx, test_idx=both_later, name=name)

    if name == "P2":
        return ProtocolSplit(train_idx=enroll_idx, enroll_idx=enroll_idx, validation_idx=validation_idx, test_idx=test_idx, name=name)

    if name == "P3":
        enroll_stimulus = "RC"
        enroll_idx = np.where((stimuli == enroll_stimulus) & np.isin(sessions, list(enroll_sessions)))[0]
        probe_mask = base_mask & (stimuli != enroll_stimulus)
        validation_idx = np.where(probe_mask & np.isin(sessions, list(val_sessions)))[0]
        test_idx = np.where(probe_mask & np.isin(sessions, list(test_sessions)))[0]
        return ProtocolSplit(train_idx=enroll_idx, enroll_idx=enroll_idx, validation_idx=validation_idx, test_idx=test_idx, name=name)

    raise ValueError(f"Unsupported protocol: {name}")


def assert_split_valid(split: ProtocolSplit, y: np.ndarray) -> None:
    if len(split.train_idx) == 0 or len(split.enroll_idx) == 0 or len(split.test_idx) == 0:
        raise ValueError(f"Protocol {split.name} produced an empty train/enroll/test split.")
    train_subjects = set(np.unique(y[split.train_idx]))
    test_subjects = set(np.unique(y[split.test_idx]))
    if train_subjects != test_subjects:
        missing = sorted(train_subjects.symmetric_difference(test_subjects))
        raise ValueError(f"Train/test subject mismatch for {split.name}: {missing}")
