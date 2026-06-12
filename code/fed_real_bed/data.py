from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import signal
import torch
from tqdm import tqdm

from .config import ExperimentConfig
from .utils import ensure_dir


DEFAULT_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
BED_PARSED_COLUMNS = ["COUNTER", "INTERPOLATED", "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6", "F4", "UNIX_TIMESTAMP"]
SUPPORTED_RAW_EXT = {".csv", ".txt", ".mat", ".npz", ".npy", ".edf", ".bdf"}


@dataclass(frozen=True)
class RawRecord:
    path: Path
    subject: str
    session: str
    stimulus: str


def normalize_session(value: object) -> str:
    s = str(value).strip().lower()
    mapping = {
        "1": "r01",
        "s1": "r01",
        "session1": "r01",
        "session_1": "r01",
        "r1": "r01",
        "run1": "r01",
        "2": "r02",
        "s2": "r02",
        "session2": "r02",
        "session_2": "r02",
        "r2": "r02",
        "run2": "r02",
        "3": "r03",
        "s3": "r03",
        "session3": "r03",
        "session_3": "r03",
        "r3": "r03",
        "run3": "r03",
    }
    return mapping.get(s, s)


def session_set(values: Iterable[object]) -> set[str]:
    return {normalize_session(v) for v in values}


def infer_record(path: Path) -> RawRecord:
    text = " ".join([p.name for p in path.parents[:3]] + [path.stem]).lower()
    bed_match = re.search(r"s(\d{1,2})_s([123])", path.stem.lower())
    if bed_match:
        return RawRecord(
            path=path,
            subject=f"s{bed_match.group(1).zfill(2)}",
            session=f"r0{bed_match.group(2)}",
            stimulus="SESSION",
        )
    subj_match = re.search(r"(?:subject|subj|user|participant|p)[_\-\s]?(\d{1,3})", text)
    sess_match = re.search(r"(?:session|sess|run|r|s)[_\-\s]?0?([123])", text)
    subject = f"s{subj_match.group(1).zfill(2)}" if subj_match else "unknown"
    session = f"r0{sess_match.group(1)}" if sess_match else "unknown"
    stimulus = "unknown"
    known = ["rc", "ro", "vc1", "vc7", "vf1", "vf10", "math", "baseline", "vep", "affective", "cognitive"]
    for token in known:
        if re.search(rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)", text):
            stimulus = token.upper()
            break
    return RawRecord(path=path, subject=subject, session=session, stimulus=stimulus)


def _mat_struct_get(obj: object, field: str, default: object = None) -> object:
    return getattr(obj, field, default)


def normalize_bed_event(info: object) -> str:
    stim = str(_mat_struct_get(info, "STIMULI", "UNKNOWN")).upper()
    if stim == "EYES":
        state = str(_mat_struct_get(info, "OPEN_CLOSED", "")).upper()
        if state == "CLOSED":
            return "RC"
        if state == "OPEN":
            return "RO"
    if stim in {"SSVEP", "SSVEPC"}:
        freq = str(_mat_struct_get(info, "FREQ", "")).upper().replace("HZ", "")
        return f"{stim}{freq}".strip()
    if stim == "REST":
        return "REST"
    if stim == "COGNITIVE":
        return "COGNITIVE"
    if stim == "IMAGE":
        return "IMAGE"
    return stim


def discover_manifest(cfg: ExperimentConfig) -> pd.DataFrame:
    manifest_csv = cfg.raw["paths"].get("manifest_csv")
    if manifest_csv:
        df = pd.read_csv(manifest_csv)
        required = {"path", "subject", "session", "stimulus"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest CSV missing columns: {sorted(missing)}")
        df["path"] = df["path"].map(lambda p: str(Path(p).expanduser()))
    else:
        root = Path(cfg.raw["paths"]["bed_raw_root"]).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"BED raw root not found: {root}")
        records = [infer_record(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_RAW_EXT]
        if not records:
            raise FileNotFoundError(f"No supported raw EEG files found under {root}")
        df = pd.DataFrame([r.__dict__ for r in records])
        df["path"] = df["path"].astype(str)
    df["session"] = df["session"].map(normalize_session)
    df["stimulus"] = df["stimulus"].astype(str).str.upper()
    df["subject"] = df["subject"].astype(str)
    return df.sort_values(["subject", "session", "stimulus", "path"]).reset_index(drop=True)


def _select_numeric_columns(df: pd.DataFrame, channels: list[str]) -> np.ndarray:
    cols = [c for c in channels if c in df.columns]
    if len(cols) == len(channels):
        return df[cols].to_numpy(dtype=np.float32).T
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < len(channels):
        raise ValueError(f"Expected at least {len(channels)} numeric EEG columns, found {numeric.shape[1]}")
    return numeric.iloc[:, : len(channels)].to_numpy(dtype=np.float32).T


def load_raw_signal(path: str | Path, channels: list[str]) -> np.ndarray:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".csv", ".txt"}:
        sep = None if ext == ".csv" else r"\s+"
        df = pd.read_csv(p, sep=sep, engine="python")
        return _select_numeric_columns(df, channels)
    if ext == ".npy":
        x = np.load(p).astype(np.float32)
    elif ext == ".npz":
        npz = np.load(p, allow_pickle=True)
        key = "X" if "X" in npz.files else npz.files[0]
        x = npz[key].astype(np.float32)
    elif ext == ".mat":
        mat = scipy_io.loadmat(p)
        candidates = [v for k, v in mat.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2]
        if not candidates:
            raise ValueError(f"No 2D numeric matrix found in MAT file: {p}")
        x = max(candidates, key=lambda a: a.size).astype(np.float32)
    elif ext in {".edf", ".bdf"}:
        try:
            import mne
        except ImportError as exc:
            raise ImportError("Install with `pip install -e .[raw-edf]` to read EDF/BDF files.") from exc
        raw = mne.io.read_raw(str(p), preload=True, verbose=False)
        picks = [ch for ch in channels if ch in raw.ch_names]
        data = raw.get_data(picks=picks or None).astype(np.float32)
        return data[: len(channels)]
    else:
        raise ValueError(f"Unsupported raw EEG extension: {p.suffix}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D signal in {p}, got shape {x.shape}")
    if x.shape[0] != len(channels) and x.shape[1] == len(channels):
        x = x.T
    if x.shape[0] < len(channels):
        raise ValueError(f"Expected at least {len(channels)} channels in {p}, got {x.shape}")
    return x[: len(channels)].astype(np.float32)


def is_bed_parsed_mat(path: str | Path) -> bool:
    p = Path(path)
    if p.suffix.lower() != ".mat":
        return False
    try:
        mat = scipy_io.loadmat(p, variable_names=["recording", "events"], squeeze_me=True, struct_as_record=False)
    except Exception:
        return False
    return "recording" in mat and "events" in mat


def iter_bed_parsed_events(path: str | Path, channels: list[str]) -> list[tuple[str, np.ndarray]]:
    """Return event-specific EEG segments from BED RAW_PARSED/sN_sM.mat files.

    BED parsed files store `recording` as time x 17 and `events` as start/end UNIX
    timestamps plus metadata. This function maps the 14 named EPOC+ channels into
    the repository channel order and extracts only the timestamp-bounded event.
    """
    mat = scipy_io.loadmat(path, squeeze_me=True, struct_as_record=False)
    recording = np.asarray(mat["recording"], dtype=np.float32)
    events = np.asarray(mat["events"], dtype=object)
    col_index = {name: i for i, name in enumerate(BED_PARSED_COLUMNS)}
    eeg_cols = [col_index[ch] for ch in channels]
    timestamps = recording[:, col_index["UNIX_TIMESTAMP"]]
    out: list[tuple[str, np.ndarray]] = []
    for row in events:
        start, end, info = float(row[0]), float(row[1]), row[2]
        mask = (timestamps >= start) & (timestamps <= end)
        if np.count_nonzero(mask) < 16:
            continue
        stimulus = normalize_bed_event(info)
        segment = recording[mask][:, eeg_cols].T.astype(np.float32)
        out.append((stimulus, segment))
    return out


def preprocess_signal(x: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    fs_in = float(cfg.raw["data"]["input_sampling_rate"])
    fs_out = float(cfg.raw["data"]["target_sampling_rate"])
    band_low, band_high = cfg.raw["data"]["bandpass_hz"]
    sos = signal.butter(4, [band_low, band_high], btype="bandpass", fs=fs_in, output="sos")
    y = signal.sosfiltfilt(sos, x, axis=1).astype(np.float32)
    notch = cfg.raw["data"].get("notch_hz")
    if notch:
        b, a = signal.iirnotch(float(notch), Q=30.0, fs=fs_in)
        y = signal.filtfilt(b, a, y, axis=1).astype(np.float32)
    if fs_out != fs_in:
        target_len = int(round(y.shape[1] * fs_out / fs_in))
        y = signal.resample(y, target_len, axis=1).astype(np.float32)
    if cfg.raw["data"].get("zscore") == "channel":
        y = (y - y.mean(axis=1, keepdims=True)) / (y.std(axis=1, keepdims=True) + 1e-6)
    return y.astype(np.float32)


def quality_score(window: np.ndarray, cfg: ExperimentConfig) -> tuple[float, dict[str, float]]:
    qcfg = cfg.raw["data"]["quality"]
    abs_bad = np.max(np.abs(window), axis=1) > float(qcfg["max_abs_uv"])
    std = np.std(window, axis=1)
    std_bad = (std < float(qcfg["min_std_uv"])) | (std > float(qcfg["max_std_uv"])) | (std < float(qcfg["flatline_std_uv"]))
    bad_channel_frac = float(np.mean(abs_bad | std_bad))
    quality = max(0.0, 1.0 - bad_channel_frac / max(float(qcfg["max_bad_channel_frac"]), 1e-6))
    return float(min(1.0, quality)), {"bad_channel_frac": bad_channel_frac, "std_mean": float(std.mean())}


def make_windows(signal_x: np.ndarray, cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    fs = int(cfg.raw["data"]["target_sampling_rate"])
    win = int(round(cfg.raw["data"]["window_sec"] * fs))
    step = int(round(cfg.raw["data"]["step_sec"] * fs))
    xs: list[np.ndarray] = []
    qs: list[float] = []
    for start in range(0, signal_x.shape[1] - win + 1, step):
        w = signal_x[:, start : start + win]
        q, _ = quality_score(w, cfg)
        if q >= float(cfg.raw["data"]["quality"]["min_quality"]):
            xs.append(w.astype(np.float32))
            qs.append(q)
    if not xs:
        return np.empty((0, signal_x.shape[0], win), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(xs), np.asarray(qs, dtype=np.float32)


def build_window_cache(cfg: ExperimentConfig) -> Path:
    cache_dir = ensure_dir(cfg.cache_dir)
    stimulus_name = str(cfg.raw["data"].get("stimulus", "all")).lower().replace("*", "all")
    out = cache_dir / f"bed_windows_{stimulus_name}_{cfg.raw['data']['target_sampling_rate']}hz_{cfg.raw['data']['window_sec']}s.npz"
    manifest = discover_manifest(cfg)
    stimulus = str(cfg.raw["data"].get("stimulus", "all")).upper()
    if stimulus not in {"ALL", "*", ""}:
        manifest = manifest[(manifest["stimulus"].str.upper() == stimulus) | (manifest["stimulus"].str.upper() == "SESSION")]
    if manifest.empty:
        raise ValueError(f"No records left after stimulus filter: {stimulus}")

    subjects = sorted(manifest["subject"].unique())
    subject_to_id = {s: i for i, s in enumerate(subjects)}
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sessions: list[np.ndarray] = []
    stimuli: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    channels = cfg.raw["data"].get("channels", DEFAULT_CHANNELS)
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Preprocess raw BED"):
        if is_bed_parsed_mat(row.path):
            segments = iter_bed_parsed_events(row.path, channels)
        else:
            segments = [(str(row.stimulus).upper(), load_raw_signal(row.path, channels))]
        for event_stimulus, raw in segments:
            if stimulus not in {"ALL", "*", ""} and event_stimulus.upper() != stimulus:
                continue
            proc = preprocess_signal(raw, cfg)
            xw, qw = make_windows(proc, cfg)
            if len(xw) == 0:
                continue
            Xs.append(xw)
            ys.append(np.full(len(xw), subject_to_id[str(row.subject)], dtype=np.int64))
            sessions.append(np.full(len(xw), normalize_session(row.session), dtype=object))
            stimuli.append(np.full(len(xw), event_stimulus.upper(), dtype=object))
            qualities.append(qw)
    if not Xs:
        raise ValueError("No valid EEG windows were produced. Check raw format and quality thresholds.")
    y_raw = np.concatenate(ys)
    session_raw = np.concatenate(sessions)
    stimulus_raw = np.concatenate(stimuli)
    quality_raw = np.concatenate(qualities).astype(np.float32)
    lengths = np.asarray([len(x) for x in Xs], dtype=np.int64)
    source_raw = np.repeat(np.arange(len(Xs)), lengths)
    offset_raw = np.concatenate([np.arange(n, dtype=np.int64) for n in lengths])

    selected = np.arange(len(y_raw))
    max_per_group = cfg.raw["data"].get("max_windows_per_subject_session_stimulus")
    if max_per_group:
        rng = np.random.default_rng(int(cfg.raw["experiment"]["seed"]))
        keep: list[np.ndarray] = []
        df_groups = pd.DataFrame({"idx": selected, "y": y_raw, "session": session_raw, "stimulus": stimulus_raw})
        for _, grp in df_groups.groupby(["y", "session", "stimulus"], sort=False):
            values = grp["idx"].to_numpy()
            if len(values) > int(max_per_group):
                values = rng.choice(values, size=int(max_per_group), replace=False)
            keep.append(np.sort(values))
        selected = np.sort(np.concatenate(keep))

    X = np.empty((len(selected), Xs[0].shape[1], Xs[0].shape[2]), dtype=np.float32)
    for out_i, raw_i in enumerate(selected):
        X[out_i] = Xs[source_raw[raw_i]][offset_raw[raw_i]]
    y = y_raw[selected]
    session_arr = session_raw[selected]
    stimulus_arr = stimulus_raw[selected]
    quality_arr = quality_raw[selected]
    np.savez_compressed(
        out,
        X=X,
        y=y,
        session=session_arr,
        stimulus=stimulus_arr,
        quality=quality_arr,
        subjects=np.asarray(subjects, dtype=object),
        channels=np.asarray(cfg.raw["data"].get("channels", DEFAULT_CHANNELS), dtype=object),
    )
    return out


def load_window_cache(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


class WindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quality: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        idx = np.arange(len(X)) if indices is None else indices
        self.X = torch.tensor(X[idx], dtype=torch.float32)
        self.y = torch.tensor(y[idx], dtype=torch.long)
        q = np.ones(len(X), dtype=np.float32) if quality is None else quality.astype(np.float32)
        self.quality = torch.tensor(q[idx], dtype=torch.float32)
        self.indices = torch.tensor(idx, dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self.X))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i], self.quality[i], self.indices[i]
