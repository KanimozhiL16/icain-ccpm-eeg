#!/usr/bin/env python3
"""
Stage 1 for the Cueless cross-dataset experiment (R1.4b).
Downloads the Cueless EEG imagined-speech dataset (HF: Alidr79/cueless_EEG_subject_identification),
prints RAW + PREPROCESSED statistics, and builds a leakage-free window cache in the
SAME .npz format as the BED pipeline (keys: X, y, session, stimulus, quality, subjects, channels).

Run on Brev from the repo root (the folder with the fed_real_bed/ package):
    pip install "mne>=1.6" huggingface_hub --quiet
    python cueless_prep.py --out artifacts/cache --channels bed14 --rate 128 --win 2.0

Outputs:
    artifacts/cache/cueless_windows_<chan>_<rate>hz_<win>s.npz    (the window cache)
    artifacts/cueless_stats/raw_stats.csv                          (per subject/session raw stats)
    artifacts/cueless_stats/preprocessed_stats.csv                (per subject/session window counts)
    artifacts/cueless_stats/summary.json                          (headline dataset statistics)

Nothing here trains a model; it only prepares data and reports statistics.
Paste the printed RAW STATS and PREPROCESSED STATS blocks back, and we build Stage 2 (train + CCPM eval + figures/tables).
"""
from __future__ import annotations
import argparse, json, glob, os, re, sys
from pathlib import Path
import numpy as np

BED14 = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]

def log(*a): print(*a, flush=True)

def download(local_dir: str) -> str:
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id="Alidr79/cueless_EEG_subject_identification",
                          repo_type="dataset", local_dir=local_dir)
    log(f"[download] dataset at {p}")
    return p

def find_fif(root: str) -> list[str]:
    # prefer the BIDS 'derivatives/preprocessed_eeg' tree; fall back to any .fif
    pref = sorted(glob.glob(os.path.join(root, "**", "derivatives", "preprocessed_eeg", "**", "*.fif"), recursive=True))
    if pref:
        return pref
    return sorted(glob.glob(os.path.join(root, "**", "*.fif"), recursive=True))

def parse_ids(path: str) -> tuple[int|None, int|None]:
    s = re.search(r"sub-?(\d+)", path); e = re.search(r"ses-?(\d+)", path)
    return (int(s.group(1)) if s else None, int(e.group(1)) if e else None)

def read_fif(path: str):
    """Return (data[n_trials,n_ch,n_times], ch_names, sfreq, labels[list])."""
    import mne
    mne.set_log_level("ERROR")
    try:
        ep = mne.read_epochs(path, preload=True)
        data = ep.get_data(copy=True)
        labels = None
        if ep.metadata is not None:
            for col in ep.metadata.columns:
                if "word" in col.lower() or "key" in col.lower() or "label" in col.lower():
                    labels = [str(v) for v in ep.metadata[col].tolist()]; break
        if labels is None and len(ep.event_id):
            inv = {v:k for k,v in ep.event_id.items()}
            labels = [inv.get(e, "W") for e in ep.events[:,2]]
        if labels is None:
            labels = ["W"]*len(data)
        return data, ep.ch_names, float(ep.info["sfreq"]), labels
    except Exception:
        raw = mne.io.read_raw_fif(path, preload=True)
        d = raw.get_data()[None]  # 1 "trial" = whole recording
        return d, raw.ch_names, float(raw.info["sfreq"]), ["W"]

def bandpass_notch_resample(x, sf_in, sf_out, low=1.0, high=45.0, notch=50.0):
    from scipy import signal
    x = np.asarray(x, dtype=np.float64)
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sf_in, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=-1)
    b, a = signal.iirnotch(notch, Q=30.0, fs=sf_in); x = signal.filtfilt(b, a, x, axis=-1)
    if abs(sf_out - sf_in) > 1e-6:
        n = int(round(x.shape[-1]*sf_out/sf_in)); x = signal.resample(x, n, axis=-1)
    return x.astype(np.float32)

def quality(w):  # simple amplitude/flatline gate -> [0,1]
    std = w.std(axis=1); bad = (std < 1e-3) | (np.max(np.abs(w),axis=1) > 500.0)
    return float(1.0 - bad.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="cueless_EEG_data")
    ap.add_argument("--out", default="artifacts/cache")
    ap.add_argument("--stats", default="artifacts/cueless_stats")
    ap.add_argument("--channels", default="bed14", choices=["bed14","all"])
    ap.add_argument("--rate", type=float, default=128.0)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--max-per-group", type=int, default=80)
    ap.add_argument("--min-quality", type=float, default=0.35)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True); Path(args.stats).mkdir(parents=True, exist_ok=True)

    root = args.data_dir if args.skip_download else download(args.data_dir)
    fifs = find_fif(root)
    if not fifs:
        sys.exit(f"No .fif files found under {root}. Check the download / path.")
    log(f"[scan] {len(fifs)} .fif files")

    # ---- discover channels + sfreq from first file ----
    d0, ch0, sf0, _ = read_fif(fifs[0])
    log(f"[probe] first file: trials={d0.shape[0]} ch={d0.shape[1]} times={d0.shape[2]} sfreq={sf0}")
    if args.channels == "bed14":
        present = [c for c in BED14 if c in ch0]
        log(f"[channels] BED-14 present in Cueless montage: {len(present)}/14 -> {present}")
        sel = present if len(present) >= 10 else ch0  # fall back to all if too few match
        if sel is ch0: log("[channels] <10 BED channels matched; using ALL channels (config must set n_channels).")
    else:
        sel = ch0
    ch_idx_cache: dict = {}

    win = int(round(args.win*args.rate))
    Xs, ys, sess, stim, qual = [], [], [], [], []
    raw_rows, subj_set = [], set()
    from collections import defaultdict
    group_count = defaultdict(int)

    for f in fifs:
        sub, ses = parse_ids(f)
        if sub is None:
            continue
        data, chn, sf, labels = read_fif(f)
        subj_set.add(sub)
        raw_rows.append({"subject":sub,"session":ses,"trials":int(data.shape[0]),
                         "channels":int(data.shape[1]),"sfreq":sf,
                         "duration_s":round(data.shape[2]/sf,2)})
        # channel picks
        key = tuple(chn)
        if key not in ch_idx_cache:
            ch_idx_cache[key] = [chn.index(c) for c in sel if c in chn]
        pick = ch_idx_cache[key]
        if len(pick) < len(sel):
            continue
        for ti in range(data.shape[0]):
            trial = data[ti, pick, :]                       # (C, T)
            trial = bandpass_notch_resample(trial, sf, args.rate)
            # per-channel z-score
            trial = (trial - trial.mean(axis=1, keepdims=True)) / (trial.std(axis=1, keepdims=True)+1e-6)
            word = str(labels[ti]).upper() if ti < len(labels) else "W"
            for s in range(0, trial.shape[1]-win+1, win):   # non-overlapping windows
                w = trial[:, s:s+win].astype(np.float32)
                q = quality(w)
                if q < args.min_quality:
                    continue
                gk = (sub, ses, word)
                if group_count[gk] >= args.max_per_group:
                    break
                group_count[gk]+=1
                Xs.append(w); ys.append(sub); sess.append(f"ses-{ses:02d}"); stim.append(word); qual.append(q)

    if not Xs:
        sys.exit("No windows produced. Inspect the printed probe line and adjust --channels/--win.")
    X = np.stack(Xs).astype(np.float32)
    y_raw = np.asarray(ys); subs = sorted(set(y_raw.tolist()))
    remap = {s:i for i,s in enumerate(subs)}
    y = np.asarray([remap[v] for v in y_raw], dtype=np.int64)
    session = np.asarray(sess, dtype=object); stimulus = np.asarray(stim, dtype=object)
    qv = np.asarray(qual, dtype=np.float32)
    chan_tag = "bed14" if args.channels=="bed14" and len(sel)==14 else f"{X.shape[1]}ch"
    out = Path(args.out)/f"cueless_windows_{chan_tag}_{int(args.rate)}hz_{args.win}s.npz"
    np.savez_compressed(out, X=X, y=y, session=session, stimulus=stimulus, quality=qv,
                        subjects=np.asarray([f"sub-{s:02d}" for s in subs],dtype=object),
                        channels=np.asarray(sel,dtype=object))

    # ---- statistics ----
    import pandas as pd
    raw_df = pd.DataFrame(raw_rows).sort_values(["subject","session"])
    raw_df.to_csv(Path(args.stats)/"raw_stats.csv", index=False)
    pp = pd.DataFrame({"subject":y, "session":session, "word":stimulus})
    per = pp.groupby(["subject","session"]).size().reset_index(name="windows")
    per.to_csv(Path(args.stats)/"preprocessed_stats.csv", index=False)
    summary = {
        "n_subjects": len(subs), "n_sessions": int(pd.Series([r["session"] for r in raw_rows]).nunique()),
        "n_fif_files": len(fifs), "raw_channels": int(d0.shape[1]), "raw_sfreq": sf0,
        "used_channels": sel, "n_used_channels": len(sel),
        "target_rate": args.rate, "window_s": args.win, "window_samples": win,
        "total_windows": int(len(X)), "words": sorted(pp["word"].unique().tolist()),
        "windows_per_subject": pp.groupby("subject").size().to_dict(),
        "cache_path": str(out),
    }
    json.dump(summary, open(Path(args.stats)/"summary.json","w"), indent=2, default=str)

    log("\n================ RAW DATASET STATISTICS ================")
    log(raw_df.to_string(index=False))
    log(f"\nsubjects={len(subs)}  sessions={summary['n_sessions']}  raw_channels={d0.shape[1]}  raw_sfreq={sf0}Hz")
    log("\n============ PREPROCESSED (WINDOW) STATISTICS ============")
    log(f"used_channels ({len(sel)}): {sel}")
    log(f"target_rate={args.rate}Hz  window={args.win}s ({win} samples)  total_windows={len(X)}")
    log("windows per subject: " + ", ".join(f"s{k}:{v}" for k,v in summary['windows_per_subject'].items()))
    log("words: " + ", ".join(summary["words"]))
    log(f"\n[done] cache -> {out}\n[done] stats -> {args.stats}/  (raw_stats.csv, preprocessed_stats.csv, summary.json)")

if __name__ == "__main__":
    main()
