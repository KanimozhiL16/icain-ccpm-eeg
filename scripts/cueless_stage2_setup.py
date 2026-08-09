#!/usr/bin/env python3
"""
Stage 2 setup for the Cueless cross-dataset experiment.
(1) Relabels the Stage-1 cache so the BED pipeline can consume it:
      - sessions ses-0X -> r0X   (protocol splitter expects r01..)
      - stimulus  ->  CUELESS    (protocol base_mask filters by stimulus)
      - saved as artifacts/cache/bed_windows_cueless_128hz_2.0s.npz  (name the loader expects)
(2) Generates configs/cueless_ccpm.yaml from your known-good P0 CCPM config,
    changing only: name, channels (10), stimulus, protocol (session hold-out), CCPM global.

Run on Brev from the repo root:
    python cueless_stage2_setup.py
    python -m fed_real_bed.cli train --config configs/cueless_ccpm.yaml

Everything the paper needs is then written under artifacts/results/fed_real_bed_cueless_ccpm/:
    metrics.json, per_subject_metrics.csv, per_session_metrics.csv, roc.png, train_log.csv, scores_*.csv
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, yaml

CUE_CH = ["F7","F3","T7","P7","O1","O2","P8","T8","F4","F8"]   # 10 BED channels present in Cueless
BASE_CFG = "configs/brev_gpu_p0_ecapa_ccpm.yaml"

def relabel_cache():
    src = sorted(glob.glob("artifacts/cache/cueless_windows_*hz_*.npz"))
    if not src:
        sys.exit("Stage-1 cache not found (artifacts/cache/cueless_windows_*). Run cueless_prep.py first.")
    d = dict(np.load(src[0], allow_pickle=True))
    sess = np.asarray([str(s).replace("ses-","r") for s in d["session"]], dtype=object)  # ses-01 -> r01
    stim = np.full(len(d["y"]), "CUELESS", dtype=object)
    out = Path("artifacts/cache/bed_windows_cueless_128hz_2.0s.npz")
    np.savez_compressed(out, X=d["X"], y=d["y"], session=sess, stimulus=stim,
                        quality=d["quality"], subjects=d["subjects"], channels=d["channels"])
    print(f"[relabel] {src[0]} -> {out}  (X={d['X'].shape}, sessions={sorted(set(sess))})")

def make_config():
    if not Path(BASE_CFG).exists():
        cands = sorted(glob.glob("configs/*ecapa_ccpm*.yaml"))
        if not cands: sys.exit(f"Base config not found ({BASE_CFG}).")
        base = cands[0]
    else:
        base = BASE_CFG
    cfg = yaml.safe_load(open(base))
    cfg.setdefault("experiment", {})["name"] = "fed_real_bed_cueless_ccpm"
    cfg.setdefault("data", {})
    cfg["data"]["channels"] = CUE_CH
    cfg["data"]["stimulus"] = "cueless"
    cfg["data"]["target_sampling_rate"] = 128
    cfg["data"]["window_sec"] = 2.0
    # leakage-free session hold-out: enrol r01-03, threshold on r04, test on untouched r05
    cfg["protocol"] = {"name":"P2","enroll_sessions":["r01","r02","r03"],
                       "validation_sessions":["r04"],"test_sessions":["r05"],
                       "threshold_source":"validation"}
    cfg.setdefault("verification", {})
    cfg["verification"]["condition_on"] = "none"     # no per-word context in continuous data
    cfg["verification"]["score_norm"] = "ccpm"
    cfg["verification"].setdefault("prototype","kmeans")
    cfg["verification"].setdefault("kmeans_k",3)
    cfg["verification"].setdefault("fusion",{"methods":["single","mean_5","mean_10"]})
    cfg.setdefault("federated",{})["enabled"] = False
    out = Path("configs/cueless_ccpm.yaml")
    yaml.safe_dump(cfg, open(out,"w"), sort_keys=False)
    print(f"[config] wrote {out} from {base}  (channels={len(CUE_CH)}, protocol=P2 r01-03/r04/r05, CCPM)")

if __name__ == "__main__":
    relabel_cache(); make_config()
    print("\nNext:  python -m fed_real_bed.cli train --config configs/cueless_ccpm.yaml")
