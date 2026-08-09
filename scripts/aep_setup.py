#!/usr/bin/env python3
"""
Generate AEP configs (CCPM + cosine) from the known-good P0 CCPM config.
Protocol P0 = enrol r01+r02 (eyes-open sessions 1,2), test held-out r03 (session 3).
Run on Brev from the repo root, AFTER aep_prep.py has written the cache:
    python aep_setup.py
    python -m fed_real_bed.cli train --config configs/aep_ccpm.yaml
    python -m fed_real_bed.cli train --config configs/aep_cosine.yaml
"""
import glob, sys
from pathlib import Path
import yaml

AEP_CH = ["T7","F8","Cz","P4"]
BASE = "configs/brev_gpu_p0_ecapa_ccpm.yaml"

def base_cfg():
    b = BASE if Path(BASE).exists() else (sorted(glob.glob("configs/*ecapa_ccpm*.yaml")) or [None])[0]
    if not b: sys.exit("Base config not found.")
    return yaml.safe_load(open(b)), b

def build(name, score_norm):
    cfg, b = base_cfg()
    cfg.setdefault("experiment", {})["name"] = name
    cfg.setdefault("data", {})
    cfg["data"]["channels"] = AEP_CH
    cfg["data"]["stimulus"] = "aep"
    cfg["data"]["target_sampling_rate"] = 128
    cfg["data"]["window_sec"] = 2.0
    cfg["protocol"] = {"name": "P0"}          # enrol r01+r02 -> test r03 (hardcoded in pipeline)
    cfg.setdefault("verification", {})
    cfg["verification"]["condition_on"] = "none"
    cfg["verification"]["score_norm"] = score_norm
    cfg["verification"].setdefault("prototype", "kmeans")
    cfg["verification"].setdefault("kmeans_k", 3)
    cfg["verification"].setdefault("fusion", {"methods": ["single","mean_5","mean_10"]})
    cfg.setdefault("federated", {})["enabled"] = False
    out = Path(f"configs/{name.replace('fed_real_bed_','')}.yaml")
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
    print(f"[config] {out}  (from {b}; channels=4, P0, score_norm={score_norm})")

if __name__ == "__main__":
    # sanity: cache present?
    if not glob.glob("artifacts/cache/bed_windows_aep_128hz_*.npz"):
        sys.exit("AEP cache missing (artifacts/cache/bed_windows_aep_128hz_*.npz). Run aep_prep.py first.")
    build("fed_real_bed_aep_ccpm", "ccpm")
    build("fed_real_bed_aep_cosine", "none")
    print("\nNext:\n  python -m fed_real_bed.cli train --config configs/aep_ccpm.yaml"
          "\n  python -m fed_real_bed.cli train --config configs/aep_cosine.yaml")
