from __future__ import annotations

import argparse
from pathlib import Path

from fed_real_bed.data import discover_manifest
from fed_real_bed.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed-raw-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig(
        {
            "paths": {"bed_raw_root": args.bed_raw_root, "manifest_csv": None, "cache_root": "artifacts/cache", "results_root": "artifacts/results"},
            "data": {"target_sampling_rate": 256, "window_sec": 2.0, "step_sec": 1.0},
            "experiment": {"name": "manifest", "seed": 1},
            "protocol": {},
            "model": {},
            "loss": {},
            "train": {},
            "verification": {},
        }
    )
    df = discover_manifest(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
