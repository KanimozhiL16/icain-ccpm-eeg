from __future__ import annotations

import argparse
import json

from .config import load_config
from .training import run_preprocess, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="FED-REAL-BED EEG biometric pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("--config", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.cmd == "preprocess":
        print(run_preprocess(cfg))
    elif args.cmd == "train":
        print(json.dumps(run_training(cfg), indent=2))


if __name__ == "__main__":
    main()

