from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from fed_real_bed.config import ExperimentConfig, load_config, set_by_dotted_key
from fed_real_bed.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--grid", required=True)
    args = parser.parse_args()

    base = load_config(args.config).raw
    grid = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))["experiments"]
    rows = []
    for exp in grid:
        cfg = deepcopy(base)
        name = exp["name"]
        cfg["experiment"]["name"] = f"{base['experiment']['name']}_{name}"
        for key, value in exp.items():
            if key == "name":
                continue
            set_by_dotted_key(cfg, key, value)
        result = run_training(ExperimentConfig(cfg))
        single = result["test_metrics"]["single"]
        rows.append({"name": name, "eer_percent": single["eer_percent"], "auc": single["auc"], "run_dir": str(Path(cfg["paths"]["results_root"]) / cfg["experiment"]["name"])})
    out = Path(base["paths"]["results_root"]) / f"{base['experiment']['name']}_ablation_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()

