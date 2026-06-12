from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


QUERY_FIELDS = [
    "timestamp",
    "index",
    "name",
    "uuid",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
]


def run_nvidia_smi() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    rows = []
    for line in out.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        rows.append(dict(zip(QUERY_FIELDS, values)))
    return rows


def query_processes() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return []
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        values = [v.strip() for v in line.split(",")]
        rows.append(dict(zip(["gpu_uuid", "pid", "process_name", "used_memory"], values)))
    return rows


def free_gpus(max_util: float, max_mem_mb: float) -> list[int]:
    rows = run_nvidia_smi()
    free = []
    for row in rows:
        util = float(row["utilization.gpu"])
        mem = float(row["memory.used"])
        if util <= max_util and mem <= max_mem_mb:
            free.append(int(row["index"]))
    return free


def write_snapshot(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpus": run_nvidia_smi(),
        "processes": query_processes(),
    }
    (out_dir / "gpu_snapshot_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def monitor(out_csv: Path, interval_sec: float) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["wall_time_utc", "user", "cuda_visible_devices", *QUERY_FIELDS]
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        while True:
            now = datetime.now(timezone.utc).isoformat()
            user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            for row in run_nvidia_smi():
                writer.writerow({"wall_time_utc": now, "user": user, "cuda_visible_devices": visible, **row})
            f.flush()
            time.sleep(interval_sec)


def summarize(csv_path: Path, out_json: Path) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("GPU log is empty.")
    numeric_cols = ["utilization.gpu", "memory.used", "memory.total", "power.draw"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    summary = {
        "log_file": str(csv_path),
        "start_utc": str(df["wall_time_utc"].iloc[0]),
        "end_utc": str(df["wall_time_utc"].iloc[-1]),
        "samples": int(len(df)),
        "visible_devices": sorted(set(str(v) for v in df["cuda_visible_devices"].dropna())),
        "per_gpu": {},
    }
    for gpu, grp in df.groupby("index"):
        summary["per_gpu"][str(gpu)] = {
            "name": str(grp["name"].iloc[0]),
            "mean_utilization_percent": float(grp["utilization.gpu"].mean()),
            "max_utilization_percent": float(grp["utilization.gpu"].max()),
            "mean_memory_mb": float(grp["memory.used"].mean()),
            "max_memory_mb": float(grp["memory.used"].max()),
            "memory_total_mb": float(grp["memory.total"].max()),
            "mean_power_w": float(grp["power.draw"].mean()),
            "max_power_w": float(grp["power.draw"].max()),
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU availability and usage logger for shared NVIDIA nodes.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_free = sub.add_parser("free")
    p_free.add_argument("--max-util", type=float, default=5.0)
    p_free.add_argument("--max-mem-mb", type=float, default=1000.0)

    p_snap = sub.add_parser("snapshot")
    p_snap.add_argument("--out-dir", required=True)

    p_mon = sub.add_parser("monitor")
    p_mon.add_argument("--out-csv", required=True)
    p_mon.add_argument("--interval-sec", type=float, default=30.0)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--csv", required=True)
    p_sum.add_argument("--out-json", required=True)

    args = parser.parse_args()
    if args.cmd == "free":
        free = free_gpus(args.max_util, args.max_mem_mb)
        print(" ".join(map(str, free)))
    elif args.cmd == "snapshot":
        write_snapshot(Path(args.out_dir))
    elif args.cmd == "monitor":
        monitor(Path(args.out_csv), args.interval_sec)
    elif args.cmd == "summarize":
        summarize(Path(args.csv), Path(args.out_json))


if __name__ == "__main__":
    main()

