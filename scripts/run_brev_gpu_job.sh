#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/FED_REAL_BED}"
CONFIG="${CONFIG:-configs/brev_gpu_rc.yaml}"
RUN_NAME="${RUN_NAME:-fed_real_bed_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-auto}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_UTIL="${MAX_UTIL:-5}"
MAX_MEM_MB="${MAX_MEM_MB:-1000}"

cd "$PROJECT_DIR"

mkdir -p "artifacts/gpu_logs/$RUN_NAME"

echo "[INFO] Project: $PROJECT_DIR"
echo "[INFO] Config : $CONFIG"
echo "[INFO] Run    : $RUN_NAME"
echo "[INFO] Checking GPU availability..."
nvidia-smi | tee "artifacts/gpu_logs/$RUN_NAME/nvidia_smi_before.txt"

if [[ "$GPU_IDS" == "auto" ]]; then
  FREE_GPUS="$(python scripts/gpu_monitor.py free --max-util "$MAX_UTIL" --max-mem-mb "$MAX_MEM_MB")"
  GPU_IDS="$(echo "$FREE_GPUS" | awk -v n="$NUM_GPUS" '{for(i=1;i<=n && i<=NF;i++){printf "%s%s", (i==1?"":","), $i}}')"
fi

if [[ -z "$GPU_IDS" ]]; then
  echo "[ERROR] No free GPU found. Try later or ask current users to release the node."
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "[INFO] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python scripts/gpu_monitor.py snapshot --out-dir "artifacts/gpu_logs/$RUN_NAME"

python scripts/gpu_monitor.py monitor \
  --out-csv "artifacts/gpu_logs/$RUN_NAME/gpu_usage.csv" \
  --interval-sec 30 &
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" >/dev/null 2>&1 || true
  wait "$MONITOR_PID" >/dev/null 2>&1 || true
  nvidia-smi > "artifacts/gpu_logs/$RUN_NAME/nvidia_smi_after.txt" || true
  python scripts/gpu_monitor.py summarize \
    --csv "artifacts/gpu_logs/$RUN_NAME/gpu_usage.csv" \
    --out-json "artifacts/gpu_logs/$RUN_NAME/gpu_summary.json" || true
}
trap cleanup EXIT

echo "[INFO] Starting training..."
python -m fed_real_bed.cli train --config "$CONFIG" 2>&1 | tee "artifacts/gpu_logs/$RUN_NAME/train_console.log"

echo "[INFO] Training complete. Logs saved in artifacts/gpu_logs/$RUN_NAME"

