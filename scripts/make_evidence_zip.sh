#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?Usage: bash scripts/make_evidence_zip.sh RUN_NAME}"
OUT="gpu_evidence_${RUN_NAME}.zip"

zip -r "$OUT" \
  "artifacts/gpu_logs/$RUN_NAME" \
  artifacts/results \
  configs/brev_gpu_rc.yaml \
  README.md \
  BREV_GPU_RUN_GUIDE.md

echo "$OUT"

