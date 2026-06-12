#!/usr/bin/env bash
# Parallel 5-seed multi-seed of the BEST strict-P2 run, one seed per GPU.
# Run on Brev:  bash run_p2_multiseed_parallel.sh
set -e
cd ~/24PHD1237/BED/FED_REAL_BED_ALL

BASE=configs/brev_gpu_p2_fedprox_warmstart_calibrated.yaml
ORIG=fed_real_bed_p2_fedprox_warmstart_calibrated_brev

for i in 1 2 3 4 5; do
  G=$((i-1))                                # GPUs 0..4 (you have 8)
  cfg=configs/_p2_seed${i}.yaml
  cp "$BASE" "$cfg"
  sed -i "s/${ORIG}/p2_best_seed${i}/g" "$cfg"
  sed -i "s/seed: 2026/seed: ${i}/" "$cfg"
  echo ">>> launching seed ${i} on GPU ${G}"
  CUDA_VISIBLE_DEVICES=${G} nohup python -W ignore -m fed_real_bed.cli train \
      --config "$cfg" > logs_p2_seed${i}.txt 2>&1 &
done
wait
echo "=== ALL P2 SEED RUNS DONE ==="

python - << 'EOF'
import json, glob, statistics as st
eers=[]
for d in sorted(glob.glob("artifacts/results/p2_best_seed*")):
    m=json.load(open(d+"/metrics.json"))
    e=m["test_metrics"]["mean_5"]["eer_percent"]; eers.append(e)
    print(d.split("/")[-1], f"mean_5 EER={e:.2f}%")
if eers:
    sd = st.stdev(eers) if len(eers)>1 else 0.0
    print(f"\nP2 BEST  MEAN EER = {st.mean(eers):.2f}% +/- {sd:.2f}  (n={len(eers)})")
    print("Baseline single-run was 29.58%. Improvement only if mean+SD < 29.58.")
EOF
