#!/usr/bin/env bash
# ECAPA + PLAIN COSINE scoring (no cohort margin), 5 seeds, parallel on GPUs 0-4.
# Same encoder/protocol as CCPM -> isolates CCPM's contribution.
# Run on Brev:  bash run_p0_cosine_baseline.sh
set -e
cd ~/24PHD1237/BED/FED_REAL_BED_ALL
BASE=configs/brev_gpu_p0_ecapa_ccpm.yaml
ORIG=fed_real_bed_p0_ecapa_ccpm_brev

for i in 1 2 3 4 5; do
  G=$((i-1)); cfg=configs/_p0_cosine_seed${i}.yaml
  cp "$BASE" "$cfg"
  sed -i "s/${ORIG}/p0_cosine_seed${i}/g" "$cfg"
  sed -i "s/seed: 2026/seed: ${i}/" "$cfg"
  sed -i "s/score_norm: ccpm/score_norm: none/" "$cfg"   # plain cosine
  echo ">>> cosine seed ${i} on GPU ${G}"
  CUDA_VISIBLE_DEVICES=${G} nohup python -W ignore -m fed_real_bed.cli train \
      --config "$cfg" > logs_p0_cosine_seed${i}.txt 2>&1 &
done
wait
echo "=== COSINE BASELINE DONE ==="
python - << 'EOF'
import json, glob, statistics as st
e=[]
for d in sorted(glob.glob("artifacts/results/p0_cosine_seed*")):
    m=json.load(open(d+"/metrics.json"))["test_metrics"]["mean_5"]["eer_percent"]; e.append(m)
    print(d.split("/")[-1], f"cosine mean_5 EER={m:.2f}%")
if e:
    sd=st.stdev(e) if len(e)>1 else 0
    print(f"\nECAPA+COSINE  MEAN EER = {st.mean(e):.2f}% +/- {sd:.2f}  (n={len(e)})")
    print("Compare to CCPM 16.82 +/- 1.02 and S-norm 17.77.")
EOF
