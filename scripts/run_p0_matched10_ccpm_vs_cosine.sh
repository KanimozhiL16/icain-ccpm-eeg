#!/usr/bin/env bash
# Matched 10-seed paired comparison: CCPM vs plain cosine on the SAME ECAPA encoder,
# SAME seeds 1..10 -> enables a clean PAIRED significance test.
# Runs 20 jobs total, 8 at a time across GPUs 0-7. Run on Brev:
#   bash run_p0_matched10_ccpm_vs_cosine.sh
set -e
cd ~/24PHD1237/BED/FED_REAL_BED_ALL
BASE=configs/brev_gpu_p0_ecapa_ccpm.yaml
ORIG=fed_real_bed_p0_ecapa_ccpm_brev

launch () {  # $1=scoring(ccpm|none) $2=seed $3=gpu
  local tag=$1 seed=$2 gpu=$3 name
  [ "$tag" = "ccpm" ] && name="m10_ccpm_seed${seed}" || name="m10_cosine_seed${seed}"
  local cfg=configs/_${name}.yaml
  cp "$BASE" "$cfg"
  sed -i "s/${ORIG}/${name}/g" "$cfg"
  sed -i "s/seed: 2026/seed: ${seed}/" "$cfg"
  sed -i "s/score_norm: ccpm/score_norm: ${tag}/" "$cfg"
  CUDA_VISIBLE_DEVICES=${gpu} nohup python -W ignore -m fed_real_bed.cli train \
      --config "$cfg" > logs_${name}.txt 2>&1 &
}

i=0
for tag in ccpm none; do
  for s in 1 2 3 4 5 6 7 8 9 10; do
    gpu=$(( i % 8 ))
    echo ">>> $tag seed $s on GPU $gpu"
    launch "$tag" "$s" "$gpu"
    i=$((i+1))
    # throttle to 8 concurrent jobs
    if (( i % 8 == 0 )); then wait; fi
  done
done
wait
echo "=== MATCHED 10-SEED RUNS DONE ==="

python - << 'EOF'
import json, glob, numpy as np
from scipy import stats
def load(pat):
    d={}
    for p in glob.glob(f"artifacts/results/{pat}_seed*/metrics.json"):
        s=int(p.split("seed")[1].split("/")[0])
        d[s]=json.load(open(p))["test_metrics"]["mean_5"]["eer_percent"]
    return d
c=load("m10_ccpm"); k=load("m10_cosine")
seeds=sorted(set(c)&set(k))
cc=np.array([c[s] for s in seeds]); kk=np.array([k[s] for s in seeds])
print("seeds:",seeds)
print(f"CCPM   mean={cc.mean():.2f} +/- {cc.std(ddof=1):.2f}")
print(f"Cosine mean={kk.mean():.2f} +/- {kk.std(ddof=1):.2f}")
print(f"CCPM lower in {int((kk>cc).sum())}/{len(seeds)} seeds")
t,p=stats.ttest_rel(kk,cc)
w,pw=stats.wilcoxon(kk,cc)
print(f"Paired t={t:.2f} p(2-tail)={p:.4f}")
print(f"Wilcoxon p={pw:.4f}")
EOF
