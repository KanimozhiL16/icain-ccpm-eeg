# =====================================================================================
# PAIRED EFFECT SIZE for CCPM vs cosine (P0, matched seeds)
# Reads the per-seed metrics.json on Brev, pairs CCPM vs cosine by seed, and computes:
#   - per-seed EERs, paired differences
#   - paired t-test, Wilcoxon signed-rank
#   - relative EER reduction
#   - Cohen's d_z (paired)  =  mean(diff) / SD(diff)     <-- the strict paired effect size
# Paste into a Jupyter cell in ~/24PHD1237/BED/FED_REAL_BED_ALL.
# =====================================================================================
import glob, json, os, numpy as np
from scipy import stats

# Where the matched 10-seed runs were written (adjust the glob if your folder names differ).
# Each run dir must contain metrics.json with the test EER.
CCPM_GLOB   = "artifacts/results/*ccpm*seed*/metrics.json"   # CCPM runs
COSINE_GLOB = "artifacts/results/*cos*seed*/metrics.json"    # cosine runs
# fallbacks if your names differ:
if not glob.glob(CCPM_GLOB):   CCPM_GLOB   = "artifacts/results/m10_ccpm_seed*/metrics.json"
if not glob.glob(COSINE_GLOB): COSINE_GLOB = "artifacts/results/m10_cosine_seed*/metrics.json"

def eer_of(path):
    d = json.load(open(path))
    # try common keys; EER may be a fraction (0.16) or percent (16.0)
    for k in ["test_eer","eer","test/eer","EER","test_EER"]:
        if k in d:
            v = float(d[k]); return v*100 if v <= 1.0 else v
    # search nested
    for v in d.values():
        if isinstance(v,dict):
            for k in ["test_eer","eer","EER"]:
                if k in v:
                    x=float(v[k]); return x*100 if x<=1.0 else x
    raise KeyError(f"no EER key in {path}: keys={list(d.keys())}")

def seed_of(path):
    import re
    m = re.search(r'seed[_-]?(\d+)', path)
    return int(m.group(1)) if m else os.path.dirname(path)

ccpm   = {seed_of(p): eer_of(p) for p in glob.glob(CCPM_GLOB)}
cosine = {seed_of(p): eer_of(p) for p in glob.glob(COSINE_GLOB)}
seeds  = sorted(set(ccpm) & set(cosine))      # only seeds present in BOTH (matched pairs)
assert seeds, f"No matched seeds. CCPM seeds={list(ccpm)} cosine seeds={list(cosine)}"

a = np.array([ccpm[s]   for s in seeds])      # CCPM EER per seed (%)
b = np.array([cosine[s] for s in seeds])      # cosine EER per seed (%)
diff = b - a                                  # positive = CCPM lower (better)

print("seed :  CCPM   cosine   diff(cos-ccpm)")
for s,x,y in zip(seeds,a,b): print(f"{s:>4} : {x:6.2f}  {y:6.2f}   {y-x:+.2f}")
print(f"\nn pairs                 = {len(seeds)}")
print(f"CCPM   mean +/- SD      = {a.mean():.2f} +/- {a.std(ddof=1):.2f}")
print(f"cosine mean +/- SD      = {b.mean():.2f} +/- {b.std(ddof=1):.2f}")
print(f"CCPM lower in           = {int((diff>0).sum())}/{len(seeds)} seeds")

t,pt   = stats.ttest_rel(b,a)                 # paired t-test
try:    w,pw = stats.wilcoxon(b,a)
except Exception as e: w,pw = float('nan'),float('nan')
print(f"paired t-test           t={t:.3f}, p={pt:.4f}")
print(f"Wilcoxon signed-rank    W={w:.1f}, p={pw:.4f}")

abs_red = b.mean()-a.mean()
rel_red = abs_red/b.mean()*100
dz      = diff.mean()/diff.std(ddof=1)        # PAIRED Cohen's d_z  <-- report this
print(f"\nabsolute EER reduction  = {abs_red:.2f} percentage points")
print(f"relative EER reduction  = {rel_red:.1f}%")
print(f"Cohen's d_z (paired)    = {dz:.2f}   "
      f"({'small' if abs(dz)<0.5 else 'medium' if abs(dz)<0.8 else 'large'} effect)")
print("\n-> Put in the paper: 'a relative reduction of "
      f"{rel_red:.1f}% (paired Cohen's d_z = {dz:.2f}).'")
