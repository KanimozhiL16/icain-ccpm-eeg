#!/usr/bin/env python3
"""
Single-subject transparency inspection for the CCPM verification pipeline.
Uses ONLY the saved, verified score files (no model reload) -> fully trustworthy.

It picks one subject and shows, end-to-end:
  (a) one genuine probe: the CCPM score for EVERY claimed identity -> the true identity
      should win and exceed the decision threshold (correct ACCEPT);
  (b) the subject's genuine-vs-impostor CCPM score distributions with the EER threshold;
  (c) a plain-language printout of the decision trace.

Run on Brev:
  cd ~/24PHD1237/BED/FED_REAL_BED_ALL
  python single_subject_inspection.py            # auto-finds a CCPM result dir
  python single_subject_inspection.py --run artifacts/results/ccpm_run1 --subject 1
"""
import argparse, glob, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def find_run():
    cands = glob.glob("artifacts/results/*ccpm*") + glob.glob("artifacts/results/m10_ccpm_seed1")
    for d in cands:
        if glob.glob(os.path.join(d, "scores*.csv")): return d
    raise SystemExit("No CCPM result dir with scores*.csv found. Pass --run explicitly.")

def pick_scores(run):
    for f in ["scores_mean_5.csv","scores.csv","scores_single.csv"]:
        p=os.path.join(run,f)
        if os.path.exists(p): return p, f
    raise SystemExit(f"No scores file in {run}")

def col(df,*names):
    low={c.lower():c for c in df.columns}
    for n in names:
        if n in low: return low[n]
    return None

def eer_threshold(labels, scores):
    from sklearn.metrics import roc_curve
    fpr,tpr,thr=roc_curve(labels,scores,pos_label=1); fnr=1-tpr
    i=int(np.nanargmin(np.abs(fpr-fnr))); return float(thr[i]), (fpr[i]+fnr[i])/2*100

ap=argparse.ArgumentParser()
ap.add_argument("--run", default=None)
ap.add_argument("--subject", type=int, default=None)
a=ap.parse_args()
run=a.run or find_run()
sp,fname=pick_scores(run)
df=pd.read_csv(sp)
sc=col(df,"score"); lb=col(df,"label"); cl=col(df,"claimed_id","claimed"); tr=col(df,"true_id","true"); pi=col(df,"probe_index","probe")
if lb is None: df["_l"]=(df[cl]==df[tr]).astype(int); lb="_l"
print(f"[run] {run}\n[scores] {fname}  rows={len(df)}  subjects={df[cl].nunique()}")

thr, eer = eer_threshold(df[lb].values, df[sc].values)
print(f"[global] EER={eer:.2f}%  decision threshold (EER point)={thr:.4f}")

subj = a.subject if a.subject is not None else int(df[tr].mode().iloc[0])
gen = df[(df[tr]==subj)&(df[cl]==subj)]            # genuine trials for this subject
imp = df[(df[tr]==subj)&(df[cl]!=subj)]            # impostor trials targeting this subject
print(f"\n[subject {subj}] genuine trials={len(gen)}  impostor trials={len(imp)}")
print(f"  genuine  CCPM score mean={gen[sc].mean():.3f}")
print(f"  impostor CCPM score mean={imp[sc].mean():.3f}")

# one worked probe: take a genuine probe and show score for every claimed identity
probe = int(gen.sort_values(sc, ascending=False)[pi].iloc[0]) if pi in df.columns else None
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
if probe is not None:
    row=df[df[pi]==probe].sort_values(cl)
    claims=row[cl].astype(int).values; vals=row[sc].values
    colors=["#2ca02c" if c==subj else "#9aa0a6" for c in claims]
    ax[0].bar(range(len(claims)), vals, color=colors)
    ax[0].axhline(thr, color="crimson", ls="--", lw=1, label=f"threshold {thr:.2f}")
    ax[0].set_xticks(range(len(claims))); ax[0].set_xticklabels(claims, fontsize=7)
    ax[0].set_xlabel("claimed identity"); ax[0].set_ylabel("CCPM score")
    ax[0].set_title(f"Probe {probe} of subject {subj}: score per claimed id\n(green=true id)")
    ax[0].legend(fontsize=8)
    decide = "ACCEPT (correct)" if row[row[cl]==subj][sc].iloc[0]>=thr else "REJECT (miss)"
    print(f"\n[worked probe {probe}] true id {subj} score="
          f"{row[row[cl]==subj][sc].iloc[0]:.3f}  vs threshold {thr:.3f}  -> {decide}")

ax[1].hist(imp[sc], bins=40, alpha=0.6, label="impostor", color="#d62728", density=True)
ax[1].hist(gen[sc], bins=40, alpha=0.6, label="genuine", color="#2ca02c", density=True)
ax[1].axvline(thr, color="crimson", ls="--", lw=1, label=f"threshold {thr:.2f}")
ax[1].set_xlabel("CCPM score"); ax[1].set_ylabel("density")
ax[1].set_title(f"Subject {subj}: genuine vs impostor scores")
ax[1].legend(fontsize=8)
fig.suptitle(f"Single-subject CCPM transparency — subject {subj} ({run.split('/')[-1]})")
fig.tight_layout()
out="single_subject_transparency.png"; fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\n[saved] {out}  -> download and show your supervisor.")
