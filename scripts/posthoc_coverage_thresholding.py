#!/usr/bin/env python3
"""
Leakage-free post-hoc analysis for the best strict-P2 BED run.
NO retraining. NO Session-3 tuning. Two analyses:

  (1) Session-2 thresholding: pick a GLOBAL and PER-USER threshold on Session-2
      validation scores, apply UNCHANGED to Session-3 test. Report HTER=(FAR+FRR)/2.
  (2) EER@coverage (error-reject curve): rank Session-3 trials by confidence
      = |score - val_global_threshold| (boundary fixed from Session 2), reject the
      least-confident fraction, recompute threshold-free EER on the retained trials.

Run on Brev:
  cd ~/24PHD1237/BED/FED_REAL_BED_ALL
  python knowledge_base/scripts/posthoc_coverage_thresholding.py \
     --run artifacts/results/fed_real_bed_p2_fedprox_warmstart_calibrated_brev \
     --fusion scores_mean_5.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd

def find_col(df, prefs):
    low = {c.lower(): c for c in df.columns}
    for p in prefs:
        if p in low: return low[p]
    return None

def load_trials(path):
    df = pd.read_csv(path)
    sc = find_col(df, ["score","fused_score","mean_score","value"])
    lb = find_col(df, ["label","is_genuine","genuine","y"])
    cl = find_col(df, ["claimed_id","claimed","subject","claim","enrolled_id"])
    tr = find_col(df, ["true_id","true","probe_subject","actual_id"])
    if sc is None:
        sys.exit(f"[ERR] no score column in {path}; columns={list(df.columns)}")
    if lb is None and (cl is not None and tr is not None):
        df["_label"] = (df[cl] == df[tr]).astype(int); lb = "_label"
    if lb is None:
        sys.exit(f"[ERR] no label column in {path}; columns={list(df.columns)}")
    out = pd.DataFrame({"score": df[sc].astype(float), "label": df[lb].astype(int)})
    out["claimed"] = df[cl] if cl is not None else -1
    return out

def eer_and_threshold(labels, scores):
    """Threshold-free EER + the score threshold at the EER operating point."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = (fpr[i] + fnr[i]) / 2.0
    return eer * 100.0, float(thr[i])

def far_frr_at(labels, scores, threshold):
    acc = scores >= threshold
    g = labels == 1; imp = labels == 0
    frr = float(np.mean(~acc[g])) if g.any() else 0.0   # genuine rejected
    far = float(np.mean(acc[imp])) if imp.any() else 0.0  # impostor accepted
    return far, frr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to the P2 result dir")
    ap.add_argument("--fusion", default="scores_mean_5.csv",
                    help="test fusion file inside --run (matches reported fusion)")
    ap.add_argument("--val", default="validation_scores.csv")
    ap.add_argument("--min-user-trials", type=int, default=30)
    a = ap.parse_args()

    test = load_trials(os.path.join(a.run, a.fusion))
    val  = load_trials(os.path.join(a.run, a.val))
    print(f"[info] test trials={len(test)}  val trials={len(val)}  "
          f"users(test)={test['claimed'].nunique()}")

    # Baseline threshold-free EER on test (sanity vs metrics.json)
    base_eer, _ = eer_and_threshold(test.label.values, test.score.values)
    print(f"\n[0] Baseline threshold-free Test EER = {base_eer:.2f}%  (should match metrics.json)")

    # ---- (1a) GLOBAL Session-2 threshold applied to Session-3 ----
    _, gthr = eer_and_threshold(val.label.values, val.score.values)
    far, frr = far_frr_at(test.label.values, test.score.values, gthr)
    print(f"\n[1a] GLOBAL threshold from Session-2 -> Session-3: "
          f"FAR={far*100:.2f}% FRR={frr*100:.2f}% HTER={(far+frr)/2*100:.2f}%")

    # ---- (1b) PER-USER Session-2 thresholds applied to Session-3 ----
    user_thr = {}
    for u, g in val.groupby("claimed"):
        if len(g) >= a.min_user_trials and g.label.nunique() == 2:
            _, t = eer_and_threshold(g.label.values, g.score.values)
            user_thr[u] = t
    acc = np.empty(len(test), dtype=bool)
    for idx, row in enumerate(test.itertuples(index=False)):
        thr = user_thr.get(row.claimed, gthr)
        acc[idx] = row.score >= thr
    lab = test.label.values
    g = lab == 1; imp = lab == 0
    frr_u = float(np.mean(~acc[g])); far_u = float(np.mean(acc[imp]))
    print(f"[1b] PER-USER thresholds from Session-2 -> Session-3 "
          f"({len(user_thr)} users had own threshold): "
          f"FAR={far_u*100:.2f}% FRR={frr_u*100:.2f}% HTER={(far_u+frr_u)/2*100:.2f}%")

    # ---- (2) EER@coverage (confidence = |score - val global threshold|) ----
    conf = np.abs(test.score.values - gthr)
    order = np.argsort(-conf)  # most confident first
    print("\n[2] EER@coverage (reject least-confident; boundary fixed on Session-2):")
    print("    coverage   retained   EER%")
    for cov in [1.00, 0.95, 0.90, 0.80, 0.70]:
        k = max(1, int(round(cov * len(order))))
        keep = order[:k]
        e, _ = eer_and_threshold(lab[keep], test.score.values[keep])
        print(f"    {int(cov*100):3d}%      {k:8d}   {e:.2f}")

    print("\n[note] Report HTER for thresholding (1a/1b) and the FULL coverage curve (2).")
    print("[note] Do NOT report only the best coverage point. No Session-3 tuning was used.")

if __name__ == "__main__":
    main()
