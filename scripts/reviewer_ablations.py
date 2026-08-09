#!/usr/bin/env python3
"""
Reviewer ablations for ICAIN Paper 194 (CCPM / BED).
Runs EVAL-STAGE only on an existing trained checkpoint (no retraining).

Covers reviewer comments:
  R1.5(b) subject-wise EER variation  -> per_subject_eer.csv + subject_eer.png
  R1.5(a,c) qualitative error analysis / failure cases -> score_hist.png + failure_cases.csv
  R1.2(b) prototype-size k ablation (k in 1,2,3,5) -> k_ablation.csv
  R1.3(c) quality impact (quality-weighted vs plain fusion) -> quality_ablation.csv

USAGE (on the Brev box, from the repo root that contains the `fed_real_bed/` package):
    python reviewer_ablations.py \
        --config   configs/<the P0 ecapa_ccpm config>.yaml \
        --run-dir  artifacts/results/fed_real_bed_p0_ecapa_ccpm_brev \
        --out      artifacts/reviewer_ablations

Notes:
  * --run-dir must contain checkpoints/best.pt (the trained frozen encoder).
  * The config's data.cache_path must point at the same window cache used for the run
    (e.g. artifacts/cache/bed_windows_all_128hz_2.0s.npz).
  * This reproduces the paper's P0 protocol exactly via fed_real_bed.protocols.
  * ALL numbers printed here are verified outputs of YOUR pipeline — copy them into the
    paper / response letter as-is. Do NOT hand-edit.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fed_real_bed.config import load_config
from fed_real_bed.data import load_window_cache
from fed_real_bed.protocols import build_protocol_split, assert_split_valid
from fed_real_bed.models import EEGAuthenticator
from fed_real_bed.evaluation import (
    build_prototypes, verification_trials, temporal_fusion, roc_metrics, bootstrap_eer_ci,
)


def extract_embeddings(model, X, idx, device, batch=512):
    embs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            xb = torch.tensor(X[b], dtype=torch.float32, device=device)
            emb, _ = model(xb)
            embs.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(embs, axis=0)


def fused_scores(enroll_emb, enroll_y, test_emb, test_y, k, method, fusion="mean_5",
                 test_quality=None, seed=2026):
    proto_method = "mean" if k == 1 else "kmeans"
    bank = build_prototypes(enroll_emb, enroll_y, method=proto_method, k=k, seed=seed)
    df = verification_trials(test_emb, test_y, bank, score_norm=method)
    if test_quality is not None:
        df = df.copy()
        df["quality"] = test_quality[df["probe_index"].to_numpy()]
    df = temporal_fusion(df, fusion)
    return df


def eer_from(df):
    m = roc_metrics(df["label"].to_numpy(), df["score"].to_numpy())
    return m["eer_percent"], m["auc"], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="artifacts/reviewer_ablations")
    ap.add_argument("--fusion", default="mean_5")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config).raw
    # Resolve the window cache path: prefer the value recorded in the run's metrics.json
    # (guaranteed to match the checkpoint); fall back to config if present.
    cache_path = cfg.get("data", {}).get("cache_path")
    if not cache_path:
        import json as _json
        _mj = Path(args.run_dir) / "metrics.json"
        if _mj.exists():
            cache_path = _json.load(open(_mj)).get("cache_path")
    if not cache_path:
        raise SystemExit("Could not resolve cache_path; add --cache or check metrics.json.")
    data = load_window_cache(cache_path)
    X, y = data["X"], data["y"].astype(int)
    quality = data.get("quality")
    split = build_protocol_split(data, cfg)
    assert_split_valid(split, y)

    samples = int(cfg["data"]["target_sampling_rate"] * cfg["data"]["window_sec"])
    model = EEGAuthenticator(cfg, samples=samples).to(device)
    ckpt = torch.load(Path(args.run_dir) / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()

    enroll_idx, test_idx = split.enroll_idx, split.test_idx
    enroll_emb = extract_embeddings(model, X, enroll_idx, device)
    test_emb = extract_embeddings(model, X, test_idx, device)
    enroll_y, test_y = y[enroll_idx], y[test_idx]
    test_q = quality[test_idx] if quality is not None else None
    print(f"[info] enroll={len(enroll_idx)}  test={len(test_idx)}  subjects={len(np.unique(y))}")

    # ---- Headline (k=3, CCPM, mean_5) ----
    df3 = fused_scores(enroll_emb, enroll_y, test_emb, test_y, k=3, method="ccpm",
                       fusion=args.fusion, test_quality=test_q, seed=args.seed)
    eer, auc_, m = eer_from(df3)
    lo, hi = bootstrap_eer_ci(df3["label"].to_numpy(), df3["score"].to_numpy(), seed=args.seed, n=200)
    print(f"[headline] CCPM k=3 {args.fusion}: EER={eer:.2f}%  AUC={auc_:.3f}  CI95=[{lo:.2f},{hi:.2f}]")

    # ---- R1.2(b) k-ablation ----
    krows = []
    for k in (1, 2, 3, 5):
        dfk = fused_scores(enroll_emb, enroll_y, test_emb, test_y, k=k, method="ccpm",
                           fusion=args.fusion, test_quality=test_q, seed=args.seed)
        e, a, _ = eer_from(dfk)
        krows.append({"k": k, "eer_percent": round(e, 2), "auc": round(a, 3)})
        print(f"[k-ablation] k={k}: EER={e:.2f}%  AUC={a:.3f}")
    pd.DataFrame(krows).to_csv(out / "k_ablation.csv", index=False)

    # ---- R1.3(c) quality impact: plain mean vs quality-weighted fusion ----
    qrows = []
    win = args.fusion.split("_")[-1]
    for fus in (f"mean_{win}", f"qmean_{win}"):
        dfq = fused_scores(enroll_emb, enroll_y, test_emb, test_y, k=3, method="ccpm",
                           fusion=fus, test_quality=test_q, seed=args.seed)
        e, a, _ = eer_from(dfq)
        qrows.append({"fusion": fus, "eer_percent": round(e, 2), "auc": round(a, 3)})
        print(f"[quality] {fus}: EER={e:.2f}%  AUC={a:.3f}")
    pd.DataFrame(qrows).to_csv(out / "quality_ablation.csv", index=False)

    # ---- R1.5(b) per-subject EER (from headline k=3 CCPM scores) ----
    subs = sorted(np.unique(df3["claimed_id"]))
    prows = []
    for sid in subs:
        sdf = df3[df3["claimed_id"] == sid]
        if sdf["label"].nunique() < 2:
            continue
        sm = roc_metrics(sdf["label"].to_numpy(), sdf["score"].to_numpy())
        prows.append({"subject": int(sid), "eer_percent": round(sm["eer_percent"], 2),
                      "auc": round(sm["auc"], 3)})
    pdf = pd.DataFrame(prows).sort_values("eer_percent")
    pdf.to_csv(out / "per_subject_eer.csv", index=False)
    e = pdf["eer_percent"]
    print(f"[per-subject] mean={e.mean():.2f}  sd={e.std(ddof=1):.2f}  "
          f"min={e.min():.2f}(s{int(pdf.iloc[0].subject)})  max={e.max():.2f}(s{int(pdf.iloc[-1].subject)})")

    plt.figure(figsize=(6, 3.2))
    order = pdf.sort_values("subject")
    plt.bar(order["subject"].astype(str), order["eer_percent"], color="#1C7293")
    plt.axhline(eer, color="#B23A48", ls="--", lw=1, label=f"overall {eer:.1f}%")
    plt.xlabel("Subject"); plt.ylabel("EER (%)"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "subject_eer.png", dpi=300); plt.close()

    # ---- R1.5(a,c) score distributions + failure cases ----
    gen = df3[df3["label"] == 1]["score"].to_numpy()
    imp = df3[df3["label"] == 0]["score"].to_numpy()
    plt.figure(figsize=(6, 3.2))
    plt.hist(imp, bins=60, alpha=0.6, density=True, label="impostor", color="#B23A48")
    plt.hist(gen, bins=60, alpha=0.6, density=True, label="genuine", color="#1B7F5B")
    plt.xlabel("CCPM score"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "score_hist.png", dpi=300); plt.close()
    pdf.tail(3).assign(note="worst / failure cases").to_csv(out / "failure_cases.csv", index=False)

    json.dump({"headline_eer_percent": round(eer, 2), "auc": round(auc_, 3),
               "ci95": [round(lo, 2), round(hi, 2)], "fusion": args.fusion},
              open(out / "summary.json", "w"), indent=2)
    print(f"[done] wrote outputs to {out}/")


if __name__ == "__main__":
    main()
