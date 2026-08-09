#!/usr/bin/env python3
"""
AEP (PhysioNet auditory-eeg) prep for the cross-dataset experiment.
Uses resting EYES-OPEN (ex01), which has 3 within-day sessions (s01,s02,s03),
mapped to r01/r02/r03 so the BED P0 protocol (enrol r01+r02, test r03) applies directly.

Builds the window cache in the pipeline's format, named so _maybe_cache finds it:
    artifacts/cache/bed_windows_aep_128hz_2.0s.npz   (session=r0X, stimulus=AEP, 4 channels)
and prints RAW + PREPROCESSED statistics.

Prereq (download Filtered_Data on Brev, ~open access, no login):
    wget -r -N -c -np -nH --cut-dirs=3 -R "index.html*" \
      https://physionet.org/files/auditory-eeg/1.0.0/Filtered_Data/
  (or:  aws s3 sync --no-sign-request s3://physionet-open/auditory-eeg/1.0.0/Filtered_Data/ Filtered_Data/ )

Run:
    python aep_prep.py --data-dir Filtered_Data --experiment ex01
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd

AEP_CH = ["T7","F8","Cz","P4"]        # OpenBCI Ganglion 4-channel montage
FS_IN = 200.0

def log(*a): print(*a, flush=True)

def bandpass_notch_resample(x, sf_in, sf_out, low=1.0, high=40.0, notch=50.0, already_filtered=True):
    from scipy import signal
    x = np.asarray(x, dtype=np.float64)
    if not already_filtered:   # Filtered_Data is already 1-40Hz + 50Hz notch; only resample
        sos = signal.butter(4,[low,high],btype="bandpass",fs=sf_in,output="sos"); x=signal.sosfiltfilt(sos,x,axis=-1)
        b,a=signal.iirnotch(notch,30.0,fs=sf_in); x=signal.filtfilt(b,a,x,axis=-1)
    if abs(sf_out-sf_in)>1e-6:
        n=int(round(x.shape[-1]*sf_out/sf_in)); x=signal.resample(x,n,axis=-1)
    return x.astype(np.float32)

def quality(w):
    std=w.std(axis=1); bad=(std<1e-3)|(np.max(np.abs(w),axis=1)>500.0); return float(1.0-bad.mean())

def read_csv_4ch(path):
    df = pd.read_csv(path, header=None, comment="%")
    # coerce numeric; drop non-numeric header row if present
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    arr = df.to_numpy(dtype=np.float32)
    if arr.shape[1] < 5:
        arr = df.dropna().to_numpy(dtype=np.float32)
    # columns: [sample_index, T7, F8, Cz, P4, ...] -> take cols 1..4
    return arr[:, 1:5].T   # (4, T)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="Filtered_Data")
    ap.add_argument("--experiment", default="ex01")   # resting eyes-open (has 3 sessions)
    ap.add_argument("--out", default="artifacts/cache")
    ap.add_argument("--stats", default="artifacts/aep_stats")
    ap.add_argument("--rate", type=float, default=128.0)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--max-per-group", type=int, default=80)
    ap.add_argument("--min-quality", type=float, default=0.35)
    args=ap.parse_args()
    Path(args.out).mkdir(parents=True,exist_ok=True); Path(args.stats).mkdir(parents=True,exist_ok=True)

    files=sorted(glob.glob(os.path.join(args.data_dir,"**",f"*{args.experiment}*"),recursive=True))
    files=[f for f in files if f.lower().endswith((".csv",".txt"))]
    if not files: sys.exit(f"No {args.experiment} files under {args.data_dir}. Download Filtered_Data first.")
    log(f"[scan] {len(files)} {args.experiment} files")

    win=int(round(args.win*args.rate))
    Xs,ys,sess,stim,qual=[],[],[],[],[]; raw_rows=[]; subj=set()
    from collections import defaultdict; gc=defaultdict(int)
    for f in files:
        m=re.search(r"s(\d+)_ex(\d+)_s(\d+)", os.path.basename(f))
        if not m:  # some files may be sXX_exXX only
            m2=re.search(r"s(\d+)_ex(\d+)", os.path.basename(f))
            if not m2: continue
            sub, ses = int(m2.group(1)), 1
        else:
            sub, ses = int(m.group(1)), int(m.group(3))
        try: sig=read_csv_4ch(f)
        except Exception as e: log(f"[skip] {f}: {e}"); continue
        subj.add(sub)
        raw_rows.append({"subject":sub,"session":ses,"channels":sig.shape[0],"sfreq":FS_IN,
                         "duration_s":round(sig.shape[1]/FS_IN,1)})
        proc=bandpass_notch_resample(sig,FS_IN,args.rate,already_filtered=True)
        proc=(proc-proc.mean(axis=1,keepdims=True))/(proc.std(axis=1,keepdims=True)+1e-6)
        for s in range(0,proc.shape[1]-win+1,win):
            w=proc[:,s:s+win].astype(np.float32); q=quality(w)
            if q<args.min_quality: continue
            gk=(sub,ses)
            if gc[gk]>=args.max_per_group: break
            gc[gk]+=1
            Xs.append(w); ys.append(sub); sess.append(f"r{ses:02d}"); stim.append("AEP"); qual.append(q)

    if not Xs: sys.exit("No windows produced.")
    X=np.stack(Xs).astype(np.float32); y_raw=np.asarray(ys); subs=sorted(set(y_raw.tolist()))
    remap={s:i for i,s in enumerate(subs)}; y=np.asarray([remap[v] for v in y_raw],dtype=np.int64)
    session=np.asarray(sess,dtype=object); stimulus=np.asarray(stim,dtype=object); qv=np.asarray(qual,dtype=np.float32)
    out=Path(args.out)/"bed_windows_aep_128hz_2.0s.npz"
    np.savez_compressed(out,X=X,y=y,session=session,stimulus=stimulus,quality=qv,
                        subjects=np.asarray([f"s{s:02d}" for s in subs],dtype=object),
                        channels=np.asarray(AEP_CH,dtype=object))

    raw_df=pd.DataFrame(raw_rows).sort_values(["subject","session"]); raw_df.to_csv(Path(args.stats)/"raw_stats.csv",index=False)
    pp=pd.DataFrame({"subject":y,"session":session}); per=pp.groupby(["subject","session"]).size().reset_index(name="windows")
    per.to_csv(Path(args.stats)/"preprocessed_stats.csv",index=False)
    summary={"n_subjects":len(subs),"n_sessions":int(pd.Series([r['session'] for r in raw_rows]).nunique()),
             "channels":AEP_CH,"raw_sfreq":FS_IN,"target_rate":args.rate,"window_s":args.win,
             "window_samples":win,"total_windows":int(len(X)),
             "windows_per_subject":pp.groupby("subject").size().to_dict(),"cache_path":str(out)}
    json.dump(summary,open(Path(args.stats)/"summary.json","w"),indent=2,default=str)

    log("\n============ RAW DATASET STATISTICS (AEP ex01 eyes-open) ============")
    log(raw_df.to_string(index=False))
    log(f"\nsubjects={len(subs)}  sessions={summary['n_sessions']}  channels={AEP_CH}  sfreq={FS_IN}Hz")
    log("\n============ PREPROCESSED (WINDOW) STATISTICS ============")
    log(f"target_rate={args.rate}Hz  window={args.win}s ({win} samp)  total_windows={len(X)}  channels={len(AEP_CH)}")
    log("windows per subject: "+", ".join(f"s{k}:{v}" for k,v in summary['windows_per_subject'].items()))
    log(f"\n[done] cache -> {out}\n[done] stats -> {args.stats}/")

if __name__=="__main__": main()
