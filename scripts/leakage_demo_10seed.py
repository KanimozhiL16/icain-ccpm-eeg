# =====================================================================================
# LEAKAGE DEMONSTRATION over the MATCHED 10 SEEDS (paper-ready).
# For each of the 10 CCPM checkpoints, compute on IDENTICAL data + encoder:
#   HONEST       : enrol r01+r02 -> held-out r03                 (our protocol)
#   LEAKY (mixed): sessions mixed, split by 10-window BLOCKS     (session-shared evaluation)
# Reports mean +/- SD for both over 10 seeds + the leakage inflation. Saves a CSV.
#   cd ~/24PHD1237/BED/FED_REAL_BED_ALL && python leakage_demo_10seed.py
# =====================================================================================
import glob, re, numpy as np, torch
from sklearn.cluster import KMeans
from fed_real_bed.config import load_config
from fed_real_bed.models import EEGAuthenticator

cfg=load_config("configs/brev_gpu_p0_ecapa_ccpm.yaml")
samples=int(round(cfg.raw["data"]["window_sec"]*cfg.raw["data"]["target_sampling_rate"]))
dev="cuda" if torch.cuda.is_available() else "cpu"
z=np.load(sorted(glob.glob("artifacts/cache/*all*128hz*.npz"))[0],allow_pickle=True)
X=z["X"]; y=np.asarray(z["y"]).ravel(); names=np.asarray(z["subjects"]).ravel()
sess=np.asarray(z["session"]).ravel(); subj=np.array([names[i] for i in y])
subjects=sorted(np.unique(subj)); S=len(subjects); s2i={s:i for i,s in enumerate(subjects)}
cks=sorted(glob.glob("artifacts/results/m10_ccpm_seed*/checkpoints/best.pt"),
           key=lambda p:int(re.search(r'seed(\d+)',p).group(1)))
print(f"{len(cks)} checkpoints")

def load(ck):
    m=EEGAuthenticator(cfg.raw,samples=samples).to(dev)
    try: st=torch.load(ck,map_location=dev,weights_only=False)
    except TypeError: st=torch.load(ck,map_location=dev)
    m.load_state_dict(st["model"] if isinstance(st,dict) and "model" in st else st); m.eval(); return m
def embed(m,M):
    o=[]
    for i in range(0,len(M),512):
        with torch.no_grad(): e,_=m(torch.tensor(M[i:i+512],dtype=torch.float32,device=dev))
        o.append(e.cpu().numpy())
    return np.concatenate(o) if o else np.zeros((0,128),np.float32)
def eer(g,im):
    t=np.linspace(min(g.min(),im.min()),max(g.max(),im.max()),1000)
    frr=np.array([(g<x).mean() for x in t]); far=np.array([(im>=x).mean() for x in t])
    k=np.argmin(np.abs(far-frr)); return (far[k]+frr[k])/2*100
def ccpm_eer(m, enrol_idx, probe_idx):
    gid=np.array([s2i[s] for s in subj[probe_idx]])
    P=np.zeros((S,3,128),np.float32)
    for s in subjects:
        ii=enrol_idx[subj[enrol_idx]==s][:600]
        E=embed(m,X[ii]); k=min(3,len(E)); C=KMeans(k,n_init=10,random_state=0).fit(E).cluster_centers_
        if k<3: C=np.repeat(C,3,axis=0)[:3]
        P[s2i[s]]=C/(np.linalg.norm(C,axis=1,keepdims=True)+1e-9)
    Z=embed(m,X[probe_idx]); Zn=Z/(np.linalg.norm(Z,axis=1,keepdims=True)+1e-9)
    C=np.max(np.einsum('id,ukd->iuk',Zn,P),axis=2)
    M_=C-np.array([np.max(np.delete(C,u,1),1) for u in range(S)]).T
    g=[];im=[]
    for u in range(S):
        rows=np.where(gid==u)[0]
        for k in range(0,len(rows)-5+1,5):
            blk=M_[rows[k:k+5]].mean(0); g.append(blk[u]); im.extend(np.delete(blk,u))
    return eer(np.array(g),np.array(im))

en=np.where((sess=="r01")|(sess=="r02"))[0]; pr=np.where(sess=="r03")[0]
def block_split(seed,block=10):
    rng=np.random.default_rng(seed); enL=[];prL=[]
    for s in subjects:
        idx=np.where(subj==s)[0]; bl=[idx[i:i+block] for i in range(0,len(idx),block)]
        rng.shuffle(bl); h=len(bl)//2
        if h>0: enL+=list(np.concatenate(bl[:h])); prL+=list(np.concatenate(bl[h:]))
    return np.array(enL),np.array(prL)

H=[];L=[]
for ck in cks:
    sd=int(re.search(r'seed(\d+)',ck).group(1)); m=load(ck)
    h=ccpm_eer(m,en,pr)
    eL,pL=block_split(sd); l=ccpm_eer(m,eL,pL)
    H.append(h); L.append(l); print(f"seed{sd}: honest={h:.2f}%  leaky(mixed)={l:.2f}%")
H=np.array(H); L=np.array(L)
print(f"\nHONEST (held-out r03)      : {H.mean():.2f}% ± {H.std(ddof=1):.2f}  (n={len(cks)})")
print(f"LEAKY  (sessions mixed)    : {L.mean():.2f}% ± {L.std(ddof=1):.2f}")
print(f">>> leakage inflation = {H.mean()-L.mean():.1f} points")
import csv
with open("artifacts/baselines/leakage_demo_10seed.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["seed","honest_eer","leaky_eer"])
    for ck,h,l in zip(cks,H,L): w.writerow([int(re.search(r'seed(\d+)',ck).group(1)),round(h,2),round(l,2)])
print("[saved] artifacts/baselines/leakage_demo_10seed.csv")
print(f"\nPAPER SENTENCE: 'On identical data and encoder, a session-mixed split yields {L.mean():.1f}±{L.std(ddof=1):.1f}% EER,")
print(f" versus {H.mean():.1f}±{H.std(ddof=1):.1f}% under our held-out-session protocol (10 seeds) — a {H.mean()-L.mean():.0f}-point inflation from session leakage.'")
