# =====================================================================================
# LEAKAGE DEMONSTRATION (corrected) — same encoder, same CCPM score; only the SPLIT changes.
# Compares three evaluation protocols on identical data to quantify how leakage inflates EER:
#   (1) HONEST       : enrol r01+r02 -> test held-out r03           (our protocol)
#   (2) LEAKY-BLOCK  : sessions mixed, split by 10-window BLOCKS    (realistic leakage, credible)
#   (3) LEAKY-WINDOW : sessions mixed, random per-window split      (degenerate: near-duplicate
#                      adjacent windows -> ~0% EER; shown only to expose the worst case)
# Also prints a genuine/impostor score summary so you can sanity-check the separation.
# Run:  cd ~/24PHD1237/BED/FED_REAL_BED_ALL && python leakage_demo.py
# =====================================================================================
import glob, numpy as np, torch
from sklearn.cluster import KMeans
from fed_real_bed.config import load_config
from fed_real_bed.models import EEGAuthenticator

cfg=load_config("configs/brev_gpu_p0_ecapa_ccpm.yaml")
samples=int(round(cfg.raw["data"]["window_sec"]*cfg.raw["data"]["target_sampling_rate"]))
dev="cuda" if torch.cuda.is_available() else "cpu"
ck=sorted(glob.glob("artifacts/results/*ccpm*/checkpoints/best.pt")+
          glob.glob("artifacts/results/m10_ccpm_seed*/checkpoints/best.pt"))[0]
m=EEGAuthenticator(cfg.raw,samples=samples).to(dev)
try: st=torch.load(ck,map_location=dev,weights_only=False)
except TypeError: st=torch.load(ck,map_location=dev)
m.load_state_dict(st["model"] if isinstance(st,dict) and "model" in st else st); m.eval()
print("checkpoint:",ck)

z=np.load(sorted(glob.glob("artifacts/cache/*all*128hz*.npz"))[0],allow_pickle=True)
X=z["X"]; y=np.asarray(z["y"]).ravel(); names=np.asarray(z["subjects"]).ravel()
sess=np.asarray(z["session"]).ravel(); subj=np.array([names[i] for i in y])
subjects=sorted(np.unique(subj)); S=len(subjects); s2i={s:i for i,s in enumerate(subjects)}

def embed(M):
    o=[]
    for i in range(0,len(M),512):
        with torch.no_grad(): e,_=m(torch.tensor(M[i:i+512],dtype=torch.float32,device=dev))
        o.append(e.cpu().numpy())
    return np.concatenate(o) if o else np.zeros((0,128),np.float32)

def scores(enrol_idx, probe_idx, gid_probe):
    """Return fused genuine & impostor CCPM scores for one split."""
    P=np.zeros((S,3,128),np.float32)
    for s in subjects:
        ii=enrol_idx[subj[enrol_idx]==s][:600]
        if len(ii)<3: ii=enrol_idx[subj[enrol_idx]==s]
        E=embed(X[ii]); k=min(3,len(E))
        C=KMeans(k,n_init=10,random_state=0).fit(E).cluster_centers_
        if k<3: C=np.repeat(C,3,axis=0)[:3]
        P[s2i[s]]=C/(np.linalg.norm(C,axis=1,keepdims=True)+1e-9)
    Z=embed(X[probe_idx]); Zn=Z/(np.linalg.norm(Z,axis=1,keepdims=True)+1e-9)
    C=np.max(np.einsum('id,ukd->iuk',Zn,P),axis=2)
    M_=C-np.array([np.max(np.delete(C,u,1),1) for u in range(S)]).T
    g=[];im=[]
    for u in range(S):
        rows=np.where(gid_probe==u)[0]
        for k in range(0,len(rows)-5+1,5):
            blk=M_[rows[k:k+5]].mean(0); g.append(blk[u]); im.extend(np.delete(blk,u))
    return np.array(g), np.array(im)

def eer_of(g,im):
    t=np.linspace(min(g.min(),im.min()),max(g.max(),im.max()),1000)
    frr=np.array([(g<x).mean() for x in t]); far=np.array([(im>=x).mean() for x in t])
    k=np.argmin(np.abs(far-frr)); return (far[k]+frr[k])/2*100

# ---------------- (1) HONEST ----------------
en=np.where((sess=="r01")|(sess=="r02"))[0]; pr=np.where(sess=="r03")[0]
g,im=scores(en,pr,np.array([s2i[s] for s in subj[pr]])); honest=eer_of(g,im)
print(f"\n(1) HONEST       enrol r01+r02 -> held-out r03 : EER = {honest:.2f}%")
print(f"      genuine  mean={g.mean():+.3f} sd={g.std():.3f} | impostor mean={im.mean():+.3f} sd={im.std():.3f}")

# ---------------- (2) LEAKY-BLOCK (credible) ----------------
def block_split(seed,block=10):
    rng=np.random.default_rng(seed); enL=[];prL=[]
    for s in subjects:
        idx=np.where(subj==s)[0]
        blocks=[idx[i:i+block] for i in range(0,len(idx),block)]
        rng.shuffle(blocks); h=len(blocks)//2
        if h>0:
            enL+=list(np.concatenate(blocks[:h])); prL+=list(np.concatenate(blocks[h:]))
    return np.array(enL),np.array(prL)
blk=[]
for sd in range(5):
    eL,pL=block_split(sd); g,im=scores(eL,pL,np.array([s2i[s] for s in subj[pL]])); blk.append(eer_of(g,im))
blk=np.array(blk)
print(f"\n(2) LEAKY-BLOCK  sessions mixed, block split  : EER = {blk.mean():.2f}% ± {blk.std(ddof=1):.2f}  (5 seeds)")

# ---------------- (3) LEAKY-WINDOW (degenerate, worst case) ----------------
def win_split(seed):
    rng=np.random.default_rng(seed); enL=[];prL=[]
    for s in subjects:
        idx=np.where(subj==s)[0].copy(); rng.shuffle(idx); h=len(idx)//2
        enL+=list(idx[:h]); prL+=list(idx[h:])
    return np.array(enL),np.array(prL)
eL,pL=win_split(0); g,im=scores(eL,pL,np.array([s2i[s] for s in subj[pL]])); win=eer_of(g,im)
print(f"(3) LEAKY-WINDOW random per-window split       : EER = {win:.2f}%  (degenerate: near-duplicate windows)")

print(f"\n>>> CREDIBLE leakage inflation = {honest-blk.mean():.1f} points "
      f"(block-leaky {blk.mean():.1f}% vs honest {honest:.1f}%).")
print("PAPER SENTENCE (use the BLOCK number, not the window one):")
print(f"  'On identical data and encoder, a session-mixed split reports {blk.mean():.1f}% EER,")
print(f"   versus {honest:.1f}% under our held-out-session protocol, quantifying leakage inflation.'")
