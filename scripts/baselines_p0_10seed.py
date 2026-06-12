# =====================================================================================
# P0 SCORING BASELINES over the MATCHED 10 SEEDS  (leakage-free) -> mean +/- SD per method.
# Post-hoc scoring on the SAME ECAPA embeddings; cohort/fit stats use r01+r02 only,
# r03 is never used to fit anything. Reproduces the Table-1 scale (cosine ~17.4, CCPM ~16.1)
# and places all baselines on it. Saves a CSV. Paste in a Brev Jupyter cell / run with python.
#   cd ~/24PHD1237/BED/FED_REAL_BED_ALL && python baselines_p0_10seed.py
# =====================================================================================
import glob, re, numpy as np, torch
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from fed_real_bed.config import load_config
from fed_real_bed.models import EEGAuthenticator

cfg=load_config("configs/brev_gpu_p0_ecapa_ccpm.yaml")
samples=int(round(cfg.raw["data"]["window_sec"]*cfg.raw["data"]["target_sampling_rate"]))
dev="cuda" if torch.cuda.is_available() else "cpu"
CAP=400; N_FUSE=5

z=np.load(sorted(glob.glob("artifacts/cache/*all*128hz*.npz"))[0],allow_pickle=True)
X=z["X"]; y=np.asarray(z["y"]).ravel(); names=np.asarray(z["subjects"]).ravel()
sess=np.asarray(z["session"]).ravel(); subj=np.array([names[i] for i in y])
subjects=sorted(np.unique(subj)); S=len(subjects); s2i={s:i for i,s in enumerate(subjects)}

cks=sorted(glob.glob("artifacts/results/m10_ccpm_seed*/checkpoints/best.pt"),
           key=lambda p:int(re.search(r'seed(\d+)',p).group(1)))
print(f"{len(cks)} seed checkpoints found")

def load_model(ck):
    m=EEGAuthenticator(cfg.raw,samples=samples).to(dev)
    try: st=torch.load(ck,map_location=dev,weights_only=False)
    except TypeError: st=torch.load(ck,map_location=dev)
    m.load_state_dict(st["model"] if isinstance(st,dict) and "model" in st else st); m.eval(); return m

def embed(m,M):
    out=[]
    for i in range(0,len(M),512):
        with torch.no_grad(): e,_=m(torch.tensor(M[i:i+512],dtype=torch.float32,device=dev))
        out.append(e.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,128),np.float32)

def eer(g,im):
    t=np.linspace(min(g.min(),im.min()),max(g.max(),im.max()),600)
    frr=np.array([(g<th).mean() for th in t]); far=np.array([(im>=th).mean() for th in t])
    k=np.argmin(np.abs(far-frr)); return (far[k]+frr[k])/2*100

def fuse_eer(Sf,gid,N=N_FUSE):
    g=[];im=[]
    for u in range(S):
        rows=np.where(gid==u)[0]; col=Sf[rows]
        for k in range(0,len(rows)-N+1,N):
            blk=col[k:k+N].mean(0); g.append(blk[u]); im.extend(np.delete(blk,u))
    return eer(np.array(g),np.array(im))

def run_seed(ck):
    m=load_model(ck)
    P=np.zeros((S,3,128),np.float32); Eenr={}
    for s in subjects:
        msk=(subj==s)&((sess=="r01")|(sess=="r02")); E=embed(m,X[msk][:600]); Eenr[s]=E
        C=KMeans(3,n_init=10,random_state=0).fit(E).cluster_centers_
        P[s2i[s]]=C/(np.linalg.norm(C,axis=1,keepdims=True)+1e-9)
    Z=[];gid=[]
    for s in subjects:
        idx=np.where((subj==s)&(sess=="r03"))[0][:CAP]; e=embed(m,X[idx]); Z.append(e); gid+=[s2i[s]]*len(e)
    Z=np.concatenate(Z); gid=np.array(gid); Zn=Z/(np.linalg.norm(Z,axis=1,keepdims=True)+1e-9)
    C=np.max(np.einsum('id,ukd->iuk',Zn,P),axis=2)                      # (Nt,S)
    Eall=np.concatenate([Eenr[s] for s in subjects]); esid=np.concatenate([[s2i[s]]*len(Eenr[s]) for s in subjects])
    Ealln=Eall/(np.linalg.norm(Eall,axis=1,keepdims=True)+1e-9)
    D=np.max(np.einsum('jd,ukd->juk',Ealln,P),axis=2)
    mu_t=np.array([D[esid!=u,u].mean() for u in range(S)]); sd_t=np.array([D[esid!=u,u].std()+1e-9 for u in range(S)])
    res={}
    res["cosine"]=fuse_eer(C,gid)
    res["CCPM"]=fuse_eer(C-np.array([np.max(np.delete(C,u,1),1) for u in range(S)]).T,gid)
    res["T-norm"]=fuse_eer((C-mu_t)/sd_t,gid)
    Zr=np.stack([(lambda r:np.array([(r[u]-np.delete(r,u).mean())/(np.delete(r,u).std()+1e-9) for u in range(S)]))(C[i]) for i in range(C.shape[0])])
    res["Z-norm"]=fuse_eer(Zr,gid); res["S-norm"]=fuse_eer(0.5*(Zr+(C-mu_t)/sd_t),gid)
    mns=np.stack([Eenr[s].mean(0) for s in subjects])
    inv=np.linalg.pinv(np.cov(np.concatenate([Eenr[s]-Eenr[s].mean(0) for s in subjects]).T)+1e-3*np.eye(128))
    diff=Z[:,None,:]-mns[None,:,:]; res["Mahalanobis"]=fuse_eer(-np.einsum('isd,de,ise->is',diff,inv,diff),gid)
    lda=np.zeros((Z.shape[0],S)); svm=np.zeros((Z.shape[0],S)); rng=np.random.default_rng(0)
    for u,s in enumerate(subjects):
        pos=Eenr[s]; negp=np.concatenate([Eenr[o] for o in subjects if o!=s])
        neg=negp[rng.choice(len(negp),min(len(pos)*3,len(negp)),replace=False)]
        Xtr=np.concatenate([pos,neg]); ytr=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
        lda[:,u]=LDA().fit(Xtr,ytr).decision_function(Z); svm[:,u]=SVC(kernel="linear",C=1).fit(Xtr,ytr).decision_function(Z)
    res["LDA(per-user)"]=fuse_eer(lda,gid); res["SVM(per-user)"]=fuse_eer(svm,gid)
    return res

allres={}
for ck in cks:
    sd=int(re.search(r'seed(\d+)',ck).group(1)); r=run_seed(ck)
    for k,v in r.items(): allres.setdefault(k,[]).append(v)
    print(f"seed{sd}: "+" ".join(f"{k}={v:.2f}" for k,v in r.items()))

import csv
order=["CCPM","S-norm","Z-norm","T-norm","cosine","SVM(per-user)","LDA(per-user)","Mahalanobis"]
print(f"\n{'method':16s} mean+/-SD (n={len(cks)})")
with open("artifacts/baselines/baselines_p0_10seed.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["method","mean_EER","sd_EER","n"])
    for k in order:
        a=np.array(allres[k]); print(f"{k:16s} {a.mean():.2f} +/- {a.std(ddof=1):.2f}")
        w.writerow([k,round(a.mean(),2),round(a.std(ddof=1),2),len(a)])
print("\nLaTeX rows (Table 1, P0 scoring-rule benchmark):")
for k in order:
    a=np.array(allres[k]); star=" \\textbf" if k=="CCPM" else ""
    print(f"{k} & {a.mean():.2f} $\\pm$ {a.std(ddof=1):.2f} \\\\")
print("\n[saved] artifacts/baselines/baselines_p0_10seed.csv")
