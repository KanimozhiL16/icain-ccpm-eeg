# =====================================================================================
# P0 SCORING BASELINES (leakage-free) on existing embeddings — for the Table 1 benchmark.
# Methods: cosine | CCPM | Z-norm | T-norm | S-norm | Mahalanobis | per-user LDA | per-user SVM
# Cohort statistics use ONLY enrolment sessions r01+r02; test session r03 is never used to
# fit anything (thresholds are swept post-hoc to read EER). Paste in a Brev Jupyter cell in
# ~/24PHD1237/BED/FED_REAL_BED_ALL.
# =====================================================================================
import glob, numpy as np, torch
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from fed_real_bed.config import load_config
from fed_real_bed.models import EEGAuthenticator

cfg=load_config("configs/brev_gpu_p0_ecapa_ccpm.yaml")
samples=int(round(cfg.raw["data"]["window_sec"]*cfg.raw["data"]["target_sampling_rate"]))
dev="cuda" if torch.cuda.is_available() else "cpu"
ck=sorted(glob.glob("artifacts/results/*ccpm*/checkpoints/best.pt")+
          glob.glob("artifacts/results/m10_ccpm_seed*/checkpoints/best.pt"))[0]
model=EEGAuthenticator(cfg.raw,samples=samples).to(dev)
try: st=torch.load(ck,map_location=dev,weights_only=False)
except TypeError: st=torch.load(ck,map_location=dev)
model.load_state_dict(st["model"] if isinstance(st,dict) and "model" in st else st); model.eval()
print("checkpoint:",ck)

z=np.load(sorted(glob.glob("artifacts/cache/*all*128hz*.npz"))[0],allow_pickle=True)
X=z["X"]; y=np.asarray(z["y"]).ravel(); names=np.asarray(z["subjects"]).ravel()
sess=np.asarray(z["session"]).ravel(); subj=np.array([names[i] for i in y])
def embed(M):
    out=[]
    for i in range(0,len(M),512):
        with torch.no_grad(): e,_=model(torch.tensor(M[i:i+512],dtype=torch.float32,device=dev))
        out.append(e.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,128),np.float32)

subjects=sorted(np.unique(subj)); S=len(subjects); s2i={s:i for i,s in enumerate(subjects)}
CAP=300  # probe windows per subject (speed); raise for full set

# ---- enrol (r01+r02): prototypes (k=3) + raw enrol embeddings per subject ----
P=np.zeros((S,3,128),np.float32); Eenr={};
for s in subjects:
    m=(subj==s)&((sess=="r01")|(sess=="r02"))
    E=embed(X[m][:600]); Eenr[s]=E
    C=KMeans(3,n_init=10,random_state=0).fit(E).cluster_centers_
    P[s2i[s]]=C/ (np.linalg.norm(C,axis=1,keepdims=True)+1e-9)

# ---- test (r03): probe embeddings per subject (capped) ----
Ztest=[]; gid=[]
for s in subjects:
    idx=np.where((subj==s)&(sess=="r03"))[0][:CAP]
    e=embed(X[idx]); Ztest.append(e); gid+= [s2i[s]]*len(e)
Z=np.concatenate(Ztest); gid=np.array(gid)            # Z:(Ntest,128), gid=true subject idx
Zn=Z/(np.linalg.norm(Z,axis=1,keepdims=True)+1e-9)

# ---- core cosine-to-each-subject matrix C[i,u] = max over u's 3 prototypes ----
# (Ntest, S, 3) -> max over 3
C=np.max(np.einsum('id,ukd->iuk', Zn, P), axis=2)      # (Ntest, S)

# cohort matrix D[j,u]: enrol embeddings (all subjects) scored vs subject u prototypes
Eall=np.concatenate([Eenr[s] for s in subjects]); esid=np.concatenate([[s2i[s]]*len(Eenr[s]) for s in subjects])
Ealln=Eall/(np.linalg.norm(Eall,axis=1,keepdims=True)+1e-9)
D=np.max(np.einsum('jd,ukd->juk', Ealln, P), axis=2)   # (Nenrol, S)

# T-norm stats per model u from IMPOSTOR cohort (enrol rows whose subject != u)
mu_t=np.zeros(S); sd_t=np.zeros(S)
for u in range(S):
    imp=D[esid!=u, u]; mu_t[u]=imp.mean(); sd_t[u]=imp.std()+1e-9

# ---- score matrices for cohort-norm methods (all derived from C) ----
def znorm_row(Crow):  # per-probe over impostor models
    out=np.zeros_like(Crow)
    for u in range(S):
        oth=np.delete(Crow,u); out[u]=(Crow[u]-oth.mean())/(oth.std()+1e-9)
    return out
SCORES={}
SCORES["cosine"]=C.copy()
SCORES["CCPM"]=C - np.array([np.max(np.delete(C,u,axis=1),axis=1) for u in range(S)]).T
SCORES["T-norm"]=(C-mu_t[None,:])/sd_t[None,:]
SCORES["Z-norm"]=np.stack([znorm_row(C[i]) for i in range(C.shape[0])])
SCORES["S-norm"]=0.5*(SCORES["Z-norm"]+SCORES["T-norm"])

# ---- Mahalanobis: per-subject mean + shared pooled covariance from enrol embeddings ----
mns=np.stack([Eenr[s].mean(0) for s in subjects])               # (S,128)
cov=np.cov(np.concatenate([Eenr[s]-Eenr[s].mean(0) for s in subjects]).T)
inv=np.linalg.pinv(cov+1e-3*np.eye(128))
diff=Z[:,None,:]-mns[None,:,:]                                   # (Ntest,S,128)
maha=np.einsum('isd,de,ise->is', diff, inv, diff)               # squared distance
SCORES["Mahalanobis"]=-maha                                     # higher = better

# ---- per-user LDA / SVM: pos=subject enrol, neg=other-subject enrol (subsampled) ----
ldaS=np.zeros((Z.shape[0],S)); svmS=np.zeros((Z.shape[0],S))
rng=np.random.default_rng(0)
for u,s in enumerate(subjects):
    pos=Eenr[s]
    negpool=np.concatenate([Eenr[o] for o in subjects if o!=s])
    neg=negpool[rng.choice(len(negpool),min(len(pos)*3,len(negpool)),replace=False)]
    Xtr=np.concatenate([pos,neg]); ytr=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    ldaS[:,u]=LDA().fit(Xtr,ytr).decision_function(Z)
    svmS[:,u]=SVC(kernel="linear",C=1.0).fit(Xtr,ytr).decision_function(Z)
SCORES["LDA(per-user)"]=ldaS; SCORES["SVM(per-user)"]=svmS

# ---- fuse mean_5 over consecutive probe windows of the same subject, then EER ----
def fuse_scores(Sf, gid, N=5):
    g=[]; im=[]
    for u in range(S):
        rows=np.where(gid==u)[0]                # genuine probes of subject u
        col=Sf[rows]                            # (n, S) scores for these probes
        for k in range(0,len(rows)-N+1,N):
            blk=col[k:k+N].mean(0)              # fused score vs every claim
            g.append(blk[u]); im.extend(np.delete(blk,u))
    return np.array(g), np.array(im)
def eer(gscores,iscores):
    t=np.linspace(min(gscores.min(),iscores.min()),max(gscores.max(),iscores.max()),600)
    frr=np.array([(gscores<th).mean() for th in t]); far=np.array([(iscores>=th).mean() for th in t])
    k=np.argmin(np.abs(far-frr)); return (far[k]+frr[k])/2*100

print(f"\n{'method':16s}  EER%(mean_5)")
rows=[]
for name,Sf in SCORES.items():
    g,im=fuse_scores(Sf,gid,5); e=eer(g,im); rows.append((name,e))
    print(f"{name:16s}  {e:6.2f}")
print("\nLaTeX rows:")
for n,e in sorted(rows,key=lambda r:r[1]): print(f"{n} & {e:.2f} \\\\")
