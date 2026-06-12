# =====================================================================================
# measure_latency.py  --  VERIFIED inference latency / throughput for the paper.
# Reports: model parameters, encoder forward-pass time per window, CCPM scoring time,
# end-to-end per-window latency, throughput, and W=5 fused-decision latency.
# These are REAL measured numbers -- paste the printout into the paper / tracker.
#
# RUN (on Brev, inside the repo that has the trained checkpoint):
#   cd ~/24PHD1237/BED/FED_REAL_BED_ALL
#   python measure_latency.py
# =====================================================================================
import glob, time, numpy as np, torch
from fed_real_bed.config import load_config
from fed_real_bed.models import EEGAuthenticator

cfg = load_config("configs/brev_gpu_p0_ecapa_ccpm.yaml")
samples = int(round(cfg.raw["data"]["window_sec"] * cfg.raw["data"]["target_sampling_rate"]))
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = sorted(glob.glob("artifacts/results/m10_ccpm_seed*/checkpoints/best.pt") +
            glob.glob("artifacts/results/*ccpm*/checkpoints/best.pt"))[0]
m = EEGAuthenticator(cfg.raw, samples=samples).to(dev)
st = torch.load(ck, map_location=dev, weights_only=False)
m.load_state_dict(st["model"] if isinstance(st, dict) and "model" in st else st)
m.eval()
print(f"device      : {dev}")
print(f"checkpoint  : {ck}")
print(f"parameters  : {sum(p.numel() for p in m.parameters()):,}")

def bench(fn, N=500, warm=20):
    for _ in range(warm): fn()
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): fn()
    if dev == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N * 1000.0   # ms

# ---- encoder forward, batch = 1 (one 2 s window) ----
x = torch.randn(1, 14, samples, device=dev)
def enc():
    with torch.no_grad(): m(x)
enc_ms = bench(enc)
print(f"encoder fwd (batch=1)         : {enc_ms:.2f} ms/window")

# ---- CCPM scoring: 1 probe vs 21 identities x 3 prototypes (128-D, unit norm) ----
S = 21
P = torch.randn(S, 3, 128, device=dev); P = P / P.norm(dim=2, keepdim=True)
z = torch.randn(1, 128, device=dev);    z = z / z.norm()
def ccpm():
    with torch.no_grad():
        c = torch.einsum('id,skd->isk', z, P).amax(2)          # best self per identity (1,S)
        rival = torch.stack([torch.cat([c[:, :u], c[:, u+1:]], 1).amax(1)
                             for u in range(S)], 1)             # best rival (1,S)
        _ = c - rival                                          # CCPM margin
ccpm_ms = bench(ccpm)
print(f"CCPM scoring (1 probe, {S} ids) : {ccpm_ms:.4f} ms")

tot = enc_ms + ccpm_ms
print(f"end-to-end per window         : {tot:.2f} ms  (~{1000.0/tot:.0f} windows/s)")
print(f"fused decision (W=5 windows)  : {5*tot:.1f} ms")
print("\nPAPER SENTENCE (fill with YOUR printed numbers):")
print(f"  'On {dev.upper()}, the {sum(p.numel() for p in m.parameters()):,}-parameter encoder runs in")
print(f"   {enc_ms:.2f} ms/window; CCPM scoring adds {ccpm_ms:.3f} ms, giving a fused (W=5)")
print(f"   verification latency of {5*tot:.1f} ms.'")
