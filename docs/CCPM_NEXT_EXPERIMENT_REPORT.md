# CCPM Next Experiment Report

## Purpose

The current BED results show that strict cross-session EEG authentication is still difficult. The best implemented strict P2 result so far is around 34.73% EER, while the reproduced prior P0 protocol gives 22.91% single-window EER and 19.32% EER with 10-window averaging.

The next implemented method is **Cohort-Competing Prototype Margin (CCPM)**, a verification-stage method inspired by face and speaker biometric systems.

## Research Idea

Standard prototype verification uses:

`score(x, u) = cos(f(x), p_u)`

where `f(x)` is the EEG embedding and `p_u` is the enrolled prototype for claimed user `u`.

CCPM uses a cohort-normalized competing score:

`score_ccpm(x, u) = max_k cos(f(x), p_{u,k}) - max_{v != u, j} cos(f(x), p_{v,j})`

If stimulus-conditioned prototypes are enabled, the competing cohort is restricted to the same stimulus context when possible.

## Why This Is Scientifically Valid

- It uses only enrolled cohort prototypes.
- It does not use the probe/test label.
- It is compatible with real-time authentication because it is only a matrix similarity and max operation.
- It directly follows likelihood-ratio logic used in face and voice verification: accept a claim only if the claimed identity beats the strongest competing identity.

## Implemented Files

- `fed_real_bed/evaluation.py`
  - Added `score_norm: ccpm`
  - Aliases: `cohort_margin`, `adaptive_snorm_margin`
- `configs/brev_gpu_p0_domcs_ccpm.yaml`
  - DOMCS-style encoder, P0 protocol, kmeans prototypes, CCPM scoring
- `configs/brev_gpu_p0_ecapa_ccpm.yaml`
  - ECAPA-style encoder, P0 protocol, stimulus-conditioned kmeans prototypes, CCPM scoring

## Next Runs

Run DOMCS+CCPM first because it directly compares against the reproduced prior result:

`fed_real_bed_p0_domcs_reproduction_brev`

Then run ECAPA+CCPM:

`fed_real_bed_p0_ecapa_ccpm_brev`

## Evidence Baseline Before CCPM

Observed P0 DOMCS reproduction:

- Validation EER: 1.6535%
- Test single-window EER: 22.9083%
- Test mean-5 EER: 19.8411%
- Test mean-10 EER: 19.3230%
- Test mean-10 AUC: 0.8919

The valid claim is the r03 test EER, not the optimistic validation EER.
