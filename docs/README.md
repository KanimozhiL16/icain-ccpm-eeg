# Cross-Session Consumer-EEG Verification with Cohort-Competing Prototype Margins (CCPM)

**ICAIN 2026 (Springer LNNS).** A reproducible, **leakage-free** cross-session EEG
biometric **verification** benchmark on the public **Biometric EEG Dataset (BED)**,
introducing **CCPM** — a *parameter-free* cohort-competing prototype-margin score that
accepts a claim only when the claimed identity's enrolled prototype outscores the
strongest competing identity in the same stimulus context.

> **Scope (read me):** every reported result uses a single, **frozen** ECAPA-EEG
> encoder with **CCPM** scoring. **No federated learning is used in any reported
> result.** The repository also retains FedProx/FedAvg/personalization variants that
> were implemented and evaluated but **excluded** from the paper because they gave no
> reliable improvement (paper, *Limitations*); they are kept only for transparency.

## Abstract
EEG signals are an emerging biometric trait, but their reliability across recording
sessions on consumer hardware is poorly characterised, and many reported results are
inflated by evaluation leakage (mixing sessions in enrolment, or tuning the threshold
on the test session). We present a leakage-free cross-session benchmark on BED (21
subjects, 14-channel Emotiv EPOC+, three sessions) under two protocols — a two-session
calibrated-enrolment protocol (P0) and a strict single-session protocol (P2) whose
test session is never seen during training or thresholding — and introduce CCPM, a
parameter-free verification-stage score. CCPM gives a statistically significant,
large-effect improvement over cosine scoring on the same encoder. The contribution is
a rigorous benchmark and a calibration-level scoring improvement, **not** a deployable
authentication system.

## Headline results (executed, multi-seed)
| Setting | Method | EER | Notes |
|---|---|---|---|
| **P0** (two-session enrolment, held-out r03) | **ECAPA + CCPM** | **16.09 ± 1.10%** | 10 seeds |
| P0 | ECAPA + cosine | 17.38 ± 0.93% | 10 seeds; CCPM lower in 8/10 |
| P0 significance | CCPM vs cosine | paired *t* p=0.005, Wilcoxon p=0.010, Cohen's *d_z*=1.17 | large effect |
| **P2** (strict single-session) | ECAPA + CCPM | **26.52 ± 1.26%** | 5 seeds |
| Leakage control | session-mixed split | **0.35 ± 0.05%** vs 15.25 ± 1.16% held-out | ~15-pt inflation removed |
| Efficiency | 398,973-param encoder | 1.82 ms/window; CCPM +0.81 ms; ~380 windows/s | single GPU |

### Scoring-rule benchmark (P0, same ECAPA embeddings, 10 seeds — ranking is the finding)
| Score | EER (%) | | Score | EER (%) |
|---|---|---|---|---|
| **CCPM** | **15.48 ± 1.09** | | Plain cosine | 17.19 ± 1.17 |
| Z-norm | 16.41 ± 1.12 | | SVM (per-user) | 16.87 ± 1.25 |
| S-norm | 16.46 ± 0.83 | | LDA (per-user) | 18.96 ± 1.40 |
| T-norm | 16.77 ± 0.79 | | Mahalanobis | 22.37 ± 1.33 |

## Repository map
```
code/        ECAPA-EEG encoder (models.py), CCPM scoring + EER (evaluation.py),
             leakage-free protocols r01+r02 -> r03 (protocols.py), data / training
configs/     per-seed experiment YAMLs (m10_ccpm_*, m10_cosine_*, p2_*, ...)
scripts/     run + analysis: baselines_p0[_10seed].py, leakage_demo[_10seed].py,
             compute_cohens_d.py, measure_latency.py, posthoc_coverage_thresholding.py
results/     every metrics.json + config.resolved.yaml + logs/ + baselines/
checkpoints/ frozen ECAPA encoders (best.pt) -- re-score without retraining
inspection/  explanatory figures (embeddings, prototypes, CCPM vectors)
paper/       manuscript (main.tex) + figures
RUN_INDEX.csv, MANIFEST.txt   run index + SHA-256 checksums
```

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt
# P0 headline (10 seeds): CCPM 16.09±1.10 vs cosine 17.38±0.93
python scripts/compute_cohens_d.py
# scoring-rule benchmark (8 scores): CCPM 15.48 best
python scripts/baselines_p0_10seed.py
# leakage control: honest 15.25 vs session-mixed 0.35
python scripts/leakage_demo_10seed.py
# inference latency: 398,973 params; 1.82 ms/window
python scripts/measure_latency.py
```
The BED preprocessed cache and model checkpoints/large `roc.csv` are archived on
Zenodo (below); place the cache under `artifacts/cache/` before re-running.

## Data & checkpoints (archive)
- **BED dataset (cite this):** Arnau-González, P., Katsigiannis, S., Arevalillo-Herráez, M.,
  Ramzan, N. *BED: A new dataset for EEG-based biometrics.* IEEE Internet of Things J. **8**(15),
  12219–12230 (2021). DOI `10.1109/JIOT.2021.3061727`.
  Data record: Zenodo DOI `10.5281/zenodo.4309472` — https://doi.org/10.5281/zenodo.4309472
  (all versions: `10.5281/zenodo.4309471`). BED is governed by its original license; this
  repository does **not** redistribute BED.
- **This work's preprocessed cache + checkpoints + full `roc.csv`:** Zenodo DOI
  **`10.5281/zenodo.XXXXXXX`** *(replace after you upload `icain_heavy_for_zenodo.tar.gz`)*.

## Citation
```bibtex
@inproceedings{kanimozhi2026ccpm,
  title     = {A Leakage-Free Cross-Session Benchmark for Consumer-Grade EEG
               Biometric Verification with Cohort-Competing Prototype Margins},
  author    = {Kanimozhi, L. and Shridevi, S.},
  booktitle = {Proc. Int. Conf. on Artificial Intelligence and Networking (ICAIN),
               Springer LNNS},
  year      = {2026}
}
```
For the **dataset**, also cite:
```bibtex
@article{arnaugonzalez2021bed,
  title   = {{BED}: A New Dataset for {EEG}-Based Biometrics},
  author  = {Arnau-Gonz\'alez, Pablo and Katsigiannis, Stamos and
             Arevalillo-Herr\'aez, Miguel and Ramzan, Naeem},
  journal = {IEEE Internet of Things Journal},
  volume  = {8}, number = {15}, pages = {12219--12230}, year = {2021},
  doi     = {10.1109/JIOT.2021.3061727}
}
```

## Acknowledgements
Experiments ran on NVIDIA A100 GPUs provided through the **NVIDIA Academic Grant
Program** (awarded to S. Shridevi).

## License
Code: MIT (suggested). Paper/figures: CC BY 4.0 (suggested). BED dataset is governed
by its original license.
