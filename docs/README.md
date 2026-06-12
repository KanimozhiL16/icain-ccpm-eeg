# FED-REAL-BED

Research-grade pipeline for **real-time cross-session EEG biometric authentication** on the BED dataset using consumer-grade Emotiv EPOC+ EEG.

This repository is designed to move beyond the previous notebook result (`r01+r02 -> r03`, best DOMCS-EEG EER `22.93%`, AUC `0.846`) by adding stricter session protocols, raw preprocessing, quality-aware metric learning, prototype verification, temporal fusion, session adaptation, and federated personalization.

## Core Pipeline

```text
raw BED -> manifest -> filtering/artifact scoring -> 2s windows
        -> CNN/TCN/Transformer/ECAPA encoder -> 128D embedding
        -> enrollment prototypes -> cosine verification
        -> EER/AUC/ROC/FAR/FRR/TAR/latency reports
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[api,dev]"
```

On NVIDIA Brev, use a CUDA PyTorch image or install the CUDA wheel recommended by PyTorch before `pip install -e .`.

## Configure

Copy and edit:

```bash
cp configs/bed_p2_quality_fed.yaml configs/local.yaml
```

Set:

```yaml
paths:
  bed_raw_root: /path/to/BED/raw
  cache_root: ./artifacts/cache
  results_root: ./artifacts/results
```

The raw loader supports CSV/TXT, MAT, NPZ, and optional EDF/BDF through `mne`. It infers subject, session/run, and stimulus from filenames or directory names. If your BED copy uses a different naming convention, create a manifest CSV with columns:

```text
path,subject,session,stimulus
```

and set `paths.manifest_csv`.

## Run

Preprocess raw BED into reproducible windows:

```bash
fed-real-bed preprocess --config configs/local.yaml
```

Train/evaluate one experiment:

```bash
fed-real-bed train --config configs/local.yaml
```

Run the ablation grid:

```bash
python scripts/run_ablation.py --config configs/local.yaml --grid configs/ablation_grid.yaml
```

Start the real-time API after a checkpoint and prototype file exist:

```bash
uvicorn fed_real_bed.api:app --host 0.0.0.0 --port 8000
```

## Protocols

- `P0`: prior reproduction, `r01+r02 train/enroll -> r03 probe`.
- `P1`: original BED verification, `Session 1 enrollment -> Session 2/3 test`.
- `P2`: primary paper protocol, `Session 1 train/enroll -> Session 2 validation/threshold selection -> Session 3 final test`.
- `P3`: cross-task protocol, `RC enrollment -> non-RC probe` for baseline-to-task robustness.

For Q1 publication strength, report all protocols but make `P2` the primary claim because Session 3 remains untouched until final evaluation.

## Outputs

Each run saves:

- `config.resolved.yaml`
- `checkpoints/best.pt`
- `prototypes.npz`
- `metrics.json`
- `scores.csv`
- `roc.csv`
- `roc.png`
- per-subject/session/stimulus EER tables
- training log CSV

## Novelty Matrix

Implemented as ablation-ready modules:

- Face biometrics: ArcFace, Sub-center ArcFace, AdaFace-style quality-aware angular margin.
- Voice biometrics: ECAPA-style temporal channel attention and adaptive score normalization.
- Gaze/behavioral biometrics: temporal decision fusion and liveness/quality rejection.
- Privacy-preserving biometrics: FedAvg, FedProx, personalized federated updates, local prototypes.
- EEG-specific validity: cross-session/template-aging protocols, RC-to-task protocol, artifact-aware rejection.

