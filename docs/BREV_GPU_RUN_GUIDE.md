# Running FED-REAL-BED Privately on NVIDIA Brev

## Upload Package

Upload `fed_real_bed_brev_upload.zip` to your Brev JupyterLab home folder, preferably under:

```text
/home/24PHD1237/
```

Then open a JupyterLab terminal and run:

```bash
cd /home/24PHD1237
unzip fed_real_bed_brev_upload.zip -d FED_REAL_BED
cd FED_REAL_BED
```

The package includes the preprocessed RC cache:

```text
artifacts/cache/bed_windows_256hz_2.0s.npz
```

So the first GPU experiment does **not** require the full raw BED dataset upload.

## Private Environment

Create your own environment inside your own project folder:

```bash
cd /home/24PHD1237/FED_REAL_BED
python3 -m venv .venv_fed_real_bed
source .venv_fed_real_bed/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[api,dev]"
```

Check GPU visibility:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
PY
```

## Protect Your Work on a Shared Machine

Use folder permissions so other Linux users cannot accidentally edit/delete your files:

```bash
chmod -R go-rwx /home/24PHD1237/FED_REAL_BED
```

If the same shared login is used by everyone, permissions cannot fully protect you. In that case:

- Keep your project in a uniquely named folder: `/home/24PHD1237/FED_REAL_BED_KANIMOZHI`.
- Keep daily backup ZIPs in another folder.
- Push code to a private GitHub repo.
- Download `artifacts/results` after each important run.
- Do not store work in common folders like `/tmp`, `Downloads`, or shared class folders.

The grant email says the node is **one 8xA100 instance used by one user at a time**. Coordinate a time slot with your supervisor/lab before long runs.

## Run Training on GPU

Start a persistent session so training continues if the browser disconnects:

```bash
tmux new -s fedbed
```

Inside tmux:

```bash
cd /home/24PHD1237/FED_REAL_BED
source .venv_fed_real_bed/bin/activate
python -m fed_real_bed.cli train --config configs/brev_gpu_rc.yaml
```

Detach without stopping training:

```text
Ctrl+B, then D
```

Resume later:

```bash
tmux attach -t fedbed
```

Watch GPU use:

```bash
watch -n 2 nvidia-smi
```

## Save Results

After training:

```bash
zip -r fed_real_bed_results_$(date +%Y%m%d_%H%M).zip artifacts/results
```

Download that ZIP from JupyterLab.

