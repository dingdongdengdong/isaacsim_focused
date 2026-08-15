# DACL10K crack/rust YOLO segmentation

The uploaded Dataset Ninja export contains zero-byte image and annotation files,
and its bundled `yolo11n-seg.pt` is truncated. The preparation script rebuilds a
usable two-class dataset from the verified official DACL10K v2 development ZIP.

## Prepare

```bash
python3 prepare_dacl10k_yolo.py \
  /home/dong/ai/data/dacl10k/dacl10k_v2_devphase.zip \
  dacl10k-DatasetNinja/dacl10k-DatasetNinja/dacl10k_yolo
```

## Train

Use the valid local YOLO26 nano segmentation checkpoint. The command below
writes all artifacts under `runs/dacl10k-crack-rust-yolo26n-seg-30e-b24/`.

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/dong/ai/.venv/bin/yolo segment train \
  model=dacl10k-DatasetNinja/dacl10k-DatasetNinja/dacl10k_yolo/yolo26n-seg.pt \
  data=dacl10k-DatasetNinja/dacl10k-DatasetNinja/dacl10k_yolo/dacl10k.yaml \
  epochs=30 imgsz=640 batch=24 device=0 seed=42 workers=8 \
  project=runs name=dacl10k-crack-rust-yolo26n-seg-30e-b24
```

If training is interrupted after a checkpoint has been written, resume it with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/dong/ai/.venv/bin/yolo segment train \
  resume model=runs/dacl10k-crack-rust-yolo26n-seg-30e-b24/weights/last.pt
```

Validate the best checkpoint independently with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/dong/ai/.venv/bin/yolo segment val \
  model=runs/dacl10k-crack-rust-yolo26n-seg-30e-b24/weights/best.pt \
  data=dacl10k-DatasetNinja/dacl10k-DatasetNinja/dacl10k_yolo/dacl10k.yaml \
  imgsz=640 batch=24 device=0 workers=8 \
  project=runs name=dacl10k-crack-rust-yolo26n-seg-val
```

The dataset is licensed CC BY-NC 4.0. It is suitable for non-commercial
research unless separate permission is obtained.
