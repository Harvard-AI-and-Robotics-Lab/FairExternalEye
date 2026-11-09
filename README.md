# FairExternalEye

Training and evaluation code for external-eye/fundus disease classification and
demographic fairness. It includes WideResNet, EfficientNet, ViT/DeiT, DRO,
adversarial training, FSCL, FIS, LLaVA, Qwen2-VL, and MedGemma.

## Setup

The tested environment used Python 3.9.21, PyTorch 2.5.0, and CUDA 12.4.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Optional dependencies:

```bash
# WideResNet MXNet weight conversion
pip install -r requirements-wideresnet.txt

# LLaVA, Qwen2-VL, and MedGemma
pip install -r requirements-vlm.txt
```

## Data

Arrange approved data as:

```text
data/
├── images/
├── splits/
│   ├── train_final.npy
│   └── val_final.npy
└── metadata/
    └── metadata.csv
```

Split arrays use this column order:

```text
grade, image, sex, race, ethnic, language, dr, amd, glaucoma
```

Outcome definitions:

- DR: `0` negative, `1` positive.
- AMD-BIN: `0`, `1`, `2`, and `255` negative; `3` positive.
- Glaucoma: `255` excluded; `-1` and `0` negative; `1` positive.

See `data/README.md` for the complete schema and privacy requirements. Validate
the setup before training:

```bash
python validate_layout.py
python -m unittest discover -s tests -v
```

## Training

`--task` accepts `dr`, `amd`, or `glaucoma`. For fairness methods,
`--group_type` is `0` sex, `1` race, `2` ethnicity, or `3` language.

```bash
# Baselines
CUDA_VISIBLE_DEVICES=0 python train_wideresnet.py \
  --name wresnet_dr_baseline --task dr
CUDA_VISIBLE_DEVICES=0 python train_eff.py \
  --name eff_dr_baseline --task dr
CUDA_VISIBLE_DEVICES=0 python train_trm.py \
  --name trm_dr_baseline --task dr

# DRO, adversarial training, and FIS by race
CUDA_VISIBLE_DEVICES=0 python train_trm.py \
  --name trm_dr_DRO_race --task dr --group_type 1
CUDA_VISIBLE_DEVICES=0 python train_trm_adv.py \
  --name trm_dr_ADV_race --task dr --group_type 1
CUDA_VISIBLE_DEVICES=0 python train_trm_fis.py \
  --name trm_dr_FIS_race --task dr --group_type 1

# FSCL stage 1, then fine-tune the checkpoint with the best validation AUC
CUDA_VISIBLE_DEVICES=0 python train_trm_fscl.py \
  --name trm_dr_FSCL_race --task dr --group_type 1
CUDA_VISIBLE_DEVICES=0 python train_trm_fsclft.py \
  --name trm_dr_FSCL_race_finetune --task dr --group_type 1 \
  --fscl-checkpoint experiments/trm_dr_FSCL_race/ckpt/044net_trm.pth

# VLM example
CUDA_VISIBLE_DEVICES=0 python train_llava_ft_final.py \
  --name dr_llava --task dr
```

Qwen2-VL and MedGemma use `train_qwen_ft.py` and `train_medgemma_ft2.py`.
Add `--wandb` for optional W&B logging or `--adapter-checkpoint PATH` to
evaluate a saved VLM adapter. Exact experiment names are listed in
`experiments.yaml`; all hyperparameters are available through `--help`.

## Analysis

Training writes checkpoints to `experiments/`. Run the bootstrap analysis first
to create `outputs/preds_cache/`, then run paired comparisons.

```bash
python analysis/bootstrap_cnn_vit_results.py \
  --output_dir outputs --n_boot 1000 --bootstrap_sample_size 20380

python analysis/metadata_only_baselines.py
python analysis/metadata_vs_image_delta_metrics.py

python analysis/fairness_strategy_paired_tests_corrected.py \
  --inference delong \
  --preds-cache-dir outputs/preds_cache \
  --output-csv outputs/fairness_delong.csv \
  --output-xlsx outputs/fairness_delong.xlsx
```
