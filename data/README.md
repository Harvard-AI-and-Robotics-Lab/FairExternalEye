# Data preparation

Project data are intentionally excluded from source control. Use only data for
which you have appropriate institutional approval and data-use authorization.

## Required files

```text
data/images/
data/splits/train_final.npy
data/splits/val_final.npy
data/metadata/metadata.csv
```

The split arrays must contain nine columns in this order:

```text
grade, image, sex, race, ethnic, language, dr, amd, glaucoma
```

`image` must match a file beneath `data/images/`. `grade` is retained because
the original evaluation cache identifies a sample as `<image>_<grade>`.

`metadata.csv` must contain `jpgfile`, `age`, `race`, and `ethnicity`. Rename
site-specific source columns to this public schema before running the analyses.

## Labels

- DR: `0` negative, `1` positive.
- AMD-BIN: `0`, `1`, `2`, `255` negative; `3` positive.
- Glaucoma: exclude `255`; map `-1` and `0` to negative and `1` to positive.

## Demographic categories

- Sex: `0`, `1`
- Race: `Asian`, `Black or African American`, `White`, `Other or Unknown`
- Ethnicity: `0`, `1`, `Unknown or Not Reported`
- Language: `English`, `Spanish`, `Other or Unknown`

## Required checks

Before training:

1. Confirm there is no patient overlap between train and validation splits.
2. Confirm every image identifier resolves to exactly one image file.
3. Confirm all disease and demographic values use the categories above.
4. Remove direct identifiers and protected health information.
5. Run `python validate_layout.py`.
