#!/usr/bin/env python3
"""Validate the public repository's expected local data layout."""

from pathlib import Path
import csv

import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "data" / "images"
SPLIT_ROOT = ROOT / "data" / "splits"
SPLIT_COLUMNS = ["grade", "image", "sex", "race", "ethnic", "language", "dr", "amd", "glaucoma"]
VALID_RACE = {"Asian", "Black or African American", "White", "Other or Unknown"}
VALID_ETHNIC = {"0", "1", "Unknown or Not Reported"}
VALID_LANGUAGE = {"English", "Spanish", "Other or Unknown"}
VALID_LABELS = {
    "dr": {0.0, 1.0},
    "amd": {0.0, 1.0, 2.0, 3.0, 255.0},
    "glaucoma": {-1.0, 0.0, 1.0, 255.0},
}


def load_split(name: str) -> np.ndarray:
    path = SPLIT_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")
    split = np.load(path, allow_pickle=True)
    if split.ndim != 2 or split.shape[1] != len(SPLIT_COLUMNS):
        raise ValueError(
            f"{path} must have shape (N, {len(SPLIT_COLUMNS)}); found {split.shape}"
        )
    return split


def values(split: np.ndarray, column: str) -> set[str]:
    idx = SPLIT_COLUMNS.index(column)
    return {str(value) for value in split[:, idx]}


def check_categories(split: np.ndarray, split_name: str) -> None:
    checks = {
        "sex": {"0", "1"},
        "race": VALID_RACE,
        "ethnic": VALID_ETHNIC,
        "language": VALID_LANGUAGE,
    }
    for column, allowed in checks.items():
        unexpected = values(split, column) - allowed
        if unexpected:
            raise ValueError(f"{split_name}: unexpected {column} values: {sorted(unexpected)}")


def check_labels(split: np.ndarray, split_name: str) -> None:
    for column, allowed in VALID_LABELS.items():
        index = SPLIT_COLUMNS.index(column)
        try:
            observed = {float(value) for value in split[:, index]}
        except (TypeError, ValueError) as error:
            raise ValueError(f"{split_name}: nonnumeric {column} label") from error
        unexpected = observed - allowed
        if unexpected:
            raise ValueError(f"{split_name}: unexpected {column} labels: {sorted(unexpected)}")


def check_metadata(train: np.ndarray, val: np.ndarray) -> None:
    path = ROOT / "data" / "metadata" / "metadata.csv"
    if not path.exists():
        print("Metadata CSV not found; metadata-only analyses will be unavailable.")
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        required = {"jpgfile", "age", "race", "ethnicity"}
        if not required.issubset(columns):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        metadata_images = {row["jpgfile"] for row in reader}
    split_images = values(train, "image") | values(val, "image")
    missing = split_images - metadata_images
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} split images; examples: {sorted(missing)[:10]}"
        )


def check_images(split: np.ndarray, split_name: str) -> None:
    image_idx = SPLIT_COLUMNS.index("image")
    missing = []
    for value in split[:, image_idx]:
        image = IMAGE_ROOT / str(value)
        if not image.exists():
            missing.append(str(value))
            if len(missing) == 10:
                break
    if missing:
        raise FileNotFoundError(
            f"{split_name}: missing images beneath {IMAGE_ROOT}; examples: {missing}"
        )


def main() -> None:
    train = load_split("train_final.npy")
    val = load_split("val_final.npy")
    check_categories(train, "train")
    check_categories(val, "validation")
    check_labels(train, "train")
    check_labels(val, "validation")
    check_images(train, "train")
    check_images(val, "validation")
    check_metadata(train, val)

    train_images = values(train, "image")
    val_images = values(val, "image")
    overlap = train_images & val_images
    if overlap:
        examples = sorted(overlap)[:10]
        raise ValueError(f"Image overlap between splits ({len(overlap)}); examples: {examples}")

    print(f"Train samples: {len(train):,}")
    print(f"Validation samples: {len(val):,}")
    print("Data layout validation passed.")
    print("Note: patient-level separation must be verified using the protected patient identifier table.")


if __name__ == "__main__":
    main()
