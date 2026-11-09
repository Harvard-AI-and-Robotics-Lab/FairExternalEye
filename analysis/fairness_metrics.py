"""Shared metrics and prediction alignment for fairness-strategy analyses."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


GROUP_LABELS = {
    "sex": ["0", "1"],
    "race": ["Asian", "Black or African American", "White", "Other or Unknown"],
    "ethnic": ["0", "1", "Unknown or Not Reported"],
    "language": ["English", "Spanish", "Other or Unknown"],
}

METHOD_TO_EXPERIMENT = {
    "DRO": "trm_{disease}_DRO_{group}",
    "Adv": "trm_{disease}_ADV_{group}",
    "FSCL": "trm_{disease}_FSCL_{group}_finetune",
    "FIS": "trm_{disease}_FIS_{group}",
}

TASKS = {
    "DR": {"disease": "dr", "baseline": "trm_dr_baseline"},
    "AMD-BIN": {"disease": "amdbin", "baseline": "trm_amdbin_baseline"},
}


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def es_auc(global_auc: float, group_aucs: list[float]) -> float:
    values = np.asarray([value for value in group_aucs if np.isfinite(value)], dtype=float)
    if values.size == 0 or not np.isfinite(global_auc):
        return np.nan
    penalty = np.sum(np.abs(values - global_auc))
    return float(global_auc / (1.0 + penalty))


def group_auc_values(df: pd.DataFrame, group: str, score_col: str) -> list[float]:
    values = []
    for label in GROUP_LABELS[group]:
        subgroup = df[df[group].astype(str) == label]
        values.append(
            safe_auc(
                subgroup["y_true"].to_numpy(dtype=float),
                subgroup[score_col].to_numpy(dtype=float),
            )
        )
    return values


def metric_values(df: pd.DataFrame, group: str, score_col: str) -> dict[str, float]:
    y_true = df["y_true"].to_numpy(dtype=float)
    scores = df[score_col].to_numpy(dtype=float)
    overall_auc = safe_auc(y_true, scores)
    group_aucs = group_auc_values(df, group, score_col)
    gaps = np.abs(np.asarray(group_aucs, dtype=float) - overall_auc)
    return {
        "overall_auc": overall_auc,
        "es_auc": es_auc(overall_auc, group_aucs),
        "mean_abs_group_gap": float(np.nanmean(gaps)),
        "max_abs_group_gap": float(np.nanmax(gaps)),
    }


def load_prediction(preds_dir: Path, experiment: str) -> pd.DataFrame:
    path = preds_dir / f"{experiment}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    columns = [
        "image_name",
        "y_true",
        "y_score",
        "sex",
        "race",
        "ethnic",
        "language",
        "checkpoint_path",
        "checkpoint_log",
    ]
    return pd.read_csv(path, usecols=columns)


def make_pair_df(base_df: pd.DataFrame, fair_df: pd.DataFrame) -> pd.DataFrame:
    baseline = base_df.rename(columns={"y_score": "baseline_score"})
    fairness = fair_df[["image_name", "y_true", "y_score"]].rename(
        columns={"y_score": "fair_score", "y_true": "fair_y_true"}
    )
    pair = baseline.merge(fairness, on="image_name", how="inner")
    return pair[pair["y_true"].astype(float) == pair["fair_y_true"].astype(float)].copy()
