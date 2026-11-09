from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.labels import binary_label, is_valid_label


SPLIT_COLUMNS = ["grade", "image", "sex", "race", "ethnic", "language", "dr", "amd", "glaucoma"]
FEATURES = ["age", "race", "ethnicity"]
SPECIFICITIES = [0.80, 0.85, 0.95]


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race"] = out["race"].astype(str).apply(
        lambda x: x if x in {"Asian", "Black or African American", "White"} else "Other or Unknown"
    )
    out["ethnicity"] = out["ethnicity"].astype(str).apply(
        lambda x: x if x in {"0", "1"} else "Unknown or Not Reported"
    )
    return out


@dataclass
class TaskDef:
    name: str
    source_col: str
    positive_rule: str


TASKS = [
    TaskDef(name="dr", source_col="dr", positive_rule="label == 1"),
    TaskDef(name="amd", source_col="amd", positive_rule="label > 0"),
    TaskDef(name="glaucoma", source_col="glaucoma", positive_rule="label > 0"),
]


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int,
    seed: int,
    bootstrap_sample_size: int | None = None,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draw_size = n if bootstrap_sample_size is None else int(bootstrap_sample_size)
    if draw_size <= 0:
        raise ValueError("bootstrap_sample_size must be positive")
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=draw_size)
        yb = y_true[idx]
        if np.unique(yb).size < 2:
            continue
        samples.append(roc_auc_score(yb, y_score[idx]))
    if not samples:
        return np.nan, np.nan
    arr = np.array(samples, dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def sensitivity_at_specificity(y_true: np.ndarray, y_score: np.ndarray, specificity: float) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score)
    target_fpr = 1.0 - specificity
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return np.nan
    return float(np.max(tpr[valid]))


def bootstrap_sensitivity_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    specificity: float,
    n_boot: int,
    seed: int,
    bootstrap_sample_size: int | None = None,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draw_size = n if bootstrap_sample_size is None else int(bootstrap_sample_size)
    if draw_size <= 0:
        raise ValueError("bootstrap_sample_size must be positive")
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=draw_size)
        yb = y_true[idx]
        if np.unique(yb).size < 2:
            continue
        samples.append(sensitivity_at_specificity(yb, y_score[idx], specificity))
    if not samples:
        return np.nan, np.nan
    arr = np.array(samples, dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def build_merged_dataframe(datasplit_dir: Path, metadata_csv: Path) -> pd.DataFrame:
    train = np.load(datasplit_dir / "train_final.npy", allow_pickle=True)
    val = np.load(datasplit_dir / "val_final.npy", allow_pickle=True)
    train_df = pd.DataFrame(train, columns=SPLIT_COLUMNS)
    val_df = pd.DataFrame(val, columns=SPLIT_COLUMNS)
    train_df["split"] = "train"
    val_df["split"] = "val"
    split_df = pd.concat([train_df, val_df], ignore_index=True)

    meta = pd.read_csv(metadata_csv, low_memory=False)
    join_df = meta[["jpgfile"] + FEATURES].copy()
    join_df = join_df.drop_duplicates(subset=["jpgfile"])
    join_df = normalize_categories(join_df)

    merged = split_df.merge(join_df, left_on="image", right_on="jpgfile", how="left")
    if merged["jpgfile"].isna().any():
        missing = int(merged["jpgfile"].isna().sum())
        raise ValueError(f"Metadata are missing for {missing} split images.")
    return merged


def valid_task_rows(df: pd.DataFrame, task: TaskDef) -> pd.DataFrame:
    tmp = df.copy()
    tmp["label"] = pd.to_numeric(tmp[task.source_col], errors="coerce")
    valid = tmp["label"].notna() & tmp["label"].map(
        lambda value: is_valid_label(value, task.name) if pd.notna(value) else False
    )
    return tmp[valid]


def to_binary_label(label: pd.Series, task_name: str) -> np.ndarray:
    return np.asarray([binary_label(value, task_name) for value in label], dtype=int)


def logistic_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["age"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["race", "ethnicity"],
            ),
        ]
    )
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    return Pipeline([("preprocess", pre), ("model", clf)])


def gbm_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                ["age"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                ["race", "ethnicity"],
            ),
        ]
    )
    clf = HistGradientBoostingClassifier(random_state=42)
    return Pipeline([("preprocess", pre), ("model", clf)])


def logistic_pipeline_age_only() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["age"],
            ),
        ]
    )
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    return Pipeline([("preprocess", pre), ("model", clf)])


def gbm_pipeline_age_only() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                ["age"],
            ),
        ]
    )
    clf = HistGradientBoostingClassifier(random_state=42)
    return Pipeline([("preprocess", pre), ("model", clf)])


def evaluate_task(
    df: pd.DataFrame,
    task: TaskDef,
    n_boot: int,
    seed: int,
    bootstrap_sample_size: int | None = None,
    feature_mode: str = "age_race_ethnicity",
) -> list[dict]:
    task_df = valid_task_rows(df, task)
    train_df = task_df[task_df["split"] == "train"].copy()
    val_df = task_df[task_df["split"] == "val"].copy()

    y_train = to_binary_label(train_df["label"], task.name)
    y_val = to_binary_label(val_df["label"], task.name)
    if feature_mode == "age_only":
        feature_cols = ["age"]
        model_specs = [
            ("logistic_regression", logistic_pipeline_age_only()),
            ("gbm", gbm_pipeline_age_only()),
        ]
        feature_label = "age only"
    elif feature_mode == "age_race_ethnicity":
        feature_cols = FEATURES
        model_specs = [
            ("logistic_regression", logistic_pipeline()),
            ("gbm", gbm_pipeline()),
        ]
        feature_label = "age + race + ethnicity"
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    x_train = train_df[feature_cols]
    x_val = val_df[feature_cols]

    if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
        return []

    outputs = []
    for model_name, pipe in model_specs:
        pipe.fit(x_train, y_train)
        y_score = pipe.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, y_score)
        ci_low, ci_high = bootstrap_auc_ci(
            y_val,
            y_score,
            n_boot=n_boot,
            seed=seed,
            bootstrap_sample_size=bootstrap_sample_size,
        )
        sens_vals = {}
        for spec in SPECIFICITIES:
            sval = sensitivity_at_specificity(y_val, y_score, spec)
            slo, shi = bootstrap_sensitivity_ci(
                y_val,
                y_score,
                specificity=spec,
                n_boot=n_boot,
                seed=seed,
                bootstrap_sample_size=bootstrap_sample_size,
            )
            key = f"{int(spec * 100)}"
            sens_vals[f"sens_at_spec_{key}"] = float(sval)
            sens_vals[f"sens_at_spec_{key}_ci_low_95"] = float(slo)
            sens_vals[f"sens_at_spec_{key}_ci_high_95"] = float(shi)

        outputs.append(
            {
                "task": task.name,
                "model": model_name,
                "features": feature_label,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_val_positive": int(y_val.sum()),
                "n_val_negative": int((1 - y_val).sum()),
                "auc": float(auc),
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "bootstrap_n": int(n_boot),
                "bootstrap_sample_size": int(bootstrap_sample_size) if bootstrap_sample_size is not None else int(len(y_val)),
                **sens_vals,
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train metadata-only baselines on manuscript train/val splits.")
    parser.add_argument(
        "--datasplit-dir",
        type=Path,
        default=Path("data/splits"),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/metadata/metadata.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/metadata_only_baseline_auc.csv"),
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="age_race_ethnicity",
        choices=["age_race_ethnicity", "age_only"],
        help="Feature set for metadata baselines.",
    )
    parser.add_argument(
        "--bootstrap-sample-size",
        type=int,
        default=None,
        help="Optional fixed sample size for each bootstrap draw (with replacement).",
    )
    args = parser.parse_args()

    merged = build_merged_dataframe(args.datasplit_dir, args.metadata_csv)
    rows = []
    for task in TASKS:
        rows.extend(
            evaluate_task(
                merged,
                task,
                n_boot=args.n_boot,
                seed=args.seed,
                bootstrap_sample_size=args.bootstrap_sample_size,
                feature_mode=args.feature_mode,
            )
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No evaluable task found. Check label filtering and class balance.")
    out = out.sort_values(["task", "model"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
