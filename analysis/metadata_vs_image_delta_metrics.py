import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
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
TASKS = ["dr", "amd", "glaucoma"]
METRICS = {
    "auc": None,
    "sens_at_spec_80": 0.80,
    "sens_at_spec_85": 0.85,
    "sens_at_spec_95": 0.95,
}


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race"] = out["race"].astype(str).apply(
        lambda x: x if x in {"Asian", "Black or African American", "White"} else "Other or Unknown"
    )
    out["ethnicity"] = out["ethnicity"].astype(str).apply(
        lambda x: x if x in {"0", "1"} else "Unknown or Not Reported"
    )
    return out


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def sensitivity_at_specificity(y_true: np.ndarray, y_score: np.ndarray, specificity: float) -> float:
    if np.unique(y_true).size < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score)
    target_fpr = 1.0 - specificity
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return np.nan
    return float(np.max(tpr[valid]))


def metric_value(y_true: np.ndarray, y_score: np.ndarray, metric_name: str) -> float:
    if metric_name == "auc":
        return safe_auc(y_true, y_score)
    specificity = METRICS[metric_name]
    return sensitivity_at_specificity(y_true, y_score, float(specificity))


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Return 1-based midranks, matching the standard fast DeLong implementation."""
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def fast_delong(predictions_sorted: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance for one or more correlated ROC AUC estimates."""
    m = int(label_1_count)
    n = predictions_sorted.shape[1] - m
    if m <= 0 or n <= 0:
        raise ValueError("DeLong test requires both positive and negative samples.")

    positive_examples = predictions_sorted[:, :m]
    negative_examples = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = compute_midrank(positive_examples[r])
        ty[r] = compute_midrank(negative_examples[r])
        tz[r] = compute_midrank(predictions_sorted[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def paired_delong_delta_ci_p_value(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    alpha: float = 0.95,
) -> tuple[float, float, float, float]:
    """Paired DeLong delta, CI, and two-sided p-value for AUC(score_a) - AUC(score_b)."""
    y_true = np.asarray(y_true, dtype=int)
    if np.unique(y_true).size < 2:
        return np.nan, np.nan, np.nan, np.nan

    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    predictions_sorted = np.vstack([y_score_a, y_score_b])[:, order]
    aucs, covariance = fast_delong(predictions_sorted, label_1_count)

    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    delta = float(aucs[0] - aucs[1])
    if variance <= 0 or not np.isfinite(variance):
        p_value = 0.0 if delta != 0.0 else 1.0
        return delta, delta, delta, p_value

    se = float(np.sqrt(variance))
    z = abs(delta) / se
    z_ci = float(norm.ppf(0.5 + alpha / 2.0))
    p_value = float(2.0 * norm.sf(z))
    return delta, float(delta - z_ci * se), float(delta + z_ci * se), p_value


def paired_delong_p_value(y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray) -> float:
    """Two-sided paired DeLong p-value for AUC(score_a) - AUC(score_b)."""
    return paired_delong_delta_ci_p_value(y_true, y_score_a, y_score_b)[3]


def paired_delta_ci(
    y_true: np.ndarray,
    y_image: np.ndarray,
    y_meta: np.ndarray,
    metric_name: str,
    n_boot: int,
    seed: int,
    bootstrap_sample_size: int,
) -> tuple[float, float, float, str, str]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=bootstrap_sample_size)
        yb = y_true[idx]
        if np.unique(yb).size < 2:
            continue
        m_img = metric_value(yb, y_image[idx], metric_name)
        m_meta = metric_value(yb, y_meta[idx], metric_name)
        if np.isfinite(m_img) and np.isfinite(m_meta):
            samples.append(m_img - m_meta)
    if not samples:
        return np.nan, np.nan, np.nan, "not_available", "not_available"
    arr = np.array(samples, dtype=float)
    if metric_name == "auc":
        _, ci_low, ci_high, p_value = paired_delong_delta_ci_p_value(y_true, y_image, y_meta)
        p_value_method = "paired_delong"
        ci_method = "paired_delong"
    else:
        ci_low = float(np.quantile(arr, 0.025))
        ci_high = float(np.quantile(arr, 0.975))
        p_value = np.nan
        p_value_method = "not_applicable_for_non_auc"
        ci_method = "paired_bootstrap_percentile"
    return ci_low, ci_high, p_value, p_value_method, ci_method


def logistic_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
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
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=2000, solver="lbfgs"))])


def gbm_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), ["age"]),
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
    return Pipeline([("preprocess", pre), ("model", HistGradientBoostingClassifier(random_state=42))])


def prepare_merged_split(datasplit_dir: Path, metadata_csv: Path) -> pd.DataFrame:
    train = np.load(datasplit_dir / "train_final.npy", allow_pickle=True)
    val = np.load(datasplit_dir / "val_final.npy", allow_pickle=True)
    train_df = pd.DataFrame(train, columns=SPLIT_COLUMNS)
    val_df = pd.DataFrame(val, columns=SPLIT_COLUMNS)
    train_df["split"] = "train"
    val_df["split"] = "val"
    split_df = pd.concat([train_df, val_df], ignore_index=True)
    split_df["image_name"] = split_df["image"].astype(str) + "_" + split_df["grade"].astype(str)

    meta = pd.read_csv(metadata_csv, low_memory=False)
    join_df = meta[["jpgfile"] + FEATURES].drop_duplicates("jpgfile")
    join_df = normalize_categories(join_df)
    merged = split_df.merge(join_df, left_on="image", right_on="jpgfile", how="left")
    if merged["jpgfile"].isna().any():
        missing = int(merged["jpgfile"].isna().sum())
        raise ValueError(f"Metadata are missing for {missing} split images.")
    return merged


def build_meta_predictions(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for task in TASKS:
        task_df = df.copy()
        task_df["label"] = pd.to_numeric(task_df[task], errors="coerce")
        valid = task_df["label"].notna() & task_df["label"].map(
            lambda value: is_valid_label(value, task) if pd.notna(value) else False
        )
        task_df = task_df[valid]

        tr = task_df[task_df["split"] == "train"].copy()
        va = task_df[task_df["split"] == "val"].copy()
        y_tr = np.asarray([binary_label(value, task) for value in tr["label"]], dtype=int)
        y_va = np.asarray([binary_label(value, task) for value in va["label"]], dtype=int)

        lr = logistic_pipeline().fit(tr[FEATURES], y_tr)
        gbm = gbm_pipeline().fit(tr[FEATURES], y_tr)
        out[task] = pd.DataFrame(
            {
                "image_name": va["image_name"].astype(str).values,
                "y_true": y_va.astype(int),
                "meta_lr_score": lr.predict_proba(va[FEATURES])[:, 1],
                "meta_gbm_score": gbm.predict_proba(va[FEATURES])[:, 1],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired image-vs-metadata deltas for all metrics.")
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
        "--preds-cache-dir",
        type=Path,
        default=Path("outputs/preds_cache"),
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path("outputs/metadata_vs_image_delta_metrics_n100_n20380.xlsx"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/metadata_vs_image_delta_metrics_n100_n20380.csv"),
    )
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--bootstrap-sample-size", type=int, default=20380)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merged = prepare_merged_split(args.datasplit_dir, args.metadata_csv)
    task_meta = build_meta_predictions(merged)

    rows: list[dict] = []
    for pred_file in sorted(args.preds_cache_dir.glob("*.csv")):
        pred_df = pd.read_csv(pred_file)
        if pred_df.empty:
            continue
        task = str(pred_df["task"].iloc[0])
        if task not in task_meta:
            continue
        image_df = pred_df[["experiment", "task", "image_name", "y_true", "y_score"]].copy()
        image_df["image_name"] = image_df["image_name"].astype(str)
        pair_df = image_df.merge(task_meta[task], on="image_name", how="inner", suffixes=("", "_meta"))
        if pair_df.empty:
            continue

        y_true = pair_df["y_true"].to_numpy(dtype=int)
        y_image = pair_df["y_score"].to_numpy(dtype=float)
        y_lr = pair_df["meta_lr_score"].to_numpy(dtype=float)
        y_gbm = pair_df["meta_gbm_score"].to_numpy(dtype=float)

        for metric_name in METRICS:
            v_image = metric_value(y_true, y_image, metric_name)
            v_lr = metric_value(y_true, y_lr, metric_name)
            v_gbm = metric_value(y_true, y_gbm, metric_name)
            if np.isnan(v_gbm) or (not np.isnan(v_lr) and v_lr >= v_gbm):
                best_model = "logistic_regression"
                v_best = v_lr
                y_best = y_lr
            else:
                best_model = "gbm"
                v_best = v_gbm
                y_best = y_gbm

            lo_best, hi_best, p_best, p_method_best, ci_method_best = paired_delta_ci(
                y_true,
                y_image,
                y_best,
                metric_name=metric_name,
                n_boot=args.n_boot,
                seed=args.seed,
                bootstrap_sample_size=args.bootstrap_sample_size,
            )
            lo_lr, hi_lr, p_lr, p_method_lr, ci_method_lr = paired_delta_ci(
                y_true,
                y_image,
                y_lr,
                metric_name=metric_name,
                n_boot=args.n_boot,
                seed=args.seed,
                bootstrap_sample_size=args.bootstrap_sample_size,
            )
            lo_gbm, hi_gbm, p_gbm, p_method_gbm, ci_method_gbm = paired_delta_ci(
                y_true,
                y_image,
                y_gbm,
                metric_name=metric_name,
                n_boot=args.n_boot,
                seed=args.seed,
                bootstrap_sample_size=args.bootstrap_sample_size,
            )

            rows.append(
                {
                    "experiment": str(pred_df["experiment"].iloc[0]),
                    "task": task,
                    "metric": metric_name,
                    "n_overlap": int(len(pair_df)),
                    "image_value": v_image,
                    "meta_logit_value": v_lr,
                    "meta_gbm_value": v_gbm,
                    "best_meta_model": best_model,
                    "meta_best_value": v_best,
                    "delta_image_minus_meta_best": v_image - v_best,
                    "delta_ci_low_95_best": lo_best,
                    "delta_ci_high_95_best": hi_best,
                    "delta_ci_method_best": ci_method_best,
                    "delta_p_value_best": p_best,
                    "delta_p_value_method_best": p_method_best,
                    "delta_image_minus_meta_logit": v_image - v_lr,
                    "delta_ci_low_95_logit": lo_lr,
                    "delta_ci_high_95_logit": hi_lr,
                    "delta_ci_method_logit": ci_method_lr,
                    "delta_p_value_logit": p_lr,
                    "delta_p_value_method_logit": p_method_lr,
                    "delta_image_minus_meta_gbm": v_image - v_gbm,
                    "delta_ci_low_95_gbm": lo_gbm,
                    "delta_ci_high_95_gbm": hi_gbm,
                    "delta_ci_method_gbm": ci_method_gbm,
                    "delta_p_value_gbm": p_gbm,
                    "delta_p_value_method_gbm": p_method_gbm,
                    "bootstrap_n": int(args.n_boot),
                    "bootstrap_sample_size": int(args.bootstrap_sample_size),
                    "bootstrap_seed": int(args.seed),
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["task", "experiment", "metric"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    # Wide sheet for easier reading in Excel.
    wide = out_df.pivot_table(
        index=["experiment", "task"],
        columns="metric",
        values=[
            "image_value",
            "meta_best_value",
            "delta_image_minus_meta_best",
            "delta_ci_low_95_best",
            "delta_ci_high_95_best",
            "delta_ci_method_best",
            "delta_p_value_best",
            "delta_p_value_method_best",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="delta_long", index=False)
        wide.to_excel(writer, sheet_name="delta_wide_best", index=False)

    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved XLSX: {args.output_xlsx}")


if __name__ == "__main__":
    main()
