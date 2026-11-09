import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

try:
    from . import metadata_vs_image_delta_metrics as base
except ImportError:
    import metadata_vs_image_delta_metrics as base


METHOD = "paired_bootstrap_wald_t"


def t_ci_p(observed_delta: float, delta_samples: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(delta_samples, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan, np.nan, np.nan
    se = float(np.std(vals, ddof=1))
    if not np.isfinite(se) or se == 0.0:
        p_value = 0.0 if observed_delta != 0.0 else 1.0
        return float(observed_delta), float(observed_delta), p_value
    df = vals.size - 1
    statistic = float(observed_delta / se)
    p_value = float(2.0 * t.sf(abs(statistic), df=df))
    critical_value = float(t.ppf(0.975, df=df))
    return float(observed_delta - critical_value * se), float(observed_delta + critical_value * se), p_value


def bootstrap_delta_samples(
    y_true: np.ndarray,
    y_image: np.ndarray,
    y_lr: np.ndarray,
    y_gbm: np.ndarray,
    metric_name: str,
    bootstrap_indices: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    lr_samples = []
    gbm_samples = []
    for idx in bootstrap_indices:
        yb = y_true[idx]
        if np.unique(yb).size < 2:
            continue
        img_b = base.metric_value(yb, y_image[idx], metric_name)
        lr_b = base.metric_value(yb, y_lr[idx], metric_name)
        gbm_b = base.metric_value(yb, y_gbm[idx], metric_name)
        if np.isfinite(img_b) and np.isfinite(lr_b):
            lr_samples.append(img_b - lr_b)
        if np.isfinite(img_b) and np.isfinite(gbm_b):
            gbm_samples.append(img_b - gbm_b)
    return np.asarray(lr_samples, dtype=float), np.asarray(gbm_samples, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paired image-vs-metadata deltas with corrected bootstrap-Wald Student t inference."
    )
    parser.add_argument("--datasplit-dir", type=Path, default=Path("data/splits"))
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
        "--output-csv",
        type=Path,
        default=Path("outputs/metadata_vs_image_delta_metrics_student_t_n100_n20380.csv"),
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path("outputs/metadata_vs_image_delta_metrics_student_t_n100_n20380.xlsx"),
    )
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--bootstrap-sample-size", type=int, default=20380)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merged = base.prepare_merged_split(args.datasplit_dir, args.metadata_csv)
    task_meta = base.build_meta_predictions(merged)

    rows = []
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

        rng = np.random.default_rng(args.seed)
        bootstrap_indices = [
            rng.integers(0, len(y_true), size=args.bootstrap_sample_size) for _ in range(args.n_boot)
        ]

        for metric_name in base.METRICS:
            v_image = base.metric_value(y_true, y_image, metric_name)
            v_lr = base.metric_value(y_true, y_lr, metric_name)
            v_gbm = base.metric_value(y_true, y_gbm, metric_name)
            if np.isnan(v_gbm) or (not np.isnan(v_lr) and v_lr >= v_gbm):
                best_model = "logistic_regression"
                v_best = v_lr
                best_suffix = "logit"
            else:
                best_model = "gbm"
                v_best = v_gbm
                best_suffix = "gbm"

            lr_samples, gbm_samples = bootstrap_delta_samples(
                y_true, y_image, y_lr, y_gbm, metric_name, bootstrap_indices
            )
            observed = {
                "logit": float(v_image - v_lr),
                "gbm": float(v_image - v_gbm),
            }
            samples = {"logit": lr_samples, "gbm": gbm_samples}
            observed["best"] = observed[best_suffix]
            samples["best"] = samples[best_suffix]

            stats = {}
            for suffix in ["best", "logit", "gbm"]:
                lo, hi, p_value = t_ci_p(observed[suffix], samples[suffix])
                stats[suffix] = (observed[suffix], lo, hi, p_value)

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
                    "delta_image_minus_meta_best": stats["best"][0],
                    "delta_ci_low_95_best": stats["best"][1],
                    "delta_ci_high_95_best": stats["best"][2],
                    "delta_ci_method_best": METHOD,
                    "delta_p_value_best": stats["best"][3],
                    "delta_p_value_method_best": METHOD,
                    "delta_image_minus_meta_logit": stats["logit"][0],
                    "delta_ci_low_95_logit": stats["logit"][1],
                    "delta_ci_high_95_logit": stats["logit"][2],
                    "delta_ci_method_logit": METHOD,
                    "delta_p_value_logit": stats["logit"][3],
                    "delta_p_value_method_logit": METHOD,
                    "delta_image_minus_meta_gbm": stats["gbm"][0],
                    "delta_ci_low_95_gbm": stats["gbm"][1],
                    "delta_ci_high_95_gbm": stats["gbm"][2],
                    "delta_ci_method_gbm": METHOD,
                    "delta_p_value_gbm": stats["gbm"][3],
                    "delta_p_value_method_gbm": METHOD,
                    "bootstrap_n": int(args.n_boot),
                    "bootstrap_sample_size": int(args.bootstrap_sample_size),
                    "bootstrap_seed": int(args.seed),
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["task", "experiment", "metric"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

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
