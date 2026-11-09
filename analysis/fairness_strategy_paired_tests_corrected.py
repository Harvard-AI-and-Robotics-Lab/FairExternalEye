import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

try:
    from .fairness_metrics import (
        GROUP_LABELS,
        METHOD_TO_EXPERIMENT,
        TASKS,
        load_prediction,
        make_pair_df,
        metric_values,
    )
    from .metadata_vs_image_delta_metrics import paired_delong_delta_ci_p_value
except ImportError:
    from fairness_metrics import (
        GROUP_LABELS,
        METHOD_TO_EXPERIMENT,
        TASKS,
        load_prediction,
        make_pair_df,
        metric_values,
    )
    from metadata_vs_image_delta_metrics import paired_delong_delta_ci_p_value


STUDENT_T_METHOD = "paired_bootstrap_wald_t"
DELONG_METHOD = "paired_delong"
NON_AUC_METHOD = "not_applicable_for_non_auc"
BOOTSTRAP_CI_METHOD = "paired_bootstrap_percentile"
METRICS = ["overall_auc", "es_auc", "mean_abs_group_gap", "max_abs_group_gap"]


def student_t_ci_p(observed_delta: float, delta_samples: np.ndarray) -> tuple[float, float, float]:
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


def bootstrap_percentile_ci(delta_samples: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(delta_samples, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def paired_bootstrap_all_deltas(
    pair_df: pd.DataFrame,
    group: str,
    n_boot: int,
    sample_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(pair_df)
    deltas = {metric: [] for metric in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=sample_size)
        boot = pair_df.iloc[idx]
        base_values = metric_values(boot, group, "baseline_score")
        fair_values = metric_values(boot.rename(columns={"fair_score": "y_score"}), group, "y_score")
        for metric in METRICS:
            base = base_values[metric]
            fair = fair_values[metric]
            if np.isfinite(base) and np.isfinite(fair):
                deltas[metric].append(fair - base)
    return {metric: np.asarray(values, dtype=float) for metric, values in deltas.items()}


def is_significant_improvement(delta: float, p_value: float, higher_is_better: bool) -> bool:
    if not np.isfinite(p_value) or p_value >= 0.05:
        return False
    return delta > 0 if higher_is_better else delta < 0


def row_for_metric(
    pair_df: pd.DataFrame,
    group: str,
    task_label: str,
    baseline_exp: str,
    fair_exp: str,
    method: str,
    baseline_df: pd.DataFrame,
    fair_df: pd.DataFrame,
    metric: str,
    inference: str,
    base_pair_values: dict[str, float],
    fair_pair_values: dict[str, float],
    delta_samples: np.ndarray,
    n_boot: int,
    bootstrap_sample_size: int,
    seed: int,
) -> dict:
    baseline_value = base_pair_values[metric]
    fairness_value = fair_pair_values[metric]
    observed_delta = float(fairness_value - baseline_value)
    higher_is_better = metric in {"overall_auc", "es_auc"}
    direction_improved = observed_delta > 0 if higher_is_better else observed_delta < 0

    if inference == "student_t":
        ci_low, ci_high, p_value = student_t_ci_p(observed_delta, delta_samples)
        ci_method = STUDENT_T_METHOD
        p_method = STUDENT_T_METHOD
    elif inference == "delong":
        if metric == "overall_auc":
            y_true = pair_df["y_true"].to_numpy(dtype=int)
            fair_score = pair_df["fair_score"].to_numpy(dtype=float)
            baseline_score = pair_df["baseline_score"].to_numpy(dtype=float)
            _, ci_low, ci_high, p_value = paired_delong_delta_ci_p_value(y_true, fair_score, baseline_score)
            ci_method = DELONG_METHOD
            p_method = DELONG_METHOD
        else:
            ci_low, ci_high = bootstrap_percentile_ci(delta_samples)
            p_value = np.nan
            ci_method = BOOTSTRAP_CI_METHOD
            p_method = NON_AUC_METHOD
    else:
        raise ValueError(f"Unsupported inference: {inference}")

    significant = is_significant_improvement(observed_delta, p_value, higher_is_better)
    return {
        "task": task_label,
        "group": group,
        "method": method,
        "baseline_experiment": baseline_exp,
        "fairness_experiment": fair_exp,
        "metric": metric,
        "n_overlap": int(len(pair_df)),
        "baseline_value": baseline_value,
        "fairness_value": fairness_value,
        "delta": observed_delta,
        "delta_ci_low_95": ci_low,
        "delta_ci_high_95": ci_high,
        "delta_ci_method": ci_method,
        "delta_p_value": p_value,
        "delta_p_value_method": p_method,
        "higher_is_better": higher_is_better,
        "direction_improved": direction_improved,
        "significant_p_lt_0_05": bool(np.isfinite(p_value) and p_value < 0.05),
        "significant_improvement": significant,
        "bootstrap_n": int(n_boot),
        "bootstrap_sample_size": int(bootstrap_sample_size),
        "bootstrap_seed": int(seed),
        "baseline_checkpoint": str(baseline_df["checkpoint_path"].iloc[0]),
        "fairness_checkpoint": str(fair_df["checkpoint_path"].iloc[0]),
    }


def write_outputs(out: pd.DataFrame, output_csv: Path, output_xlsx: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    wide = out.pivot_table(
        index=["task", "group", "method", "baseline_experiment", "fairness_experiment", "n_overlap"],
        columns="metric",
        values=[
            "baseline_value",
            "fairness_value",
            "delta",
            "delta_ci_low_95",
            "delta_ci_high_95",
            "delta_ci_method",
            "delta_p_value",
            "delta_p_value_method",
            "significant_improvement",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    summary = (
        out[out["metric"].isin(["es_auc", "max_abs_group_gap"])]
        .groupby(["task", "group", "method", "metric"], as_index=False)
        .agg(
            delta=("delta", "first"),
            delta_ci_low_95=("delta_ci_low_95", "first"),
            delta_ci_high_95=("delta_ci_high_95", "first"),
            delta_p_value=("delta_p_value", "first"),
            delta_p_value_method=("delta_p_value_method", "first"),
            significant_improvement=("significant_improvement", "first"),
        )
    )

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="paired_tests_long", index=False)
        wide.to_excel(writer, sheet_name="paired_tests_wide", index=False)
        summary.to_excel(writer, sheet_name="fairness_summary", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected paired fairness strategy inference.")
    parser.add_argument(
        "--preds-cache-dir",
        type=Path,
        default=Path("outputs/preds_cache"),
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-xlsx", type=Path, required=True)
    parser.add_argument("--inference", choices=["student_t", "delong"], required=True)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--bootstrap-sample-size", type=int, default=20380)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for task_label, task_info in TASKS.items():
        baseline_exp = task_info["baseline"]
        baseline_df = load_prediction(args.preds_cache_dir, baseline_exp)
        for group in GROUP_LABELS:
            for method, template in METHOD_TO_EXPERIMENT.items():
                fair_exp = template.format(disease=task_info["disease"], group=group)
                fair_path = args.preds_cache_dir / f"{fair_exp}.csv"
                if not fair_path.exists():
                    continue
                fair_df = load_prediction(args.preds_cache_dir, fair_exp)
                pair_df = make_pair_df(baseline_df, fair_df)
                base_pair_values = metric_values(pair_df, group, "baseline_score")
                fair_pair_values = metric_values(pair_df.rename(columns={"fair_score": "y_score"}), group, "y_score")
                bootstrap_deltas = paired_bootstrap_all_deltas(
                    pair_df,
                    group,
                    n_boot=args.n_boot,
                    sample_size=args.bootstrap_sample_size,
                    seed=args.seed,
                )
                for metric in METRICS:
                    rows.append(
                        row_for_metric(
                            pair_df,
                            group,
                            task_label,
                            baseline_exp,
                            fair_exp,
                            method,
                            baseline_df,
                            fair_df,
                            metric,
                            args.inference,
                            base_pair_values,
                            fair_pair_values,
                            bootstrap_deltas[metric],
                            args.n_boot,
                            args.bootstrap_sample_size,
                            args.seed,
                        )
                    )

    out = pd.DataFrame(rows).sort_values(["task", "group", "method", "metric"]).reset_index(drop=True)
    write_outputs(out, args.output_csv, args.output_xlsx)
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved XLSX: {args.output_xlsx}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
