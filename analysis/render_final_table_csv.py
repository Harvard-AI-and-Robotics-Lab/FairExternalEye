import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def normalize_section_name(raw: str) -> str:
    s = str(raw).strip().upper().replace(" ", "")
    return s


def row_experiment(section: str, model: str) -> Optional[str]:
    model_key = " ".join(str(model).strip().lower().replace("+", " + ").split())
    sec = normalize_section_name(section)

    baseline = {
        ("DR", "wresnet38"): "wresnet_dr_baseline",
        ("DR", "vit (deit-based)"): "trm_dr_baseline",
        ("DR", "efficientnet"): "eff_dr_baseline",
        ("AMD-BIN", "wresnet38"): "wresnet_amd_baseline",
        ("AMD-BIN", "vit (deit-based)"): "trm_amdbin_baseline",
        ("AMD-BIN", "efficientnet"): "eff_amd",
        ("GLAUCOMA", "wresnet38"): "wresnet_glaucoma_E2",
        ("GLAUCOMA", "vit (deit-based)"): "vit_glaucoma",
        ("GLAUCOMA", "efficientnet"): "eff_glaucoma",
    }
    if (sec, model_key) in baseline:
        return baseline[(sec, model_key)]

    fairness_key = {
        "vit": "baseline",
        "vit + dro": "DRO",
        "vit + adv": "ADV",
        "vit + fscl": "FSCL",
        "vit + fis": "FIS",
    }
    if model_key not in fairness_key:
        return None
    if "LLAVA" in str(model).upper() or "MEDGEMMA" in str(model).upper():
        return None

    if sec.startswith("DR/"):
        disease = "dr"
    elif sec.startswith("AMD/"):
        disease = "amdbin"
    else:
        return None

    group = sec.split("/", 1)[1].strip().lower()
    method = fairness_key[model_key]
    if method == "baseline":
        return f"trm_{disease}_baseline"
    if method == "FSCL":
        return f"trm_{disease}_FSCL_{group}_finetune"
    return f"trm_{disease}_{method}_{group}"


def format_ci(value: float, lo: float, hi: float) -> str:
    if pd.isna(value):
        return ""
    v = value * 100.0
    if pd.isna(lo) or pd.isna(hi):
        return f"{v:.1f}"
    return f"{v:.1f} ({lo * 100.0:.1f}, {hi * 100.0:.1f})"


def get_metric(df: pd.DataFrame, experiment: str, **conds) -> tuple[float, float, float]:
    m = df[df["experiment"] == experiment]
    for k, v in conds.items():
        m = m[m[k] == v]
    if m.empty:
        return (np.nan, np.nan, np.nan)
    r = m.iloc[0]
    return (float(r["value"]) if pd.notna(r["value"]) else np.nan, float(r["ci_low"]) if pd.notna(r["ci_low"]) else np.nan, float(r["ci_high"]) if pd.notna(r["ci_high"]) else np.nan)


def get_checkpoint(manifest_df: pd.DataFrame, experiment: str) -> str:
    m = manifest_df[manifest_df["experiment"] == experiment]
    if m.empty:
        return ""
    r = m.iloc[0]
    ckpt = str(r["checkpoint_path"]) if pd.notna(r["checkpoint_path"]) else ""
    if not ckpt:
        return ""
    if pd.notna(r.get("checkpoint_log", np.nan)) and str(r["checkpoint_log"]).strip():
        return f"{ckpt} | log={r['checkpoint_log']}"
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_xlsx", required=True)
    parser.add_argument("--bootstrap_xlsx", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    table = pd.read_excel(args.template_xlsx, sheet_name="Final", header=None)
    overall = pd.read_excel(args.bootstrap_xlsx, sheet_name="overall_auc_ci")
    sens = pd.read_excel(args.bootstrap_xlsx, sheet_name="sensitivity_ci")
    sub = pd.read_excel(args.bootstrap_xlsx, sheet_name="subgroup_auc_ci")
    es = pd.read_excel(args.bootstrap_xlsx, sheet_name="es_auc_ci")
    manifest = pd.read_excel(args.bootstrap_xlsx, sheet_name="manifest")

    label_map = {
        ("race", "Black"): "Black or African American",
        ("race", "Other"): "Other or Unknown",
        ("ethnic", "Non-hispanic"): "0",
        ("ethnic", "Hispanic"): "1",
        ("ethnic", "Other"): "Unknown or Not Reported",
        ("language", "Other"): "Other or Unknown",
    }

    table[23] = ""
    table.iloc[2, 23] = "checkpoint_used"

    current_section = ""
    for i in range(len(table)):
        section_cell = table.iloc[i, 0]
        if pd.notna(section_cell) and str(section_cell).strip():
            current_section = str(section_cell)
        model = table.iloc[i, 1]
        if pd.isna(model):
            continue
        exp = row_experiment(current_section, model)
        if not exp:
            continue

        o = get_metric(overall, exp, metric="overall_auc")
        table.iloc[i, 2] = format_ci(*o)

        for col, spec in zip([3, 4, 5, 6], [0.8, 0.85, 0.9, 0.95]):
            s = get_metric(sens, exp, metric="sensitivity", specificity=spec)
            table.iloc[i, col] = format_ci(*s)

        def subgroup_fill(col: int, group_type: str, label_name: str):
            normalized = label_map.get((group_type, label_name), label_name)
            v = get_metric(sub, exp, metric_type="subgroup_auc", group_type=group_type, group_label=normalized)
            table.iloc[i, col] = format_ci(*v)

        subgroup_fill(7, "sex", "0")
        subgroup_fill(8, "sex", "1")
        subgroup_fill(10, "race", "Asian")
        subgroup_fill(11, "race", "Black")
        subgroup_fill(12, "race", "White")
        subgroup_fill(13, "race", "Other")
        subgroup_fill(15, "ethnic", "Non-hispanic")
        subgroup_fill(16, "ethnic", "Hispanic")
        subgroup_fill(17, "ethnic", "Other")
        subgroup_fill(19, "language", "English")
        subgroup_fill(20, "language", "Spanish")
        subgroup_fill(21, "language", "Other")

        for col, g in [(9, "sex"), (14, "race"), (18, "ethnic"), (22, "language")]:
            v = get_metric(es, exp, metric_type="es_auc", group_type=g, group_label="ALL")
            table.iloc[i, col] = format_ci(*v)

        table.iloc[i, 23] = get_checkpoint(manifest, exp)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    table.to_csv(args.out_csv, header=False, index=False)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
