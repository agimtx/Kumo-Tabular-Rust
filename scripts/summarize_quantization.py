"""Summarize Rust ONNX reports and the original-.pt comparisons.

Writes output/quantization-comparison-report.{md,json}: the 32-context
precision comparison plus the INT4 context-size sensitivity check.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("classification", "regression")
PRECISIONS = ("fp16", "fp8", "int4")
CONTEXTS = (16, 32, 64)


def rust_report(task: str, precision: str, context: int) -> Path:
    if context == 32:
        return ROOT / "output" / f"medium-{task}-{precision}-full-evaluation-rust.json"
    return ROOT / "output" / f"medium-{task}-{precision}-full-context-{context}-evaluation-rust.json"


def official_report(task: str, precision: str, context: int) -> Path:
    suffix = "" if context == 32 else f"-context-{context}"
    return ROOT / "output" / f"medium-{task}-{precision}{suffix}-official-comparison.json"


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data[0] if isinstance(data, list) else data


def main() -> None:
    comparison = []
    for task in TASKS:
        for precision in PRECISIONS:
            rust = load(rust_report(task, precision, 32))
            official = load(official_report(task, precision, 32))
            comparison.append({
                "task": task,
                "precision": precision,
                "onnx_model": rust["model"],
                "official_model": official["model"],
                "test_rows": rust["test_rows"],
                "onnx_metrics": rust["metrics"],
                "official_metrics": official["metrics"],
                "official_minus_onnx": official["comparison_to_rust"]["official_minus_rust_metrics"],
                "prediction_pearson": official["comparison_to_rust"]["prediction_pearson"],
                "prediction_mae_difference": official["comparison_to_rust"]["prediction_mae_difference"],
            })

    sensitivity = {"classification": [], "regression": []}
    for task in TASKS:
        for context in CONTEXTS:
            entry = {"context_rows": context}
            for precision in ("fp16", "int4"):
                entry[f"onnx_{precision}"] = load(rust_report(task, precision, context))["metrics"]
            entry["official"] = load(official_report(task, "int4", context))["metrics"]
            entry["official_minus_int4"] = {
                key: entry["official"][key] - entry["onnx_int4"][key]
                for key in (("roc_auc_yes", "balanced_accuracy") if task == "classification" else ("mae", "r2"))
            }
            sensitivity[task].append(entry)

    report = {
        "size": "medium",
        "dataset": {
            "classification": "dataset/classification/bad_customer_detection.parquet",
            "regression": "dataset/regression/airfoil_self_noise.parquet",
        },
        "seed": 42,
        "context_rows": 32,
        "split": "held-out test rows only",
        "note": (
            "official = original pt FP32 checkpoint replayed on the same context/query batches; "
            "deltas are official minus ONNX. INT4 models use per-output-channel scales and keep "
            "small tensors in FP32."
        ),
        "rows": comparison,
        "context_sensitivity": sensitivity,
    }
    (ROOT / "output" / "quantization-comparison-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    lines = [
        "# Quantization comparison: ONNX vs original PyTorch checkpoints",
        "",
        "*Size:* medium · *seed:* 42 · *context rows:* 32 · "
        "*test rows:* 344 classification / 301 regression",
        "",
        "## Classification",
        "",
        "| Precision | ONNX AUC | ONNX bal.acc | ONNX acc | PT AUC | PT bal.acc | PT acc | "
        "ΔAUC | Δbal.acc | Δacc | ρ(pred) | MAE(pred diff) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        if row["task"] != "classification":
            continue
        onnx = row["onnx_metrics"]
        official = row["official_metrics"]
        delta = row["official_minus_onnx"]
        lines.append(
            f"| {row['precision']} | {onnx['roc_auc_yes']:.4f} | {onnx['balanced_accuracy']:.4f} | "
            f"{onnx['accuracy']:.4f} | {official['roc_auc_yes']:.4f} | "
            f"{official['balanced_accuracy']:.4f} | {official['accuracy']:.4f} | "
            f"{delta['roc_auc_yes']:+.4f} | {delta['balanced_accuracy']:+.4f} | "
            f"{delta['accuracy']:+.4f} | {row['prediction_pearson']:.4f} | "
            f"{row['prediction_mae_difference']:.4f} |"
        )
    lines += [
        "",
        "## Regression",
        "",
        "| Precision | ONNX MAE | ONNX RMSE | ONNX R2 | PT MAE | PT RMSE | PT R2 | "
        "ΔMAE | ΔRMSE | ΔR2 | ρ(pred) | MAE(pred diff) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        if row["task"] != "regression":
            continue
        onnx = row["onnx_metrics"]
        official = row["official_metrics"]
        delta = row["official_minus_onnx"]
        lines.append(
            f"| {row['precision']} | {onnx['mae']:.4f} | {onnx['rmse']:.4f} | {onnx['r2']:.4f} | "
            f"{official['mae']:.4f} | {official['rmse']:.4f} | {official['r2']:.4f} | "
            f"{delta['mae']:+.4f} | {delta['rmse']:+.4f} | {delta['r2']:+.4f} | "
            f"{row['prediction_pearson']:.4f} | {row['prediction_mae_difference']:.4f} |"
        )

    lines += [
        "",
        "## Context-size sensitivity (INT4 vs FP16 control vs original .pt)",
        "",
        "| Task | Context | FP16 AUC / bal.acc | INT4 AUC / bal.acc | PT AUC / bal.acc | "
        "ΔAUC / Δbal.acc (PT − INT4) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in sensitivity["classification"]:
        fp16, int4, official = entry["onnx_fp16"], entry["onnx_int4"], entry["official"]
        lines.append(
            f"| classification | {entry['context_rows']} | "
            f"{fp16['roc_auc_yes']:.4f} / {fp16['balanced_accuracy']:.4f} | "
            f"{int4['roc_auc_yes']:.4f} / {int4['balanced_accuracy']:.4f} | "
            f"{official['roc_auc_yes']:.4f} / {official['balanced_accuracy']:.4f} | "
            f"{entry['official_minus_int4']['roc_auc_yes']:+.4f} / "
            f"{entry['official_minus_int4']['balanced_accuracy']:+.4f} |"
        )
    lines += [
        "",
        "| Task | Context | FP16 MAE / R2 | INT4 MAE / R2 | PT MAE / R2 | ΔMAE / ΔR2 (PT − INT4) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in sensitivity["regression"]:
        fp16, int4, official = entry["onnx_fp16"], entry["onnx_int4"], entry["official"]
        lines.append(
            f"| regression | {entry['context_rows']} | "
            f"{fp16['mae']:.4f} / {fp16['r2']:.4f} | "
            f"{int4['mae']:.4f} / {int4['r2']:.4f} | "
            f"{official['mae']:.4f} / {official['r2']:.4f} | "
            f"{entry['official_minus_int4']['mae']:+.4f} / "
            f"{entry['official_minus_int4']['r2']:+.4f} |"
        )
    (ROOT / "output" / "quantization-comparison-report.md").write_text("\n".join(lines) + "\n")
    print("Wrote output/quantization-comparison-report.{md,json}")


if __name__ == "__main__":
    main()