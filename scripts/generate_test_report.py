#!/usr/bin/env python3
"""Build TEST_REPORT.md from the Rust and official PyTorch evaluation JSON.

Rust reports are written by `evaluate_full`; official reports are written by
`compare_official_medium.py`. Both are cached under the ignored output/
directory, so this script only formats an existing run.
"""

import importlib
import json
import math
import os
import platform
import re
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
REPORT = ROOT / "TEST_REPORT.md"
README = ROOT / "README.md"
README_START = "<!-- TEST_REPORT_START -->"
README_END = "<!-- TEST_REPORT_END -->"
SEED = 42
CONTEXTS = (16, 32, 64)
TASKS = ("classification", "regression")
PRECISIONS = ("fp16", "fp8", "int4")
PRECISION_LABEL = {"fp16": "FP16", "fp8": "FP8", "int4": "INT4"}
TASK_LABEL = {"classification": "Classification", "regression": "Regression"}

FINDINGS = """
- **Classification:** FP16, FP8 and INT4 ROC AUC stay within 0.08 of the
  official FP32 model at every context size (FP8 is closest at 16 and 64 rows,
  INT4 at 32). All thresholded accuracies are below the 0.8866 all-No baseline,
  so balanced accuracy and AUC are the meaningful comparison; prediction
  correlation with PT is 0.81–0.94.
- **Regression:** at 32 and 64 context rows every ONNX export is within about
  0.07 MAE and 0.01 R2 of the official checkpoint, and at 16 rows INT4 has the
  lowest MAE (4.375 vs 4.550 for PT). Prediction correlation is 0.97–0.995.
- **Context size:** going from 16 to 64 context rows roughly halves regression
  MAE (4.375–4.607 to 2.281–2.321). Classification AUC peaks at 32 rows and is
  not monotonic, so more context is not uniformly better.
- **Quantization:** INT4 is 38.3 MB (classification) and 58.0 MB (regression),
  versus 134–136 MB for FP16, with no consistent AUC or regression penalty on
  this benchmark. INT4 and FP8 latency is close to FP16; dequantization offsets
  the smaller weights.
- **Latency:** ONNX mean time per batch rises from 57–97 ms (regression) and
  104–173 ms (classification) as context grows from 16 to 64. The official
  PyTorch API call is roughly 1.6–2.1x slower for classification and 1.9–2.7x
  slower for regression, but it also includes recipe work.
"""


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    return value[0] if isinstance(value, list) else value


def rust_report(task: str, precision: str, context: int) -> Path:
    if context == 32:
        name = f"medium-{task}-{precision}-full-evaluation-rust.json"
    else:
        name = f"medium-{task}-{precision}-full-context-{context}-evaluation-rust.json"
    return OUTPUT / name


def official_report(task: str, precision: str, context: int) -> Path:
    suffix = "" if context == 32 else f"-context-{context}"
    return OUTPUT / f"medium-{task}-{precision}{suffix}-official-comparison.json"


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def latency(stats: dict | None) -> str:
    if not stats or not stats.get("calls"):
        return "n/a / n/a"
    return f"{stats['mean_ms']:.2f} / {stats['p95_ms']:.2f}"


def queries_per_second(stats: dict | None, queries: int = 2) -> str:
    if not stats or not stats.get("calls") or not stats.get("total_seconds"):
        return "n/a"
    return f"{stats['calls'] * queries / stats['total_seconds']:.2f}"


def module_version(name: str) -> str:
    try:
        return getattr(importlib.import_module(name), "__version__", "unknown")
    except Exception:
        return "not installed"


def machine_info() -> tuple[str, str]:
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        memory = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        memory_gib = int(memory) / (1024**3)
    except Exception:
        chip = platform.processor() or "unknown CPU"
        memory_gib = 0
    system = platform.platform()
    return f"{chip}, {os.cpu_count()} cores, {memory_gib:.0f} GB", system


def collect() -> list[dict]:
    rows = []
    missing = []
    for task in TASKS:
        for context in CONTEXTS:
            for precision in PRECISIONS:
                rust_path = rust_report(task, precision, context)
                official_path = official_report(task, precision, context)
                for path in (rust_path, official_path):
                    if not path.is_file():
                        missing.append(path)
                if not rust_path.is_file() or not official_path.is_file():
                    continue
                rust = load(rust_path)
                official = load(official_path)
                rows.append(
                    {
                        "task": task,
                        "context": context,
                        "precision": precision,
                        "rust": rust,
                        "official": official,
                    }
                )
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"missing reports; rerun scripts/run_test_report.sh:\n{listing}")
    return rows


def classification_tables(rows: list[dict]) -> list[str]:
    lines = [
        "| Context | Precision | ONNX AUC | ONNX bal. acc | ONNX acc | "
        "PT AUC | PT bal. acc | ΔAUC | Δbal. acc | ρ(pred) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (entry for entry in rows if entry["task"] == "classification"):
        onnx = row["rust"]["metrics"]
        pt = row["official"]["metrics"]
        delta = row["official"]["comparison_to_rust"]["official_minus_rust_metrics"]
        lines.append(
            f"| {row['context']} | {PRECISION_LABEL[row['precision']]} | "
            f"{fmt(onnx['roc_auc_yes'])} | {fmt(onnx['balanced_accuracy'])} | "
            f"{fmt(onnx['accuracy'])} | {fmt(pt['roc_auc_yes'])} | "
            f"{fmt(pt['balanced_accuracy'])} | {delta['roc_auc_yes']:+.4f} | "
            f"{delta['balanced_accuracy']:+.4f} | "
            f"{fmt(row['official']['comparison_to_rust']['prediction_pearson'], 3)} |"
        )
    return lines


def regression_tables(rows: list[dict]) -> list[str]:
    lines = [
        "| Context | Precision | ONNX MAE | ONNX RMSE | ONNX R2 | "
        "PT MAE | PT RMSE | PT R2 | ΔMAE | ΔR2 | ρ(pred) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (entry for entry in rows if entry["task"] == "regression"):
        onnx = row["rust"]["metrics"]
        pt = row["official"]["metrics"]
        delta = row["official"]["comparison_to_rust"]["official_minus_rust_metrics"]
        lines.append(
            f"| {row['context']} | {PRECISION_LABEL[row['precision']]} | "
            f"{fmt(onnx['mae'])} | {fmt(onnx['rmse'])} | {fmt(onnx['r2'])} | "
            f"{fmt(pt['mae'])} | {fmt(pt['rmse'])} | {fmt(pt['r2'])} | "
            f"{delta['mae']:+.4f} | {delta['r2']:+.4f} | "
            f"{fmt(row['official']['comparison_to_rust']['prediction_pearson'], 3)} |"
        )
    return lines


def latency_table(rows: list[dict], task: str) -> list[str]:
    lines = [
        "| Context | Precision | ONNX mean / p95 (ms) | PT mean / p95 (ms) | "
        "ONNX queries/s | PT queries/s |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in (entry for entry in rows if entry["task"] == task):
        rust_latency = row["rust"].get("inference_latency", {}).get("test")
        pt_latency = row["official"].get("latency", {}).get("test")
        lines.append(
            f"| {row['context']} | {PRECISION_LABEL[row['precision']]} | "
            f"{latency(rust_latency)} | {latency(pt_latency)} | "
            f"{queries_per_second(rust_latency)} | {queries_per_second(pt_latency)} |"
        )
    return lines


def model_size_table(rows: list[dict]) -> list[str]:
    sizes = {}
    for row in rows:
        path = ROOT / row["rust"]["model"]
        sizes[(row["task"], row["precision"])] = path.stat().st_size / (1024 * 1024)
    lines = [
        "| Task | FP16 | FP8 | INT4 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        values = " | ".join(
            f"{sizes[(task, precision)]:.1f} MB" for precision in PRECISIONS
        )
        lines.append(f"| {TASK_LABEL[task]} | {values} |")
    return lines


def sync_readme(report: str) -> None:
    if not README.is_file():
        return
    readme = README.read_text()
    if README_START not in readme or README_END not in readme:
        return
    body = report.split("\n", 1)[1].strip()
    body = re.sub(r"^(#{2,}) ", r"#\1 ", body, flags=re.MULTILINE)
    start = readme.index(README_START) + len(README_START)
    end = readme.index(README_END)
    README.write_text(readme[:start] + "\n\n" + body + "\n\n" + readme[end:])


def main() -> None:
    rows = collect()
    machine, system = machine_info()
    classification_rows = [row for row in rows if row["task"] == "classification"]
    regression_rows = [row for row in rows if row["task"] == "regression"]
    test_rows = classification_rows[0]["rust"]["test_rows"]
    regression_test_rows = regression_rows[0]["rust"]["test_rows"]

    lines = [
        "# Kumo Tabular Rust — test report",
        "",
        f"*Snapshot:* {date.today().isoformat()} · *seed:* {SEED} · "
        f"*models:* medium (FP16 / FP8 / INT4)",
        "",
        "## Environment",
        "",
        f"- Machine: {machine}",
        f"- System: {system}",
        f"- Rust inference: ONNX Runtime {module_version('onnxruntime')} (CPU), release build",
        f"- Official comparison: PyTorch {module_version('torch')} (CPU), "
        f"structured-data-models {module_version('sdm')}",
        "",
        "## Models",
        "",
        "All ONNX files are medium dynamic exports under `models/`; every official "
        "comparison loads the corresponding FP32 checkpoint from `pt/`.",
        "",
        *model_size_table(rows),
        "",
        "## Datasets",
        "",
        f"- **Classification:** `dataset/classification/bad_customer_detection.parquet` "
        f"(13 features, {classification_rows[0]['rust']['fit_rows'] + classification_rows[0]['rust']['validation_rows'] + test_rows} rows; "
        f"{test_rows} held-out test rows).",
        f"- **Regression:** `dataset/regression/airfoil_self_noise.parquet` "
        f"(5 numerical features, {regression_rows[0]['rust']['fit_rows'] + regression_rows[0]['rust']['validation_rows'] + regression_test_rows} rows; "
        f"{regression_test_rows} held-out test rows).",
        "",
        "## Protocol",
        "",
        f"Rust reads `models/kumo-tabular-medium-{{task}}-dynamic-{{fp16,fp8,int4}}.onnx`; "
        f"the official API loads `pt/classifier.pt` or `pt/regressor.pt`. Each context "
        f"size is a separate run with a seed-{SEED} stratified 80/20 train/test split and "
        f"20% of the training rows held out for validation. Context rows are sampled "
        f"class-balanced for classification and uniformly for regression; every batch "
        f"contains the configured context (16, 32 or 64 rows) plus up to 2 query rows. "
        f"The classifier threshold is fitted on validation balanced accuracy; regression "
        f"uses the median of 999 quantiles. Metrics use held-out test rows only.",
        "",
        "Latency is wall time per batch on CPU: ONNX is the `session.run` call, PT is one "
        "official API call (which includes its recipe). The first call is included, so "
        "mean and p95 include warm-up. ONNX and PT latency are indicative, not a "
        "framework-only microbenchmark. PT latency repeats within a context size because "
        "one FP32 replay serves all three precision comparisons.",
        "",
        "## Classification dataset",
        "",
        "The test split is imbalanced: the all-No baseline is "
        f"{fmt(classification_rows[0]['rust']['metrics']['majority_no_accuracy'], 4)}. "
        "Accuracy is reported for completeness; balanced accuracy and ROC AUC carry "
        "more signal there.",
        "",
        "Δ and ρ compare the official FP32 model with each ONNX export on the same "
        "context/query row indices (Δ = PT − ONNX; positive ΔAUC favours PT).",
        "",
        *classification_tables(rows),
        "",
        f"### Classification inference latency ({test_rows} test rows, 2 queries per batch)",
        "",
        *latency_table(rows, "classification"),
        "",
        "## Regression dataset",
        "",
        "ΔMAE and ΔR2 are PT − ONNX; positive ΔMAE means PT has larger error and "
        "positive ΔR2 favours PT.",
        "",
        *regression_tables(rows),
        "",
        f"### Regression inference latency ({regression_test_rows} test rows, 2 queries per batch)",
        "",
        *latency_table(rows, "regression"),
        "",
        "## Findings",
        *[line for line in FINDINGS.strip().splitlines()],
        "",
        "## Caveats",
        "",
        "This is an end-to-end comparison, not bit-for-bit parity: Rust and the official "
        "Python predictor use different recipe RNGs and arithmetic paths. The same row "
        "indices are replayed through both pipelines; prediction correlation (ρ) and "
        "per-metric deltas quantify the remaining difference. The 16/32/64 runs reuse "
        "the same held-out rows and are a context-size study, not independent validation "
        "of a selected configuration.",
        "",
        "## Reproduce",
        "",
        "```sh",
        "export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib",
        "scripts/run_test_report.sh",
        "```",
        "",
        "The script resumes missing reports under `output/` and regenerates this file; "
        "individual commands are in the Evaluation section above.",
        "",
    ]
    REPORT.write_text("\n".join(lines))
    sync_readme("\n".join(lines))
    print(f"Wrote {REPORT.relative_to(ROOT)} and updated README.md")


if __name__ == "__main__":
    main()