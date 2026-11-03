"""Compare official FP32 KumoTabular with traced Rust medium ONNX batches.

Run with the isolated Python environment described in README. Each context and
query row index is read from Rust's --trace output; no Python split is drawn.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import sdm
import torch


def note(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def batches_fingerprint(trace: dict) -> str:
    digest = hashlib.sha256()
    for split in ("validation", "test"):
        for batch in trace[f"{split}_batches"]:
            digest.update(np.asarray(batch["context"], dtype=np.int64).tobytes())
            digest.update(np.asarray(batch["query"], dtype=np.int64).tobytes())
    return digest.hexdigest()[:32]


def auc(truth: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    rank_sum = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        rank_sum += ((start + 1 + end) / 2.0) * sorted_truth[start:end].sum()
        start = end
    positive = truth.sum()
    negative = len(truth) - positive
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def counts(truth: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    positive = scores > threshold
    yes = truth == 1
    return [int((~positive & ~yes).sum()), int((positive & ~yes).sum()),
            int((~positive & yes).sum()), int((positive & yes).sum())]


def balanced(truth: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    tn, fp, fn, tp = counts(truth, scores, threshold)
    return (tn / (tn + fp) + tp / (tp + fn)) / 2


def select_threshold(truth: np.ndarray, scores: np.ndarray) -> float:
    return max(np.unique(scores), key=lambda cut: (balanced(truth, scores, cut), cut))


def load_model(task: str, device: str) -> sdm.models.KumoTabular:
    model = sdm.models.KumoTabular(task=task, size="medium", pretrained=False, device="cpu")
    path = Path("pt") / ("classifier.pt" if task == "classification" else "regressor.pt")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model.models[sdm.Task(task)].load_state_dict(weights, strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("classification", "regression"))
    parser.add_argument("--context-rows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--precision", choices=("fp16", "fp8", "int4"), default="fp16")
    args = parser.parse_args()
    trace_path = Path(f"output/medium-{args.task}-context-{args.context_rows}-seed-{args.seed}-trace.json")
    if args.context_rows == 32:
        report_path = Path(f"output/medium-{args.task}-{args.precision}-full-evaluation-rust.json")
    else:
        report_path = Path(
            f"output/medium-{args.task}-{args.precision}-full-context-{args.context_rows}"
            "-evaluation-rust.json"
        )
    trace = json.loads(trace_path.read_text())
    rust_report = json.loads(report_path.read_text())[0]
    if rust_report["task"] != args.task or rust_report["precision"] != args.precision:
        raise ValueError(f"{report_path} does not match requested task and precision")
    dataframe = pd.read_parquet(trace["dataset"])
    assert len(dataframe) > max(trace["fit"] + trace["validation"] + trace["test"])
    assert list(dataframe.drop(columns=trace["target"]).columns) == trace["features"]
    fingerprint = batches_fingerprint(trace)
    # The FP32 checkpoint and its input batches do not depend on the ONNX
    # precision, so all three comparisons share one replay result.
    cache_path = Path(
        f"output/medium-{args.task}-context-{args.context_rows}-seed-{args.seed}"
        "-official-predictions.json"
    )
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else None
    if cached is not None and cached.get("fingerprint") != fingerprint:
        note(f"official predictions cache is stale for {args.task}; replaying")
        cached = None
    truth = dataframe[trace["target"]].to_numpy()

    if cached is not None:
        note(f"official predictions cache hit: {cache_path}")
        validation_predictions = np.asarray(cached["validation_predictions"], dtype=np.float64)
        test_predictions = np.asarray(cached["test_predictions"], dtype=np.float64)
        cut = cached["classification_threshold"]
        metrics = cached["metrics"]
        elapsed = cached["evaluation_seconds"]
    else:
        stypes = {column: ("categorical" if pd.api.types.is_string_dtype(dataframe[column])
                             else "numerical") for column in trace["features"]}
        stypes[trace["target"]] = "categorical" if args.task == "classification" else "numerical"
        table = sdm.TableTensor.from_pandas(dataframe, stypes=stypes, device=args.device)
        note(f"loading official checkpoint and replaying batches for {args.task}")
        model = load_model(args.task, args.device)

        def predict(batches: list[dict], label: str) -> np.ndarray:
            predictions = []
            for index, batch in enumerate(batches):
                context = table[batch["context"]]
                query = table[batch["query"]]
                generator = torch.Generator(device=args.device).manual_seed(args.seed + index)
                result = model(
                    x_context=context.drop_columns(trace["target"]),
                    y_context=context.select_columns([trace["target"]]),
                    x_query=query.drop_columns(trace["target"]),
                    num_estimators=1,
                    generator=generator,
                )
                if args.task == "classification":
                    names = list(result.columns[sdm.Stype.numerical])
                    probabilities = result.numerical.detach().cpu().numpy()
                    scores = np.log(probabilities[:, names.index("Yes")]) - np.log(probabilities[:, names.index("No")])
                    predictions.extend(scores.tolist())
                else:
                    quantiles = result.numerical.detach().cpu().numpy()
                    predictions.extend(quantiles[:, 499].tolist())
                if (index + 1) % 50 == 0:
                    note(f"{args.task} {label}: {index+1}/{len(batches)} batches")
            return np.asarray(predictions, dtype=np.float64)

        start = time.monotonic()
        validation_predictions = predict(trace["validation_batches"], "validation")
        test_predictions = predict(trace["test_batches"], "test")
        elapsed = time.monotonic() - start
        if args.task == "classification":
            validation_truth = (truth[trace["validation"]] == "Yes").astype(np.float64)
            test_truth = (truth[trace["test"]] == "Yes").astype(np.float64)
            cut = float(select_threshold(validation_truth, validation_predictions))
            tn, fp, fn, tp = counts(test_truth, test_predictions, cut)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            metrics = {
                "accuracy": (tn + tp) / len(test_truth),
                "balanced_accuracy": balanced(test_truth, test_predictions, cut),
                "roc_auc_yes": auc(test_truth, test_predictions),
                "precision_yes": precision, "recall_yes": recall,
                "f1_yes": 2 * precision * recall / max(precision + recall, float(np.finfo(float).eps)),
                "confusion_matrix_no_yes": [[tn, fp], [fn, tp]],
            }
        else:
            cut = None
            test_truth = truth[trace["test"]].astype(float)
            error = test_predictions - test_truth
            metrics = {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
                       "r2": float(1 - np.square(error).sum() / np.square(test_truth - test_truth.mean()).sum())}
        cache_path.write_text(json.dumps({
            "fingerprint": fingerprint, "task": args.task, "context_rows": args.context_rows,
            "seed": args.seed, "classification_threshold": cut, "metrics": metrics,
            "evaluation_seconds": elapsed,
            "validation_predictions": validation_predictions.tolist(),
            "test_predictions": test_predictions.tolist(),
        }, indent=2) + "\n")
        note(f"cached official predictions: {cache_path}")
    if not all(math.isfinite(value) for value in (validation_predictions.tolist() + test_predictions.tolist())):
        raise ValueError("official predictions contain nonfinite values")
    rust = np.asarray(rust_report["test_predictions"], dtype=float)
    if len(rust) != len(test_predictions):
        raise ValueError("Rust and PyTorch held-out predictions have different lengths")
    difference = test_predictions - rust
    compared_metrics = ("roc_auc_yes", "balanced_accuracy", "accuracy", "f1_yes") if args.task == "classification" else ("mae", "rmse", "r2")
    comparison = {
        "rust_precision": args.precision,
        "rust_metrics": rust_report["metrics"],
        "official_minus_rust_metrics": {key: metrics[key] - rust_report["metrics"][key]
                                         for key in compared_metrics},
        "prediction_pearson": float(np.corrcoef(rust, test_predictions)[0, 1]),
        "prediction_mae_difference": float(np.abs(difference).mean()),
        "prediction_rmse_difference": float(np.sqrt(np.square(difference).mean())),
        "prediction_bias_official_minus_rust": float(difference.mean()),
    }
    output = {"engine": "official-python-torch", "precision": "fp32", "size": "medium", "task": args.task,
              "model": str(Path("pt") / ("classifier.pt" if args.task == "classification" else "regressor.pt")),
              "trace": str(trace_path), "rust_report": str(report_path),
              "context_rows": args.context_rows, "seed": args.seed,
              "num_estimators": 1, "device": args.device, "evaluation_seconds": elapsed,
              "classification_threshold": cut, "metrics": metrics, "comparison_to_rust": comparison,
              "test_predictions": test_predictions.tolist(), "validation_predictions": validation_predictions.tolist()}
    suffix = "" if args.context_rows == 32 else f"-context-{args.context_rows}"
    output_path = Path(f"output/medium-{args.task}-{args.precision}{suffix}-official-comparison.json")
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    summary = {"task": args.task, "precision": args.precision, "metrics": metrics,
               "official_minus_onnx": comparison["official_minus_rust_metrics"],
               "prediction_pearson": comparison["prediction_pearson"],
               "prediction_mae_difference": comparison["prediction_mae_difference"]}
    note(f"{json.dumps(summary)}\nReport: {output_path}")


if __name__ == "__main__":
    main()
