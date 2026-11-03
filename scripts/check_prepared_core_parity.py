"""Compare official PyTorch and ONNX core outputs on identical Rust tensors."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sdm.models.kumo.tabular.model import MODEL_KWARGS, _KumoTabular

ROOT = Path(__file__).resolve().parents[1]


def load_core(task: str) -> torch.nn.Module:
    model = _KumoTabular(
        num_classes=10 if task == "classification" else 0,
        num_quantiles=0 if task == "classification" else 999,
        device="meta",
        **MODEL_KWARGS["medium"],
    )
    filename = "classifier.pt" if task == "classification" else "regressor.pt"
    state = torch.load(ROOT / "pt" / filename, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--context-rows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    trace_path = ROOT / "output" / f"medium-{args.task}-context-{args.context_rows}-seed-{args.seed}-trace.json"
    trace = json.loads(trace_path.read_text())
    batches = trace[f"{args.split}_prepared"]
    model = load_core(args.task)
    torch.set_num_threads(4)
    rust_scores = []
    torch_scores = []
    max_abs = 0.0
    mean_abs_sum = 0.0
    value_count = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            x = torch.tensor(batch["x"], dtype=torch.float32)
            y = torch.tensor(
                batch["y"],
                dtype=torch.int64 if args.task == "classification" else torch.float32,
            )
            mask = torch.tensor(batch["categorical_mask"], dtype=torch.bool)
            torch_raw = model(x, y, mask, cache=None).cpu().numpy()
            rust_raw = np.asarray(batch["rust_output"], dtype=np.float32)
            if torch_raw.shape != rust_raw.shape:
                raise ValueError(f"batch {batch_index}: shape mismatch {torch_raw.shape} != {rust_raw.shape}")
            delta = np.abs(torch_raw.astype(np.float64) - rust_raw.astype(np.float64))
            max_abs = max(max_abs, float(delta.max(initial=0)))
            mean_abs_sum += float(delta.sum())
            value_count += delta.size
            if args.task == "classification":
                classes = batch["class_permutation"]
                no_index = classes.index(0.0)
                yes_index = classes.index(1.0)
                rust_score = rust_raw[:, yes_index] - rust_raw[:, no_index]
                torch_score = torch_raw[:, yes_index] - torch_raw[:, no_index]
            else:
                rust_sorted = np.sort(rust_raw, axis=-1)
                torch_sorted = np.sort(torch_raw, axis=-1)
                center = batch["target_center"]
                scale = batch["target_scale"]
                sign = batch["target_sign"]
                rust_score = rust_sorted[:, 499] * scale * sign + center
                torch_score = torch_sorted[:, 499] * scale * sign + center
            rust_scores.extend(rust_score.tolist())
            torch_scores.extend(torch_score.tolist())

    rust_scores = np.asarray(rust_scores, dtype=np.float64)
    torch_scores = np.asarray(torch_scores, dtype=np.float64)
    prediction_delta = np.abs(rust_scores - torch_scores)
    report = {
        "task": args.task,
        "split": args.split,
        "context_rows": args.context_rows,
        "batches": len(batches),
        "rows": len(rust_scores),
        "core_output_max_abs_error": max_abs,
        "core_output_mean_abs_error": mean_abs_sum / max(value_count, 1),
        "prediction_mean_abs_error": float(prediction_delta.mean()),
        "prediction_max_abs_error": float(prediction_delta.max(initial=0)),
        "prediction_correlation": float(np.corrcoef(rust_scores, torch_scores)[0, 1]),
    }
    output = ROOT / "output" / f"medium-{args.task}-context-{args.context_rows}-prepared-core-parity.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
