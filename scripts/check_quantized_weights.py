"""Compare dequantized ONNX initializers with the original PyTorch checkpoint.

Every INT4 matrix is reconstructed from its packed nibbles and per-output-row
scale.  Named initializers are matched to the FP32 state dict by key; the
anonymous ``onnx::MatMul_*`` tensors are matched by shape and dequantized-value
similarity.  The FP16 export is a lossless reference for all remaining
tensors, so coverage and per-layer error can be reported for the whole model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "classification": "classifier.pt",
    "regression": "regressor.pt",
}
HELPERS = (
    "_scale", "_shape", "_flat_shape", "_slice_start", "_slice_end",
    "_slice_axis", "_slice_step", "_indices",
)


def decode_int4(packed: np.ndarray, scale: np.ndarray, size: int) -> np.ndarray:
    nibbles = np.empty(packed.size * 2, dtype=np.int8)
    nibbles[0::2] = (packed & 0x0F).astype(np.int8)
    nibbles[1::2] = (packed >> 4).astype(np.int8)
    nibbles[nibbles >= 8] -= 16
    quantized = nibbles[:size].astype(np.float64)
    rows = int(scale.size)
    if rows <= 0 or size % rows:
        raise ValueError(f"cannot reshape {size} values into {rows} rows")
    values = quantized.reshape(rows, size // rows) * scale.reshape(rows, 1)
    return values.astype(np.float32)


def decode_fp8(indices: np.ndarray) -> np.ndarray:
    import ml_dtypes

    return indices.view(ml_dtypes.float8_e4m3fn).astype(np.float32)


def load_onnx_weights(path: Path) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    """Return (base-name -> float32 tensor, base-name -> was quantized)."""
    graph = onnx.load(str(path), load_external_data=True).graph
    initializers = {init.name: init for init in graph.initializer}
    arrays: dict[str, np.ndarray] = {}
    quantized: dict[str, bool] = {}
    for name, init in initializers.items():
        if any(name.endswith(suffix) for suffix in HELPERS):
            continue
        if name.endswith("_int4_packed_uint8"):
            base = name[: -len("_int4_packed_uint8")]
            scale = numpy_helper.to_array(initializers[base + "_scale"]).astype(np.float64)
            shape = numpy_helper.to_array(initializers[base + "_shape"])
            size = int(np.prod(shape))
            arrays[base] = decode_int4(numpy_helper.to_array(init), scale, size)
            quantized[base] = True
        elif name.endswith("_fp8_uint8"):
            base = name[: -len("_fp8_uint8")]
            arrays[base] = decode_fp8(numpy_helper.to_array(init))
            quantized[base] = True
        elif name.endswith("_fp16"):
            base = name[: -len("_fp16")]
            arrays[base] = numpy_helper.to_array(init).astype(np.float32)
            quantized[base] = True
        elif init.data_type == onnx.TensorProto.FLOAT:
            arrays[name] = numpy_helper.to_array(init).astype(np.float32)
            quantized[name] = False
    return arrays, quantized


def errors(values: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64).reshape(-1)
    reference = reference.astype(np.float64).reshape(-1)
    norm = np.linalg.norm(reference)
    delta = values - reference
    max_abs = float(np.abs(delta).max(initial=0))
    ref_max = float(np.abs(reference).max(initial=1))
    return {
        "relative_l2": float(np.linalg.norm(delta) / norm) if norm else 0.0,
        "max_abs": max_abs,
        "relative_max_abs": max_abs / ref_max if ref_max else 0.0,
        "cosine": float(values @ reference / (np.linalg.norm(values) * norm))
        if norm and np.linalg.norm(values)
        else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(TASKS))
    parser.add_argument("--precision", required=True, choices=("int4", "fp8", "fp16"))
    parser.add_argument("--size", default="medium")
    args = parser.parse_args()

    model_path = ROOT / "models" / f"kumo-tabular-{args.size}-{args.task}-dynamic-{args.precision}.onnx"
    fp16_path = ROOT / "models" / f"kumo-tabular-{args.size}-{args.task}-dynamic-fp16.onnx"
    state = torch.load(ROOT / "pt" / TASKS[args.task], map_location="cpu", weights_only=True)
    state = {key: value.float().numpy() for key, value in state.items()}
    lossless, _ = load_onnx_weights(fp16_path)
    values_by_name, was_quantized = load_onnx_weights(model_path)
    # Drop helper constants that only exist in the quantized graph.
    values_by_name = {k: v for k, v in values_by_name.items() if k in lossless}

    records = []
    matched_pt = anonymous = matched_fp16 = 0
    for name, values in values_by_name.items():
        record = {"tensor": name, "shape": list(values.shape),
                  "precision_weight": bool(was_quantized.get(name, False))}
        key = name[len("model."):] if name.startswith("model.") else name
        if key in state and state[key].shape == values.shape:
            record.update(errors(values, state[key]))
            record["reference"] = "pt"
            record["pt_key"] = key
            matched_pt += 1
        else:
            candidates = [k for k, v in state.items() if v.shape == values.shape]
            scored = sorted((errors(values, state[c])["relative_l2"], c) for c in candidates)
            best_error, best = scored[0] if scored else (float("inf"), None)
            second = scored[1][0] if len(scored) > 1 else float("inf")
            if best is not None and best_error < 0.5 and second > 2 * best_error:
                record.update(errors(values, state[best]))
                record["reference"] = "pt-shape-match"
                record["pt_key"] = best
                anonymous += 1
            else:
                record.update(errors(values, lossless[name]))
                record["reference"] = "fp16-onnx"
                matched_fp16 += 1
        records.append(record)

    quantized_records = [r for r in records if r["precision_weight"]]
    relative = np.asarray([r["relative_l2"] for r in quantized_records])
    relative_all = np.asarray([r["relative_l2"] for r in records])
    summary = {
        "task": args.task,
        "size": args.size,
        "precision": args.precision,
        "model": str(model_path.relative_to(ROOT)),
        "reference_pt": str((ROOT / "pt" / TASKS[args.task]).relative_to(ROOT)),
        "tensors_total": len(records),
        "quantized_tensors": len(quantized_records),
        "fp32_retained_tensors": len(records) - len(quantized_records),
        "named_pt_matches": matched_pt,
        "anonymous_shape_matches": anonymous,
        "fp16_reference_matches": matched_fp16,
        "relative_l2_mean": float(relative.mean()),
        "relative_l2_median": float(np.median(relative)),
        "relative_l2_p95": float(np.percentile(relative, 95)),
        "relative_l2_max": float(relative.max()),
        "relative_l2_all_tensors_mean": float(relative_all.mean()),
        "cosine_min": float(min(r["cosine"] for r in quantized_records)),
        "relative_l2_zero_tensors": int(np.sum(relative == 0.0)),
        "worst_tensors": sorted(quantized_records, key=lambda r: -r["relative_l2"])[:10],
    }
    output = ROOT / "output" / f"{args.size}-{args.task}-{args.precision}-weight-check.json"
    output.write_text(json.dumps({"summary": summary, "tensors": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()