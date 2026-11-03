"""Fail the export job unless its ONNX graph handles real context/feature shapes."""

import argparse

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    args = parser.parse_args()
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    for context, query, features in ((32, 2, 13 if args.task == "classification" else 5),
                                      (16, 1, 10 if args.task == "classification" else 4)):
        x = rng.standard_normal((context + query, features)).astype(np.float32)
        x[0, 0] = np.nan
        y = (rng.integers(0, 2, context, dtype=np.int64) if args.task == "classification"
             else rng.standard_normal(context).astype(np.float32))
        mask = np.zeros(features, dtype=np.bool_)
        if args.task == "classification":
            mask[-1] = True
            x[:, -1] = rng.integers(0, 3, context + query).astype(np.float32)
        inputs = {"x": x, "y": y, "categorical_mask": mask}
        out = session.run(["predictions"], inputs)[0]
        expected = (query, 10 if args.task == "classification" else 999)
        if out.shape != expected or not np.isfinite(out).all():
            raise ValueError(f"Expected finite {expected}, got {out.shape}")
        modified = x.copy()
        modified[-1, 1] += 10.0
        inputs["x"] = modified
        changed = session.run(["predictions"], inputs)[0]
        if np.allclose(out[-1], changed[-1], atol=1e-6, rtol=1e-6):
            raise ValueError("Model output did not respond to changed query input")
        print(f"Validated context={context}, query={query}, features={features}: {out.shape}")


if __name__ == "__main__":
    main()
