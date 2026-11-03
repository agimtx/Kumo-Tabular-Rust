"""Compare local medium FP32 cores against corrected dynamic FP16 ONNX graphs."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sdm
import torch
from sdm.models.kumo.tabular.model import MODEL_KWARGS, _KumoTabular


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "classification": {
        "checkpoint": "pt/classifier.pt",
        "onnx": "medium/kumo-tabular-medium-classification-dynamic-fp16.onnx",
        "features": 13,
    },
    "regression": {
        "checkpoint": "pt/regressor.pt",
        "onnx": "medium/kumo-tabular-medium-regression-dynamic-fp16.onnx",
        "features": 5,
    },
}


def load_core(task: str) -> torch.nn.Module:
    model = _KumoTabular(
        num_classes=10 if task == "classification" else 0,
        num_quantiles=0 if task == "classification" else 999,
        device="meta",
        **MODEL_KWARGS["medium"],
    )
    state = torch.load(ROOT / TASKS[task]["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[4, 8, 16, 24, 32])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(context < 1 or context >= 34 for context in args.contexts):
        parser.error("context sizes must be between 1 and 33 (34 rows total)")

    torch.set_num_threads(4)
    rng = np.random.default_rng(args.seed)
    for task, config in TASKS.items():
        model = load_core(task)
        session = ort.InferenceSession(
            str(ROOT / config["onnx"]), providers=["CPUExecutionProvider"]
        )
        output_name = session.get_outputs()[0].name
        features = config["features"]
        x = rng.standard_normal((34, features)).astype(np.float32)
        mask = np.zeros(features, dtype=np.bool_)
        if task == "classification":
            y = rng.integers(0, 10, size=33, dtype=np.int64)
        else:
            y = rng.standard_normal(33).astype(np.float32)

        for context in args.contexts:
            # Keep all 34 rows fixed; context/query boundary is varied in y length.
            x_tensor = torch.from_numpy(x)
            y_tensor = torch.from_numpy(y[:context])
            with torch.no_grad():
                torch_output = model(x_tensor, y_tensor, torch.from_numpy(mask), cache=None)
            onnx_output = session.run(
                [output_name], {"x": x, "y": y_tensor.numpy(), "categorical_mask": mask}
            )[0]
            torch_array = torch_output.detach().cpu().numpy()
            if torch_array.shape != onnx_output.shape:
                raise ValueError(
                    f"{task} context={context}: output shapes differ: "
                    f"PyTorch {torch_array.shape}, ONNX {onnx_output.shape}"
                )
            difference = np.abs(torch_array.astype(np.float64) - onnx_output.astype(np.float64))
            correlation = np.corrcoef(torch_array.ravel(), onnx_output.ravel())[0, 1]
            print(
                f"{task} context={context} x={x.shape} y={y_tensor.shape} "
                f"output={torch_array.shape} max_abs_error={difference.max():.8g} "
                f"mean_abs_error={difference.mean():.8g} correlation={correlation:.8g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
