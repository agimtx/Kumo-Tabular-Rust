"""Export the tensor-level Kumo Tabular classifier or regressor to ONNX."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import torch
from onnx import TensorProto, numpy_helper
from torch.onnx import symbolic_helper
from sdm.models.kumo.tabular.model import MODEL_KWARGS, _KumoTabular


class ExportableKumo(torch.nn.Module):
    def __init__(self, model: _KumoTabular) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, categorical_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(x, y, categorical_mask, cache=None)


@symbolic_helper.parse_args("v", "is", "i", "none")
def export_nanmean(g, values, dimensions, keepdim, dtype):
    missing = g.op("IsNaN", values)
    zero = g.op("Constant", value_t=torch.tensor(0.0))
    clean = g.op("Where", missing, zero, values)
    valid = g.op("Cast", g.op("Not", missing), to_i=onnx.TensorProto.FLOAT)
    axes = g.op("Constant", value_t=torch.tensor(dimensions, dtype=torch.int64))
    total = g.op("ReduceSum", clean, axes, keepdims_i=keepdim)
    count = g.op("ReduceSum", valid, axes, keepdims_i=keepdim)
    return g.op("Div", total, count)


def export_matrix_transpose(g, values):
    rank = symbolic_helper._get_tensor_rank(values)
    if rank is None or rank < 2:
        raise ValueError("mT export requires a known rank of at least two")
    permutation = list(range(rank))
    permutation[-2:] = permutation[-2:][::-1]
    return g.op("Transpose", values, perm_i=permutation)


@symbolic_helper.parse_args("v", "is", "v", "f")
def export_rms_norm(g, values, normalized_shape, weight, eps):
    square = g.op("Mul", values, values)
    mean = g.op("ReduceMean", square, axes_i=[-1], keepdims_i=1)
    epsilon = g.op(
        "Constant",
        value_t=torch.tensor(
            eps if eps is not None else torch.finfo(torch.float32).eps,
            dtype=torch.float32,
        ),
    )
    normalized = g.op("Div", values, g.op("Sqrt", g.op("Add", mean, epsilon)))
    if symbolic_helper._is_none(weight):
        return normalized
    return g.op("Mul", normalized, weight)


def export(checkpoint: Path, output: Path, size: str, task: str) -> None:
    classifier = task == "classification"
    model = _KumoTabular(
        num_classes=10 if classifier else 0,
        num_quantiles=0 if classifier else 999,
        device="meta",
        **MODEL_KWARGS[size],
    )
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval()

    # The public KumoTabular classifier also runs ECOC outside this core network.
    x = torch.randn(6, 3, dtype=torch.float32)
    y = (
        torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        if classifier
        else torch.randn(4)
    )
    categorical_mask = torch.zeros(3, dtype=torch.bool)
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=output.parent) as temporary_dir:
        fp32_path = Path(temporary_dir) / "model-fp32.onnx"
        # The gradient-enabled branch avoids inference-only output buffers and
        # chunking, which are not part of the mathematical forward pass.
        with torch.enable_grad():
            torch.onnx.register_custom_op_symbolic(
                "aten::nanmean", export_nanmean, 17
            )
            torch.onnx.register_custom_op_symbolic(
                "aten::mT", export_matrix_transpose, 17
            )
            torch.onnx.register_custom_op_symbolic(
                "aten::rms_norm", export_rms_norm, 17
            )
            torch.onnx.export(
                ExportableKumo(model),
                (x, y, categorical_mask),
                str(fp32_path),
                input_names=["x", "y", "categorical_mask"],
                output_names=["predictions"],
                opset_version=17,
                dynamo=False,
                verbose=False,
            )
        del model
        graph = onnx.load(str(fp32_path))
        casts = []
        for weight in graph.graph.initializer:
            if weight.data_type != TensorProto.FLOAT:
                continue
            name = weight.name
            half = numpy_helper.from_array(
                numpy_helper.to_array(weight).astype("float16"), name=name + "_fp16"
            )
            weight.CopyFrom(half)
            casts.append(
                onnx.helper.make_node(
                    "Cast", [half.name], [name], name=name + "_to_fp32",
                    to=TensorProto.FLOAT,
                )
            )
        for cast in reversed(casts):
            graph.graph.node.insert(0, cast)
        onnx.save_model(
            graph,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            size_threshold=1024 * 1024 * 1024,
            location=f"{output.name}.data",
        )
    onnx.checker.check_model(str(output))
    print(f"Exported FP16 {task} {size} model to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, choices=tuple(MODEL_KWARGS))
    parser.add_argument(
        "--task", required=True, choices=("classification", "regression")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.size, args.task)


if __name__ == "__main__":
    main()
