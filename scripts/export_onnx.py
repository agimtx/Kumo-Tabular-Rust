"""Export the original FP32 Kumo Tabular core or compressed weights to ONNX."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import onnx
import numpy as np
import ml_dtypes
import torch
import torch.nn.functional as functional

original_scaled_dot_product_attention = functional.scaled_dot_product_attention
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


def export_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    if not enable_gqa:
        return original_scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale=scale,
        )

    repeats = query.size(-3) // key.size(-3)
    key = key.repeat_interleave(repeats, dim=-3)
    value = value.repeat_interleave(repeats, dim=-3)
    factor = scale if scale is not None else query.size(-1) ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * factor
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    if is_causal:
        causal = torch.ones(
            query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
        ).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    if dropout_p:
        probabilities = functional.dropout(probabilities, dropout_p)
    return torch.matmul(probabilities, value)


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


def fix_sdpa_key_length_gathers(graph: onnx.ModelProto, size: str) -> None:
    """Repair legacy-exporter key-length gathers in ICL attention scaling."""
    expected_casts = {"small": 12, "medium": 47, "large": 47}[size]
    expected_gathers = {"small": 0, "medium": 45, "large": 45}[size]
    producers = {
        output: node
        for node in graph.graph.node
        for output in node.output
    }
    casts = [
        node for node in graph.graph.node
        if node.name.startswith("/model/icl_block/layers.")
        and node.name.endswith(("/query_scaling/Cast", "/query_scaling_1/Cast"))
    ]
    if len(casts) != expected_casts:
        raise RuntimeError(
            f"expected {expected_casts} ICL query-scaling casts for {size}, found {len(casts)}"
        )

    gathers = 0
    for cast in casts:
        source = producers.get(cast.input[0])
        if source is None:
            raise RuntimeError(f"missing query-scaling input producer: {cast.name}")
        if source.op_type == "Squeeze":
            slice_node = producers.get(source.input[0])
            if slice_node is None or slice_node.op_type != "Slice":
                raise RuntimeError(f"unexpected query-scaling squeeze path: {cast.name}")
            constants = [producers.get(name) for name in slice_node.input[1:]]
            if any(item is None or item.op_type != "Constant" for item in constants):
                raise RuntimeError(f"unexpected key-length slice constants: {slice_node.name}")
            values = [int(numpy_helper.to_array(
                next(attribute.t for attribute in item.attribute if attribute.name == "value")
            ).reshape(-1)[0]) for item in constants]
            if values[:3] != [-3, -2, 0]:
                raise RuntimeError(f"unexpected key-length slice axes: {slice_node.name}: {values}")
            continue
        if source.op_type != "Gather" or len(source.input) != 2:
            raise RuntimeError(f"unexpected query-scaling path: {cast.name} -> {source.op_type}")
        shape = producers.get(source.input[0])
        key = producers.get(shape.input[0]) if shape is not None and shape.op_type == "Shape" else None
        if key is None or "/attn/key_transform/" not in key.name and "/model/icl_block/Slice_" not in key.name:
            raise RuntimeError(f"Gather is not reading the ICL attention key length: {source.name}")
        axis = next((attribute.i for attribute in source.attribute if attribute.name == "axis"), 0)
        if axis != 0:
            raise RuntimeError(f"unexpected Gather axis {axis}: {source.name}")
        index_node = producers.get(source.input[1])
        if index_node is None or index_node.op_type != "Constant":
            raise RuntimeError(f"key-length Gather index is not a Constant: {source.name}")
        index = next(
            (attribute.t for attribute in index_node.attribute if attribute.name == "value"),
            None,
        )
        if index is None or np.asarray(numpy_helper.to_array(index)).shape != ():
            raise RuntimeError(f"key-length Gather index is not scalar: {source.name}")
        value = int(numpy_helper.to_array(index).item())
        if value == 1:
            index.CopyFrom(numpy_helper.from_array(np.asarray(-3, dtype=np.int64)))
        elif value not in (0, -3):
            raise RuntimeError(f"unexpected key-length index {value}: {source.name}")
        gathers += 1

    if gathers != expected_gathers:
        raise RuntimeError(
            f"expected {expected_gathers} ICL key-length gathers for {size}, found {gathers}"
        )


INT4_MIN_MATRIX_ELEMENTS = 4096
REGRESSION_HEAD_WEIGHT = "model.icl_block.head.2.weight"
PRECISION_SUFFIX = {"fp32": "f32", "fp16": "f16", "fp8": "f8", "int4": "i4"}
TASK_SUFFIX = {"classification": "cls", "regression": "reg"}


def model_filename(task: str, size: str, precision: str) -> str:
    """Name an export by task, upstream model size and stored weight precision."""
    return f"tabular-{TASK_SUFFIX[task]}-{size}-{PRECISION_SUFFIX[precision]}.onnx"


def keep_int4_weight_fp32(name: str, task: str) -> bool:
    """Keep the regression quantile projection exact in an INT4 export."""
    return task == "regression" and name == REGRESSION_HEAD_WEIGHT


def int4_group_size_for_task(task: str, requested: int | None) -> int:
    """Use finer groups for regression without changing classification exports."""
    if requested is not None and requested < 0:
        raise ValueError("group_size must be nonnegative")
    return requested if requested is not None else (16 if task == "regression" else 0)


def quantize_int4(values: np.ndarray, group_size: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrically quantize each output channel, optionally in input groups."""
    source = np.asarray(values, dtype=np.float32)
    if source.ndim < 2:
        raise ValueError("INT4 is only used for matrix weights")
    if group_size < 0:
        raise ValueError("group_size must be nonnegative")
    width = (
        group_size if group_size and source.shape[-1] % group_size == 0
        else source.shape[-1]
    )
    rows = source.reshape(-1, width).astype(np.float64)
    maxima = np.max(np.abs(rows), axis=-1)
    scales = np.where(maxima > 0, maxima / 7.0, 1.0).astype(np.float32)
    quantized = np.clip(np.rint(rows / scales[:, None]), -8, 7).astype(np.int8)
    groups = source.shape[-1] // width
    scale_shape = source.shape[:-1] + ((groups,) if groups > 1 else ())
    return quantized.reshape(source.shape), scales.reshape(scale_shape)


def pack_int4(values: np.ndarray) -> np.ndarray:
    """Pack signed INT4 values in row-major order, low nibble first."""
    quantized = np.asarray(values, dtype=np.int8).reshape(-1)
    if quantized.size % 2:
        quantized = np.pad(quantized, (0, 1))
    nibbles = quantized.astype(np.uint8) & 0x0F
    return nibbles[0::2] | (nibbles[1::2] << 4)


def unpack_int4(values: np.ndarray, count: int) -> np.ndarray:
    """Unpack low-nibble-first signed INT4 values for exporter tests."""
    packed = np.asarray(values, dtype=np.uint8).reshape(-1)
    nibbles = np.empty(packed.size * 2, dtype=np.int8)
    nibbles[0::2] = (packed & 0x0F).astype(np.int8)
    nibbles[1::2] = (packed >> 4).astype(np.int8)
    nibbles[nibbles >= 8] -= 16
    return nibbles[:count]


def export(checkpoint: Path, output: Path, size: str, task: str, quantization: str,
           int4_group_size: int | None = None) -> None:
    if quantization not in PRECISION_SUFFIX:
        raise ValueError(f"unsupported weight precision: {quantization}")
    if output.name != model_filename(task, size, quantization):
        raise ValueError(f"output filename must be {model_filename(task, size, quantization)}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing {size} {task} checkpoint: {checkpoint}")
    classifier = task == "classification"
    group_size = int4_group_size_for_task(task, int4_group_size) if quantization == "int4" else 0
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
    x = torch.randn(34, 13 if classifier else 5, dtype=torch.float32)
    y = (
        torch.tensor([0, 1] * 16, dtype=torch.int64)
        if classifier
        else torch.randn(32)
    )
    categorical_mask = torch.zeros(x.size(1), dtype=torch.bool)
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=output.parent) as temporary_dir:
        fp32_path = Path(temporary_dir) / "model-fp32.onnx"
        # The gradient-enabled branch avoids inference-only output buffers and
        # chunking, which are not part of the mathematical forward pass.
        with torch.enable_grad(), patch.object(
            functional,
            "scaled_dot_product_attention",
            export_scaled_dot_product_attention,
        ):
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
                dynamic_axes={
                    "x": {0: "total_rows", 1: "features"},
                    "y": {0: "context_rows"},
                    "categorical_mask": {0: "features"},
                    "predictions": {0: "query_rows"},
                },
                opset_version=17,
                dynamo=False,
                verbose=False,
            )
        del model
        graph = onnx.load(str(fp32_path))
        fix_sdpa_key_length_gathers(graph, size)
        casts = []
        fp8_lookup_name = "__fp8_e4m3fn_to_fp32"
        original_initializers = list(graph.graph.initializer)
        for weight in original_initializers:
            if weight.data_type != TensorProto.FLOAT:
                continue
            name = weight.name
            values = numpy_helper.to_array(weight)
            if quantization == "fp32":
                # Keep the exact FP32 initializers produced from the official checkpoint.
                continue
            if quantization == "fp16":
                compressed = numpy_helper.from_array(
                    values.astype(np.float16), name=name + "_fp16"
                )
            elif quantization == "fp8":
                encoded = values.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
                compressed = numpy_helper.from_array(
                    encoded, name=name + "_fp8_uint8"
                )
            elif quantization == "int4" and keep_int4_weight_fp32(name, task):
                if values.ndim != 2 or values.shape[0] != 999:
                    raise ValueError(f"unexpected regression head shape: {values.shape}")
                continue
            elif quantization == "int4" and values.ndim >= 2 and values.size >= INT4_MIN_MATRIX_ELEMENTS:
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite weight values in {name}")
                quantized, scales = quantize_int4(values, group_size)
                encoded = pack_int4(quantized)
                compressed = numpy_helper.from_array(encoded, name=name + "_int4_packed_uint8")
            elif quantization == "int4":
                # Small vectors and scalars are numerically sensitive and save
                # negligible space, so retain them in FP32.
                continue
            weight.CopyFrom(compressed)
            if quantization == "fp8":
                indices_name = name + "_fp8_indices"
                casts.extend((
                    onnx.helper.make_node(
                        "Cast", [compressed.name], [indices_name],
                        name=name + "_fp8_indices_cast", to=TensorProto.INT64,
                    ),
                    onnx.helper.make_node(
                        "Gather", [fp8_lookup_name, indices_name], [name],
                        name=name + "_fp8_decode", axis=0,
                    ),
                ))
            elif quantization == "int4":
                shape = np.asarray(values.shape, dtype=np.int64)
                groups = scales.size // int(np.prod(values.shape[:-1]))
                group_shape = values.shape[:-1] + (groups, values.shape[-1] // groups)
                flat_name = name + "_int4_flat"
                sliced_name = name + "_int4_unpadded"
                gathered_name = name + "_int4_pairs"
                indices_name = name + "_int4_indices"
                graph.graph.initializer.extend((
                    numpy_helper.from_array(
                        scales.reshape(group_shape[:-1] + (1,) if groups > 1 else values.shape[:-1] + (1,)),
                        name=name + "_scale",
                    ),
                    numpy_helper.from_array(np.asarray([-1], dtype=np.int64), name=name + "_flat_shape"),
                    numpy_helper.from_array(np.asarray([0], dtype=np.int64), name=name + "_slice_start"),
                    numpy_helper.from_array(np.asarray([values.size], dtype=np.int64), name=name + "_slice_end"),
                    numpy_helper.from_array(np.asarray([0], dtype=np.int64), name=name + "_slice_axis"),
                    numpy_helper.from_array(np.asarray([1], dtype=np.int64), name=name + "_slice_step"),
                    numpy_helper.from_array(shape, name=name + "_shape"),
                ))
                if groups > 1:
                    graph.graph.initializer.append(numpy_helper.from_array(
                        np.asarray(group_shape, dtype=np.int64), name=name + "_group_shape"
                    ))
                casts.extend((
                    onnx.helper.make_node(
                        "Cast", [compressed.name], [indices_name],
                        name=name + "_int4_indices_cast", to=TensorProto.INT64,
                    ),
                    onnx.helper.make_node(
                        "Gather", ["__int4_byte_to_pair", indices_name], [gathered_name],
                        name=name + "_int4_unpack", axis=0,
                    ),
                    onnx.helper.make_node(
                        "Reshape", [gathered_name, name + "_flat_shape"], [flat_name],
                        name=name + "_int4_flatten",
                    ),
                    onnx.helper.make_node(
                        "Slice", [flat_name, name + "_slice_start", name + "_slice_end",
                                  name + "_slice_axis", name + "_slice_step"], [sliced_name],
                        name=name + "_int4_remove_padding",
                    ),
                    onnx.helper.make_node(
                        "Reshape", [sliced_name, name + ("_group_shape" if groups > 1 else "_shape")],
                        [name + "_int4_values"],
                        name=name + "_int4_restore_shape",
                    ),
                    onnx.helper.make_node(
                        "Mul", [name + "_int4_values", name + "_scale"],
                        [name + "_int4_group_values" if groups > 1 else name],
                        name=name + "_int4_dequantize",
                    ),
                ))
                if groups > 1:
                    casts.append(onnx.helper.make_node(
                        "Reshape", [name + "_int4_group_values", name + "_shape"], [name],
                        name=name + "_int4_restore_matrix",
                    ))
            else:
                casts.append(onnx.helper.make_node(
                    "Cast", [compressed.name], [name], name=name + "_to_fp32",
                    to=TensorProto.FLOAT,
                ))
        if quantization == "fp8":
            fp8_values = np.arange(256, dtype=np.uint8).view(
                ml_dtypes.float8_e4m3fn
            ).astype(np.float32)
            graph.graph.initializer.append(
                numpy_helper.from_array(fp8_values, name=fp8_lookup_name)
            )
        elif quantization == "int4":
            byte_values = np.arange(256, dtype=np.uint8)
            low = (byte_values & 0x0F).astype(np.int8)
            high = (byte_values >> 4).astype(np.int8)
            low[low >= 8] -= 16
            high[high >= 8] -= 16
            int4_lookup = np.stack((low, high), axis=1).astype(np.float32)
            graph.graph.initializer.append(
                numpy_helper.from_array(int4_lookup, name="__int4_byte_to_pair")
            )
        for cast in reversed(casts):
            graph.graph.node.insert(0, cast)
        checkpoint_hash = hashlib.sha256()
        with checkpoint.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                checkpoint_hash.update(chunk)
        metadata = {
            "checkpoint_sha256": checkpoint_hash.hexdigest(),
            "checkpoint_file": checkpoint.name,
            "model_size": size,
            "task": task,
            "weight_storage": quantization,
            "compute_dtype": "float32",
            "int4_group_size": str(group_size) if quantization == "int4" else "not_applicable",
            "int4_min_matrix_elements": str(INT4_MIN_MATRIX_ELEMENTS) if quantization == "int4" else "not_applicable",
            "int4_regression_head_fp32": str(task == "regression" and quantization == "int4").lower(),
        }
        for key, value in metadata.items():
            graph.metadata_props.add(key=key, value=value)
        onnx.save_model(
            graph,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            size_threshold=1024 * 1024 * 1024,
            location=f"{output.name}.data",
        )
    onnx.checker.check_model(str(output))
    print(f"Exported {quantization.upper()} {task} {size} model to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, choices=tuple(MODEL_KWARGS))
    parser.add_argument(
        "--task", required=True, choices=("classification", "regression")
    )
    parser.add_argument(
        "--quantization", choices=tuple(PRECISION_SUFFIX), default="fp32"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Output directory (default: models/)")
    parser.add_argument("--int4-group-size", type=int,
                        help="Input-channel group width (default: regression 16, classification 0; 0 uses per-output-channel scales)")
    args = parser.parse_args()
    if args.int4_group_size is not None and (args.int4_group_size < 0 or args.quantization != "int4"):
        parser.error("--int4-group-size requires INT4 and a nonnegative group width")
    output_dir = args.output if args.output is not None else Path("models")
    export(args.checkpoint, output_dir / model_filename(args.task, args.size, args.quantization),
           args.size, args.task, args.quantization, args.int4_group_size)


if __name__ == "__main__":
    main()
