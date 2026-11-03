# Kumo-Tabular-Rust

## Export ONNX

In GitHub Actions, run **Export Kumo Tabular ONNX** manually and choose
`small`, `medium`, or `large` and either `classification` or `regression`.
The workflow downloads the matching checkpoint from
[NVIDIA Kumo-Tabular](https://huggingface.co/nvidia/Kumo-Tabular), converts it
with Python, stores the model weights as FP16, and provides the `.onnx` file as
a run artifact. The graph casts the weights back to FP32 for computation, so
the float32 inputs and outputs remain unchanged. FP16 quantization is lossy.

The exported graph is the core Kumo Tabular network, **not** the public
`KumoTabular` inference wrapper. It takes `x` (6 rows by 3 numerical features),
`y` (4 labeled context rows), and `categorical_mask` (3 boolean values).
The last 2 rows are queries. For classification, `y` is int64 with class codes
0 through 9 and the output contains 10 raw scores per query; the public API's
ECOC class-probability decoding is not included. For regression, `y` is float32
and the output has 999 quantiles per query. This export uses fixed example
shapes; other row or feature counts require exporting with new example inputs.
For models exceeding ONNX's 2 GB limit, the artifact also contains external
weight data that must remain beside the `.onnx` file.

Local export with an existing small classifier checkpoint:

```sh
python -m pip install 'torch>=2.7' structured-data-models onnx
python scripts/export_onnx.py --size small --task classification \
  --checkpoint smail/classifier.pt --output output/kumo-tabular-small-classification-fp16.onnx
```
