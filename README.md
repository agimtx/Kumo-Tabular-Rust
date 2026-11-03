# Kumo-Tabular-Rust

Rust library for extensible tabular preprocessing and ONNX Runtime inference. It currently includes a generic table/preprocessor pipeline, a named-tensor ONNX adapter, a model registry, and a validator for the fixed tensor contract of the Kumo ONNX exports in this repository. FFI is intentionally not part of this crate yet.

## Rust API

`Table` is a typed, nullable, column-oriented input model. It converts to and from Polars `DataFrame`; callers can also construct it from rows. Define a serializable `PreprocessConfig` alongside each trained model. Numeric features apply a missing-value replacement and optional standardization using persisted training statistics. Categorical features use a persisted category-to-code dictionary with explicit unknown and missing codes. Input feature order is the order in `PreprocessConfig::features`.

`Predictor` is the extension point for a model family. Implementations own their preprocessing and tensor contracts, and return task-tagged `Prediction`s. `ModelRegistry` enables runtime registration by model name. `OnnxSession` is a low-level ORT session wrapper for named float32 tensor inputs; model adapters assemble model-specific inputs above it.

## Kumo tensor contract

The current exported graph is the core network, not the public `KumoTabular` wrapper:

- Inputs are `x: float32[6, 3]`, `y: int64[4]` for classification or `float32[4]` for regression, and `categorical_mask: bool[3]`.
- Rows 0..4 provide context labels; rows 4..6 are the two query predictions.
- Classification output is 10 raw ECOC scores, not decoded class probabilities.
- Regression output contains 999 quantiles.
- The export has fixed example dimensions. The adapter validates these constraints; a differently shaped export requires a separate model contract/configuration.

The export script and workflow are described below. For external-data ONNX models, keep the generated `.onnx.data` file beside its model.

## Export ONNX

In GitHub Actions, run **Export Kumo Tabular ONNX** manually and choose `small`, `medium`, or `large` and either `classification` or `regression`. The workflow downloads the matching checkpoint from NVIDIA Kumo-Tabular, converts it with Python, stores model weights as FP16, and provides the `.onnx` file as a run artifact. The graph casts weights back to FP32 for computation. FP16 quantization is lossy.

Local export with an existing small classifier checkpoint:

```sh
python -m pip install 'torch>=2.7' structured-data-models onnx
python scripts/export_onnx.py --size small --task classification \
  --checkpoint smail/classifier.pt --output output/kumo-tabular-small-classification-fp16.onnx
```

## Validation

Run `cargo test` to test table conversion, preprocessing, and model-contract handling.

## FP16 and FP32 model dataset evaluation

With the two local Parquet files and FP16/FP32 ONNX exports in place, make the
DuckDB CLI and a compatible ONNX Runtime dynamic library available. The Rust
CLI uses `OnnxSession` for inference; Python is not involved. For example, on
this machine:

```sh
CARGO_HOME="$PWD/.cargo-home" \
ORT_DYLIB_PATH=/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.23.0.dylib \
cargo run --offline --release --bin evaluate_quantized -- --seed 42
```

Use `--precision fp32` to evaluate only the unquantized models, or
`--precision both` to compare FP16 and FP32 using identical splits and context
samples. The default is `fp16`. Output defaults to
`output/quantized-evaluation-rust.json`,
`output/unquantized-evaluation-rust.json`, or
`output/precision-comparison-rust.json`, respectively; override with
`--output PATH`. To run the FP32 end-to-end Rust test explicitly with local
models and `ORT_DYLIB_PATH` configured:

```sh
cargo test --bin evaluate_quantized fp32_models_evaluate_both_datasets -- --ignored
```

Set `ORT_DYLIB_PATH` to the ONNX Runtime shared library on your machine; the
local `CARGO_HOME` and `--offline` flags are optional when Cargo can access its
normal cache and registry. The CLI uses a fixed, stratified 80/20 train/test
split for classification and a seeded 80/20
split for regression. Only training labels are supplied as context; each model
call uses four context rows and two held-out query rows. Since the exported graph
only accepts three columns, the selected features are recorded in the JSON.
Numeric features are standardized using training-set statistics. The binary
classifier uses the first two raw logits (class 0 = No, class 1 = Yes), rather
than treating the 10 output scores as probabilities. Regression predictions
use the median of 999 output quantiles, reversing the context-label scaling.

With seed 42, Rust classification (344 test rows) gave FP16 accuracy 0.485 and
ROC AUC 0.543 versus FP32 accuracy 0.483 and ROC AUC 0.543. Predicting No for
every row has accuracy 0.887. Rust regression (301 test rows) gave FP16 MAE
6.2496 and RMSE 7.9871 versus FP32 MAE 6.2499 and RMSE 7.9873. The training-mean
baseline MAE is 5.648 and RMSE is 6.978. These are limited-context,
three-feature evaluations of the *core* exports, not estimates for the public
KumoTabular pipeline with its full
preprocessing and larger context windows.
