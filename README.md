# Kumo-Tabular-Rust

Rust library for tabular preprocessing and ONNX Runtime inference. It provides a
typed table model, a serializable preprocessor, named-tensor ONNX sessions, a
model registry, adapters for the Kumo Tabular exports in this repository, and a
C ABI for loading a dynamic ONNX model and running Kumo inference.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | Library and the `evaluate_full` benchmark CLI |
| `models/` | Dynamic ONNX exports: original FP32, FP16, FP8 and INT4 weights (currently medium) |
| `pt/` | Official FP32 PyTorch checkpoints (`classifier.pt`, `regressor.pt`) |
| `dataset/` | Classification and regression Parquet datasets |
| `scripts/` | Export, comparison and report tooling |
| `tests/` | Unit and ignored model integration tests |
| `output/` | Generated reports and traces (ignored by Git) |

Model and checkpoint files are ignored by Git; place them in `models/` and `pt/`
before running the model-backed tests.

## C ABI

The crate builds both an Rust library and a C-compatible dynamic library. The public API is in `include/onnx_ffi.h`; create a model handle, call `tabular_onnx_predict` with row-major `float32` features, context labels and a categorical mask, then release the returned values and handle with the matching API functions. Classification returns 10 raw logits and regression returns 999 quantiles. This is a tensor-level Kumo interface: callers remain responsible for preparing feature columns and encoded labels.

The **Build dynamic libraries** GitHub Actions workflow (manual dispatch, `arch` input: `all`, `x86_64`, or `aarch64`) builds bundles for Linux x86_64, Windows x86_64, and macOS arm64. Each bundle contains `include/onnx_ffi.h`, the platform library at `lib/libkumo-onnx-{arch}` with the platform extension (`.so`, `.dll`, or `.dylib`), the pinned ONNX Runtime shared library as a single unversioned file (`libonnxruntime.so`, `libonnxruntime.dylib`, or `onnxruntime.dll`) in the same `lib/` directory (with `ONNXRUNTIME_LICENSE`), `BUILD_INFO.txt`, and `RUNTIME_DEPS.txt`. The library loads the bundled runtime from its own directory on first use, so hosts need no separate ONNX Runtime install; set `ORT_DYLIB_PATH` to override it with another build. Local builds still require `ORT_DYLIB_PATH` because they are not packaged with a runtime.

### High-level table ABI

For hosts that should not build tensors, `include/onnx_ffi.h` also exposes a
column-oriented table API. A `TabularOnnxTable` holds named columns; numeric
columns are `double` arrays (`NaN` = missing) and categorical columns are UTF-8
string arrays (`NULL` = missing). The target column is named separately by
`target_column`. A table containing only the target column is accepted as a
univariate series; Rust adds a deterministic row-index feature.

`tabular_onnx_predict_table` treats the first `context_rows` rows as labeled
context and the rest as queries. Rust runs the Kumo recipe, target
standardization, class encoding and model inference. Classification returns
`[rows, class_count]` probabilities plus class labels; regression returns
`[rows, 3]` median/q05/q95 plus the full sorted 999-quantile matrix in original
target units.

`tabular_onnx_forecast` takes chronological history rows and a horizon. The last
`context_rows` history rows are the context. An optional future table supplies
feature values for each horizon step; when it is absent the last history row's
feature values are repeated. The future target column is ignored. This API is a
horizon wrapper around the same table pipeline, not an automatic time-series
feature generator: supply lag/rolling/calendar columns as ordinary features, and
supply future rows when those features change over the horizon. This function
returns the same probability/quantile structures as `predict_table`. Forecast is
per series: call once per series, or include a series identifier as a
categorical feature with enough context rows for each series.

Classification class sets are inferred from the context target column, so every
class to predict must appear in the context. For more than ten classes Rust
builds an ECOC codebook, runs the required core tasks and aggregates their
log-probabilities; callers see only the final class probabilities. All
prediction memory is released with `tabular_onnx_prediction_free`, and a table
handle remains single-threaded like the tensor API.

A minimal C call looks like this; the arrays are owned by the caller and only
borrowed for the duration of the call:

```c
TabularOnnxColumn columns[2] = {
    { .name = "age", .kind = TABULAR_ONNX_COLUMN_NUMERIC,
      .numeric_values = age, .category_values = NULL },
    { .name = "target", .kind = TABULAR_ONNX_COLUMN_CATEGORICAL,
      .numeric_values = NULL, .category_values = target },
};
TabularOnnxTable table = { columns, 2, rows };
TabularOnnxPrediction prediction = { 0 };
if (tabular_onnx_predict_table(handle, TABULAR_ONNX_CLASSIFICATION, &table,
                               16, "target", &prediction) != TABULAR_ONNX_OK) {
    /* tabular_onnx_last_error() has the reason. */
}
/* prediction.values is row-major probabilities; prediction.class_labels names
   the columns. */
tabular_onnx_prediction_free(&prediction);
```

## Rust API

`Table` is a typed, nullable, column-oriented input model. It converts to and
from Polars `DataFrame`; callers can also construct it from rows. A serializable
`PreprocessConfig` defines numeric missing-value handling, optional
standardization using training statistics, and categorical code dictionaries.

`Predictor` is the extension point for a model family; implementations own
preprocessing and tensor contracts and return task-tagged `Prediction`s.
`ModelRegistry` supports runtime registration by model name. `OnnxSession` is a
low-level ORT wrapper for named float32 tensors.

## Kumo tensor contract

Dynamic exports accept:

- `x: float32[context_rows + query_rows, features]`
- `y`: `int64[context_rows]` for classification or `float32[context_rows]` for
  regression
- `categorical_mask: bool[features]`
- output `predictions: float32[query_rows, 10]` (class logits) or
  `float32[query_rows, 999]` (regression quantiles)

The Rust evaluator uses the public core contract directly with 16, 32 or 64
context rows and up to 2 query rows per inference. Classification scores use the
difference between the No and Yes raw logits after the fitted class
permutation; regression uses the median of the 999 quantiles. The fixed-shape
`[6, 3]` `KumoBatch::from_features` adapter remains for legacy exports.

The public Python `KumoTabular` predictor additionally fits its preprocessing
recipe on every context table and aggregates estimators. The ONNX exports
contain the core network but not that full recipe, so the Rust path is an
all-feature reimplementation of the default single-estimator route rather than
bit-for-bit Python parity. Relevant upstream code:
[model](https://github.com/NVIDIA/structured-data-models/blob/main/sdm/models/kumo/tabular/model.py),
[recipe](https://github.com/NVIDIA/structured-data-models/blob/main/sdm/models/kumo/tabular/recipe.py),
[ICLModel](https://github.com/NVIDIA/structured-data-models/blob/main/sdm/models/base/icl.py).

## Export ONNX

Run the **Export Kumo Tabular ONNX** GitHub Action and choose the model size,
task and weight precision (`fp32`, `fp16`, `fp8`, `int4`, or `all`). It downloads the
matching NVIDIA FP32 checkpoint and uploads the dynamic `.onnx` artifact. Keep a
generated `.onnx.data` file beside its model. The canonical filename is
`tabular-{cls|reg}-{small|medium|large}-{f32|f16|f8|i4}.onnx`;
`f32` retains the original checkpoint weights, while the smaller variants
compress weight storage and decode to FP32 for computation. Each export records
its checkpoint SHA-256, task, size and weight-storage precision in ONNX metadata
for newly generated exports; the six renamed legacy compressed exports predate
this metadata. INT4 also records its task-dependent group width (classification:
per output channel; regression: 16 input channels).

Local export:

```sh
python -m pip install 'torch>=2.7' structured-data-models onnx
python scripts/export_onnx.py --size medium --task classification \
  --checkpoint pt/classifier.pt --quantization fp32 --output models
# writes models/tabular-cls-medium-f32.onnx
```

INT4 stores matrix weights in signed 4-bit groups and converts them to FP32 for
computation. The regression head and small vectors/scalars stay in FP32; see
`scripts/export_onnx.py --help` for group-size options.

## Testing

```sh
cargo test
```

Unit tests do not need ONNX Runtime. The ignored integration test loads
`models/tabular-cls-medium-f16.onnx`:

```sh
ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib \
  cargo test --test pipeline -- --ignored
```

All Rust evaluations read models from `models/`; official comparisons load the
FP32 checkpoints from `pt/`.

## Evaluation

Set `ORT_DYLIB_PATH` and run the medium models on both datasets:

```sh
cargo run --release --bin evaluate_full -- \
  --seed 42 --task both --precision all --context-rows 32 --trace
```

`--context-rows` accepts 16, 32 or 64. Each report includes held-out metrics and
per-batch inference latency (mean, median, p95, total). `--trace` records the
context/query row indices for the official replay.

Compare the original FP32 checkpoints on the same batches:

```sh
.venv-official/bin/python scripts/compare_official_medium.py \
  --task classification --precision int4 --context-rows 32
```

Run the complete matrix (2 datasets × 3 precisions × 16/32/64 context rows) and
regenerate the report:

```sh
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib
scripts/run_test_report.sh
```

Raw JSON and traces are written to `output/`; the same results are kept in
[TEST_REPORT.md](TEST_REPORT.md).

## Test report

<!-- TEST_REPORT_START -->

*Snapshot:* 2026-11-03 · *seed:* 42 · *models:* medium (FP16 / FP8 / INT4)

### Environment

- Machine: Apple M1 Pro, 10 cores, 16 GB
- System: macOS-14.6.1-arm64-arm-64bit
- Rust inference: ONNX Runtime 1.23.0 (CPU), release build
- Official comparison: PyTorch 2.7.1 (CPU), structured-data-models 0.2.1

### Models

All ONNX files are medium dynamic exports under `models/`; every official comparison loads the corresponding FP32 checkpoint from `pt/`.

| Task | FP16 | FP8 | INT4 |
| --- | ---: | ---: | ---: |
| Classification | 134.4 MB | 69.5 MB | 38.3 MB |
| Regression | 136.3 MB | 70.5 MB | 58.0 MB |

### Datasets

- **Classification:** `dataset/classification/bad_customer_detection.parquet` (13 features, 1723 rows; 344 held-out test rows).
- **Regression:** `dataset/regression/airfoil_self_noise.parquet` (5 numerical features, 1503 rows; 301 held-out test rows).

### Protocol

Rust reads `models/tabular-{cls,reg}-medium-{f16,f8,i4}.onnx`; the official API loads `pt/classifier.pt` or `pt/regressor.pt`. Each context size is a separate run with a seed-42 stratified 80/20 train/test split and 20% of the training rows held out for validation. Context rows are sampled class-balanced for classification and uniformly for regression; every batch contains the configured context (16, 32 or 64 rows) plus up to 2 query rows. The classifier threshold is fitted on validation balanced accuracy; regression uses the median of 999 quantiles. Metrics use held-out test rows only.

Latency is wall time per batch on CPU: ONNX is the `session.run` call, PT is one official API call (which includes its recipe). The first call is included, so mean and p95 include warm-up. ONNX and PT latency are indicative, not a framework-only microbenchmark. PT latency repeats within a context size because one FP32 replay serves all three precision comparisons.

### Classification dataset

The test split is imbalanced: the all-No baseline is 0.8866. Accuracy is reported for completeness; balanced accuracy and ROC AUC carry more signal there.

Δ and ρ compare the official FP32 model with each ONNX export on the same context/query row indices (Δ = PT − ONNX; positive ΔAUC favours PT).

| Context | Precision | ONNX AUC | ONNX bal. acc | ONNX acc | PT AUC | PT bal. acc | ΔAUC | Δbal. acc | ρ(pred) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | FP16 | 0.5912 | 0.5710 | 0.4971 | 0.6460 | 0.5749 | +0.0548 | +0.0039 | 0.871 |
| 16 | FP8 | 0.6077 | 0.5839 | 0.5000 | 0.6460 | 0.5749 | +0.0383 | -0.0089 | 0.868 |
| 16 | INT4 | 0.5707 | 0.5532 | 0.3663 | 0.6460 | 0.5749 | +0.0753 | +0.0218 | 0.814 |
| 32 | FP16 | 0.6740 | 0.6227 | 0.6483 | 0.6557 | 0.6175 | -0.0183 | -0.0052 | 0.902 |
| 32 | FP8 | 0.6715 | 0.6016 | 0.5116 | 0.6557 | 0.6175 | -0.0158 | +0.0159 | 0.897 |
| 32 | INT4 | 0.6588 | 0.6354 | 0.5320 | 0.6557 | 0.6175 | -0.0031 | -0.0179 | 0.860 |
| 64 | FP16 | 0.6256 | 0.6089 | 0.6831 | 0.6559 | 0.6351 | +0.0303 | +0.0262 | 0.942 |
| 64 | FP8 | 0.6365 | 0.6364 | 0.7122 | 0.6559 | 0.6351 | +0.0194 | -0.0013 | 0.925 |
| 64 | INT4 | 0.6356 | 0.6116 | 0.6483 | 0.6559 | 0.6351 | +0.0203 | +0.0235 | 0.850 |

#### Classification inference latency (344 test rows, 2 queries per batch)

| Context | Precision | ONNX mean / p95 (ms) | PT mean / p95 (ms) | ONNX queries/s | PT queries/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | FP16 | 104.22 / 114.73 | 214.93 / 265.55 | 19.19 | 9.31 |
| 16 | FP8 | 113.38 / 132.23 | 214.93 / 265.55 | 17.64 | 9.31 |
| 16 | INT4 | 107.17 / 123.33 | 214.93 / 265.55 | 18.66 | 9.31 |
| 32 | FP16 | 127.50 / 143.94 | 231.33 / 248.50 | 15.69 | 8.65 |
| 32 | FP8 | 130.13 / 145.37 | 231.33 / 248.50 | 15.37 | 8.65 |
| 32 | INT4 | 145.58 / 213.17 | 231.33 / 248.50 | 13.74 | 8.65 |
| 64 | FP16 | 168.56 / 184.85 | 283.58 / 304.22 | 11.87 | 7.05 |
| 64 | FP8 | 172.50 / 193.90 | 283.58 / 304.22 | 11.59 | 7.05 |
| 64 | INT4 | 170.70 / 194.74 | 283.58 / 304.22 | 11.72 | 7.05 |

### Regression dataset

ΔMAE and ΔR2 are PT − ONNX; positive ΔMAE means PT has larger error and positive ΔR2 favours PT.

| Context | Precision | ONNX MAE | ONNX RMSE | ONNX R2 | PT MAE | PT RMSE | PT R2 | ΔMAE | ΔR2 | ρ(pred) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | FP16 | 4.6066 | 6.0511 | 0.2436 | 4.5496 | 5.9830 | 0.2605 | -0.0570 | +0.0169 | 0.984 |
| 16 | FP8 | 4.5768 | 5.9714 | 0.2634 | 4.5496 | 5.9830 | 0.2605 | -0.0272 | -0.0029 | 0.978 |
| 16 | INT4 | 4.3753 | 5.7973 | 0.3057 | 4.5496 | 5.9830 | 0.2605 | +0.1744 | -0.0452 | 0.969 |
| 32 | FP16 | 2.9421 | 3.9460 | 0.6783 | 2.9675 | 4.0062 | 0.6684 | +0.0254 | -0.0099 | 0.986 |
| 32 | FP8 | 3.0223 | 4.0427 | 0.6624 | 2.9675 | 4.0062 | 0.6684 | -0.0548 | +0.0061 | 0.984 |
| 32 | INT4 | 3.0328 | 4.0186 | 0.6664 | 2.9675 | 4.0062 | 0.6684 | -0.0653 | +0.0021 | 0.977 |
| 64 | FP16 | 2.2806 | 3.0384 | 0.8093 | 2.3425 | 3.1214 | 0.7987 | +0.0618 | -0.0106 | 0.995 |
| 64 | FP8 | 2.3212 | 3.0733 | 0.8049 | 2.3425 | 3.1214 | 0.7987 | +0.0213 | -0.0062 | 0.994 |
| 64 | INT4 | 2.3171 | 3.0435 | 0.8086 | 2.3425 | 3.1214 | 0.7987 | +0.0254 | -0.0099 | 0.992 |

#### Regression inference latency (301 test rows, 2 queries per batch)

| Context | Precision | ONNX mean / p95 (ms) | PT mean / p95 (ms) | ONNX queries/s | PT queries/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | FP16 | 57.35 / 67.25 | 152.01 / 188.52 | 34.88 | 13.16 |
| 16 | FP8 | 57.86 / 68.92 | 152.01 / 188.52 | 34.57 | 13.16 |
| 16 | INT4 | 59.74 / 76.68 | 152.01 / 188.52 | 33.48 | 13.16 |
| 32 | FP16 | 71.05 / 80.61 | 172.19 / 210.14 | 28.15 | 11.62 |
| 32 | FP8 | 79.42 / 123.99 | 172.19 / 210.14 | 25.18 | 11.62 |
| 32 | INT4 | 72.27 / 83.45 | 172.19 / 210.14 | 27.67 | 11.62 |
| 64 | FP16 | 96.76 / 111.73 | 188.13 / 226.43 | 20.67 | 10.63 |
| 64 | FP8 | 96.98 / 112.15 | 188.13 / 226.43 | 20.62 | 10.63 |
| 64 | INT4 | 95.98 / 104.14 | 188.13 / 226.43 | 20.84 | 10.63 |

### Findings
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

### Caveats

This is an end-to-end comparison, not bit-for-bit parity: Rust and the official Python predictor use different recipe RNGs and arithmetic paths. The same row indices are replayed through both pipelines; prediction correlation (ρ) and per-metric deltas quantify the remaining difference. The 16/32/64 runs reuse the same held-out rows and are a context-size study, not independent validation of a selected configuration.

### Reproduce

```sh
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib
scripts/run_test_report.sh
```

The script resumes missing reports under `output/` and regenerates this file; individual commands are in the Evaluation section above.

<!-- TEST_REPORT_END -->