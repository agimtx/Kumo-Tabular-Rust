# Kumo-Tabular-Rust

Rust library for extensible tabular preprocessing and ONNX Runtime inference. It currently includes a generic table/preprocessor pipeline, a named-tensor ONNX adapter, a model registry, and a validator for the fixed tensor contract of the Kumo ONNX exports in this repository. FFI is intentionally not part of this crate yet.

## Rust API

`Table` is a typed, nullable, column-oriented input model. It converts to and from Polars `DataFrame`; callers can also construct it from rows. Define a serializable `PreprocessConfig` alongside each trained model. Numeric features apply a missing-value replacement and optional standardization using persisted training statistics. Categorical features use a persisted category-to-code dictionary with explicit unknown and missing codes. Input feature order is the order in `PreprocessConfig::features`.

`Predictor` is the extension point for a model family. Implementations own their preprocessing and tensor contracts, and return task-tagged `Prediction`s. `ModelRegistry` enables runtime registration by model name. `OnnxSession` is a low-level ORT session wrapper for named float32 tensor inputs; model adapters assemble model-specific inputs above it. `KumoPredictor::load` requires the four context feature rows alongside their four labels; the model cannot infer from context labels paired with zero-filled features.

## Kumo tensor contract

The current exported graph is the core network, not the public `KumoTabular` wrapper:

- Inputs are `x: float32[6, 3]`, `y: int64[4]` for classification or `float32[4]` for regression, and `categorical_mask: bool[3]`.
- Rows 0..4 provide context labels; rows 4..6 are the two query predictions.
- Classification output is 10 raw class logits (the ECOC wrapper only handles targets with more than 10 classes), not decoded class probabilities.
- Regression output contains 999 quantiles.
- The export has fixed example dimensions. The adapter validates these constraints; a differently shaped export requires a separate model contract/configuration.

### Difference from the public Python predictor

The Python `KumoTabular` public predictor fits a recipe on each supplied
context table and transforms queries using the fitted recipe. Its default
recipe handles numerical and categorical columns separately, drops constant
columns, standardizes and clips numeric values, applies alternative nonlinear
transforms and random sign flips, shuffles/selects columns, transforms targets,
then aggregates multiple estimators. Classification applies softmax to the
class logits (the ECOC wrapper only remaps codes above ten classes), while
regression sorts quantiles and inverts the target transform. The ONNX export
calls the internal `_KumoTabular` directly and contains **none** of that
recipe. Its fixed `[6,3]` input and four context labels also prevent feeding
the full table or a realistic context set. `KumoBatch` also currently rejects
NaNs even though the Python model handles missing cells. The Rust evaluator is therefore a
limited core-network experiment, not a parity test with Python `predict`.
Simply applying a per-batch four-row standardization was checked against a
training-only validation set and was not selected for any of the four tasks;
it cannot substitute for the full recipe. To test public-predictor accuracy,
export a graph supporting variable context/query sizes and feature counts and
implement the full recipe, or evaluate the Python public predictor separately.
Source: [Kumo model](https://github.com/NVIDIA/structured-data-models/blob/main/sdm/models/kumo/tabular/model.py),
[default recipe](https://github.com/NVIDIA/structured-data-models/blob/main/sdm/models/kumo/tabular/recipe.py),
[ICLModel](https://github.com/NVIDIA/structured-data-models/blob/main/sdm/models/base/icl.py).

The export script and workflow are described below. For external-data ONNX models, keep the generated `.onnx.data` file beside its model.

## Export ONNX

In GitHub Actions, run **Export Kumo Tabular ONNX** manually and choose the model size, task, and weight precision (`fp16`, `fp8`, `int4`, or `all` to export all three). The workflow downloads the matching checkpoint from NVIDIA Kumo-Tabular and provides the `.onnx` file as a run artifact. Weights are stored in the selected low-precision format and converted to FP32 for computation; all three formats are lossy. INT4 uses symmetric per-tensor scaling and stores signed 4-bit weights packed into bytes, which the graph unpacks with standard ONNX operators. The exporter also repairs a legacy tracing bug in ICL attention key-length selection (`key.size(-3)`), validates all expected query-scaling paths, then stores the graph. This correction is required for correct context-dependent logits.

Local export with an existing small classifier checkpoint:

```sh
python -m pip install 'torch>=2.7' structured-data-models onnx
python scripts/export_onnx.py --size small --task classification \
  --checkpoint smail/classifier.pt --output small/kumo-tabular-small-classification-fp16.onnx
```

## Validation

Run `cargo test` to test table conversion, preprocessing, and model-contract handling.

## Small and medium FP16 model dataset evaluation

With the two local Parquet files and FP16 ONNX exports in `small/` and `medium/`, make the
DuckDB CLI and a compatible ONNX Runtime dynamic library available. The Rust
CLI uses `OnnxSession` for inference; Python is not involved. For example, on
this machine:

```sh
CARGO_HOME="$PWD/.cargo-home" \
ORT_DYLIB_PATH=/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.23.0.dylib \
cargo run --offline --release --bin evaluate_quantized -- --seed 42 --size both
```

The default `--size both` runs small and medium independently and writes
`output/small-evaluation-rust.json` and `output/medium-evaluation-rust.json`.
Use `--size small` or `--size medium` to run only one size. Only FP16 models
are evaluated; unquantized models are not used. Both sizes use the same seed,
train/test split, validation split and context-row selection rules. To run the
end-to-end Rust test explicitly with local models and `ORT_DYLIB_PATH` configured:

```sh
cargo test --bin evaluate_quantized quantized_models_evaluate_both_sizes_and_datasets -- --ignored
```

Set `ORT_DYLIB_PATH` to the ONNX Runtime shared library on your machine; the
local `CARGO_HOME` and `--offline` flags are optional when Cargo can access its
normal cache and registry. The CLI uses a fixed stratified 80/20 train/test
split for classification and a seeded 80/20 split for regression. It reserves
20% of the training rows for validation. Using only validation labels, it
selects random or nearest-feature context by ROC AUC (classification) or MAE
(regression), and sets the binary classification threshold by balanced accuracy.
The held-out test rows never influence these choices; only fit-set labels are
supplied as model context. Each call uses four context rows and two query rows.
Since the exported graphs accept only three features, the selected columns are
recorded in the JSON. Features are standardized using fit-set statistics.
The classifier uses the difference between the first two raw logits (class 0 =
No, class 1 = Yes), not the 10 output scores as probabilities. Regression
predictions use the median of 999 quantiles, reversing context-label scaling.

With seed 42, classification (344 test rows) gives small accuracy 0.727,
balanced accuracy 0.589, Yes-class F1 0.254, ROC AUC 0.606; medium accuracy
0.509, balanced accuracy 0.533, Yes-class F1 0.207, ROC AUC 0.567. The
all-No baseline accuracy is 0.887, so raw accuracy alone is misleading for
this imbalanced dataset (about 11% Yes). Regression (301 test rows) gives
small MAE 5.395, RMSE 7.080; medium MAE 5.776, RMSE 7.274. The training-mean
baseline has MAE 5.648 and RMSE 6.978. The earlier 2 Yes/2 No context with a
zero-logit threshold ignored class prevalence; a raw accuracy-tuned threshold
instead predicted nearly all No. Both issues are addressed with the
validation-only balanced threshold. Nearest-feature context improves small
regression MAE, but medium regression and both classification models remain
weak. Fixed three-feature, four-context-row core exports cannot recover the
omitted features or emulate the public KumoTabular pipeline's larger context
and preprocessing; the remaining gap is not evidence of a quantization bug.

## Full-feature Rust evaluation (dynamic FP16 exports)

The separate `evaluate_full` binary consumes `*-dynamic-fp16.onnx` models
and never reuses the old fixed-shape reports. The updated **Export Kumo Tabular
ONNX** GitHub Action takes `small`/`medium` and `classification`/`regression`
as inputs; each run downloads NVIDIA's `v1.0.1` checkpoint and checks two
input shapes through ONNX Runtime. All four resulting models have been
validated locally with 32 context/2 query rows and 16 context/1 query row,
different feature counts, and changed query inputs. Place them at:

```text
small/kumo-tabular-small-classification-dynamic-fp16.onnx
small/kumo-tabular-small-regression-dynamic-fp16.onnx
medium/kumo-tabular-medium-classification-dynamic-fp16.onnx
medium/kumo-tabular-medium-regression-dynamic-fp16.onnx
```

Then run Rust inference (DuckDB CLI and `ORT_DYLIB_PATH` are required):

```sh
CARGO_HOME="$PWD/.cargo-home" \
ORT_DYLIB_PATH=/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.23.0.dylib \
cargo run --locked --offline --release --bin evaluate_full -- --seed 42 --size both
```

This reads every non-target column: 13 features (including categorical
columns) in customer classification and 5 numerical features in airfoil
regression. Each inference uses 32 fitted context examples and up to 2 query
rows by default; `--context-rows 16` or `--context-rows 64` changes the
context size and writes a separate report. The Rust recipe fits numeric
statistics and category vocabularies on
context rows, drops constants, standardizes/clips, handles high-cardinality
counts, applies the default one-estimator identity option and fitted sigma
clipping, sign/category shifts, column permutation, and inverse target
transforms. Classification's logit difference is used for ROC AUC and a
validation-only balanced-accuracy threshold; regression uses sorted quantile
median. The public Kumo API defaults to one estimator for two-dimensional
inputs; multi-estimator output aggregation and the remaining nonlinear recipe
options are **not implemented** here. Rust and PyTorch random generators also
do not produce identical permutations from the same seed. This is an
all-feature Rust implementation of the default single-estimator route, **not
bit-for-bit Python predictor parity**. The full-feature reports for seed 42
are in `output/{small,medium}-full-evaluation-rust.json` (32 context rows)
and `output/{small,medium}-full-context-{16,64}-evaluation-rust.json`.
The earlier `*-evaluation-rust.json` reports still describe the old `[6,3]`
graphs, not these dynamic models.

| Context rows | Model | Classification AUC | Balanced accuracy | Accuracy | Regression MAE | RMSE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | small | 0.596 | 0.514 | 0.177 | 4.618 | 6.045 |
| 32 | small | 0.621 | 0.569 | 0.770 | 3.270 | 4.444 |
| 64 | small | 0.664 | 0.639 | 0.698 | 2.330 | 3.084 |
| 16 | medium | 0.634 | 0.605 | 0.497 | 4.855 | 6.056 |
| 32 | medium | 0.675 | 0.606 | 0.677 | 4.527 | 5.646 |
| 64 | medium | 0.659 | 0.632 | 0.744 | 4.221 | 5.279 |

All runs use the same held-out rows (344 classification, 301 regression),
with thresholds fitted only on validation rows. The classification all-No
accuracy baseline is 0.887; regression fit-mean MAE is 5.657. More context
improves regression MAE in these runs, especially for small, but classification
remains below the majority-class accuracy baseline and threshold selection is
unstable at 16 rows. The 16/32/64 comparison reuses the same test split: it is
an exploratory granularity study, **not** independent validation of a
context-size choice. For a final selected setting, evaluate on a fresh holdout.

## Quantization comparison against original PyTorch checkpoints

The original medium FP32 weights in `pt/classifier.pt` and `pt/regressor.pt`
are loaded into the official `structured-data-models==0.2.1` API (PyTorch
2.7.1, CPU). Each ONNX quantization level (FP16, FP8, INT4) is evaluated by
Rust on the same dataset split (`seed=42`, 32 context rows), then the official
`.pt` model is replayed on the exact same context/query row batches. Classification
thresholds are selected independently on each pipeline's validation predictions
using balanced accuracy. All metrics below use held-out test rows (344
classification, 301 regression). Differences are **official FP32 minus ONNX**;
positive MAE/RMSE means the official model has larger error, while positive R2,
AUC, balanced accuracy, accuracy, or F1 means it scores higher.

| Task | ONNX precision | ONNX MAE / AUC | ONNX RMSE / balanced accuracy | ONNX R2 / accuracy | Official MAE / AUC | Official RMSE / balanced accuracy | Official R2 / accuracy | Difference (official - ONNX), MAE / AUC | Difference, RMSE / balanced accuracy | Difference, R2 / accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification | FP16 | - / 0.6740 | - / 0.6227 | - / 0.6483 | - / 0.6557 | - / 0.6175 | - / 0.6192 | - / -0.0183 | - / -0.0052 | - / -0.0291 |
| Classification | FP8 | - / 0.6715 | - / 0.6016 | - / 0.5116 | - / 0.6557 | - / 0.6175 | - / 0.6192 | - / -0.0158 | - / +0.0159 | - / +0.1076 |
| Classification | INT4 | - / 0.4274 | - / 0.4567 | - / 0.2151 | - / 0.6557 | - / 0.6175 | - / 0.6192 | - / +0.2282 | - / +0.1608 | - / +0.4041 |
| Regression | FP16 | 2.9421 / - | 3.9460 / - | 0.6783 / - | 2.9675 / - | 4.0062 / - | 0.6684 / - | +0.0254 / - | +0.0602 / - | -0.0099 / - |
| Regression | FP8 | 3.0223 / - | 4.0427 / - | 0.6624 / - | 2.9675 / - | 4.0062 / - | 0.6684 / - | -0.0548 / - | -0.0365 / - | +0.0061 / - |
| Regression | INT4 | 5.6742 / - | 6.9951 / - | -0.0108 / - | 2.9675 / - | 4.0062 / - | 0.6684 / - | -2.7067 / - | -2.9889 / - | +0.6793 / - |

The `.pt` metrics repeat for all three rows because this is the same FP32
checkpoint and its input batches are fixed; the differences isolate the result
of each ONNX model/pipeline against that baseline. FP16 and FP8 regression are
close to the original model on these metrics. INT4 regression degrades sharply
(MAE 5.674 versus 2.967 and R2 -0.011 versus 0.668). For classification,
FP16/FP8 ranking AUC is close to the baseline, while INT4 falls to 0.427. All
classification accuracies are below the all-No baseline of 0.8866, so accuracy
alone is not informative for this imbalanced dataset.

This is an end-to-end comparison, **not a pure weight-quantization error test**:
Rust and Python use different recipe RNGs and arithmetic paths, although they
receive the same context/query row indices. Prediction-level Pearson
correlation and MAE are stored in the comparison JSON files. The lower
classification INT4 correlation and regression INT4 correlation indicate a
substantial prediction change beyond small floating-point rounding. Rust
reports include held-out predictions and are task/precision-specific.

Run all Rust reports (DuckDB CLI, ONNX Runtime dynamic library, and the six
medium ONNX models are required), then run the original `.pt` comparison for
each task and precision:

```sh
CARGO_HOME="$PWD/.cargo-home" ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib \
  cargo run --locked --offline --release --bin evaluate_full -- \
  --seed 42 --size medium --trace
for task in classification regression; do
  for precision in fp16 fp8 int4; do
    OMP_NUM_THREADS=4 .venv-official/bin/python scripts/compare_official_medium.py \
      --task "$task" --precision "$precision" --seed 42 --context-rows 32
  done
done
```

Rust reports are `output/medium-{classification,regression}-{fp16,fp8,int4}-full-evaluation-rust.json`;
original checkpoint comparison reports are
`output/medium-{classification,regression}-{fp16,fp8,int4}-official-comparison.json`.
The separate `scripts/check_prepared_core_parity.py` remains the tool for
checking raw core inference on identical prepared tensors.
identical tensors against PyTorch; at 32 contexts, mean absolute output error is
0.00165 for classification logits and 0.00020 for regression quantiles, with
correlations above 0.999999. This isolates the former large gap to a legacy
ONNX tracing bug, now repaired, rather than the FP16 conversion.
