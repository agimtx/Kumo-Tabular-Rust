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

In GitHub Actions, run **Export Kumo Tabular ONNX** manually and choose `small`, `medium`, or `large` and either `classification` or `regression`. The workflow downloads the matching checkpoint from NVIDIA Kumo-Tabular, converts it with Python, stores model weights as FP16, and provides the `.onnx` file as a run artifact. The graph casts weights back to FP32 for computation. FP16 quantization is lossy.

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

## Full-feature Rust evaluation (new dynamic exports required)

The separate `evaluate_full` binary consumes **new** `*-dynamic-fp16.onnx`
models and never reuses the old fixed-shape reports. Run the updated
**Export Kumo Tabular ONNX** GitHub Action four times: `small`/`medium` times
`classification`/`regression`. Each run downloads NVIDIA's `v1.0.1`
checkpoint, exports variable-length context/query rows and feature columns,
and validates two distinct shapes through ONNX Runtime before uploading the
artifact. Put each artifact's ONNX file at its corresponding path:

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
rows. The Rust recipe fits numeric statistics and category vocabularies on
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
bit-for-bit Python predictor parity**. Its new results will be written to
`output/small-full-evaluation-rust.json` and
`output/medium-full-evaluation-rust.json` only after the new models arrive.
Until then, the existing JSON reports above refer exclusively to the old
fixed `[6,3]` graphs; no new full-feature accuracy has been measured.
