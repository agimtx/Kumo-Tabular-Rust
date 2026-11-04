#ifndef TABULAR_ONNX_FFI_H
#define TABULAR_ONNX_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TabularOnnxHandle TabularOnnxHandle;

enum {
    TABULAR_ONNX_OK = 0,
    TABULAR_ONNX_ERROR = 1,
    TABULAR_ONNX_CLASSIFICATION = 0,
    TABULAR_ONNX_REGRESSION = 1
};

/*
 * Load an ONNX file. model_path must be UTF-8. ONNX Runtime is loaded from the
 * lib/ directory shipped with this library unless ORT_DYLIB_PATH selects
 * another installation. On failure, out_handle remains NULL and
 * tabular_onnx_last_error() explains why.
 */
int32_t tabular_onnx_create(const char *model_path, TabularOnnxHandle **out_handle);

/*
 * Run dynamic Kumo inference with row-major float32 x [total_rows, features],
 * context labels y [context_rows], and one categorical-mask byte per feature.
 * Classification labels are integer codes 0..9 represented as float32.
 * Returns row-major float32 logits [total_rows-context_rows, 10] or quantiles
 * [total_rows-context_rows, 999]. Release output with tabular_onnx_free_values.
 * All pointer arguments must reference valid memory of the documented length.
 */
int32_t tabular_onnx_predict(
    TabularOnnxHandle *handle,
    int32_t task,
    const float *x,
    size_t total_rows,
    size_t features,
    const float *y,
    size_t context_rows,
    const uint8_t *categorical_mask,
    float **out_values,
    size_t *out_rows,
    size_t *out_columns);

/* Free output allocated by tabular_onnx_predict. */
void tabular_onnx_free_values(float *values, size_t rows, size_t columns);

/* Destroy a model handle; NULL is accepted. */
void tabular_onnx_destroy(TabularOnnxHandle *handle);

/*
 * High-level table API
 * --------------------
 *
 * The functions below accept named columns of raw values and let Rust perform
 * the Kumo recipe, target encoding and inference. They are the recommended
 * entry point for hosts that do not already own a preprocessed tensor.
 *
 * A table is column-oriented. Numeric columns point at row_count doubles; NaN
 * means missing. Categorical columns point at row_count UTF-8 C strings; NULL
 * means missing. Every column name must be unique and non-empty. A table with
 * only the target column is accepted as a univariate series; Rust supplies a
 * deterministic row-index feature in that case.
 *
 * For predict_table, the first context_rows rows are labeled context and the
 * remaining rows are queries. target_column names the label column; it may be
 * numeric or categorical for classification and must be numeric for regression.
 * The classification class set is inferred from the context rows, so every class
 * to predict must appear in the context. Targets with more than ten classes are
 * expanded with ECOC inside the library.
 *
 * For forecast, history rows are chronological and include the target column.
 * The last context_rows rows are used as context. The optional future table
 * supplies feature values for the horizon rows; when it is NULL, the last
 * history row's feature values are repeated. The future target column, if
 * present, is ignored. The handle has the same single-threaded contract as the
 * tensor API: do not call prediction functions concurrently on one handle.
 */
enum {
    TABULAR_ONNX_COLUMN_NUMERIC = 0,
    TABULAR_ONNX_COLUMN_CATEGORICAL = 1
};

typedef struct {
    const char *name;
    int32_t kind; /* TABULAR_ONNX_COLUMN_* */
    const double *numeric_values;
    const char *const *category_values;
} TabularOnnxColumn;

typedef struct {
    const TabularOnnxColumn *columns;
    size_t column_count;
    size_t row_count;
} TabularOnnxTable;

/*
 * A prediction owns all pointers below. values is row-major:
 *   classification: [rows, class_count] probabilities in class_labels order;
 *   regression:     [rows, 3] median, q05 and q95 in original target units.
 * quantiles is regression-only row-major [rows, quantile_count] (999), sorted
 * and in original target units. class_labels is classification-only and holds
 * class_count UTF-8 strings. Call tabular_onnx_prediction_free after use; the
 * struct is zeroed and a second free call is safe.
 */
typedef struct {
    int32_t task;
    size_t rows;
    size_t columns;
    float *values;
    size_t class_count;
    char **class_labels;
    size_t quantile_count;
    float *quantiles;
} TabularOnnxPrediction;

int32_t tabular_onnx_predict_table(
    TabularOnnxHandle *handle,
    int32_t task,
    const TabularOnnxTable *table,
    size_t context_rows,
    const char *target_column,
    TabularOnnxPrediction *out_prediction);

int32_t tabular_onnx_forecast(
    TabularOnnxHandle *handle,
    int32_t task,
    const TabularOnnxTable *history,
    size_t context_rows,
    const char *target_column,
    size_t horizon,
    const TabularOnnxTable *future,
    TabularOnnxPrediction *out_prediction);

void tabular_onnx_prediction_free(TabularOnnxPrediction *prediction);

/* Thread-local error text, valid until the next API call on this thread. */
const char *tabular_onnx_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* TABULAR_ONNX_FFI_H */
