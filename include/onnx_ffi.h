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
 * Load an ONNX file. model_path must be UTF-8. The caller must make a
 * compatible ONNX Runtime shared library available to the platform loader
 * before calling this function. On failure, out_handle remains NULL.
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

/* Thread-local error text, valid until the next API call on this thread. */
const char *tabular_onnx_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* TABULAR_ONNX_FFI_H */
