//! Stable C ABI for loading a dynamic Kumo ONNX model and running predictions.

use crate::kumo::KumoBatch;
use crate::{OnnxSession, Task};
use ndarray::Array2;
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::path::PathBuf;
use std::ptr;
use std::slice;

pub const TABULAR_ONNX_OK: i32 = 0;
pub const TABULAR_ONNX_ERROR: i32 = 1;

pub struct TabularOnnxHandle {
    session: OnnxSession,
}

impl TabularOnnxHandle {
    /// Mutable access for the higher-level table ABI in `ffi_table`.
    pub(crate) fn session_mut(&mut self) -> &mut OnnxSession {
        &mut self.session
    }
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

pub(crate) fn set_error(message: impl ToString) {
    let message = message.to_string().replace('\0', "\\0");
    LAST_ERROR.with(|slot| *slot.borrow_mut() = CString::new(message).unwrap());
}

pub(crate) fn ffi_result<T>(operation: impl FnOnce() -> crate::Result<T>) -> Result<T, i32> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => {
            set_error(error);
            Err(TABULAR_ONNX_ERROR)
        }
        Err(payload) => {
            set_error(panic_message(payload));
            Err(TABULAR_ONNX_ERROR)
        }
    }
}

/// Turn a caught panic payload into an error message instead of letting it
/// unwind across the C ABI, which aborts the host process.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    let detail = payload
        .downcast_ref::<&str>()
        .copied()
        .map(str::to_owned)
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_owned());
    format!("internal panic: {detail}")
}

/// Return a thread-local error string, valid until the next API call on this thread.
#[no_mangle]
pub extern "C" fn tabular_onnx_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Load an ONNX model. `model_path` must be a UTF-8, NUL-terminated path.
/// ONNX Runtime is loaded from the `lib/` directory shipped next to this
/// library, unless `ORT_DYLIB_PATH` points at another installation.
///
/// # Safety
///
/// `model_path` must be a readable NUL-terminated string. `out_handle` must be
/// valid, writable, suitably aligned storage for one handle pointer.
#[no_mangle]
pub unsafe extern "C" fn tabular_onnx_create(
    model_path: *const c_char,
    out_handle: *mut *mut TabularOnnxHandle,
) -> i32 {
    if model_path.is_null() || out_handle.is_null() {
        set_error("model_path and out_handle must not be null");
        return TABULAR_ONNX_ERROR;
    }
    *out_handle = ptr::null_mut();
    let result = ffi_result(|| {
        let path = CStr::from_ptr(model_path)
            .to_str()
            .map_err(|error| crate::Error::Model(format!("model path is not UTF-8: {error}")))?;
        let path = PathBuf::from(path);
        let session = OnnxSession::load(path)?;
        Ok(Box::into_raw(Box::new(TabularOnnxHandle { session })))
    });
    match result {
        Ok(handle) => {
            *out_handle = handle;
            TABULAR_ONNX_OK
        }
        Err(code) => code,
    }
}

/// Run dynamic Kumo inference on row-major float32 features.
/// task is 0 for classification, 1 for regression. y contains context labels
/// only; categorical_mask uses one byte per feature (zero or nonzero).
/// On success, out_values owns a newly allocated row-major float32 array; release
/// it with tabular_onnx_free_values. Outputs are logits (10 columns) or quantiles
/// (999 columns).
///
/// # Safety
///
/// `handle` must come from `tabular_onnx_create`. `x`, `y`,
/// `categorical_mask` and the three output pointers must reference enough
/// readable/writable memory for the documented shapes; `x` is
/// `total_rows * features` floats, `y` is `context_rows` floats and the mask is
/// `features` bytes. The handle must not be used concurrently.
#[no_mangle]
pub unsafe extern "C" fn tabular_onnx_predict(
    handle: *mut TabularOnnxHandle,
    task: i32,
    x: *const f32,
    total_rows: usize,
    features: usize,
    y: *const f32,
    context_rows: usize,
    categorical_mask: *const u8,
    out_values: *mut *mut f32,
    out_rows: *mut usize,
    out_columns: *mut usize,
) -> i32 {
    if handle.is_null()
        || x.is_null()
        || y.is_null()
        || categorical_mask.is_null()
        || out_values.is_null()
        || out_rows.is_null()
        || out_columns.is_null()
    {
        set_error("prediction arguments must not be null");
        return TABULAR_ONNX_ERROR;
    }
    *out_values = ptr::null_mut();
    *out_rows = 0;
    *out_columns = 0;
    let result = ffi_result(|| {
        let task = match task {
            0 => Task::Classification,
            1 => Task::Regression,
            _ => {
                return Err(crate::Error::Model(
                    "task must be 0 (classification) or 1 (regression)".into(),
                ))
            }
        };
        let value_count = total_rows
            .checked_mul(features)
            .ok_or_else(|| crate::Error::Model("feature tensor size overflow".into()))?;
        let x_values = slice::from_raw_parts(x, value_count);
        let y_values = slice::from_raw_parts(y, context_rows);
        let mask_values = slice::from_raw_parts(categorical_mask, features)
            .iter()
            .map(|&value| value != 0)
            .collect::<Vec<_>>();
        let features = Array2::from_shape_vec((total_rows, features), x_values.to_vec())
            .map_err(|error| crate::Error::Model(error.to_string()))?;
        let batch = KumoBatch::from_dynamic_features(&features, y_values, task, &mask_values)?;
        let output = (*handle).session.run_kumo(batch)?;
        let rows = output.nrows();
        let columns = output.ncols();
        let mut values = output.into_raw_vec_and_offset().0.into_boxed_slice();
        let data = values.as_mut_ptr();
        std::mem::forget(values);
        Ok((data, rows, columns))
    });
    match result {
        Ok((values, rows, columns)) => {
            *out_values = values;
            *out_rows = rows;
            *out_columns = columns;
            TABULAR_ONNX_OK
        }
        Err(code) => code,
    }
}

/// Release a prediction array returned by tabular_onnx_predict.
///
/// # Safety
///
/// `values` must be null or a pointer returned by `tabular_onnx_predict`, with
/// the same `rows` and `columns`. It must not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn tabular_onnx_free_values(values: *mut f32, rows: usize, columns: usize) {
    if !values.is_null() {
        if let Some(length) = rows.checked_mul(columns) {
            drop(Box::from_raw(ptr::slice_from_raw_parts_mut(values, length)));
        }
    }
}

/// Unload a handle returned by tabular_onnx_create. Null is accepted.
///
/// # Safety
///
/// `handle` must be null or a handle from `tabular_onnx_create` that is no
/// longer used by any thread. It must not be destroyed twice.
#[no_mangle]
pub unsafe extern "C" fn tabular_onnx_destroy(handle: *mut TabularOnnxHandle) {
    if !handle.is_null() {
        let _ =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(Box::from_raw(handle))));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_text_is_nul_safe_and_thread_local() {
        set_error("bad\0input");
        let message = unsafe { CStr::from_ptr(tabular_onnx_last_error()) };
        assert_eq!(message.to_str().unwrap(), "bad\\0input");
    }

    #[test]
    fn null_handle_destroy_is_safe() {
        unsafe { tabular_onnx_destroy(ptr::null_mut()) };
    }

    #[test]
    fn rejects_null_create_arguments() {
        let result = unsafe { tabular_onnx_create(ptr::null(), ptr::null_mut()) };
        assert_eq!(result, TABULAR_ONNX_ERROR);
        assert!(!tabular_onnx_last_error().is_null());
    }
}
