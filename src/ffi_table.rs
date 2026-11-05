//! High-level C ABI for raw tabular input.
//!
//! The low-level `ffi` module exposes tensors. This module accepts a table of
//! named numeric/categorical columns, performs the Kumo recipe preprocessing,
//! target encoding, ECOC expansion for wide classification targets, and returns
//! simple point/probability results. Callers never build a feature matrix.

use crate::ecoc::{aggregate, Codebook, CORE_CLASSES};
use crate::ffi::{ffi_result, set_error, TabularOnnxHandle, TABULAR_ONNX_ERROR, TABULAR_ONNX_OK};
use crate::kumo::KumoBatch;
use crate::kumo_recipe::{prepare_with_feature_types, FeatureType, Prepared, Task as RecipeTask};
use crate::{Error, OnnxSession, Result, Task};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{Number, Value};
use std::collections::{BTreeSet, HashMap};
use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::slice;

/// Numeric column: `numeric_values` points at `row_count` `double`s; NaN means
/// missing.
pub const TABULAR_ONNX_COLUMN_NUMERIC: i32 = 0;
/// Categorical column: `category_values` points at `row_count` UTF-8 C strings;
/// NULL means missing.
pub const TABULAR_ONNX_COLUMN_CATEGORICAL: i32 = 1;

#[repr(C)]
pub struct TabularOnnxColumn {
    pub name: *const c_char,
    pub kind: i32,
    pub numeric_values: *const f64,
    pub category_values: *const *const c_char,
}

#[repr(C)]
pub struct TabularOnnxTable {
    pub columns: *const TabularOnnxColumn,
    pub column_count: usize,
    pub row_count: usize,
}

#[repr(C)]
pub struct TabularOnnxPrediction {
    pub task: i32,
    pub rows: usize,
    pub columns: usize,
    pub values: *mut f32,
    pub class_count: usize,
    pub class_labels: *mut *mut c_char,
    pub quantile_count: usize,
    pub quantiles: *mut f32,
}

#[derive(Clone)]
struct ParsedColumn {
    name: String,
    kind: i32,
    values: Vec<Value>,
}

struct ParsedTable {
    columns: Vec<ParsedColumn>,
    row_count: usize,
    target_index: usize,
    target_name: String,
    feature_names: Vec<String>,
}

struct ClassMap {
    labels: Vec<String>,
    codes: HashMap<String, usize>,
}

impl ClassMap {
    fn from_context(
        rows: &[Vec<Value>],
        target_index: usize,
        context_rows: usize,
        numeric: bool,
    ) -> Result<Self> {
        let mut labels = BTreeSet::new();
        for row in rows.iter().take(context_rows) {
            labels.insert(label_key(&row[target_index])?);
        }
        if labels.is_empty() {
            return Err(Error::Preprocessing(
                "classification context has no target labels".into(),
            ));
        }
        let mut labels = labels.into_iter().collect::<Vec<_>>();
        if numeric {
            labels.sort_by(|left, right| {
                left.parse::<f64>()
                    .unwrap_or(f64::NAN)
                    .total_cmp(&right.parse::<f64>().unwrap_or(f64::NAN))
            });
        }
        let codes = labels
            .iter()
            .enumerate()
            .map(|(index, label)| (label.clone(), index))
            .collect::<HashMap<_, _>>();
        Ok(Self { labels, codes })
    }

    fn len(&self) -> usize {
        self.labels.len()
    }

    fn code_of(&self, value: &Value) -> Result<usize> {
        let key = label_key(value)?;
        self.codes.get(&key).copied().ok_or_else(|| {
            Error::Preprocessing(format!(
                "classification target {key:?} does not appear in the context rows"
            ))
        })
    }
}

struct Request {
    feature_names: Vec<String>,
    feature_types: Vec<FeatureType>,
    feature_rows: Vec<Vec<Value>>,
    target_values: Vec<f64>,
    class_map: Option<ClassMap>,
    context_rows: usize,
}

struct PredictionData {
    task: Task,
    rows: usize,
    columns: usize,
    values: Vec<f32>,
    class_labels: Vec<String>,
    quantiles: Vec<f32>,
}

fn label_key(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Number(value) => value
            .as_f64()
            .filter(|value| value.is_finite())
            .map(|value| {
                if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
                    (value as i64).to_string()
                } else {
                    value.to_string()
                }
            })
            .ok_or_else(|| Error::Preprocessing("classification target is not finite".into())),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Null => Err(Error::Preprocessing(
            "classification target is missing in a context row".into(),
        )),
        _ => Err(Error::Preprocessing(
            "classification target must be a scalar".into(),
        )),
    }
}

fn numeric_target(value: &Value) -> Result<f64> {
    let number = match value {
        Value::Number(value) => value
            .as_f64()
            .ok_or_else(|| Error::Preprocessing("regression target is not finite".into()))?,
        Value::Bool(value) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        Value::Null => {
            return Err(Error::Preprocessing(
                "regression target is missing in a context row".into(),
            ))
        }
        _ => {
            return Err(Error::Preprocessing(
                "regression target must be numeric".into(),
            ))
        }
    };
    if !number.is_finite() {
        return Err(Error::Preprocessing(
            "regression target is not finite".into(),
        ));
    }
    Ok(number)
}

unsafe fn cstr_to_string(pointer: *const c_char, what: &str) -> Result<String> {
    if pointer.is_null() {
        return Err(Error::InvalidTable(format!("{what} must not be null")));
    }
    CStr::from_ptr(pointer)
        .to_str()
        .map(str::to_owned)
        .map_err(|error| Error::InvalidTable(format!("{what} is not UTF-8: {error}")))
}

unsafe fn parse_columns(table: &TabularOnnxTable) -> Result<Vec<ParsedColumn>> {
    if table.columns.is_null() {
        return Err(Error::InvalidTable("table columns must not be null".into()));
    }
    if table.column_count == 0 {
        return Err(Error::InvalidTable(
            "table must have at least one column".into(),
        ));
    }
    if table.row_count == 0 {
        return Err(Error::InvalidTable(
            "table must have at least one row".into(),
        ));
    }

    let raw_columns = slice::from_raw_parts(table.columns, table.column_count);
    let mut names = std::collections::HashSet::new();
    let mut columns = Vec::with_capacity(table.column_count);
    for raw in raw_columns {
        let name = cstr_to_string(raw.name, "column name")?;
        if name.is_empty() {
            return Err(Error::InvalidTable("column names must not be empty".into()));
        }
        if !names.insert(name.clone()) {
            return Err(Error::InvalidTable(format!("duplicate column name {name}")));
        }
        let values = match raw.kind {
            TABULAR_ONNX_COLUMN_NUMERIC => {
                if raw.numeric_values.is_null() {
                    return Err(Error::InvalidTable(format!(
                        "numeric column {name} has no values"
                    )));
                }
                let raw_values = slice::from_raw_parts(raw.numeric_values, table.row_count);
                raw_values
                    .iter()
                    .map(|value| {
                        if value.is_nan() {
                            Ok(Value::Null)
                        } else if value.is_finite() {
                            Number::from_f64(*value).map(Value::Number).ok_or_else(|| {
                                Error::InvalidTable(format!(
                                    "numeric column {name} contains an invalid value"
                                ))
                            })
                        } else {
                            Ok(Value::Object(serde_json::Map::from_iter([(
                                "__tabular_onnx_ffi_infinity__".into(),
                                Value::String(if value.is_sign_positive() {
                                    "positive".into()
                                } else {
                                    "negative".into()
                                }),
                            )])))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            TABULAR_ONNX_COLUMN_CATEGORICAL => {
                if raw.category_values.is_null() {
                    return Err(Error::InvalidTable(format!(
                        "categorical column {name} has no values"
                    )));
                }
                let raw_values = slice::from_raw_parts(raw.category_values, table.row_count);
                raw_values
                    .iter()
                    .map(|value| {
                        if value.is_null() {
                            Ok(Value::Null)
                        } else {
                            cstr_to_string(*value, "category value").map(Value::String)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            other => {
                return Err(Error::InvalidTable(format!(
                    "column {name} has unsupported kind {other}"
                )))
            }
        };
        columns.push(ParsedColumn {
            name,
            kind: raw.kind,
            values,
        });
    }
    Ok(columns)
}

unsafe fn parse_table(table: &TabularOnnxTable, target_name: &str) -> Result<ParsedTable> {
    let columns = parse_columns(table)?;
    let target_index = columns
        .iter()
        .position(|column| column.name == target_name)
        .ok_or_else(|| Error::InvalidTable(format!("missing target column {target_name}")))?;
    let feature_names = columns
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != target_index)
        .map(|(_, column)| column.name.clone())
        .collect::<Vec<_>>();
    Ok(ParsedTable {
        columns,
        row_count: table.row_count,
        target_index,
        target_name: target_name.to_owned(),
        feature_names,
    })
}

fn rows_from_columns(columns: &[ParsedColumn], row_count: usize) -> Vec<Vec<Value>> {
    (0..row_count)
        .map(|row| {
            columns
                .iter()
                .map(|column| column.values[row].clone())
                .collect()
        })
        .collect()
}

fn build_request(
    task: Task,
    table: &ParsedTable,
    rows: Vec<Vec<Value>>,
    context_rows: usize,
) -> Result<Request> {
    if rows.len() != table.row_count {
        return Err(Error::InvalidTable(
            "row count changed while reading table".into(),
        ));
    }
    if context_rows == 0 || context_rows >= rows.len() {
        return Err(Error::Preprocessing(
            "Kumo needs at least one context row and at least one query row".into(),
        ));
    }
    let feature_indices = (0..table.columns.len())
        .filter(|index| *index != table.target_index)
        .collect::<Vec<_>>();
    let (feature_names, feature_types, feature_rows) = if feature_indices.is_empty() {
        // A target-only table is still a valid univariate series. Give the model
        // a deterministic time index instead of rejecting the call.
        let rows = rows
            .iter()
            .enumerate()
            .map(|(index, _)| vec![Value::Number(Number::from(index as u64))])
            .collect::<Vec<_>>();
        (
            vec!["__row_index".to_owned()],
            vec![FeatureType::Numerical],
            rows,
        )
    } else {
        let feature_types = feature_indices
            .iter()
            .map(|&index| {
                if table.columns[index].kind == TABULAR_ONNX_COLUMN_CATEGORICAL {
                    FeatureType::Categorical
                } else {
                    FeatureType::Numerical
                }
            })
            .collect::<Vec<_>>();
        let rows = rows
            .iter()
            .map(|row| {
                feature_indices
                    .iter()
                    .map(|index| row[*index].clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        (table.feature_names.clone(), feature_types, rows)
    };

    let target_is_numeric = table.columns[table.target_index].kind == TABULAR_ONNX_COLUMN_NUMERIC;
    let class_map = match task {
        Task::Classification => Some(ClassMap::from_context(
            &rows,
            table.target_index,
            context_rows,
            target_is_numeric,
        )?),
        Task::Regression => None,
    };

    let mut target_values = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.iter().enumerate() {
        let value = &row[table.target_index];
        match task {
            Task::Regression => {
                if row_index < context_rows {
                    target_values.push(numeric_target(value)?);
                } else {
                    target_values.push(f64::NAN);
                }
            }
            Task::Classification => {
                if row_index < context_rows {
                    target_values.push(class_map.as_ref().unwrap().code_of(value)? as f64);
                } else {
                    target_values.push(0.0);
                }
            }
        }
    }

    Ok(Request {
        feature_names,
        feature_types,
        feature_rows,
        target_values,
        class_map,
        context_rows,
    })
}

fn run_batch(
    session: &mut OnnxSession,
    prepared: &Prepared,
    labels: &[f32],
    task: Task,
) -> Result<Array2<f32>> {
    let batch =
        KumoBatch::from_dynamic_features(&prepared.x, labels, task, &prepared.categorical_mask)?;
    session.run_kumo(batch)
}

fn softmax_row(row: &[f32]) -> Result<Vec<f32>> {
    if row.is_empty() || row.iter().any(|value| !value.is_finite()) {
        return Err(Error::Model(
            "classification logits must be finite and non-empty".into(),
        ));
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let values = row
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let total = values.iter().sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::Model(
            "classification logits cannot be normalized".into(),
        ));
    }
    Ok(values.into_iter().map(|value| value / total).collect())
}

fn classification_prediction(
    session: &mut OnnxSession,
    prepared: &Prepared,
    class_map: &ClassMap,
    context_codes: &[usize],
    query_rows: usize,
) -> Result<PredictionData> {
    let classes = class_map.len();
    if classes > CORE_CLASSES {
        return ecoc_prediction(session, prepared, class_map, context_codes, query_rows);
    }
    let output = run_batch(session, prepared, &prepared.labels, Task::Classification)?;
    if output.nrows() != query_rows || output.ncols() != CORE_CLASSES {
        return Err(Error::Model(format!(
            "expected classification output [{query_rows}, {CORE_CLASSES}], got {:?}",
            output.shape()
        )));
    }
    if prepared.class_permutation.len() != classes {
        return Err(Error::Model(
            "classification class permutation does not match the class count".into(),
        ));
    }
    let mut permutation = vec![false; classes];
    for &value in &prepared.class_permutation {
        if !value.is_finite() || value.fract() != 0.0 || value < 0.0 || value >= classes as f64 {
            return Err(Error::Model(
                "classification class permutation is invalid".into(),
            ));
        }
        let index = value as usize;
        if std::mem::replace(&mut permutation[index], true) {
            return Err(Error::Model(
                "classification class permutation is not bijective".into(),
            ));
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::Model(
            "classification produced a non-finite logit".into(),
        ));
    }
    let mut values = vec![0.0f32; query_rows * classes];
    for row in 0..query_rows {
        let probabilities = softmax_row(&output.row(row).to_vec())?;
        for (encoded, probability) in probabilities.iter().enumerate() {
            let original = prepared.class_permutation[encoded] as usize;
            if original >= classes {
                return Err(Error::Model(
                    "classification class permutation is out of range".into(),
                ));
            }
            values[row * classes + original] = *probability;
        }
    }
    Ok(PredictionData {
        task: Task::Classification,
        rows: query_rows,
        columns: classes,
        values,
        class_labels: class_map.labels.clone(),
        quantiles: Vec::new(),
    })
}

fn ecoc_prediction(
    session: &mut OnnxSession,
    prepared: &Prepared,
    class_map: &ClassMap,
    context_codes: &[usize],
    query_rows: usize,
) -> Result<PredictionData> {
    let classes = class_map.len();
    if context_codes.iter().any(|&class| class >= classes) {
        return Err(Error::Model(
            "ECOC context class code is out of range".into(),
        ));
    }
    let codebook = Codebook::build(classes, 0);
    let mut task_logits = Vec::with_capacity(codebook.task_count());
    for task_index in 0..codebook.task_count() {
        let labels = context_codes
            .iter()
            .map(|&class| codebook.symbol(task_index, class) as f32)
            .collect::<Vec<_>>();
        let output = run_batch(session, prepared, &labels, Task::Classification)?;
        if output.nrows() != query_rows || output.ncols() != CORE_CLASSES {
            return Err(Error::Model(format!(
                "expected ECOC task output [{query_rows}, {CORE_CLASSES}], got {:?}",
                output.shape()
            )));
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(Error::Model(
                "ECOC classification produced a non-finite logit".into(),
            ));
        }
        task_logits.push(output.iter().copied().collect::<Vec<_>>());
    }
    let values = aggregate(&codebook, &task_logits, query_rows);
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Model(
            "ECOC aggregation produced a non-finite probability".into(),
        ));
    }
    Ok(PredictionData {
        task: Task::Classification,
        rows: query_rows,
        columns: classes,
        values,
        class_labels: class_map.labels.clone(),
        quantiles: Vec::new(),
    })
}

fn regression_prediction(
    session: &mut OnnxSession,
    prepared: &Prepared,
    query_rows: usize,
) -> Result<PredictionData> {
    let output = run_batch(session, prepared, &prepared.labels, Task::Regression)?;
    if output.nrows() != query_rows || output.ncols() != 999 {
        return Err(Error::Model(format!(
            "expected regression output [{query_rows}, 999], got {:?}",
            output.shape()
        )));
    }
    let mut values = Vec::with_capacity(query_rows * 3);
    let mut quantiles = Vec::with_capacity(query_rows * 999);
    for row in 0..query_rows {
        let mut row_quantiles = output
            .row(row)
            .iter()
            .map(|value| {
                (*value as f64 * prepared.target_scale * prepared.target_sign
                    + prepared.target_center) as f32
            })
            .collect::<Vec<_>>();
        if row_quantiles.iter().any(|value| !value.is_finite()) {
            return Err(Error::Model(
                "regression produced a non-finite quantile".into(),
            ));
        }
        row_quantiles.sort_by(f32::total_cmp);
        values.push(row_quantiles[499]);
        values.push(row_quantiles[49]);
        values.push(row_quantiles[949]);
        quantiles.extend_from_slice(&row_quantiles);
    }
    Ok(PredictionData {
        task: Task::Regression,
        rows: query_rows,
        columns: 3,
        values,
        class_labels: Vec::new(),
        quantiles,
    })
}

fn predict_rows(
    handle: &mut TabularOnnxHandle,
    task: Task,
    request: &Request,
) -> Result<PredictionData> {
    let rows = request.feature_rows.len();
    let context_rows = request.context_rows;
    let query_rows = rows
        .checked_sub(context_rows)
        .ok_or_else(|| Error::Preprocessing("context rows exceed total rows".into()))?;
    if query_rows == 0 {
        return Err(Error::Preprocessing(
            "at least one query row is required".into(),
        ));
    }
    let feature_names = request
        .feature_names
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let context_indices = (0..context_rows).collect::<Vec<_>>();
    let query_indices = (context_rows..rows).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(0);
    let recipe_task = match task {
        Task::Classification => RecipeTask::Classification,
        Task::Regression => RecipeTask::Regression,
    };
    let wide_classification = task == Task::Classification
        && request
            .class_map
            .as_ref()
            .is_some_and(|map| map.len() > CORE_CLASSES);
    let prepared = if wide_classification {
        prepare_with_feature_types(
            &request.feature_rows,
            &feature_names,
            &context_indices,
            &query_indices,
            &request.target_values,
            recipe_task,
            &mut rng,
            Some(&request.feature_types),
        )
    } else {
        prepare_with_feature_types(
            &request.feature_rows,
            &feature_names,
            &context_indices,
            &query_indices,
            &request.target_values,
            recipe_task,
            &mut rng,
            Some(&request.feature_types),
        )
    }
    .map_err(Error::Preprocessing)?;

    match task {
        Task::Classification => {
            let context_codes = prepared
                .labels
                .iter()
                .map(|value| {
                    if !value.is_finite() || *value < 0.0 || value.fract() != 0.0 {
                        return Err(Error::Model(
                            "prepared context class code is invalid".into(),
                        ));
                    }
                    Ok(*value as usize)
                })
                .collect::<Result<Vec<_>>>()?;
            classification_prediction(
                handle.session_mut(),
                &prepared,
                request.class_map.as_ref().unwrap(),
                &context_codes,
                query_rows,
            )
        }
        Task::Regression => regression_prediction(handle.session_mut(), &prepared, query_rows),
    }
}

fn parse_task(task: i32) -> Result<Task> {
    match task {
        0 => Ok(Task::Classification),
        1 => Ok(Task::Regression),
        _ => Err(Error::Model(
            "task must be 0 (classification) or 1 (regression)".into(),
        )),
    }
}

fn zero_prediction() -> TabularOnnxPrediction {
    TabularOnnxPrediction {
        task: 0,
        rows: 0,
        columns: 0,
        values: ptr::null_mut(),
        class_count: 0,
        class_labels: ptr::null_mut(),
        quantile_count: 0,
        quantiles: ptr::null_mut(),
    }
}

fn leak_f32(values: Vec<f32>) -> (*mut f32, usize) {
    if values.is_empty() {
        return (ptr::null_mut(), 0);
    }
    let mut values = values.into_boxed_slice();
    let pointer = values.as_mut_ptr();
    let length = values.len();
    std::mem::forget(values);
    (pointer, length)
}

fn write_prediction(prediction: PredictionData, out: *mut TabularOnnxPrediction) {
    let PredictionData {
        task,
        rows,
        columns,
        values: prediction_values,
        class_labels: prediction_labels,
        quantiles: prediction_quantiles,
    } = prediction;
    let (values, values_len) = leak_f32(prediction_values);
    let (quantiles, quantile_length) = leak_f32(prediction_quantiles);
    let quantile_count = if rows == 0 { 0 } else { quantile_length / rows };
    let labels = prediction_labels
        .into_iter()
        .map(|label| {
            CString::new(label)
                .expect("label contains a NUL byte")
                .into_raw()
        })
        .collect::<Vec<_>>();
    let class_count = labels.len();
    let class_labels = if labels.is_empty() {
        ptr::null_mut()
    } else {
        let mut labels = labels.into_boxed_slice();
        let pointer = labels.as_mut_ptr();
        std::mem::forget(labels);
        pointer
    };
    let task = match task {
        Task::Classification => 0,
        Task::Regression => 1,
    };
    unsafe {
        ptr::write(
            out,
            TabularOnnxPrediction {
                task,
                rows,
                columns,
                values,
                class_count,
                class_labels,
                quantile_count,
                quantiles,
            },
        );
        // `values_len` and `quantile_length` are not part of the public
        // struct; freeing uses rows * columns and rows * quantile_count.
        debug_assert_eq!(values_len, rows.saturating_mul(columns));
        debug_assert_eq!(quantile_length, rows.saturating_mul(quantile_count));
    }
}

/// Predict from raw table columns.
///
/// The first `context_rows` table rows are labeled context; the remaining rows
/// are queries. `target_column` names the column holding labels. Rust performs
/// feature preprocessing, target normalization, classification encoding and
/// (when needed) ECOC expansion. Classification results are probabilities with
/// one column per class; regression results are median/q05/q95 plus the full
/// 999 quantiles. Release `out_prediction` with
/// `tabular_onnx_prediction_free`.
///
/// # Safety
///
/// `handle` must come from `tabular_onnx_create`. `table` and its nested
/// pointers must describe readable arrays of at least `row_count` elements.
/// `target_column` must be a valid NUL-terminated string and `out_prediction`
/// must be valid, writable, suitably aligned storage for one prediction.
#[no_mangle]
pub unsafe extern "C" fn tabular_onnx_predict_table(
    handle: *mut TabularOnnxHandle,
    task: i32,
    table: *const TabularOnnxTable,
    context_rows: usize,
    target_column: *const c_char,
    out_prediction: *mut TabularOnnxPrediction,
) -> i32 {
    if out_prediction.is_null() {
        set_error("out_prediction must not be null");
        return TABULAR_ONNX_ERROR;
    }
    ptr::write(out_prediction, zero_prediction());
    if handle.is_null() || table.is_null() || target_column.is_null() {
        set_error("handle, table and target_column must not be null");
        return TABULAR_ONNX_ERROR;
    }
    let result = ffi_result(|| {
        let task = parse_task(task)?;
        let target = cstr_to_string(target_column, "target_column")?;
        let parsed = parse_table(&*table, &target)?;
        let rows = rows_from_columns(&parsed.columns, parsed.row_count);
        let request = build_request(task, &parsed, rows, context_rows)?;
        predict_rows(&mut *handle, task, &request)
    });
    match result {
        Ok(prediction) => {
            write_prediction(prediction, out_prediction);
            TABULAR_ONNX_OK
        }
        Err(code) => code,
    }
}

/// Forecast a horizon from a history table.
///
/// `history` rows are chronological and include the target column; the last
/// `context_rows` rows become the labeled context. `future` is optional: when
/// supplied it must contain `horizon` rows and the same feature columns, and its
/// target column (if present) is ignored. When `future` is NULL, Rust repeats
/// the last history row's feature values for every horizon step. Supply future
/// rows when the model uses known covariates or lag/rolling columns; this API
/// does not invent time-series features.
///
/// # Safety
///
/// `handle` must come from `tabular_onnx_create`. `history` and the optional
/// `future` table must describe readable arrays of the documented lengths;
/// `target_column` must be a valid NUL-terminated string and `out_prediction`
/// must be valid, writable, suitably aligned storage for one prediction.
#[no_mangle]
pub unsafe extern "C" fn tabular_onnx_forecast(
    handle: *mut TabularOnnxHandle,
    task: i32,
    history: *const TabularOnnxTable,
    context_rows: usize,
    target_column: *const c_char,
    horizon: usize,
    future: *const TabularOnnxTable,
    out_prediction: *mut TabularOnnxPrediction,
) -> i32 {
    if out_prediction.is_null() {
        set_error("out_prediction must not be null");
        return TABULAR_ONNX_ERROR;
    }
    ptr::write(out_prediction, zero_prediction());
    if handle.is_null() || history.is_null() || target_column.is_null() {
        set_error("handle, history and target_column must not be null");
        return TABULAR_ONNX_ERROR;
    }
    if horizon == 0 {
        set_error("horizon must be at least one");
        return TABULAR_ONNX_ERROR;
    }
    let result = ffi_result(|| {
        let task = parse_task(task)?;
        let target = cstr_to_string(target_column, "target_column")?;
        let parsed = parse_table(&*history, &target)?;
        if context_rows == 0 || context_rows > parsed.row_count {
            return Err(Error::Preprocessing(
                "context_rows must be between one and the history row count".into(),
            ));
        }
        let context_start = parsed.row_count - context_rows;
        let context_rows_values = rows_from_columns(&parsed.columns, parsed.row_count)
            .into_iter()
            .skip(context_start)
            .collect::<Vec<_>>();

        let future_rows = if future.is_null() {
            let last = context_rows_values
                .last()
                .cloned()
                .ok_or_else(|| Error::Preprocessing("history has no rows".into()))?;
            (0..horizon)
                .map(|_| {
                    let mut row = last.clone();
                    row[parsed.target_index] = Value::Null;
                    row
                })
                .collect::<Vec<_>>()
        } else {
            let future_table = &*future;
            if future_table.row_count != horizon {
                return Err(Error::InvalidTable(format!(
                    "future table must contain {horizon} rows, got {}",
                    future_table.row_count
                )));
            }
            let future_columns = parse_columns(future_table)?;
            let by_name = future_columns
                .iter()
                .map(|column| (column.name.as_str(), column))
                .collect::<HashMap<_, _>>();
            for column in &parsed.columns {
                if column.name == parsed.target_name {
                    continue;
                }
                let future_column = by_name.get(column.name.as_str()).ok_or_else(|| {
                    Error::InvalidTable(format!(
                        "future table is missing feature column {}",
                        column.name
                    ))
                })?;
                if future_column.kind != column.kind {
                    return Err(Error::InvalidTable(format!(
                        "future column {} has a different kind",
                        column.name
                    )));
                }
            }
            (0..horizon)
                .map(|row| {
                    let mut values = vec![Value::Null; parsed.columns.len()];
                    for (index, column) in parsed.columns.iter().enumerate() {
                        if column.name == parsed.target_name {
                            continue;
                        }
                        values[index] = by_name[column.name.as_str()].values[row].clone();
                    }
                    values
                })
                .collect::<Vec<_>>()
        };

        let mut rows = context_rows_values;
        rows.extend(future_rows);
        let combined = ParsedTable {
            columns: parsed.columns.clone(),
            row_count: rows.len(),
            target_index: parsed.target_index,
            target_name: parsed.target_name.clone(),
            feature_names: parsed.feature_names.clone(),
        };
        let request = build_request(task, &combined, rows, context_rows)?;
        predict_rows(&mut *handle, task, &request)
    });
    match result {
        Ok(prediction) => {
            write_prediction(prediction, out_prediction);
            TABULAR_ONNX_OK
        }
        Err(code) => code,
    }
}

/// Free every allocation owned by a prediction returned by
/// `tabular_onnx_predict_table` or `tabular_onnx_forecast`. The struct is
/// zeroed afterwards; calling this twice or on a zeroed struct is safe.
///
/// # Safety
///
/// `prediction` must be null or point to a prediction initialized by one of the
/// table functions. It must not be aliased or freed concurrently.
#[no_mangle]
pub unsafe extern "C" fn tabular_onnx_prediction_free(prediction: *mut TabularOnnxPrediction) {
    if prediction.is_null() {
        return;
    }
    let prediction = &mut *prediction;
    if !prediction.values.is_null() {
        if let Some(length) = prediction.rows.checked_mul(prediction.columns) {
            drop(Box::from_raw(ptr::slice_from_raw_parts_mut(
                prediction.values,
                length,
            )));
        }
    }
    if !prediction.quantiles.is_null() {
        if let Some(length) = prediction.rows.checked_mul(prediction.quantile_count) {
            drop(Box::from_raw(ptr::slice_from_raw_parts_mut(
                prediction.quantiles,
                length,
            )));
        }
    }
    if !prediction.class_labels.is_null() {
        for index in 0..prediction.class_count {
            let label = *prediction.class_labels.add(index);
            if !label.is_null() {
                drop(CString::from_raw(label));
            }
        }
        drop(Box::from_raw(ptr::slice_from_raw_parts_mut(
            prediction.class_labels,
            prediction.class_count,
        )));
    }
    ptr::write(prediction, zero_prediction());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric_column(name: &str, values: Vec<f64>) -> ParsedColumn {
        ParsedColumn {
            name: name.into(),
            kind: TABULAR_ONNX_COLUMN_NUMERIC,
            values: values
                .into_iter()
                .map(|value| {
                    if value.is_nan() {
                        Value::Null
                    } else {
                        Value::Number(Number::from_f64(value).unwrap())
                    }
                })
                .collect(),
        }
    }

    fn categorical_column(name: &str, values: Vec<Option<&str>>) -> ParsedColumn {
        ParsedColumn {
            name: name.into(),
            kind: TABULAR_ONNX_COLUMN_CATEGORICAL,
            values: values
                .into_iter()
                .map(|value| value.map_or(Value::Null, |value| Value::String(value.into())))
                .collect(),
        }
    }

    fn table(columns: Vec<ParsedColumn>, target_name: &str) -> ParsedTable {
        let target_index = columns
            .iter()
            .position(|column| column.name == target_name)
            .unwrap();
        ParsedTable {
            row_count: columns[0].values.len(),
            feature_names: columns
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != target_index)
                .map(|(_, column)| column.name.clone())
                .collect(),
            target_index,
            target_name: target_name.into(),
            columns,
        }
    }

    #[test]
    fn prediction_roundtrips_through_c_struct_and_free() {
        let prediction = PredictionData {
            task: Task::Classification,
            rows: 2,
            columns: 3,
            values: vec![0.1, 0.2, 0.7, 0.3, 0.3, 0.4],
            class_labels: vec!["a".into(), "b".into(), "c".into()],
            quantiles: Vec::new(),
        };
        let mut output = zero_prediction();
        write_prediction(prediction, &mut output);
        assert_eq!(output.task, 0);
        assert_eq!(output.rows, 2);
        assert_eq!(output.columns, 3);
        assert_eq!(output.class_count, 3);
        assert_eq!(output.quantile_count, 0);
        assert!(!output.values.is_null());
        assert!(!output.class_labels.is_null());
        unsafe {
            assert_eq!(*output.values.add(2), 0.7);
            assert_eq!(
                CStr::from_ptr(*output.class_labels.add(1))
                    .to_str()
                    .unwrap(),
                "b"
            );
            tabular_onnx_prediction_free(&mut output);
        }
        assert!(output.values.is_null());
        assert!(output.class_labels.is_null());
    }

    #[test]
    fn regression_prediction_reports_per_row_quantile_count() {
        let prediction = PredictionData {
            task: Task::Regression,
            rows: 2,
            columns: 3,
            values: vec![1.0; 6],
            class_labels: Vec::new(),
            quantiles: vec![0.0; 2 * 999],
        };
        let mut output = zero_prediction();
        write_prediction(prediction, &mut output);
        assert_eq!(output.task, 1);
        assert_eq!(output.quantile_count, 999);
        unsafe {
            assert!(!output.quantiles.is_null());
            tabular_onnx_prediction_free(&mut output);
        }
        assert!(output.quantiles.is_null());
    }

    #[test]
    fn null_output_is_rejected_and_zeroed() {
        let mut prediction = zero_prediction();
        let status = unsafe {
            tabular_onnx_predict_table(
                ptr::null_mut(),
                TABULAR_ONNX_COLUMN_NUMERIC,
                ptr::null(),
                0,
                ptr::null(),
                &mut prediction,
            )
        };
        assert_eq!(status, TABULAR_ONNX_ERROR);
        assert!(prediction.values.is_null());
        assert_eq!(prediction.rows, 0);
    }

    #[test]
    fn prediction_free_is_safe_on_null_and_zeroed_values() {
        unsafe {
            tabular_onnx_prediction_free(ptr::null_mut());
            let mut prediction = zero_prediction();
            tabular_onnx_prediction_free(&mut prediction);
            tabular_onnx_prediction_free(&mut prediction);
        }
    }

    #[test]
    fn forecast_rejects_zero_horizon_before_touching_history() {
        let mut prediction = zero_prediction();
        let status = unsafe {
            tabular_onnx_forecast(
                ptr::null_mut(),
                1,
                ptr::null(),
                1,
                ptr::null(),
                0,
                ptr::null(),
                &mut prediction,
            )
        };
        assert_eq!(status, TABULAR_ONNX_ERROR);
        assert!(prediction.values.is_null());
    }

    #[test]
    fn class_map_sorts_labels_and_maps_numeric_codes() {
        let rows = vec![
            vec![Value::Number(Number::from_f64(1.0).unwrap())],
            vec![Value::Number(Number::from_f64(0.0).unwrap())],
            vec![Value::Number(Number::from_f64(1.0).unwrap())],
        ];
        let map = ClassMap::from_context(&rows, 0, 3, true).unwrap();
        assert_eq!(map.labels, vec!["0", "1"]);
        assert_eq!(
            map.code_of(&Value::Number(Number::from_f64(1.0).unwrap()))
                .unwrap(),
            1
        );
        assert!(map
            .code_of(&Value::Number(Number::from_f64(3.0).unwrap()))
            .is_err());
    }

    #[test]
    fn class_map_orders_numeric_and_categorical_labels_by_declared_kind() {
        let rows = ["1", "10", "2"]
            .iter()
            .map(|value| vec![Value::String((*value).into())])
            .collect::<Vec<_>>();
        let numeric = ClassMap::from_context(&rows, 0, rows.len(), true).unwrap();
        let categorical = ClassMap::from_context(&rows, 0, rows.len(), false).unwrap();
        assert_eq!(numeric.labels, vec!["1", "2", "10"]);
        assert_eq!(categorical.labels, vec!["1", "10", "2"]);
    }

    #[test]
    fn logits_softmax_uses_all_core_classes_and_rejects_non_finite_values() {
        let logits = [0.0; CORE_CLASSES];
        let probabilities = softmax_row(&logits).unwrap();
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((probabilities[0] - 0.1).abs() < 1e-6);
        assert!(softmax_row(&[f32::NAN, 0.0]).is_err());
    }

    #[test]
    fn build_request_splits_features_and_context_labels() {
        let table = table(
            vec![
                numeric_column("x", vec![1.0, 2.0, 3.0, 4.0]),
                categorical_column("city", vec![Some("Oslo"), None, Some("Rome"), None]),
                numeric_column("target", vec![0.0, 1.0, 0.0, 1.0]),
            ],
            "target",
        );
        let rows = rows_from_columns(&table.columns, table.row_count);
        let request = build_request(Task::Classification, &table, rows, 2).unwrap();
        assert_eq!(request.feature_names, vec!["x", "city"]);
        assert_eq!(request.feature_rows.len(), 4);
        assert_eq!(
            request.feature_rows[1],
            vec![Value::Number(Number::from_f64(2.0).unwrap()), Value::Null]
        );
        assert_eq!(request.target_values, vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(request.class_map.unwrap().labels, vec!["0", "1"]);
    }

    #[test]
    fn target_only_table_gets_a_synthetic_time_index() {
        let table = table(
            vec![numeric_column("target", vec![3.0, 4.0, 5.0])],
            "target",
        );
        let rows = rows_from_columns(&table.columns, table.row_count);
        let request = build_request(Task::Regression, &table, rows, 2).unwrap();
        assert_eq!(request.feature_names, vec!["__row_index"]);
        assert_eq!(request.feature_rows.len(), 3);
        assert_eq!(
            request.feature_rows[2],
            vec![Value::Number(Number::from(2u64))]
        );
    }

    #[test]
    fn regression_targets_require_finite_context_values() {
        let table = table(
            vec![
                numeric_column("x", vec![1.0, 2.0]),
                numeric_column("target", vec![f64::NAN, 4.0]),
            ],
            "target",
        );
        let rows = rows_from_columns(&table.columns, table.row_count);
        assert!(build_request(Task::Regression, &table, rows, 1).is_err());
    }

    #[test]
    fn forecast_future_rows_are_matched_by_name_and_target_is_nulled() {
        let history = table(
            vec![
                numeric_column("x", vec![1.0, 2.0, 3.0]),
                numeric_column("target", vec![10.0, 11.0, 12.0]),
            ],
            "target",
        );
        let future_columns = vec![numeric_column("x", vec![4.0, 5.0])];
        let future = ParsedTable {
            row_count: 2,
            feature_names: vec!["x".into()],
            target_index: 0,
            target_name: "target".into(),
            columns: future_columns,
        };
        assert_eq!(future.row_count, 2);
        // Directly exercise the row-shaping rule used by the FFI wrapper.
        let by_name = future
            .columns
            .iter()
            .map(|column| (column.name.as_str(), column))
            .collect::<HashMap<_, _>>();
        let mut row = vec![Value::Null; history.columns.len()];
        for (index, column) in history.columns.iter().enumerate() {
            if column.name != history.target_name {
                row[index] = by_name[column.name.as_str()].values[0].clone();
            }
        }
        assert_eq!(row[0], Value::Number(Number::from_f64(4.0).unwrap()));
        assert_eq!(row[1], Value::Null);
    }
}
