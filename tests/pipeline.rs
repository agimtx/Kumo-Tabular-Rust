use ndarray::array;
use ndarray::Array2;
use std::collections::BTreeMap;
use tabular_onnx::kumo::{KumoBatch, KumoPredictor};
use tabular_onnx::model::{prediction_from_tensor, Task, TensorPrediction};
use tabular_onnx::{
    Column, ColumnData, FeatureSpec, OnnxSession, PreprocessConfig, Preprocessor, Table, Value,
};

#[test]
fn constructs_table_and_preprocesses_missing_numeric_and_categories() {
    let names = vec!["age".to_string(), "city".to_string()];
    let rows = vec![
        vec![Value::Int(10), Value::String("Oslo".into())],
        vec![Value::Null, Value::String("Rome".into())],
        vec![Value::Int(30), Value::Null],
    ];
    let table = Table::from_rows(&names, &rows).unwrap();
    let mut categories = BTreeMap::new();
    categories.insert("Oslo".into(), 4);
    categories.insert("Rome".into(), 7);
    let prep = Preprocessor::new(PreprocessConfig {
        standardize_numeric: true,
        features: vec![
            FeatureSpec::Numeric {
                column: "age".into(),
                mean: 20.0,
                scale: 10.0,
                missing_value: 20.0,
            },
            FeatureSpec::Categorical {
                column: "city".into(),
                categories,
                unknown_code: -1,
                missing_code: -2,
            },
        ],
    })
    .unwrap();
    let output = prep.transform(&table).unwrap();
    assert_eq!(output.values, array![[-1.0, 4.0], [0.0, 7.0], [1.0, -2.0]]);
    assert_eq!(output.categorical_mask, vec![false, true]);
}

#[test]
fn roundtrips_through_polars() {
    let table = Table::new(vec![
        Column {
            name: "x".into(),
            data: ColumnData::Numeric(vec![Some(1.0), None]),
        },
        Column {
            name: "group".into(),
            data: ColumnData::Categorical(vec![Some("a".into()), None]),
        },
    ])
    .unwrap();
    let frame = table.to_polars().unwrap();
    let parsed = Table::from_polars(&frame).unwrap();
    assert_eq!(parsed, table);
}

#[test]
fn validates_kumo_contract_and_classification_outputs() {
    let features = Array2::zeros((6, 3));
    let batch = KumoBatch::from_features(
        &features,
        &[0.0, 1.0, 2.0, 9.0],
        Task::Classification,
        &[false; 3],
    )
    .unwrap();
    assert_eq!(batch.query_rows(), 4..6);
    assert_eq!(batch.y_classification.unwrap().to_vec(), vec![0, 1, 2, 9]);
    let prediction = prediction_from_tensor(
        Task::Classification,
        TensorPrediction {
            values: Array2::zeros((2, 10)),
        },
        None,
    )
    .unwrap();
    assert!(
        matches!(prediction, tabular_onnx::Prediction::Classification { scores, .. } if scores.len() == 2 && scores[0].len() == 10)
    );
}

#[test]
fn validates_dynamic_kumo_context_and_query_contract() {
    let mut features = Array2::<f32>::zeros((34, 13));
    features[(0, 0)] = f32::NAN;
    let batch =
        KumoBatch::from_dynamic_features(&features, &[0.0; 32], Task::Classification, &[false; 13])
            .unwrap();
    assert_eq!(batch.x.shape(), &[34, 13]);
    assert_eq!(batch.y_classification.unwrap().len(), 32);
    assert!(KumoBatch::from_dynamic_features(
        &features,
        &[0.0; 34],
        Task::Classification,
        &[false; 13]
    )
    .is_err());
    assert!(KumoBatch::from_dynamic_features(
        &features,
        &[0.0; 32],
        Task::Classification,
        &[false; 3]
    )
    .is_err());
}

#[test]
fn predictor_rejects_missing_context_feature_rows_before_loading_model() {
    let config = PreprocessConfig {
        features: ["a", "b", "c"]
            .iter()
            .map(|name| FeatureSpec::Numeric {
                column: (*name).into(),
                mean: 0.0,
                scale: 1.0,
                missing_value: 0.0,
            })
            .collect(),
        standardize_numeric: false,
    };
    let context = Table::new(
        ["a", "b", "c"]
            .iter()
            .map(|name| Column {
                name: (*name).into(),
                data: ColumnData::Numeric(vec![Some(1.0); 3]),
            })
            .collect(),
    )
    .unwrap();
    let result = KumoPredictor::load(
        "missing-model.onnx",
        Preprocessor::new(config).unwrap(),
        Task::Classification,
        &context,
        vec![0.0, 1.0, 0.0, 1.0],
        None,
    );
    assert!(
        matches!(result, Err(tabular_onnx::Error::Model(message)) if message.contains("4 context feature rows"))
    );
}

#[test]
#[ignore = "requires models/tabular-cls-medium-f16.onnx and ORT_DYLIB_PATH; run with --ignored"]
fn dynamic_medium_model_accepts_16_context_rows() {
    use std::time::Instant;

    let model = "models/tabular-cls-medium-f16.onnx";
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "set ORT_DYLIB_PATH before running this test"
    );
    assert!(
        std::path::Path::new(model).exists(),
        "missing model: {model}"
    );

    // 16 context rows + 2 query rows and the 13 classification features.
    let mut features = Array2::<f32>::zeros((18, 13));
    for row in 0..18 {
        for col in 0..13 {
            features[(row, col)] = ((row * 13 + col) % 17) as f32 / 7.0;
        }
    }
    let labels = (0..16).map(|row| (row % 2) as f32).collect::<Vec<_>>();
    let categorical_mask = [
        false, false, true, true, true, false, true, true, false, false, false, false, false,
    ];
    let batch = KumoBatch::from_dynamic_features(
        &features,
        &labels,
        Task::Classification,
        &categorical_mask,
    )
    .unwrap();

    let mut session = OnnxSession::load(model).unwrap();
    let started = Instant::now();
    let output = session.run_kumo(batch).unwrap();
    eprintln!(
        "{model} 16-context inference: shape {:?}, {:.2} ms",
        output.shape(),
        started.elapsed().as_secs_f64() * 1000.0
    );

    assert_eq!(output.shape(), &[2, 10]);
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "requires models/tabular-cls-medium-f16.onnx and ORT_DYLIB_PATH; run with --ignored"]
fn high_level_table_api_classifies_without_tensors() {
    use std::ffi::{CStr, CString};
    use tabular_onnx::ffi::{tabular_onnx_create, tabular_onnx_destroy, tabular_onnx_last_error};
    use tabular_onnx::ffi_table::{
        tabular_onnx_predict_table, tabular_onnx_prediction_free, TabularOnnxColumn,
        TabularOnnxPrediction, TabularOnnxTable, TABULAR_ONNX_COLUMN_NUMERIC,
    };

    let model = "models/tabular-cls-medium-f16.onnx";
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "set ORT_DYLIB_PATH before running this test"
    );
    assert!(
        std::path::Path::new(model).exists(),
        "missing model: {model}"
    );

    let model = CString::new(model).unwrap();
    let mut handle = std::ptr::null_mut();
    let status = unsafe { tabular_onnx_create(model.as_ptr(), &mut handle) };
    if status != 0 {
        let error = unsafe { CStr::from_ptr(tabular_onnx_last_error()) };
        panic!("create failed: {}", error.to_string_lossy());
    }

    let context_rows = 16;
    let rows = context_rows + 2;
    let feature_count = 13;
    let names = (0..feature_count)
        .map(|index| CString::new(format!("f{index}")).unwrap())
        .chain(std::iter::once(CString::new("target").unwrap()))
        .collect::<Vec<_>>();
    let mut values = (0..feature_count)
        .map(|column| {
            (0..rows)
                .map(|row| ((row * feature_count + column) % 17) as f64 / 7.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    values.push(
        (0..rows)
            .map(|row| if row % 2 == 0 { 0.0 } else { 1.0 })
            .collect(),
    );
    let columns = (0..names.len())
        .map(|index| TabularOnnxColumn {
            name: names[index].as_ptr(),
            kind: TABULAR_ONNX_COLUMN_NUMERIC,
            numeric_values: values[index].as_ptr(),
            category_values: std::ptr::null(),
        })
        .collect::<Vec<_>>();
    let table = TabularOnnxTable {
        columns: columns.as_ptr(),
        column_count: columns.len(),
        row_count: rows,
    };
    let target = CString::new("target").unwrap();
    let mut prediction = unsafe { std::mem::zeroed::<TabularOnnxPrediction>() };
    let status = unsafe {
        tabular_onnx_predict_table(
            handle,
            0,
            &table,
            context_rows,
            target.as_ptr(),
            &mut prediction,
        )
    };
    if status != 0 {
        let error = unsafe { CStr::from_ptr(tabular_onnx_last_error()) };
        panic!("predict_table failed: {}", error.to_string_lossy());
    }

    assert_eq!(prediction.rows, 2);
    assert_eq!(prediction.columns, 2);
    assert_eq!(prediction.class_count, 2);
    assert!(!prediction.values.is_null());
    for row in 0..prediction.rows {
        let p0 = unsafe { *prediction.values.add(row * prediction.columns) };
        let p1 = unsafe { *prediction.values.add(row * prediction.columns + 1) };
        assert!((p0 + p1 - 1.0).abs() < 1e-4, "probabilities {p0} + {p1}");
    }

    unsafe {
        tabular_onnx_prediction_free(&mut prediction);
        tabular_onnx_destroy(handle);
    }
}

#[test]
#[ignore = "requires models/tabular-reg-medium-f16.onnx and ORT_DYLIB_PATH; run with --ignored"]
fn high_level_table_api_regresses_without_tensors() {
    use std::ffi::{CStr, CString};
    use tabular_onnx::ffi::{tabular_onnx_create, tabular_onnx_destroy, tabular_onnx_last_error};
    use tabular_onnx::ffi_table::{
        tabular_onnx_predict_table, tabular_onnx_prediction_free, TabularOnnxColumn,
        TabularOnnxPrediction, TabularOnnxTable, TABULAR_ONNX_COLUMN_NUMERIC,
    };

    let model = "models/tabular-reg-medium-f16.onnx";
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "set ORT_DYLIB_PATH before running this test"
    );
    assert!(
        std::path::Path::new(model).exists(),
        "missing model: {model}"
    );

    let model = CString::new(model).unwrap();
    let mut handle = std::ptr::null_mut();
    let status = unsafe { tabular_onnx_create(model.as_ptr(), &mut handle) };
    if status != 0 {
        let error = unsafe { CStr::from_ptr(tabular_onnx_last_error()) };
        panic!("create failed: {}", error.to_string_lossy());
    }

    let context_rows = 16;
    let rows = context_rows + 2;
    let feature_count = 5;
    let names = (0..feature_count)
        .map(|index| CString::new(format!("f{index}")).unwrap())
        .chain(std::iter::once(CString::new("target").unwrap()))
        .collect::<Vec<_>>();
    let mut values = (0..feature_count)
        .map(|column| {
            (0..rows)
                .map(|row| ((row * feature_count + column) % 23) as f64 / 11.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    values.push((0..rows).map(|row| 100.0 + (row % 7) as f64).collect());
    let columns = (0..names.len())
        .map(|index| TabularOnnxColumn {
            name: names[index].as_ptr(),
            kind: TABULAR_ONNX_COLUMN_NUMERIC,
            numeric_values: values[index].as_ptr(),
            category_values: std::ptr::null(),
        })
        .collect::<Vec<_>>();
    let table = TabularOnnxTable {
        columns: columns.as_ptr(),
        column_count: columns.len(),
        row_count: rows,
    };
    let target = CString::new("target").unwrap();
    let mut prediction = unsafe { std::mem::zeroed::<TabularOnnxPrediction>() };
    let status = unsafe {
        tabular_onnx_predict_table(
            handle,
            1,
            &table,
            context_rows,
            target.as_ptr(),
            &mut prediction,
        )
    };
    if status != 0 {
        let error = unsafe { CStr::from_ptr(tabular_onnx_last_error()) };
        panic!("predict_table failed: {}", error.to_string_lossy());
    }

    assert_eq!(prediction.rows, 2);
    assert_eq!(prediction.columns, 3);
    assert_eq!(prediction.quantile_count, 999);
    assert!(!prediction.values.is_null());
    assert!(!prediction.quantiles.is_null());
    for index in 0..prediction.rows * prediction.columns {
        assert!(unsafe { *prediction.values.add(index) }.is_finite());
    }

    unsafe {
        tabular_onnx_prediction_free(&mut prediction);
        tabular_onnx_destroy(handle);
    }
}

#[test]
#[ignore = "requires models/tabular-reg-medium-f16.onnx and ORT_DYLIB_PATH; run with --ignored"]
fn high_level_forecast_handles_univariate_history() {
    use std::ffi::{CStr, CString};
    use tabular_onnx::ffi::{tabular_onnx_create, tabular_onnx_destroy, tabular_onnx_last_error};
    use tabular_onnx::ffi_table::{
        tabular_onnx_forecast, tabular_onnx_prediction_free, TabularOnnxColumn,
        TabularOnnxPrediction, TabularOnnxTable, TABULAR_ONNX_COLUMN_NUMERIC,
    };

    let model = "models/tabular-reg-medium-f16.onnx";
    assert!(
        std::env::var_os("ORT_DYLIB_PATH").is_some(),
        "set ORT_DYLIB_PATH before running this test"
    );
    assert!(
        std::path::Path::new(model).exists(),
        "missing model: {model}"
    );

    let model = CString::new(model).unwrap();
    let mut handle = std::ptr::null_mut();
    let status = unsafe { tabular_onnx_create(model.as_ptr(), &mut handle) };
    if status != 0 {
        let error = unsafe { CStr::from_ptr(tabular_onnx_last_error()) };
        panic!("create failed: {}", error.to_string_lossy());
    }

    let rows = 20;
    let values = (0..rows)
        .map(|row| 100.0 + row as f64 * 0.5)
        .collect::<Vec<_>>();
    let target_name = CString::new("target").unwrap();
    let column = TabularOnnxColumn {
        name: target_name.as_ptr(),
        kind: TABULAR_ONNX_COLUMN_NUMERIC,
        numeric_values: values.as_ptr(),
        category_values: std::ptr::null(),
    };
    let table = TabularOnnxTable {
        columns: &column,
        column_count: 1,
        row_count: rows,
    };
    let mut prediction = unsafe { std::mem::zeroed::<TabularOnnxPrediction>() };
    let status = unsafe {
        tabular_onnx_forecast(
            handle,
            1,
            &table,
            16,
            target_name.as_ptr(),
            3,
            std::ptr::null(),
            &mut prediction,
        )
    };
    if status != 0 {
        let error = unsafe { CStr::from_ptr(tabular_onnx_last_error()) };
        panic!("forecast failed: {}", error.to_string_lossy());
    }

    assert_eq!(prediction.rows, 3);
    assert_eq!(prediction.columns, 3);
    assert_eq!(prediction.quantile_count, 999);
    for index in 0..prediction.rows * prediction.columns {
        assert!(unsafe { *prediction.values.add(index) }.is_finite());
    }
    for index in 0..prediction.rows * prediction.quantile_count {
        assert!(unsafe { *prediction.quantiles.add(index) }.is_finite());
    }

    unsafe {
        tabular_onnx_prediction_free(&mut prediction);
        tabular_onnx_destroy(handle);
    }
}
