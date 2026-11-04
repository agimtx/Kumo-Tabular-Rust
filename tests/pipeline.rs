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
