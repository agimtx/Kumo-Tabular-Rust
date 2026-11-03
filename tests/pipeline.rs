use ndarray::array;
use ndarray::Array2;
use std::collections::BTreeMap;
use tabular_onnx::kumo::KumoBatch;
use tabular_onnx::model::{prediction_from_tensor, Task, TensorPrediction};
use tabular_onnx::{Column, ColumnData, FeatureSpec, PreprocessConfig, Preprocessor, Table, Value};

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
