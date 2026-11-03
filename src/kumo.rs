//! Adapter for the tensor contract emitted by `scripts/export_onnx.py`.
//! The current export is fixed to 6 rows, 3 features, 4 context rows and 2 queries.

use crate::model::{prediction_from_tensor, ModelKind, Prediction, Predictor, TensorPrediction};
use crate::preprocess::Preprocessor;
use crate::{Error, OnnxSession, Result, Table, Task};
use ndarray::{Array1, Array2};
use std::path::Path;
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct KumoBatch {
    pub x: Array2<f32>,
    pub y_classification: Option<Array1<i64>>,
    pub y_regression: Option<Array1<f32>>,
    pub categorical_mask: Array1<bool>,
}

impl KumoBatch {
    /// Build an in-context batch for dynamic exports; missing feature cells remain NaN.
    pub fn from_dynamic_features(
        features: &Array2<f32>,
        labels: &[f32],
        task: Task,
        categorical_mask: &[bool],
    ) -> Result<Self> {
        if labels.is_empty() || features.nrows() <= labels.len() || features.ncols() == 0 {
            return Err(Error::Preprocessing(
                "Kumo needs context rows, query rows and features".into(),
            ));
        }
        if categorical_mask.len() != features.ncols() {
            return Err(Error::Preprocessing(
                "Kumo categorical mask must match feature width".into(),
            ));
        }
        if features.iter().any(|value| value.is_infinite()) {
            return Err(Error::Preprocessing(
                "Kumo x contains infinite values".into(),
            ));
        }
        let (y_classification, y_regression) = match task {
            Task::Classification => {
                let labels = labels
                    .iter()
                    .map(|&label| {
                        if !label.is_finite()
                            || !(0.0..=9.0).contains(&label)
                            || label.fract() != 0.0
                        {
                            return Err(Error::Preprocessing(
                                "Kumo classification labels must be integer codes 0 through 9"
                                    .into(),
                            ));
                        }
                        Ok(label as i64)
                    })
                    .collect::<Result<Vec<_>>>()?;
                (Some(Array1::from(labels)), None)
            }
            Task::Regression => {
                if labels.iter().any(|label| !label.is_finite()) {
                    return Err(Error::Preprocessing(
                        "Kumo regression labels must be finite".into(),
                    ));
                }
                (None, Some(Array1::from(labels.to_vec())))
            }
        };
        Ok(Self {
            x: features.clone(),
            y_classification,
            y_regression,
            categorical_mask: Array1::from(categorical_mask.to_vec()),
        })
    }

    pub fn from_features(
        features: &Array2<f32>,
        labels: &[f32],
        task: Task,
        categorical_mask: &[bool],
    ) -> Result<Self> {
        if features.shape() != [6, 3] {
            return Err(Error::Preprocessing(format!(
                "current Kumo export requires x shape [6, 3], got {:?}",
                features.shape()
            )));
        }
        if labels.len() != 4 {
            return Err(Error::Preprocessing(
                "current Kumo export requires exactly 4 context labels".into(),
            ));
        }
        if categorical_mask.len() != 3 {
            return Err(Error::Preprocessing(
                "current Kumo export requires exactly 3 categorical mask values".into(),
            ));
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err(Error::Preprocessing(
                "Kumo x contains non-finite values".into(),
            ));
        }
        let (y_classification, y_regression) = match task {
            Task::Classification => {
                let mut encoded = Vec::with_capacity(4);
                for label in labels {
                    if !label.is_finite() || *label < 0.0 || *label > 9.0 || label.fract() != 0.0 {
                        return Err(Error::Preprocessing(
                            "Kumo classification labels must be integer class codes 0 through 9"
                                .into(),
                        ));
                    }
                    encoded.push(*label as i64);
                }
                (Some(Array1::from(encoded)), None)
            }
            Task::Regression => {
                if labels.iter().any(|value| !value.is_finite()) {
                    return Err(Error::Preprocessing(
                        "Kumo regression labels must be finite".into(),
                    ));
                }
                (None, Some(Array1::from(labels.to_vec())))
            }
        };
        Ok(Self {
            x: features.clone(),
            y_classification,
            y_regression,
            categorical_mask: Array1::from(categorical_mask.to_vec()),
        })
    }

    pub fn query_rows(&self) -> std::ops::Range<usize> {
        4..6
    }
}

pub struct KumoPredictor {
    session: Mutex<OnnxSession>,
    preprocessor: Preprocessor,
    task: Task,
    context_features: Array2<f32>,
    context_labels: Vec<f32>,
    labels: Option<Vec<String>>,
}

impl KumoPredictor {
    pub fn load(
        model_path: impl AsRef<Path>,
        preprocessor: Preprocessor,
        task: Task,
        context_table: &Table,
        context_labels: Vec<f32>,
        labels: Option<Vec<String>>,
    ) -> Result<Self> {
        if context_labels.len() != 4 {
            return Err(Error::Model(
                "current Kumo export requires 4 context labels".into(),
            ));
        }
        if preprocessor.config().features.len() != 3 {
            return Err(Error::Model(
                "current Kumo export requires exactly 3 configured features".into(),
            ));
        }
        if context_table.height != 4 {
            return Err(Error::Model(
                "current Kumo export requires exactly 4 context feature rows".into(),
            ));
        }
        let context_features = preprocessor.transform(context_table)?.values;
        Ok(Self {
            session: Mutex::new(OnnxSession::load(model_path)?),
            preprocessor,
            task,
            context_features,
            context_labels,
            labels,
        })
    }
}

impl Predictor for KumoPredictor {
    fn kind(&self) -> ModelKind {
        ModelKind::KumoTabular
    }
    fn task(&self) -> Task {
        self.task
    }

    fn predict(&self, table: &Table) -> Result<Prediction> {
        if table.height != 2 {
            return Err(Error::Preprocessing(
                "current Kumo export predicts exactly 2 query rows per call".into(),
            ));
        }
        let transformed = self.preprocessor.transform(table)?;
        let mut x = Array2::zeros((6, 3));
        x.slice_mut(ndarray::s![..4, ..])
            .assign(&self.context_features);
        x.slice_mut(ndarray::s![4..6, ..])
            .assign(&transformed.values);
        if transformed.categorical_mask.len() != 3 {
            return Err(Error::Preprocessing(
                "Kumo categorical mask must contain 3 values".into(),
            ));
        }
        let batch = KumoBatch::from_features(
            &x,
            &self.context_labels,
            self.task,
            &transformed.categorical_mask,
        )?;
        let output = self
            .session
            .lock()
            .map_err(|_| Error::Model("ONNX session lock was poisoned".into()))?
            .run_kumo(batch)?;
        prediction_from_tensor(
            self.task,
            TensorPrediction { values: output },
            self.labels.clone(),
        )
    }
}
