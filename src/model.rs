use crate::{Error, Result, Table};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Classification,
    Regression,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    KumoTabular,
    GenericOnnx,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Prediction {
    Classification {
        scores: Vec<Vec<f32>>,
        labels: Option<Vec<String>>,
    },
    Regression {
        values: Vec<Vec<f32>>,
    },
}

pub trait Predictor: Send + Sync {
    fn kind(&self) -> ModelKind;
    fn task(&self) -> Task;
    fn predict(&self, table: &Table) -> Result<Prediction>;
}

type Factory = Box<dyn Fn(&[u8]) -> Result<Box<dyn Predictor>> + Send + Sync>;

#[derive(Default)]
pub struct ModelRegistry {
    factories: HashMap<String, Factory>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(
        &mut self,
        name: impl Into<String>,
        factory: impl Fn(&[u8]) -> Result<Box<dyn Predictor>> + Send + Sync + 'static,
    ) -> Result<()> {
        let name = name.into();
        if self.factories.contains_key(&name) {
            return Err(Error::Model(format!(
                "model factory {name} is already registered"
            )));
        }
        self.factories.insert(name, Box::new(factory));
        Ok(())
    }
    pub fn load(&self, name: &str, config: &[u8]) -> Result<Box<dyn Predictor>> {
        self.factories
            .get(name)
            .ok_or_else(|| Error::UnsupportedModel(name.into()))?(config)
    }
}

#[derive(Clone, Debug)]
pub struct TensorPrediction {
    pub values: Array2<f32>,
}

pub fn prediction_from_tensor(
    task: Task,
    output: TensorPrediction,
    labels: Option<Vec<String>>,
) -> Result<Prediction> {
    let rows = output
        .values
        .outer_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    match task {
        Task::Classification => {
            if let Some(labels) = &labels {
                if rows.iter().any(|row| row.len() != labels.len()) {
                    return Err(Error::Model(
                        "classification label count must match output width".into(),
                    ));
                }
            }
            Ok(Prediction::Classification {
                scores: rows,
                labels,
            })
        }
        Task::Regression => Ok(Prediction::Regression { values: rows }),
    }
}
