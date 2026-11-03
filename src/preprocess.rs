use crate::{ColumnData, Error, Result, Table};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FeatureSpec {
    Numeric {
        column: String,
        mean: f64,
        scale: f64,
        missing_value: f64,
    },
    Categorical {
        column: String,
        categories: BTreeMap<String, i64>,
        unknown_code: i64,
        missing_code: i64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreprocessConfig {
    pub features: Vec<FeatureSpec>,
    #[serde(default)]
    pub standardize_numeric: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransformOutput {
    pub values: Array2<f32>,
    pub categorical_mask: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct Preprocessor {
    config: PreprocessConfig,
}

impl Preprocessor {
    pub fn new(config: PreprocessConfig) -> Result<Self> {
        if config.features.is_empty() {
            return Err(Error::Preprocessing(
                "at least one feature is required".into(),
            ));
        }
        let mut names = std::collections::HashSet::new();
        for feature in &config.features {
            let name = match feature {
                FeatureSpec::Numeric { column, .. } | FeatureSpec::Categorical { column, .. } => {
                    column
                }
            };
            if !names.insert(name) {
                return Err(Error::Preprocessing(format!(
                    "feature column {name} is configured more than once"
                )));
            }
            if let FeatureSpec::Numeric {
                scale,
                mean,
                missing_value,
                ..
            } = feature
            {
                if !scale.is_finite()
                    || *scale <= 0.0
                    || !mean.is_finite()
                    || !missing_value.is_finite()
                {
                    return Err(Error::Preprocessing(format!(
                        "invalid numeric statistics for {name}"
                    )));
                }
            }
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &PreprocessConfig {
        &self.config
    }

    pub fn transform(&self, table: &Table) -> Result<TransformOutput> {
        let mut values = Array2::<f32>::zeros((table.height, self.config.features.len()));
        let mut categorical_mask = Vec::with_capacity(self.config.features.len());
        for (feature_index, feature) in self.config.features.iter().enumerate() {
            match feature {
                FeatureSpec::Numeric {
                    column,
                    mean,
                    scale,
                    missing_value,
                } => {
                    let source = table.column(column)?;
                    let ColumnData::Numeric(source) = &source.data else {
                        return Err(Error::Preprocessing(format!(
                            "column {column} must be numeric"
                        )));
                    };
                    for (row, value) in source.iter().enumerate() {
                        let value = value.unwrap_or(*missing_value);
                        let transformed = if self.config.standardize_numeric {
                            (value - mean) / scale
                        } else {
                            value
                        };
                        values[(row, feature_index)] = finite_f32(transformed, column)?;
                    }
                    categorical_mask.push(false);
                }
                FeatureSpec::Categorical {
                    column,
                    categories,
                    unknown_code,
                    missing_code,
                } => {
                    let source = table.column(column)?;
                    let ColumnData::Categorical(source) = &source.data else {
                        return Err(Error::Preprocessing(format!(
                            "column {column} must be categorical"
                        )));
                    };
                    for (row, value) in source.iter().enumerate() {
                        let code = match value {
                            Some(value) => *categories.get(value).unwrap_or(unknown_code),
                            None => *missing_code,
                        };
                        values[(row, feature_index)] = code as f32;
                    }
                    categorical_mask.push(true);
                }
            }
        }
        Ok(TransformOutput {
            values,
            categorical_mask,
        })
    }
}

fn finite_f32(value: f64, column: &str) -> Result<f32> {
    if !value.is_finite() || value.abs() > f32::MAX as f64 {
        return Err(Error::Preprocessing(format!(
            "value in column {column} cannot be represented as finite f32"
        )));
    }
    Ok(value as f32)
}
