use crate::{Error, Result};
use polars::prelude::{Column as PolarsColumn, DataFrame, DataType, NamedFrom, Series};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "values", rename_all = "snake_case")]
pub enum ColumnData {
    Numeric(Vec<Option<f64>>),
    Categorical(Vec<Option<String>>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub data: ColumnData,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Table {
    pub columns: Vec<Column>,
    pub height: usize,
}

impl Table {
    pub fn new(columns: Vec<Column>) -> Result<Self> {
        let height = columns.first().map_or(0, |c| c.data.len());
        if columns.iter().any(|c| c.data.len() != height) {
            return Err(Error::InvalidTable(
                "all columns must have the same length".into(),
            ));
        }
        let mut names = std::collections::HashSet::new();
        if columns.iter().any(|c| !names.insert(c.name.as_str())) {
            return Err(Error::InvalidTable("column names must be unique".into()));
        }
        Ok(Self { columns, height })
    }

    pub fn from_rows(names: &[String], rows: &[Vec<Value>]) -> Result<Self> {
        if rows.iter().any(|row| row.len() != names.len()) {
            return Err(Error::InvalidTable(
                "each row must match the number of column names".into(),
            ));
        }
        let mut columns = Vec::with_capacity(names.len());
        for (index, name) in names.iter().enumerate() {
            let values: Vec<&Value> = rows.iter().map(|r| &r[index]).collect();
            let categorical = values.iter().any(|v| matches!(v, Value::String(_)));
            let data = if categorical {
                ColumnData::Categorical(
                    values
                        .into_iter()
                        .map(|v| match v {
                            Value::Null => Ok(None),
                            Value::String(s) => Ok(Some(s.clone())),
                            _ => Err(Error::InvalidTable(format!(
                                "mixed categorical values in column {name}"
                            ))),
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            } else {
                ColumnData::Numeric(
                    values
                        .into_iter()
                        .map(|v| match v {
                            Value::Null => Ok(None),
                            Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
                            Value::Int(n) => Ok(Some(*n as f64)),
                            Value::Float(x) if x.is_finite() => Ok(Some(*x)),
                            Value::Float(_) => Err(Error::InvalidTable(format!(
                                "non-finite value in column {name}"
                            ))),
                            Value::String(_) => unreachable!(),
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            };
            columns.push(Column {
                name: name.clone(),
                data,
            });
        }
        Self::new(columns)
    }

    pub fn from_polars(df: &DataFrame) -> Result<Self> {
        let mut columns = Vec::with_capacity(df.width());
        for column in df.get_columns() {
            let name = column.name().to_string();
            let cast = match column.dtype() {
                DataType::String => {
                    let values = column
                        .str()
                        .map_err(|e| Error::InvalidTable(e.to_string()))?;
                    ColumnData::Categorical(
                        values.into_iter().map(|v| v.map(str::to_owned)).collect(),
                    )
                }
                _ => {
                    let numeric = column
                        .cast(&DataType::Float64)
                        .map_err(|e| Error::InvalidTable(format!("column {name}: {e}")))?;
                    let values = numeric
                        .f64()
                        .map_err(|e| Error::InvalidTable(e.to_string()))?;
                    ColumnData::Numeric(values.into_iter().collect())
                }
            };
            columns.push(Column { name, data: cast });
        }
        Self::new(columns)
    }

    pub fn to_polars(&self) -> Result<DataFrame> {
        let cols = self
            .columns
            .iter()
            .map(|column| -> Result<PolarsColumn> {
                let series = match &column.data {
                    ColumnData::Numeric(values) => {
                        Series::new(column.name.clone().into(), values.clone())
                    }
                    ColumnData::Categorical(values) => {
                        Series::new(column.name.clone().into(), values.clone())
                    }
                };
                Ok(series.into())
            })
            .collect::<Result<Vec<_>>>()?;
        DataFrame::new(cols).map_err(|e| Error::InvalidTable(e.to_string()))
    }

    pub fn column(&self, name: &str) -> Result<&Column> {
        self.columns
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| Error::InvalidTable(format!("missing column {name}")))
    }
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            Self::Numeric(v) => v.len(),
            Self::Categorical(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
