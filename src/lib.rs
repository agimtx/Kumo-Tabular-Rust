//! Extensible preprocessing and ONNX inference for tabular models.
//!
//! The crate separates input preparation from model-specific tensor contracts.
//! A C ABI exposes both tensor-level Kumo ONNX inference and a high-level table
//! API that accepts raw numeric/categorical columns, performs the Kumo recipe,
//! handles multiclass ECOC, and returns probabilities or quantiles without
//! requiring callers to link Rust or Polars.

pub mod ecoc;
pub mod error;
pub mod ffi;
pub mod ffi_table;
pub mod kumo;
pub mod kumo_recipe;
pub mod model;
pub mod onnx;
pub mod preprocess;
mod runtime;
pub mod table;

pub use error::{Error, Result};
pub use model::{ModelKind, ModelRegistry, Prediction, Predictor, Task};
pub use onnx::OnnxSession;
pub use preprocess::{FeatureSpec, PreprocessConfig, Preprocessor, TransformOutput};
pub use table::{Column, ColumnData, Table, Value};
