//! Extensible preprocessing and ONNX inference for tabular models.
//!
//! The crate separates input preparation from model-specific tensor contracts.
//! A C ABI exposes dynamic Kumo ONNX inference without requiring callers to link
//! Rust or Polars.

pub mod error;
pub mod ffi;
pub mod kumo;
pub mod kumo_recipe;
pub mod model;
pub mod onnx;
pub mod preprocess;
pub mod table;

pub use error::{Error, Result};
pub use model::{ModelKind, ModelRegistry, Prediction, Predictor, Task};
pub use onnx::OnnxSession;
pub use preprocess::{FeatureSpec, PreprocessConfig, Preprocessor, TransformOutput};
pub use table::{Column, ColumnData, Table, Value};
