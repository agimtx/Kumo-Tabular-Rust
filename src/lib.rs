//! Extensible preprocessing and ONNX inference for tabular models.
//!
//! The crate separates input preparation from model-specific tensor contracts.
//! A future FFI can expose [`Table`], [`Predictor`] and serializable configs
//! without coupling callers to Polars or ONNX Runtime internals.

pub mod error;
pub mod kumo;
pub mod model;
pub mod onnx;
pub mod preprocess;
pub mod table;

pub use error::{Error, Result};
pub use model::{ModelKind, ModelRegistry, Prediction, Predictor, Task};
pub use onnx::OnnxSession;
pub use preprocess::{FeatureSpec, PreprocessConfig, Preprocessor, TransformOutput};
pub use table::{Column, ColumnData, Table, Value};
