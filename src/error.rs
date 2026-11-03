use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid table: {0}")]
    InvalidTable(String),
    #[error("preprocessing error: {0}")]
    Preprocessing(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("ONNX Runtime error: {0}")]
    Ort(String),
}
