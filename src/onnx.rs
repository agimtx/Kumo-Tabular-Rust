//! ONNX Runtime-backed tensor model adapter.
//!
//! Model-specific input assembly lives in adapters. This low-level wrapper
//! accepts named float32 tensor inputs and returns a float32 output matrix.

use crate::{Error, Result};
use ndarray::{ArrayD, Ix2};
use ort::{session::Session, value::Tensor};
use std::path::Path;

pub struct OnnxSession {
    session: Session,
}

impl OnnxSession {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort_error)?
            .commit_from_file(path)
            .map_err(ort_error)?;
        Ok(Self { session })
    }

    pub fn input_names(&self) -> Vec<String> {
        self.session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect()
    }

    pub fn output_names(&self) -> Vec<String> {
        self.session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect()
    }

    pub fn run_f32(
        &mut self,
        inputs: Vec<(String, ArrayD<f32>)>,
        output_name: &str,
    ) -> Result<ArrayD<f32>> {
        let tensors = inputs
            .into_iter()
            .map(|(name, array)| {
                Tensor::from_array(array)
                    .map(|tensor| (name, tensor))
                    .map_err(ort_error)
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = self.session.run(tensors).map_err(ort_error)?;
        let output = outputs
            .get(output_name)
            .ok_or_else(|| Error::Model(format!("ONNX output {output_name} was not returned")))?;
        let (shape, data) = output.try_extract_tensor::<f32>().map_err(ort_error)?;
        ArrayD::from_shape_vec(
            shape.iter().map(|d| *d as usize).collect::<Vec<_>>(),
            data.to_vec(),
        )
        .map_err(|e| Error::Ort(e.to_string()))
    }

    pub fn run_f32_matrix(
        &mut self,
        inputs: Vec<(String, ArrayD<f32>)>,
        output_name: &str,
    ) -> Result<ndarray::Array2<f32>> {
        self.run_f32(inputs, output_name)?
            .into_dimensionality::<Ix2>()
            .map_err(|e| Error::Model(format!("expected rank-2 output: {e}")))
    }

    pub fn run_kumo(&mut self, batch: crate::kumo::KumoBatch) -> Result<ndarray::Array2<f32>> {
        let output_name = self
            .session
            .outputs
            .first()
            .map(|output| output.name.clone())
            .ok_or_else(|| Error::Model("ONNX model has no outputs".into()))?;
        let x = Tensor::from_array(batch.x.into_dyn()).map_err(ort_error)?;
        let categorical_mask =
            Tensor::from_array(batch.categorical_mask.into_dyn()).map_err(ort_error)?;
        let outputs = match (batch.y_classification, batch.y_regression) {
            (Some(values), None) => {
                let y = Tensor::from_array(values.into_dyn()).map_err(ort_error)?;
                self.session
                    .run(ort::inputs! {
                        "x" => x,
                        "y" => y,
                        "categorical_mask" => categorical_mask,
                    })
                    .map_err(ort_error)?
            }
            (None, Some(values)) => {
                let y = Tensor::from_array(values.into_dyn()).map_err(ort_error)?;
                self.session
                    .run(ort::inputs! {
                        "x" => x,
                        "y" => y,
                        "categorical_mask" => categorical_mask,
                    })
                    .map_err(ort_error)?
            }
            _ => {
                return Err(Error::Model(
                    "Kumo batch must contain exactly one target tensor".into(),
                ))
            }
        };
        let output = outputs
            .get(output_name.as_str())
            .ok_or_else(|| Error::Model(format!("ONNX output {output_name} was not returned")))?;
        let (shape, data) = output.try_extract_tensor::<f32>().map_err(ort_error)?;
        let values = ArrayD::from_shape_vec(
            shape.iter().map(|d| *d as usize).collect::<Vec<_>>(),
            data.to_vec(),
        )
        .map_err(|e| Error::Ort(e.to_string()))?;
        values
            .into_dimensionality::<Ix2>()
            .map_err(|e| Error::Model(format!("expected rank-2 Kumo output: {e}")))
    }
}

fn ort_error(error: impl std::fmt::Display) -> Error {
    Error::Ort(error.to_string())
}
