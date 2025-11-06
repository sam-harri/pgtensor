use crate::tensor_core::{tensor_reduce, Tensor, TensorElemBuffer};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    tensor::TensorElementType,
};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("shape mismatch")]
    ShapeMismatch,
    #[error("failed to build session")]
    SessionBuilderError,
    #[error("inference failed")]
    InferenceFailed,
    #[error("unsupported output")]
    UnsupportedOutput,
    #[error("onnxruntime error: {0}")]
    Ort(#[from] ort::Error),
}

pub struct InferenceSession {
    pub model_name: String,
    session: Session,
    pub input_var: String,
    pub output_var: String,
}

impl InferenceSession {
    pub fn new(
        model_path: impl AsRef<Path>,
        input_var: impl Into<String>,
        output_var: impl Into<String>,
    ) -> Result<Self, InferenceError> {
        let model_path_ref = model_path.as_ref();

        let session = Session::builder()
            .map_err(|_| InferenceError::SessionBuilderError)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|_| InferenceError::SessionBuilderError)?
            .with_intra_threads(1)
            .map_err(|_| InferenceError::SessionBuilderError)?
            .commit_from_file(model_path_ref)
            .map_err(|_| InferenceError::SessionBuilderError)?;

        Ok(Self {
            model_name: model_path_ref
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown_model")
                .to_string(),
            session,
            input_var: input_var.into(),
            output_var: output_var.into(),
        })
    }

    pub fn infer(&mut self, t: Tensor) -> Result<Tensor, InferenceError> {
        let dims_usize: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();

        let mut outputs = tensor_reduce!(t, |v: T| {
            let input = ort::value::Tensor::<T>::from_array((dims_usize, v))
                .map_err(|_| InferenceError::InferenceFailed)?;
            self.session
                .run(ort::inputs![&self.input_var => input])
                .map_err(|_| InferenceError::InferenceFailed)?
        });

        let dyn_val = outputs
            .remove(&self.output_var)
            .or_else(|| outputs.into_iter().next().map(|(_, v)| v))
            .ok_or(InferenceError::InferenceFailed)?;

        let out_dims: Vec<u32> = dyn_val
            .shape()
            .iter()
            .map(|&d| {
                if d < 0 {
                    Err(InferenceError::InferenceFailed)
                } else {
                    u32::try_from(d).map_err(|_| InferenceError::InferenceFailed)
                }
            })
            .collect::<Result<_, _>>()?;

        let mut strides = vec![0u32; out_dims.len()];
        let mut acc: u64 = 1;
        for i in (0..out_dims.len()).rev() {
            strides[i] = u32::try_from(acc).map_err(|_| InferenceError::InferenceFailed)?;
            acc = acc
                .checked_mul(out_dims[i] as u64)
                .ok_or(InferenceError::InferenceFailed)?;
        }

        let nelems_u64 = out_dims
            .iter()
            .try_fold(1u64, |p, &d| p.checked_mul(d as u64))
            .ok_or(InferenceError::InferenceFailed)?;
        let nelems = u32::try_from(nelems_u64).map_err(|_| InferenceError::InferenceFailed)?;

        let elem_buffer = match dyn_val.data_type() {
            TensorElementType::Float16 => TensorElemBuffer::F16(
                dyn_val
                    .downcast()
                    .map_err(|_| InferenceError::InferenceFailed)?
                    .extract_tensor()
                    .1
                    .into(),
            ),
            TensorElementType::Float32 => TensorElemBuffer::F32(
                dyn_val
                    .downcast()
                    .map_err(|_| InferenceError::InferenceFailed)?
                    .extract_tensor()
                    .1
                    .into(),
            ),
            TensorElementType::Float64 => TensorElemBuffer::F64(
                dyn_val
                    .downcast()
                    .map_err(|_| InferenceError::InferenceFailed)?
                    .extract_tensor()
                    .1
                    .into(),
            ),
            TensorElementType::Int32 => TensorElemBuffer::I32(
                dyn_val
                    .downcast()
                    .map_err(|_| InferenceError::InferenceFailed)?
                    .extract_tensor()
                    .1
                    .into(),
            ),
            TensorElementType::Int64 => TensorElemBuffer::I64(
                dyn_val
                    .downcast()
                    .map_err(|_| InferenceError::InferenceFailed)?
                    .extract_tensor()
                    .1
                    .into(),
            ),
            _ => return Err(InferenceError::UnsupportedOutput),
        };

        Ok(Tensor {
            dims: out_dims,
            strides,
            elem_buffer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_core::{Tensor as T, TensorElemType};

    #[test]
    fn test_sigmoid_constant_input_runs() -> Result<(), Box<dyn std::error::Error>> {
        let t = T::ones(vec![3, 4, 5], TensorElemType::F64)?;

        let mut sess = InferenceSession::new("models/sigmoid.onnx", "x", "y")?;
        let out = sess.infer(t.clone()).expect("inference should succeed");

        assert_eq!(out.dims, vec![3, 4, 5]);
        assert_eq!(out.strides, vec![20, 5, 1], "unexpected strides");

        let expected = 0.731_058_578_630_0049_f64;

        let TensorElemBuffer::F64(v) = out.elem_buffer else {
            panic!("unexpected inference output type");
        };

        for (i, &x) in v.iter().enumerate() {
            assert!(
                (x - expected).abs() < 1e-6,
                "elem {i}: got {x}, expected ~{expected}"
            );
        }

        Ok(())
    }
}
