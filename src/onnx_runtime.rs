use crate::tensor_core::Tensor as TCTensor;
use ort::session::{builder::GraphOptimizationLevel, Session};
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

    pub fn infer(&mut self, t: TCTensor) -> Result<TCTensor, InferenceError> {
        let dims_usize: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
        let input = ort::value::Tensor::<f64>::from_array((dims_usize, t.elem_buffer))
            .map_err(|_| InferenceError::InferenceFailed)?;

        let mut outputs = self
            .session
            .run(ort::inputs![&self.input_var => input])
            .map_err(|_| InferenceError::InferenceFailed)?;

        let dyn_val = outputs
            .remove(&self.output_var)
            .or_else(|| outputs.into_iter().next().map(|(_, v)| v))
            .ok_or(InferenceError::InferenceFailed)?;

        let out_t: ort::value::Tensor<f64> = dyn_val
            .downcast()
            .map_err(|_| InferenceError::InferenceFailed)?;
        let (shape, data_f64) = out_t.extract_tensor();

        let out_dims: Vec<u32> = shape
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

        Ok(TCTensor {
            ndims: out_dims.len() as u8,
            flags: 0,
            nelems,
            dims: out_dims,
            strides,
            elem_buffer: data_f64.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_core::Tensor as T;

    #[test]
    fn test_sigmoid_constant_input_runs() -> Result<(), Box<dyn std::error::Error>> {
        let t = T::ones(vec![3, 4, 5])?;

        let mut sess = InferenceSession::new("models/sigmoid.onnx", "x", "y")?;
        let out = sess.infer(t.clone()).expect("inference should succeed");

        assert_eq!(out.dims, vec![3, 4, 5]);
        assert_eq!(out.nelems, 60);
        assert_eq!(out.ndims, 3);
        assert_eq!(out.strides, vec![20, 5, 1], "unexpected strides");

        let expected = 0.731_058_578_630_0049_f64;
        for (i, &x) in out.elem_buffer.iter().enumerate() {
            assert!(
                (x - expected).abs() < 1e-6,
                "elem {i}: got {x}, expected ~{expected}"
            );
        }

        Ok(())
    }
}
