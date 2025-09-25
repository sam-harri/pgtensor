use crate::tensor_core::Tensor;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::{path::{Path, PathBuf}, fmt};
use thiserror::Error;

#[derive(Debug, Copy, Clone)]
pub enum SessionBuildStep {
    Builder,
    OptLevel,
    IntraThreads,
    Commit,
}

impl fmt::Display for SessionBuildStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionBuildStep::Builder => f.write_str("builder()"),
            SessionBuildStep::OptLevel => f.write_str("with_optimization_level(Level3)"),
            SessionBuildStep::IntraThreads => f.write_str("with_intra_threads(1)"),
            SessionBuildStep::Commit => f.write_str("commit_from_file"),
        }
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InferenceError {
    // Model/session init
    #[error("model path does not exist: {0}")]
    InvalidModelPath(PathBuf),

    #[error("failed to build session at step {step}")]
    SessionBuild {
        step: SessionBuildStep,
        #[source]
        source: ort::Error,
    },

    // Running
    #[error("failed to run inference")]
    Run {
        #[source]
        source: ort::Error,
    },

    // Inputs
    #[error("input shape/data length mismatch: dims {dims:?} => expected {expected} elements, but buffer has {actual}")]
    InputDataLenMismatch {
        dims: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    // Outputs
    #[error("requested output '{name}' not found")]
    MissingOutput {
        name: String,
    },

    #[error("unexpected output type; expected f64 tensor")]
    OutputTypeMismatch,

    #[error("model produced a negative dimension at axis {axis}: {value}")]
    NegativeOutputDim {
        axis: usize,
        value: i64,
    },

    #[error("output dimension does not fit in u32 at axis {axis}: {value}")]
    OutputDimOverflow {
        axis: usize,
        value: i64,
    },

    #[error("stride/product overflow while computing strides for dims {dims:?}")]
    StrideOverflow {
        dims: Vec<u32>,
    },

    #[error("element count overflow for dims {dims:?}")]
    ElemCountOverflow {
        dims: Vec<u32>,
    },

    #[error("output data length mismatch: expected {expected}, got {actual}")]
    OutputLenMismatch {
        expected: usize,
        actual: usize,
    },

    // Raw passthrough (kept for convenience)
    #[error("onnxruntime error")]
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

        if !model_path_ref.exists() {
            return Err(InferenceError::InvalidModelPath(model_path_ref.to_path_buf()));
        }

        let builder = Session::builder()
            .map_err(|e| InferenceError::SessionBuild { step: SessionBuildStep::Builder, source: e })?;

        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferenceError::SessionBuild { step: SessionBuildStep::OptLevel, source: e })?;

        let builder = builder
            .with_intra_threads(1)
            .map_err(|e| InferenceError::SessionBuild { step: SessionBuildStep::IntraThreads, source: e })?;

        let session = builder
            .commit_from_file(model_path_ref)
            .map_err(|e| InferenceError::SessionBuild { step: SessionBuildStep::Commit, source: e })?;

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
        // Pre-check input shape vs buffer length for clearer errors
        let dims_usize: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
        let expected_len = dims_usize.iter().copied().fold(1usize, |acc, d| acc.saturating_mul(d));
        let actual_len = t.elem_buffer.len();
        if expected_len != actual_len {
            return Err(InferenceError::InputDataLenMismatch {
                dims: dims_usize.clone(),
                expected: expected_len,
                actual: actual_len,
            });
        }

        let input = ort::value::Tensor::<f64>::from_array((dims_usize, t.elem_buffer))
            .map_err(InferenceError::Ort)?;

        let mut outputs = self
            .session
            .run(ort::inputs![&self.input_var => input])
            .map_err(|e| InferenceError::Run { source: e })?;


        let dyn_val = outputs
            .remove(&self.output_var)
            .or_else(|| outputs.into_iter().next().map(|(_, v)| v))
            .ok_or_else(|| InferenceError::MissingOutput {
                name: self.output_var.clone()
            })?;

        let out_t: ort::value::Tensor<f64> = dyn_val
            .downcast()
            .map_err(|_| InferenceError::OutputTypeMismatch)?;
        let (shape, data_f64) = out_t.extract_tensor(); // shape: Vec<i64>, data: Vec<f64>

        // Validate/convert dims
        let mut out_dims: Vec<u32> = Vec::with_capacity(shape.len());
        for (axis, &d) in shape.iter().enumerate() {
            if d < 0 {
                return Err(InferenceError::NegativeOutputDim { axis, value: d });
            }
            let du = d as u64;
            if du > u32::MAX as u64 {
                return Err(InferenceError::OutputDimOverflow { axis, value: d });
            }
            out_dims.push(du as u32);
        }

        // Compute strides with checked math
        let mut strides = vec![0u32; out_dims.len()];
        let mut acc: u64 = 1;
        for i in (0..out_dims.len()).rev() {
            strides[i] = u32::try_from(acc).map_err(|_| InferenceError::StrideOverflow {
                dims: out_dims.clone(),
            })?;
            acc = acc
                .checked_mul(out_dims[i] as u64)
                .ok_or_else(|| InferenceError::StrideOverflow {
                    dims: out_dims.clone(),
                })?;
        }

        let nelems_u64 = out_dims
            .iter()
            .try_fold(1u64, |p, &d| p.checked_mul(d as u64))
            .ok_or_else(|| InferenceError::ElemCountOverflow {
                dims: out_dims.clone(),
            })?;
        let nelems = u32::try_from(nelems_u64).map_err(|_| InferenceError::ElemCountOverflow {
            dims: out_dims.clone(),
        })?;

        if data_f64.len() != nelems as usize {
            return Err(InferenceError::OutputLenMismatch {
                expected: nelems as usize,
                actual: data_f64.len(),
            });
        }

        Ok(Tensor {
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
