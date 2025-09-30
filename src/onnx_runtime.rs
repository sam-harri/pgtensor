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

    pub fn infer(&mut self, t: &Tensor) -> Result<Tensor, InferenceError> {
        let dims_usize: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();

        let input = ort::value::Tensor::<f64>::from_array((dims_usize, t.elem_buffer.clone()))
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
        let (shape, out_buffer) = out_t.extract_tensor();


        let mut out_dims: Vec<u32> = Vec::with_capacity(shape.len());
        for (axis, &d) in shape.iter().enumerate() {
            out_dims.push(d as u32);
        }

        let t = Tensor::from_dims_and_vec(out_dims, out_buffer.into()).unwrap();
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_core::Tensor as T;

    #[test]
    fn test_sigmoid_constant_input_runs() -> Result<(), Box<dyn std::error::Error>> {
        let mut t = T::ones(vec![3, 4, 5])?;

        let mut sess = InferenceSession::new("models/sigmoid.onnx", "x", "y")?;
        let out = sess.infer(&mut t).expect("inference should succeed");

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
