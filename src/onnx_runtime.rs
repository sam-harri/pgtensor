use crate::tensor_core::Tensor as TCTensor;
use ort::session::{builder::GraphOptimizationLevel, Session};
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


pub fn infer(t: TCTensor) -> Result<TCTensor, InferenceError> {
    let mut session = Session::builder()
        .map_err(|_| InferenceError::SessionBuilderError)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|_| InferenceError::SessionBuilderError)?
        .with_intra_threads(1)
        .map_err(|_| InferenceError::SessionBuilderError)?
        .commit_from_file("models/f64_sigmoid.onnx")
        .map_err(|_| InferenceError::SessionBuilderError)?;

    let dims_usize: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();

    let input = ort::value::Tensor::<f64>::from_array((dims_usize, t.elem_buffer))
        .map_err(|_| InferenceError::InferenceFailed)?;

    let mut outputs = session
        .run(ort::inputs!["x" => input])
        .map_err(|_| InferenceError::InferenceFailed)?;
    let dyn_val = outputs
        .remove("y")
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

    Ok(TCTensor {
        ndims: out_dims.len() as u8,
        flags: 0,
        nelems: (out_dims[0] * out_dims[1] * out_dims[2]) as u32,
        dims: out_dims,
        strides,
        elem_buffer: data_f64.iter().map(|&v| v as f64).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_core::Tensor as T;

    #[test]
    fn test_sigmoid_constant_input_runs() -> Result<(), Box<dyn std::error::Error>> {
        let t = T::ones(vec![3, 4, 5])?;

        let out = infer(t.clone()).expect("inference should succeed");

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
