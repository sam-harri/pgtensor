use pgrx::datum::numeric_support::error;
// core tensor API, no postgres leakage (or at least as much as possible)
use bytemuck::{try_cast_slice, Pod};
use serde::{Deserialize, Serialize};
use serde_json::{Number as JNumber, Value as JValue};
use std::convert::TryFrom;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataTypeError {
    #[error("Invalid dtype code in typmod")]
    InvalidDtype,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[repr(u8)] // force C repr so it can fit in a u8
pub enum DataType {
    Float64 = 0x0,
    Float32 = 0x1,
    Int64 = 0x2,
    Int32 = 0x3,
}

impl DataType {
    pub const fn size_of(self) -> usize {
        match self {
            DataType::Float64 | DataType::Int64 => 8,
            DataType::Float32 | DataType::Int32 => 4,
        }
    }
}

impl TryFrom<u8> for DataType {
    type Error = DataTypeError;
    fn try_from(code: u8) -> Result<Self, DataTypeError> {
        match code {
            0x0 => Ok(DataType::Float64),
            0x1 => Ok(DataType::Float32),
            0x2 => Ok(DataType::Int64),
            0x3 => Ok(DataType::Int32),
            _ => Err(DataTypeError::InvalidDtype),
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Float64 => f.write_str("f64"),
            DataType::Float32 => f.write_str("f32"),
            DataType::Int64 => f.write_str("i64"),
            DataType::Int32 => f.write_str("i32"),
        }
    }
}

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("dtype mismatch")]
    DTypeMismatch,
    #[error("shape mismatch")]
    ShapeMismatch,
    #[error("nelems mismatch")]
    ElemCountMismatch,
    #[error("buffer length/dtype/nelems mismatch")]
    BufferSizeMismatch,
    #[error("int32 overflow during elemwise_add")]
    Int32Overflow,
    #[error("int64 overflow during elemwise_add")]
    Int64Overflow,
    #[error("pod cast error: {0}")]
    PodCastError(String),
    #[error("to_literal only supports Float64 tensors right now")]
    UnsupportedDType,
    #[error("found non-finite f64 (NaN or ±inf) at element {index}")]
    NonFiniteF64 { index: usize },
    #[error("json serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tensor {
    // fixed sized
    // when we migrated to varlena, this will be the header
    pub ndims: u8,
    pub dtype: DataType,
    pub flags: u8,
    pub nelems: u32,
    // then everything below this will be in the allocated buffer
    pub dims: Vec<u32>,       // sizeof : ndims * 4
    pub strides: Vec<u32>,    // sizeof : ndims * 4
    pub elem_buffer: Vec<u8>, // sizeof: nelems * dtype.sizeof()
}

impl Tensor {
    pub fn elemwise_add(t1: &Tensor, t2: &Tensor) -> Result<Tensor, TensorError> {
        if t1.dtype != t2.dtype {
            return Err(TensorError::DTypeMismatch);
        }
        if t1.ndims != t2.ndims || t1.dims != t2.dims {
            return Err(TensorError::ShapeMismatch);
        }
        if t1.nelems != t2.nelems {
            return Err(TensorError::ElemCountMismatch);
        }

        let n = t1.nelems as usize;
        let es = t1.dtype.size_of();
        if t1.elem_buffer.len() != n * es || t2.elem_buffer.len() != n * es {
            return Err(TensorError::BufferSizeMismatch);
        }

        let out_bytes = match t1.dtype {
            DataType::Float64 => {
                let a: &[f64] = try_cast_slice(&t1.elem_buffer)
                    .map_err(|e| TensorError::PodCastError(e.to_string()))?;

                let b: &[f64] = try_cast_slice(&t2.elem_buffer)
                    .map_err(|e| TensorError::PodCastError(e.to_string()))?;
                let mut out: Vec<f64> = Vec::with_capacity(a.len());
                out.extend(a.iter().zip(b).map(|(&x, &y)| x + y));
                vec_to_bytes(out)
            }
            DataType::Float32 => {
                let a: &[f32] = try_cast_slice(&t1.elem_buffer)
                    .map_err(|e| TensorError::PodCastError(e.to_string()))?;
                let b: &[f32] = try_cast_slice(&t2.elem_buffer)
                    .map_err(|e| TensorError::PodCastError(e.to_string()))?;
                let mut out: Vec<f32> = Vec::with_capacity(a.len());
                out.extend(a.iter().zip(b).map(|(&x, &y)| x + y));
                vec_to_bytes(out)
            }
            DataType::Int64 => {
                let a: &[i64] = try_cast_slice(&t1.elem_buffer)
                    .map_err(|e| TensorError::PodCastError(e.to_string()))?;
                let b: &[i64] = try_cast_slice(&t2.elem_buffer)
                    .map_err(|e| TensorError::PodCastError(e.to_string()))?;
                let mut out: Vec<i64> = Vec::with_capacity(a.len());
                for (&x, &y) in a.iter().zip(b) {
                    out.push(x.checked_add(y).ok_or(TensorError::Int64Overflow)?);
                }
                vec_to_bytes(out)
            }
            DataType::Int32 => {
                let a: &[i32] = try_cast_slice(&t1.elem_buffer)
                    .map_err(|e| TensorError::PodCastError(e.to_string()))?;
                let b: &[i32] = try_cast_slice(&t2.elem_buffer)
                    .map_err(|e| TensorError::PodCastError(e.to_string()))?;
                let mut out: Vec<i32> = Vec::with_capacity(a.len());
                for (&x, &y) in a.iter().zip(b) {
                    out.push(x.checked_add(y).ok_or(TensorError::Int32Overflow)?);
                }
                vec_to_bytes(out)
            }
        };

        Ok(Tensor {
            ndims: t1.ndims,
            dtype: t1.dtype,
            flags: 0,
            nelems: t1.nelems,
            dims: t1.dims.clone(),
            strides: t1.strides.clone(),
            elem_buffer: out_bytes,
        })
    }

    pub fn to_literal(&self) -> Result<String, TensorError> {
        fn f64_at(buf: &[u8], idx: usize) -> f64 {
            let start = idx * 8;
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&buf[start..start + 8]);
            f64::from_le_bytes(bytes)
        }

        fn build_level(
            level: usize,
            dims: &[u32],
            dtype: DataType,
            buf: &[u8],
            next_idx: &mut usize,
        ) -> Result<JValue, TensorError> {
            if level == dims.len() {
                match dtype {
                    DataType::Float64 => {
                        let v = f64_at(buf, *next_idx);
                        let i = *next_idx;
                        *next_idx += 1;
                        let num = serde_json::Number::from_f64(v)
                            .ok_or(TensorError::NonFiniteF64 { index: i })?;
                        Ok(JValue::Number(num))
                    }
                    _ => Err(TensorError::UnsupportedDType),
                }
            } else {
                let len = dims[level] as usize;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(build_level(level + 1, dims, dtype, buf, next_idx)?);
                }
                Ok(JValue::Array(arr))
            }
        }

        // Only f64 currently supported
        if self.dtype != DataType::Float64 {
            return Err(TensorError::UnsupportedDType);
        }

        let mut idx = 0usize;
        let v = build_level(0, &self.dims, self.dtype, &self.elem_buffer, &mut idx)?;

        // (Optional) sanity check: consumed exactly nelems
        // if idx as u32 != tensor.nelems { return Err(TensorError::ElemCountMismatch); }

        serde_json::to_string(&v).map_err(TensorError::from)
    }

    pub fn from_literal(literal: &str, dtype: DataType) -> Result<Tensor, String> {
        // postgres string repr of multidim arrays is the same as JSON, so JSON deserialize into a JSON value
        let jv: JValue =
            serde_json::from_str(literal).map_err(|e| format!("json parse error: {e}"))?;

        let (tensor_shape, flattened_elements) = shape_and_flatten_numbers(&jv)?;

        // allocate u8 buffer with enough size of all the elements
        // then extend each elements into the vec
        let elem_size = dtype.size_of();
        let mut buffer = Vec::<u8>::with_capacity(flattened_elements.len() * elem_size);
        match dtype {
            DataType::Float64 => {
                for n in &flattened_elements {
                    buffer.extend_from_slice(&n.as_f64().unwrap().to_le_bytes());
                }
            }
            _ => return Err("only Float64 is supported here right now".into()),
        }

        // basically just turning the tensor_shape vec into u32
        // should probably just have `shape_and_flatten_numbers` return the proper type or stick to usize
        // but tbh idk whats more idiomatic (usize or u32)
        // like technically it's usize since its tracking in index but we'll clean up this messy code as we go
        let dims: Vec<u32> = tensor_shape
            .iter()
            .map(|&d| u32::try_from(d).map_err(|_| "dimension too large for u32".to_string()))
            .collect::<Result<_, _>>()?;

        // num elems is hte product of all the length of the dims
        let nelems = dims.iter().product();

        // for context stride is how far (in bytes) the next element is in any direction
        // you can calcualte it using a reversed running product with the accumulator set to the element size at the start
        let mut acc = elem_size as u32;
        let mut strides = Vec::with_capacity(dims.len());
        dims.iter().rev().for_each(|dim| {
            strides.push(acc);
            acc *= dim;
        });
        strides.reverse();

        Ok(Tensor {
            ndims: dims.len() as u8,
            dtype,
            flags: 0,
            nelems,
            dims,
            strides,
            elem_buffer: buffer,
        })
    }
}

/// Zero-copy reinterpret a Vec<T: Pod> as Vec<u8>.
/// Safety is valid because Pod guarantees no padding/invalid bit patterns,
/// and Vec<T> uses the same allocation we re-expose as bytes
fn vec_to_bytes<T: Pod>(mut v: Vec<T>) -> Vec<u8> {
    let len = v.len() * std::mem::size_of::<T>();
    let cap = v.capacity() * std::mem::size_of::<T>();
    let ptr = v.as_mut_ptr() as *mut u8;
    std::mem::forget(v);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

fn shape_and_flatten_numbers(v: &JValue) -> Result<(Vec<usize>, Vec<JNumber>), String> {
    match v {
        JValue::Array(arr) => {
            if arr.is_empty() {
                return Err("empty arrays are not supported".into());
            }
            let (sub_shape0, mut flat0) = shape_and_flatten_numbers(&arr[0])?;
            let mut flat = Vec::with_capacity(flat0.len() * arr.len());
            flat.append(&mut flat0);
            for item in &arr[1..] {
                let (sh_i, mut flat_i) = shape_and_flatten_numbers(item)?;
                if sh_i != sub_shape0 {
                    return Err("ragged arrays are not supported".into());
                }
                flat.append(&mut flat_i);
            }
            let mut shape = Vec::with_capacity(1 + sub_shape0.len());
            shape.push(arr.len());
            shape.extend(sub_shape0);
            Ok((shape, flat))
        }
        JValue::Number(n) => Ok((Vec::new(), vec![n.clone()])),
        _ => Err("expected arrays of numbers".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expect_f64_bytes(vals: &[f64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 8);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn rank1_float64_mixed_literals() {
        let t = Tensor::from_literal("[1, 2.0, 3.5, 4]", DataType::Float64).unwrap();
        assert_eq!(t.ndims, 1);
        assert_eq!(t.dims, vec![4]);
        assert_eq!(t.nelems, 4);
        assert_eq!(t.strides, vec![8]);
        assert_eq!(t.elem_buffer, expect_f64_bytes(&[1.0, 2.0, 3.5, 4.0]));

        let s = t.to_literal().unwrap();
        assert_eq!(s, "[1.0,2.0,3.5,4.0]".to_owned())
    }

    #[test]
    fn rank2_float64() {
        let t = Tensor::from_literal("[[1,2,3],[4,5,6]]", DataType::Float64).unwrap();
        assert_eq!(t.ndims, 2);
        assert_eq!(t.dims, vec![2, 3]);
        assert_eq!(t.nelems, 6);
        assert_eq!(t.strides, vec![24, 8]);
        assert_eq!(t.elem_buffer, expect_f64_bytes(&[1., 2., 3., 4., 5., 6.]));

        let s = t.to_literal().unwrap();
        assert_eq!(s, "[[1.0,2.0,3.0],[4.0,5.0,6.0]]".to_owned());
    }

    #[test]
    fn rank3_float64() {
        let t = Tensor::from_literal("[[[1,2],[3,4]],[[5,6],[7,8]]]", DataType::Float64).unwrap();
        assert_eq!(t.ndims, 3);
        assert_eq!(t.dims, vec![2, 2, 2]);
        assert_eq!(t.nelems, 8);
        assert_eq!(t.strides, vec![32, 16, 8]);
        assert_eq!(
            t.elem_buffer,
            expect_f64_bytes(&[1., 2., 3., 4., 5., 6., 7., 8.])
        );

        let s = t.to_literal().unwrap();
        assert_eq!(
            s,
            "[[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]]".to_owned()
        );
    }

    #[test]
    fn elemwise_add_f64_rank2() {
        let a = Tensor::from_literal("[[1,2,3],[4,5,6]]", DataType::Float64).unwrap();
        let b = Tensor::from_literal("[[10,20,30],[40,50,60]]", DataType::Float64).unwrap();
        let c = Tensor::elemwise_add(&a, &b).unwrap();
        assert_eq!(
            c.to_literal().unwrap(),
            "[[11.0,22.0,33.0],[44.0,55.0,66.0]]"
        );
    }
}
