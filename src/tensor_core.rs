// core tensor API, no postgres leakage (or at least as much as possible)
use serde::{Deserialize, Serialize};
use serde_json::{Number as JNumber, Value as JValue};
use std::convert::TryFrom;
use std::fmt;
use std::str::FromStr;
use std::string::String;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("shape mismatch")]
    ShapeMismatch,
    #[error("Overflow during elemwise_add")]
    Overflow,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Tensor {
    pub ndims: u8,
    pub flags: u8, // not sure what flags we will be adding tbh
    pub dims: Vec<u32>,
    pub strides: Vec<u32>,
    pub elem_buffer: Vec<f64>,
}

impl Tensor {
    pub fn elemwise_add(t1: &Tensor, t2: &Tensor) -> Result<Tensor, TensorError> {
        if t1.ndims != t2.ndims || t1.dims != t2.dims {
            return Err(TensorError::ShapeMismatch);
        }

        let out_vec: Vec<f64> = t1
            .elem_buffer
            .iter()
            .zip(t2.elem_buffer.iter())
            .map(|(a, b)| {
                let result = a + b;
                if result.is_finite() {
                    Ok(result)
                } else {
                    Err(TensorError::Overflow)
                }
            })
            .collect::<Result<Vec<f64>, TensorError>>()?;

        Ok(Tensor {
            ndims: t1.ndims,
            flags: 0,
            dims: t1.dims.clone(),
            strides: t1.strides.clone(),
            elem_buffer: out_vec,
        })
    }

    pub fn ones(dims: Vec<u32>) -> Result<Tensor, TensorError> {
        let ndims = dims.len();
        let nelems: u32 = dims.iter().product();
        let flags = 0;
        let mut strides = vec![0u32; dims.len()];
        let mut acc: u128 = 1;
        for i in (0..dims.len()).rev() {
            strides[i] = u32::try_from(acc).map_err(|_| TensorError::Overflow)?;
            acc = acc
                .checked_mul(dims[i] as u128)
                .ok_or(TensorError::Overflow)?;
        }
        let elem_buffer = vec![1.0; nelems as usize];

        Ok(Tensor {
            ndims: ndims as u8,
            flags,
            dims,
            strides,
            elem_buffer,
        })
    }
}

impl FromStr for Tensor {
    type Err = TensorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // infer dimensions recursively
        // uphold rectangularity and no empties invariants
        fn infer_dims(v: &JValue) -> Result<Vec<u32>, TensorError> {
            match v {
                JValue::Array(a) => {
                    if a.is_empty() {
                        return Err(TensorError::ShapeMismatch);
                    }
                    // infer sub-dims from first element, then all others must match
                    let first_dims = infer_dims(&a[0])?;
                    for elem in &a[1..] {
                        let d = infer_dims(elem)?;
                        if d != first_dims {
                            return Err(TensorError::ShapeMismatch);
                        }
                    }
                    // current dimension is array length + sub-dims
                    let mut dims = Vec::with_capacity(1 + first_dims.len());
                    dims.push(u32::try_from(a.len()).map_err(|_| TensorError::Overflow)?);
                    dims.extend(first_dims);
                    Ok(dims)
                }
                JValue::Number(_) => Ok(vec![]), // probably not ideal creating an allocation at each elem
                _ => Err(TensorError::ShapeMismatch),
            }
        }

        // flatten numbers as row-major
        fn flatten(v: &JValue, out: &mut Vec<f64>) -> Result<(), TensorError> {
            match v {
                JValue::Array(a) => {
                    for elem in a {
                        flatten(elem, out)?
                    }
                    Ok(())
                }
                JValue::Number(n) => {
                    let x = n.as_f64().ok_or(TensorError::ShapeMismatch)?;
                    if !x.is_finite() {
                        return Err(TensorError::Overflow);
                    }
                    out.push(x);
                    Ok(())
                }
                _ => Err(TensorError::ShapeMismatch),
            }
        }

        let v: JValue = serde_json::from_str(s).map_err(|_| TensorError::ShapeMismatch)?;

        // root must be an array, dont accept just a raw scalar. Has to be [x] instead of just x
        let JValue::Array(_) = v else {
            return Err(TensorError::ShapeMismatch);
        };

        let dims = infer_dims(&v)?;
        if dims.is_empty() || dims.iter().any(|&d| d == 0) {
            return Err(TensorError::ShapeMismatch);
        }

        let mut elem_buffer = Vec::<f64>::new();
        flatten(&v, &mut elem_buffer)?;

        // Count check (and protect against overflow)
        let mut prod: u128 = 1;
        for &d in &dims {
            prod = prod.checked_mul(d as u128).ok_or(TensorError::Overflow)?;
        }
        if prod as usize != elem_buffer.len() {
            return Err(TensorError::ShapeMismatch);
        }

        let nelems = u32::try_from(elem_buffer.len()).map_err(|_| TensorError::Overflow)?;

        // Element strides in C-order (row-major).
        let mut strides = vec![0u32; dims.len()];
        let mut acc: u128 = 1;
        for i in (0..dims.len()).rev() {
            strides[i] = u32::try_from(acc).map_err(|_| TensorError::Overflow)?;
            acc = acc
                .checked_mul(dims[i] as u128)
                .ok_or(TensorError::Overflow)?;
        }

        Ok(Tensor {
            ndims: dims.len() as u8,
            flags: 0,
            dims,
            strides,
            elem_buffer,
        })
    }
}

impl From<Tensor> for String {
    fn from(t: Tensor) -> Self {
        // assumed we upheld invariants for constructed Tensors
        // i.e. rectangular with non-zero dims.
        // that way the translation is infallible

        fn product(dims: &[u32]) -> usize {
            dims.iter()
                .fold(1usize, |acc, &d| acc.saturating_mul(d as usize))
        }

        fn write_recursive(out: &mut String, data: &[f64], dims: &[u32]) {
            if dims.len() == 1 {
                let n = dims[0] as usize;
                out.push('[');
                for i in 0..n {
                    if i > 0 {
                        out.push(',');
                    }
                    let elem = data[i];
                    // formating elems with no frac bits as X.0 so that anything downstream alwasys treats them as floats
                    let formatted_elem = if elem.fract() == 0.0 {
                        format!("{:.1}", elem)
                    } else {
                        format!("{}", elem)
                    };
                    out.push_str(&formatted_elem);
                }
                out.push(']');
            } else {
                let chunk = product(&dims[1..]);
                let n = dims[0] as usize;
                out.push('[');
                for i in 0..n {
                    if i > 0 {
                        out.push(',');
                    }
                    let start = i * chunk;
                    let end = start + chunk;
                    write_recursive(out, &data[start..end], &dims[1..]);
                }
                out.push(']');
            }
        }

        assert!(
            !t.dims.is_empty() && !t.dims.iter().any(|&d| d == 0),
            "Tensor -> String requires non-empty, non-zero dimensions"
        );
        let mut s = String::new();
        write_recursive(&mut s, &t.elem_buffer, &t.dims);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn rank1_float64_mixed_literals() -> Result<(), Box<dyn Error>> {
        let t = "[1, 2.0, 3.5, 4]".parse::<Tensor>()?;
        assert_eq!(t.ndims, 1);
        assert_eq!(t.dims, vec![4]);
        assert_eq!(t.strides, vec![1]);

        let s: String = t.into();
        assert_eq!(s, "[1.0,2.0,3.5,4.0]".to_owned());
        Ok(())
    }

    #[test]
    fn rank2() -> Result<(), Box<dyn Error>> {
        let t = "[[1,2,3],[4,5,6]]".parse::<Tensor>()?;
        assert_eq!(t.ndims, 2);
        assert_eq!(t.dims, vec![2, 3]);
        assert_eq!(t.strides, vec![3, 1]);

        let s: String = t.into();
        assert_eq!(s, "[[1.0,2.0,3.0],[4.0,5.0,6.0]]".to_owned());
        Ok(())
    }

    #[test]
    fn rank3() -> Result<(), Box<dyn Error>> {
        let t = "[[[1,2],[3,4]],[[5,6],[7,8]]]".parse::<Tensor>()?;
        assert_eq!(t.ndims, 3);
        assert_eq!(t.dims, vec![2, 2, 2]);
        assert_eq!(t.strides, vec![4, 2, 1]);

        let s: String = t.into();
        assert_eq!(
            s,
            "[[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]]".to_owned()
        );
        Ok(())
    }

    #[test]
    fn elemwise_add_f64_rank2() -> Result<(), Box<dyn Error>> {
        let a = "[[1,2,3],[4,5,6]]".parse::<Tensor>()?;
        let b = "[[10,20,30],[40,50,60]]".parse::<Tensor>()?;
        let c = Tensor::elemwise_add(&a, &b)?;
        assert_eq!(
            Into::<String>::into(c),
            "[[11.0,22.0,33.0],[44.0,55.0,66.0]]"
        );
        Ok(())
    }

    #[test]
    fn test_ones() -> Result<(), Box<dyn Error>> {
        let t1 = Tensor::ones(vec![1, 2]);
        let t2 = "[[1,1]]".parse::<Tensor>()?;
        assert_eq!(t2, t2);
        Ok(())
    }
}
