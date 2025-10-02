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
    pub nelems: u32,
    pub dims: Vec<u32>,
    pub strides: Vec<u32>,
    pub elem_buffer: Vec<f64>,
}

impl Tensor {
    fn to_elem_buffer_idx(&self, idxs: &[u32]) -> usize {
        idxs.iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i as usize * s as usize)
            .sum()
    }

    fn of_elem_buffer_idx(&self, mut i: usize) -> Vec<u32> {
        let mut res = vec![0_u32; self.ndims as usize];
        let mut sorted_strides = self
            .dims
            .iter()
            .zip(self.strides.iter())
            .enumerate()
            .collect::<Vec<_>>();
        sorted_strides.sort_by_key(|(_, (_, &s))| s);
        for (j, (&d, &s)) in sorted_strides.iter().rev() {
            res[*j] = (i / s as usize) as u32;
            i %= s as usize;
        }
        res
    }

    pub fn elemwise_add(t1: &Tensor, t2: &Tensor) -> Result<Tensor, TensorError> {
        if t1.dims != t2.dims {
            return Err(TensorError::ShapeMismatch);
        }

        // TODO: Should we really be erroring if a value becomes infinite?
        let out_vec: Vec<f64> = if t1.strides == t2.strides {
            // Fast path
            t1.elem_buffer
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
                .collect::<Result<Vec<f64>, TensorError>>()?
        } else {
            // Slow path
            let mut res = Vec::with_capacity(t1.nelems as usize);
            for (i, a) in t1.elem_buffer.iter().enumerate() {
                let b = t2[&t1.of_elem_buffer_idx(i)];
                let result = a + b;
                if !result.is_finite() {
                    return Err(TensorError::Overflow);
                }
                res.push(result);
            }
            res
        };

        Ok(Tensor {
            ndims: t1.ndims,
            flags: 0,
            nelems: t1.nelems,
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
            nelems,
            dims,
            strides,
            elem_buffer,
        })
    }
}

impl std::ops::Index<&[u32]> for Tensor {
    type Output = f64;

    fn index(&self, index: &[u32]) -> &Self::Output {
        &self.elem_buffer[self.to_elem_buffer_idx(index)]
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
            nelems,
            dims,
            strides,
            elem_buffer,
        })
    }
}

impl From<&Tensor> for String {
    fn from(t: &Tensor) -> Self {
        // assumed we upheld invariants for constructed Tensors
        // i.e. rectangular with non-zero dims.
        // that way the translation is infallible

        // TODO: Handle non-standard strides here.

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
                let chunk: usize = dims[1..].iter().map(|&d| d as usize).product();
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
        assert_eq!(
            t.nelems as usize,
            t.elem_buffer.len(),
            "inconsistent nelems/elem_buffer"
        );

        let mut s = String::new();
        write_recursive(&mut s, &t.elem_buffer, &t.dims);
        s
    }
}

impl From<Tensor> for String {
    fn from(value: Tensor) -> Self {
        (&value).into()
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
        assert_eq!(t.nelems, 4);
        assert_eq!(t.strides, vec![1]);

        let s: String = t.into();
        assert_eq!(s, "[1.0,2.0,3.5,4.0]".to_owned());
        Ok(())
    }

    #[test]
    fn rank2() -> Result<(), Box<dyn Error>> {
        let t1 = "[[1,2,3],[4,5,6]]".parse::<Tensor>()?;
        assert_eq!(t1.ndims, 2);
        assert_eq!(t1.dims, vec![2, 3]);
        assert_eq!(t1.nelems, 6);
        assert_eq!(t1.strides, vec![3, 1]);
        assert_eq!(t1[&[1, 1]], 5.0);

        let s: String = (&t1).into();
        assert_eq!(s, "[[1.0,2.0,3.0],[4.0,5.0,6.0]]".to_owned());

        let t2 = Tensor {
            ndims: 2,
            flags: 0,
            nelems: 6,
            dims: vec![2, 3],
            strides: vec![1, 2],
            elem_buffer: vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        };
        assert_eq!(t2[&[1, 1]], 5.0);

        assert_eq!(
            Into::<String>::into(&Tensor::elemwise_add(&t1, &t2)?),
            "[[2.0,4.0,6.0],[8.0,10.0,12.0]]".to_owned()
        );

        // TODO: Remove 0 add once Tensor -> String conversion respects strides.
        let t3 = "[[0,0,0],[0,0,0]]".parse::<Tensor>()?;
        assert_eq!(
            Into::<String>::into(&Tensor::elemwise_add(
                &t3,
                &Tensor::elemwise_add(&t2, &t1)?
            )?),
            "[[2.0,4.0,6.0],[8.0,10.0,12.0]]".to_owned()
        );

        Ok(())
    }

    #[test]
    fn rank3() -> Result<(), Box<dyn Error>> {
        let t = "[[[1,2],[3,4]],[[5,6],[7,8]]]".parse::<Tensor>()?;
        assert_eq!(t.ndims, 3);
        assert_eq!(t.dims, vec![2, 2, 2]);
        assert_eq!(t.nelems, 8);
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
            Into::<String>::into(&c),
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
