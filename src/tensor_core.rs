// core tensor API, no postgres leakage (or at least as much as possible)
use half::f16;
use num_traits::{Float, Num, One, Zero};
use paste::paste;
use serde::{Deserialize, Serialize};
use serde_json::{Number as JNumber, Value as JValue};
use std::convert::TryFrom;
use std::fmt::{self, Display, Write};
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;
use std::string::String;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("shape mismatch")]
    ShapeMismatch,
    #[error("element type mismatch")]
    TypeMismatch,
    #[error("overflow")]
    Overflow,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum TensorElemBuffer {
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

macro_rules! tensor_reduce {
    ($t:expr, |$v:ident| $b:expr) => {
        match &$t.elem_buffer {
            TensorElemBuffer::F16($v) => $b,
            TensorElemBuffer::F32($v) => $b,
            TensorElemBuffer::F64($v) => $b,
            TensorElemBuffer::I32($v) => $b,
            TensorElemBuffer::I64($v) => $b,
        }
    };
    ($t:expr, |$v:ident: $ty:ident| $b:expr) => {
        match $t.elem_buffer {
            TensorElemBuffer::F16($v) => {
                type $ty = half::f16;
                $b
            }
            TensorElemBuffer::F32($v) => {
                type $ty = f32;
                $b
            }
            TensorElemBuffer::F64($v) => {
                type $ty = f64;
                $b
            }
            TensorElemBuffer::I32($v) => {
                type $ty = i32;
                $b
            }
            TensorElemBuffer::I64($v) => {
                type $ty = i64;
                $b
            }
        }
    };
    ($t:expr, { F($vf:ident) => $bf:expr, I($vi:ident) => $bi:expr, }) => {
        match &$t.elem_buffer {
            TensorElemBuffer::F16($vf) => $bf,
            TensorElemBuffer::F32($vf) => $bf,
            TensorElemBuffer::F64($vf) => $bf,
            TensorElemBuffer::I32($vi) => $bi,
            TensorElemBuffer::I64($vi) => $bi,
        }
    };
}
pub(crate) use tensor_reduce;

macro_rules! tensor_map {
    ($t:expr, |$v:ident| $b:expr) => {
        match $t.elem_buffer {
            TensorElemBuffer::F16($v) => TensorElemBuffer::F16($b),
            TensorElemBuffer::F32($v) => TensorElemBuffer::F32($b),
            TensorElemBuffer::F64($v) => TensorElemBuffer::F64($b),
            TensorElemBuffer::I32($v) => TensorElemBuffer::I32($b),
            TensorElemBuffer::I64($v) => TensorElemBuffer::I64($b),
        }
    };
    ($t:expr, { F($vf:ident) => $bf:expr, I($vi:ident) => $bi:expr, }) => {
        match $t.elem_buffer {
            TensorElemBuffer::F16($vf) => TensorElemBuffer::F16($bf),
            TensorElemBuffer::F32($vf) => TensorElemBuffer::F32($bf),
            TensorElemBuffer::F64($vf) => TensorElemBuffer::F64($bf),
            TensorElemBuffer::I32($vi) => TensorElemBuffer::I32($bi),
            TensorElemBuffer::I64($vi) => TensorElemBuffer::I64($bi),
        }
    };
    ($t1:expr, $t2:expr, |$v1:ident, $v2:ident| $b:expr) => {
        match ($t1.elem_buffer, $t2.elem_buffer) {
            (TensorElemBuffer::F16($v1), TensorElemBuffer::F16($v2)) => TensorElemBuffer::F16($b),
            (TensorElemBuffer::F32($v1), TensorElemBuffer::F32($v2)) => TensorElemBuffer::F32($b),
            (TensorElemBuffer::F64($v1), TensorElemBuffer::F64($v2)) => TensorElemBuffer::F64($b),
            (TensorElemBuffer::I32($v1), TensorElemBuffer::I32($v2)) => TensorElemBuffer::I32($b),
            (TensorElemBuffer::I64($v1), TensorElemBuffer::I64($v2)) => TensorElemBuffer::I64($b),
            _ => return Err(TensorError::TypeMismatch),
        }
    };
}
pub(crate) use tensor_map;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorElemType {
    F16,
    F32,
    F64,
    I32,
    I64,
}
macro_rules! tensor_elemtype_to_buffer {
    ($ty:expr, $b:expr) => {
        match $ty {
            TensorElemType::F16 => TensorElemBuffer::F16($b),
            TensorElemType::F32 => TensorElemBuffer::F32($b),
            TensorElemType::F64 => TensorElemBuffer::F64($b),
            TensorElemType::I32 => TensorElemBuffer::I32($b),
            TensorElemType::I64 => TensorElemBuffer::I64($b),
        }
    };
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Tensor {
    pub dims: Vec<u32>,
    pub strides: Vec<u32>,
    pub elem_buffer: TensorElemBuffer,
}

macro_rules! tensor_elemwise_op {
    ($i:ident, $f:expr) => {
        paste! {
            pub fn [<elemwise_ $i>](t1: Tensor, t2: Tensor) -> Result<Tensor, TensorError> {
                if t1.dims != t2.dims {
                    return Err(TensorError::ShapeMismatch);
                }

                let elem_buffer = tensor_map!(t1, t2, |v1, v2| {
                    v1.into_iter().zip(v2).map(|(e1, e2)| $f(e1, e2)).collect()
                });

                Ok(Tensor {
                    elem_buffer,
                    ..t1
                })
            }
        }
    };
}

impl Tensor {
    pub fn len(&self) -> usize {
        tensor_reduce!(self, |v| v.len())
    }

    tensor_elemwise_op!(add, Add::add);
    tensor_elemwise_op!(sub, Sub::sub);
    tensor_elemwise_op!(mul, Mul::mul);
    tensor_elemwise_op!(div, Div::div);

    pub fn ones(dims: Vec<u32>, ty: TensorElemType) -> Result<Tensor, TensorError> {
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

        let elem_buffer = tensor_elemtype_to_buffer!(ty, vec![One::one(); nelems as usize]);

        Ok(Tensor {
            dims,
            strides,
            elem_buffer,
        })
    }

    pub fn exp(t: Tensor) -> Result<Tensor, TensorError> {
        let elem_buffer = tensor_map!(t, {
            F(v) => v.into_iter().map(|e| e.exp()).collect(),
            I(v) => return Err(TensorError::TypeMismatch),
        });

        Ok(Tensor { elem_buffer, ..t })
    }

    pub fn ln(t: Tensor) -> Result<Tensor, TensorError> {
        let elem_buffer = tensor_map!(t, {
            F(v) => v.into_iter().map(|e| e.ln()).collect(),
            I(v) => return Err(TensorError::TypeMismatch),
        });

        Ok(Tensor { elem_buffer, ..t })
    }

    pub fn powf(t: Tensor, exp: f32) -> Result<Tensor, TensorError> {
        let elem_buffer = match t.elem_buffer {
            TensorElemBuffer::F16(v) => {
                TensorElemBuffer::F16(v.into_iter().map(|e| e.powf(f16::from_f32(exp))).collect())
            }
            TensorElemBuffer::F32(v) => {
                TensorElemBuffer::F32(v.into_iter().map(|e| e.powf(exp)).collect())
            }
            TensorElemBuffer::F64(v) => {
                TensorElemBuffer::F64(v.into_iter().map(|e| e.powf(exp.into())).collect())
            }
            TensorElemBuffer::I32(_) | TensorElemBuffer::I64(_) => {
                return Err(TensorError::TypeMismatch)
            }
        };

        Ok(Tensor { elem_buffer, ..t })
    }

    pub fn powi(t: Tensor, exp: i32) -> Result<Tensor, TensorError> {
        let elem_buffer = tensor_map!(t, {
            F(v) =>  v.into_iter().map(|e| e.powi(exp)).collect() ,
            I(v) => {
                let exp: u32 = exp.try_into().map_err(|_| TensorError::Overflow)?;
                v.into_iter().map(|e| e.pow(exp)).collect()
            },
        });

        Ok(Tensor { elem_buffer, ..t })
    }

    pub fn dotf(t1: Tensor, t2: Tensor) -> Result<f64, TensorError> {
        match (t1.elem_buffer, t2.elem_buffer) {
            (TensorElemBuffer::F16(v1), TensorElemBuffer::F16(v2)) => Ok(v1
                .into_iter()
                .zip(v2)
                .map(|(e1, e2)| e1.to_f64() * e2.to_f64())
                .sum()),
            (TensorElemBuffer::F32(v1), TensorElemBuffer::F32(v2)) => Ok(v1
                .into_iter()
                .zip(v2)
                .map(|(e1, e2)| e1 as f64 * e2 as f64)
                .sum()),
            (TensorElemBuffer::F64(v1), TensorElemBuffer::F64(v2)) => {
                Ok(v1.into_iter().zip(v2).map(|(e1, e2)| e1 * e2).sum())
            }
            _ => Err(TensorError::TypeMismatch),
        }
    }

    pub fn doti(t1: Tensor, t2: Tensor) -> Result<i64, TensorError> {
        match (t1.elem_buffer, t2.elem_buffer) {
            (TensorElemBuffer::I32(v1), TensorElemBuffer::I32(v2)) => Ok(v1
                .into_iter()
                .zip(v2)
                .map(|(e1, e2)| e1 as i64 * e2 as i64)
                .sum()),
            (TensorElemBuffer::I64(v1), TensorElemBuffer::I64(v2)) => {
                Ok(v1.into_iter().zip(v2).map(|(e1, e2)| e1 * e2).sum())
            }
            _ => Err(TensorError::TypeMismatch),
        }
    }

    pub fn matvec(m: Tensor, v: Tensor) -> Result<Tensor, TensorError> {
        if m.dims.len() != 2 && v.dims.len() != 1 && m.dims[1] != v.dims[0] {
            return Err(TensorError::ShapeMismatch);
        }

        let elem_buffer = tensor_map!(m, v, |mb, vb| {
            let mut res = Vec::with_capacity(m.dims[0] as usize);
            for y in 0..m.dims[0] as usize {
                let mut e = Zero::zero();
                for x in 0..m.dims[1] as usize {
                    e += mb[m.strides[0] as usize * y + m.strides[1] as usize * x]
                        * vb[v.strides[0] as usize * x];
                }
                res.push(e);
            }
            res
        });

        Ok(Tensor {
            dims: vec![m.dims[0]],
            strides: vec![1],
            elem_buffer,
        })
    }

    pub fn vecmat(v: Tensor, m: Tensor) -> Result<Tensor, TensorError> {
        if v.dims.len() != 1 && m.dims.len() != 2 && v.dims[0] != m.dims[0] {
            return Err(TensorError::ShapeMismatch);
        }

        let elem_buffer = tensor_map!(m, v, |mb, vb| {
            let mut res = Vec::with_capacity(m.dims[1] as usize);
            for x in 0..m.dims[1] as usize {
                let mut e = Zero::zero();
                for y in 0..m.dims[0] as usize {
                    e += mb[m.strides[0] as usize * y + m.strides[1] as usize * x]
                        * vb[v.strides[0] as usize * y];
                }
                res.push(e);
            }
            res
        });

        Ok(Tensor {
            dims: vec![m.dims[1]],
            strides: vec![1],
            elem_buffer,
        })
    }

    pub fn matmul(t1: Tensor, t2: Tensor) -> Result<Tensor, TensorError> {
        if t1.dims.len() != 2 && t2.dims.len() != 2 && t1.dims[1] != t2.dims[0] {
            return Err(TensorError::ShapeMismatch);
        }

        let elem_buffer = tensor_map!(t1, t2, |v1, v2| {
            let mut res = Vec::with_capacity(t1.dims[0] as usize * t2.dims[1] as usize);
            for x in 0..t1.dims[0] as usize {
                for y in 0..t2.dims[1] as usize {
                    let mut e = Zero::zero();
                    for z in 0..t1.dims[1] as usize {
                        e += v1[t1.strides[0] as usize * x + t1.strides[1] as usize * z]
                            * v2[t2.strides[0] as usize * z + t2.strides[1] as usize * y];
                    }
                    res.push(e);
                }
            }
            res
        });

        Ok(Tensor {
            dims: vec![t1.dims[0], t2.dims[1]],
            strides: vec![t2.dims[1], 1],
            elem_buffer,
        })
    }
}

trait FromJNumber: Sized {
    fn from(n: &JNumber) -> Result<Self, TensorError>;
}

impl FromJNumber for f16 {
    fn from(n: &JNumber) -> Result<Self, TensorError> {
        let x = f16::from_f64(n.as_f64().ok_or(TensorError::ShapeMismatch)?);
        if !x.is_finite() {
            return Err(TensorError::Overflow);
        }
        Ok(x)
    }
}

impl FromJNumber for f32 {
    fn from(n: &JNumber) -> Result<Self, TensorError> {
        let x = n.as_f64().ok_or(TensorError::ShapeMismatch)? as f32;
        if !x.is_finite() {
            return Err(TensorError::Overflow);
        }
        Ok(x)
    }
}

impl FromJNumber for f64 {
    fn from(n: &JNumber) -> Result<Self, TensorError> {
        let x = n.as_f64().ok_or(TensorError::ShapeMismatch)?;
        if !x.is_finite() {
            return Err(TensorError::Overflow);
        }
        Ok(x)
    }
}

impl FromJNumber for i32 {
    fn from(n: &JNumber) -> Result<Self, TensorError> {
        n.as_i64()
            .ok_or(TensorError::ShapeMismatch)?
            .try_into()
            .map_err(|_| TensorError::Overflow)
    }
}

impl FromJNumber for i64 {
    fn from(n: &JNumber) -> Result<Self, TensorError> {
        n.as_i64().ok_or(TensorError::ShapeMismatch)
    }
}

impl FromStr for Tensor {
    type Err = TensorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (s, ty) = if let Some(s) = s.strip_suffix("::f16") {
            (s, TensorElemType::F16)
        } else if let Some(s) = s.strip_suffix("::f32") {
            (s, TensorElemType::F32)
        } else if let Some(s) = s.strip_suffix("::f64") {
            (s, TensorElemType::F64)
        } else if let Some(s) = s.strip_suffix("::i32") {
            (s, TensorElemType::I32)
        } else if let Some(s) = s.strip_suffix("::i64") {
            (s, TensorElemType::I64)
        } else {
            (s, TensorElemType::F64)
        };

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
        fn flatten<T: FromJNumber>(v: &JValue, out: &mut Vec<T>) -> Result<(), TensorError> {
            match v {
                JValue::Array(a) => {
                    for elem in a {
                        flatten(elem, out)?
                    }
                    Ok(())
                }
                JValue::Number(n) => {
                    out.push(FromJNumber::from(n)?);
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

        // Count check (and protect against overflow)
        let mut prod: u128 = 1;
        for &d in &dims {
            prod = prod.checked_mul(d as u128).ok_or(TensorError::Overflow)?;
        }

        let nelems = u32::try_from(prod).map_err(|_| TensorError::Overflow)?;

        // Element strides in C-order (row-major).
        let mut strides = vec![0u32; dims.len()];
        let mut acc: u128 = 1;
        for i in (0..dims.len()).rev() {
            strides[i] = u32::try_from(acc).map_err(|_| TensorError::Overflow)?;
            acc = acc
                .checked_mul(dims[i] as u128)
                .ok_or(TensorError::Overflow)?;
        }

        let elem_buffer = tensor_elemtype_to_buffer!(ty, {
            let mut elem_buffer = Vec::with_capacity(nelems as usize);
            flatten(&v, &mut elem_buffer)?;

            if prod as usize != elem_buffer.len() {
                return Err(TensorError::ShapeMismatch);
            }

            elem_buffer
        });

        Ok(Tensor {
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

        fn write_recursive<E, F>(out: &mut String, data: &[E], dims: &[u32], f: F)
        where
            E: Copy,
            F: Fn(E) -> String + Copy,
        {
            if dims.len() == 1 {
                let n = dims[0] as usize;
                out.push('[');
                for i in 0..n {
                    if i > 0 {
                        out.push(',');
                    }
                    let elem = data[i];
                    let formatted_elem = f(elem);
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
                    write_recursive(out, &data[start..end], &dims[1..], f);
                }
                out.push(']');
            }
        }

        fn write_float<E>(e: E) -> String
        where
            E: Float + Display,
        {
            if e.fract() == Zero::zero() {
                format!("{:.1}", e)
            } else {
                format!("{}", e)
            }
        }

        fn write_int<E>(e: E) -> String
        where
            E: Num + Display,
        {
            format!("{}", e)
        }

        assert!(
            !t.dims.is_empty() && !t.dims.iter().any(|&d| d == 0),
            "Tensor -> String requires non-empty, non-zero dimensions"
        );

        let mut s = String::new();
        tensor_reduce!(t, {
            F(v) => write_recursive(&mut s, &v, &t.dims, write_float),
            I(v) => write_recursive(&mut s, &v, &t.dims, write_int),
        });
        match t.elem_buffer {
            TensorElemBuffer::F16(_) => s.write_str("::f16"),
            TensorElemBuffer::F32(_) => s.write_str("::f32"),
            TensorElemBuffer::F64(_) => s.write_str("::f64"),
            TensorElemBuffer::I32(_) => s.write_str("::i32"),
            TensorElemBuffer::I64(_) => s.write_str("::i64"),
        };
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
        assert_eq!(t.dims, vec![4]);
        assert_eq!(t.strides, vec![1]);

        let s: String = t.into();
        assert_eq!(s, "[1.0,2.0,3.5,4.0]::f64".to_owned());
        Ok(())
    }

    #[test]
    fn rank2() -> Result<(), Box<dyn Error>> {
        let t = "[[1,2,3],[4,5,6]]".parse::<Tensor>()?;
        assert_eq!(t.dims, vec![2, 3]);
        assert_eq!(t.strides, vec![3, 1]);

        let s: String = t.into();
        assert_eq!(s, "[[1.0,2.0,3.0],[4.0,5.0,6.0]]::f64".to_owned());
        Ok(())
    }

    #[test]
    fn rank3() -> Result<(), Box<dyn Error>> {
        let t = "[[[1,2],[3,4]],[[5,6],[7,8]]]".parse::<Tensor>()?;
        assert_eq!(t.dims, vec![2, 2, 2]);
        assert_eq!(t.strides, vec![4, 2, 1]);

        let s: String = t.into();
        assert_eq!(
            s,
            "[[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]]::f64".to_owned()
        );
        Ok(())
    }

    #[test]
    fn elemwise_add_f64_rank2() -> Result<(), Box<dyn Error>> {
        let a = "[[1,2,3],[4,5,6]]".parse::<Tensor>()?;
        let b = "[[10,20,30],[40,50,60]]".parse::<Tensor>()?;
        let c = Tensor::elemwise_add(a, b)?;
        assert_eq!(
            Into::<String>::into(c),
            "[[11.0,22.0,33.0],[44.0,55.0,66.0]]::f64"
        );
        Ok(())
    }

    #[test]
    fn elemwise_add_f16_rank2() -> Result<(), Box<dyn Error>> {
        let a = "[[1,2,3],[4,5,6]]::f16".parse::<Tensor>()?;
        let b = "[[10,20,30],[40,50,60]]::f16".parse::<Tensor>()?;
        let c = Tensor::elemwise_add(a, b)?;
        assert_eq!(
            Into::<String>::into(c),
            "[[11.0,22.0,33.0],[44.0,55.0,66.0]]::f16"
        );
        Ok(())
    }

    #[test]
    fn elemwise_add_i32_rank2() -> Result<(), Box<dyn Error>> {
        let a = "[[1,2,3],[4,5,6]]::i32".parse::<Tensor>()?;
        let b = "[[10,20,30],[40,50,60]]::i32".parse::<Tensor>()?;
        let c = Tensor::elemwise_add(a, b)?;
        assert_eq!(Into::<String>::into(c), "[[11,22,33],[44,55,66]]::i32");
        Ok(())
    }

    #[test]
    fn test_ones() -> Result<(), Box<dyn Error>> {
        let t1 = Tensor::ones(vec![1, 2], TensorElemType::F64);
        let t2 = "[[1,1]]".parse::<Tensor>()?;
        assert_eq!(t2, t2);
        Ok(())
    }

    #[test]
    fn test_matvec() -> Result<(), Box<dyn Error>> {
        let m = "[[1,2,3],[4,5,6]]".parse::<Tensor>()?;
        let v = "[7, 8, 9]".parse::<Tensor>()?;
        let a = Tensor::matvec(m, v)?;
        assert_eq!(Into::<String>::into(a), "[50.0,122.0]::f64");
        Ok(())
    }

    #[test]
    fn test_vecmat() -> Result<(), Box<dyn Error>> {
        let v = "[7, 8]".parse::<Tensor>()?;
        let m = "[[1,2,3],[4,5,6]]".parse::<Tensor>()?;
        let a = Tensor::vecmat(v, m)?;
        assert_eq!(Into::<String>::into(a), "[39.0,54.0,69.0]::f64");
        Ok(())
    }

    #[test]
    fn test_matmul() -> Result<(), Box<dyn Error>> {
        let a = "[[1,2,3],[4,5,6]]".parse::<Tensor>()?;
        let b = "[[1,2],[3,4],[5,6]]".parse::<Tensor>()?;
        let c = Tensor::matmul(a, b)?;
        assert_eq!(Into::<String>::into(c), "[[22.0,28.0],[49.0,64.0]]::f64");
        Ok(())
    }
}
