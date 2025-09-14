// core tensor API, no postgres leakage (or at least as much as possible)
use serde::{Deserialize, Serialize};
use serde_json::{Number as JNumber, Value as JValue};
use std::convert::TryFrom;
use std::fmt;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    type Error = &'static str;
    fn try_from(code: u8) -> Result<Self, Self::Error> {
        match code {
            0x0 => Ok(DataType::Float64),
            0x1 => Ok(DataType::Float32),
            0x2 => Ok(DataType::Int64),
            0x3 => Ok(DataType::Int32),
            _ => Err("invalid dtype code in typmod"),
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tensor {
    // fixed sized
    // when we migrated to varlena, this will be the header
    pub ndims: u8,
    pub dtype: DataType,
    pub flags: u8,
    pub nelems: u32,
    // then everything below this will be in the allocated buffer
    pub dims: Vec<u32>,    // sizeof : ndims * 4
    pub strides: Vec<u32>, // sizeof : ndims * 4
    pub buffer: Vec<u8>,   // sizeof: nelems * dtype.sizeof()
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

pub fn parse_tensor_literal(literal: &str, dtype: DataType) -> Result<Tensor, String> {
    // postgres string repr of multidim arrays is the same as JSON, so JSON deserialize into a JSON value
    let jv: JValue = serde_json::from_str(literal).map_err(|e| format!("json parse error: {e}"))?;

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
        nelems: nelems,
        dims,
        strides,
        buffer,
    })
}

pub fn to_literal(tensor: &Tensor) -> String {

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
    ) -> JValue {
        if level == dims.len() {
            match dtype {
                DataType::Float64 => {
                    let v = f64_at(buf, *next_idx);
                    *next_idx += 1;
                    JValue::Number(JNumber::from_f64(v).expect("finite f64"))
                }
                _ => unreachable!("only Float64 supported in to_literal now"),
            }
        } else {
            let len = dims[level] as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(build_level(level + 1, dims, dtype, buf, next_idx));
            }
            JValue::Array(arr)
        }
    }

    let mut idx = 0usize;
    let v = build_level(0, &tensor.dims, tensor.dtype, &tensor.buffer, &mut idx);
    serde_json::to_string(&v).expect("serialize tensor literal")
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
        let t = parse_tensor_literal("[1, 2.0, 3.5, 4]", DataType::Float64).unwrap();
        assert_eq!(t.ndims, 1);
        assert_eq!(t.dims, vec![4]);
        assert_eq!(t.nelems, 4);
        assert_eq!(t.strides, vec![8]);
        assert_eq!(t.buffer, expect_f64_bytes(&[1.0, 2.0, 3.5, 4.0]));

        let s = to_literal(&t);
        assert_eq!(s, "[1.0,2.0,3.5,4.0]".to_owned())
    }

    #[test]
    fn rank2_float64() {
        let t = parse_tensor_literal("[[1,2,3],[4,5,6]]", DataType::Float64).unwrap();
        assert_eq!(t.ndims, 2);
        assert_eq!(t.dims, vec![2, 3]);
        assert_eq!(t.nelems, 6);
        assert_eq!(t.strides, vec![24, 8]);
        assert_eq!(t.buffer, expect_f64_bytes(&[1., 2., 3., 4., 5., 6.]));

        let s = to_literal(&t);
        assert_eq!(s, "[[1.0,2.0,3.0],[4.0,5.0,6.0]]".to_owned());
    }

    #[test]
    fn rank3_float64() {
        let t = parse_tensor_literal("[[[1,2],[3,4]],[[5,6],[7,8]]]", DataType::Float64).unwrap();
        assert_eq!(t.ndims, 3);
        assert_eq!(t.dims, vec![2, 2, 2]);
        assert_eq!(t.nelems, 8);
        assert_eq!(t.strides, vec![32, 16, 8]);
        assert_eq!(
            t.buffer,
            expect_f64_bytes(&[1., 2., 3., 4., 5., 6., 7., 8.])
        );

        let s = to_literal(&t);
        assert_eq!(s, "[[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]]".to_owned());
    }
}
