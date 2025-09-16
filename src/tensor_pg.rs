// postgres glue for the tensor_core interface
use core::ffi::CStr;
use pgrx::callconv::{ArgAbi, BoxRet};
use pgrx::datum::{FromDatum, IntoDatum, JsonB};
use pgrx::pgrx_sql_entity_graph::metadata::{
    ArgumentError, Returns, ReturnsError, SqlMapping, SqlTranslatable,
};
use pgrx::prelude::*;
use pgrx::wrappers::rust_regtypein;
use serde_json::Value as JValue;
use std::ffi::CString;

use crate::tensor_core::{self, DataType, Tensor};

// typmod is metadata we have access to on insert to validate a record
// we only have 32 bits, so gotta make them count. I have ideas to make this better later
// for now this is what im going with
// 32-bit typmod layout
// [************************|****|****]
//  ^24b                     ^4b  ^4b
// nelems                   dtype ndims
// also did you know a group of 4 bits is called a nibble haha
pub fn pack_typmod(ndims: u8, dtype: DataType, dims: &[i32]) -> i32 {
    let mut nelems: u64 = 1;
    for &d in dims {
        assert!(d >= 0, "dims must be non-negative");
        nelems *= d as u64;
    }
    assert!(nelems <= 0x00FF_FFFF, "nelem too large for 24-bit");
    assert!(ndims <= 0x0F, "ndims must fit in 4 bits (0..15)");

    let nd = ndims as u32;
    let dt = dtype as u32;
    // bit mask to stop the nel from being more than 24 bits
    // will need proper validation
    let nel = (nelems as u32) & 0x00FF_FFFF;

    ((nel << 8) | (dt << 4) | nd) as i32
}

// note: unpacking typmod is lossy.
// We only have 32 bits, so we can encode the number of dimensions,
// the data type, and the total number of elements, but not the actual
// lengths of each dimension.
//
// this means different shapes with the same dim count and product
// are indistinguishable. For example, tensor(2,3,4) and tensor(3,2,4)
// both unpack to ndims=3 and nelems=24. That’s good enough to catch
// obvious errors (e.g., wrong rank or element count), but not enough
// to validate the exact shape.
//
// TODO : for low-rank tensors (<= 3), we could pack
// the actual dimension lengths instead of just the product.
// For example, in rank 2 we could dedicate 12 bits per dimension
// (first dim in bits 0–11, second dim in bits 12–23) and still
// have room for dtype/ndims. Beyond rank-3 this breaks down,
// since 24 bits can’t represent enough per-dimension information
pub fn unpack_typmod(typmod: i32) -> (u8, DataType, u32) {
    assert!(typmod != -1, "typmod is unknown (-1)");
    let t = typmod as u32;

    let nd = (t & 0x0F) as u8;
    let dt4 = ((t >> 4) & 0x0F) as u8;
    let nel = (t >> 8) & 0x00FF_FFFF;

    let dtype = DataType::try_from(dt4).expect("invalid dtype nibble in typmod");
    (nd, dtype, nel)
}

// tells pgrx how to map the Rust Tensor type to SQL in the generated SQL
// when Tensor is used in #[pg_extern] functions, it is translated to
// the SQL type name tensor for both arguments and return values
// basically just telling pgrx to call our Rust Tensor type tensor
// in SQL land
unsafe impl SqlTranslatable for Tensor {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As("tensor".into()))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As("tensor".into())))
    }
}

// for context, in postgres, a Datum is a generic container for values.
// It's basically a void pointer that can either
// store a pass-by-value type directly (if it fits in 4 bytes) or
// points to the in-memory representation of a pass-by-reference type
// https://stackoverflow.com/questions/53543909/what-exactly-is-datum-in-postgresql-c-language-functions

// FromDatum deinfes how to take a Datum (a pointer to JSONB) and
// reconstruct a Rust Tensor. This is essentially
// deserializing JSONB into a Rust struct.
impl FromDatum for Tensor {
    unsafe fn from_polymorphic_datum(
        datum: pg_sys::Datum,
        is_null: bool,
        typoid: pg_sys::Oid,
    ) -> Option<Self> {
        let jb = JsonB::from_datum(datum, is_null)?;
        let v: JValue = jb.0;
        match serde_json::from_value::<Tensor>(v) {
            Ok(t) => Some(t),
            Err(e) => pgrx::error!("failed to deserialize Tensor from jsonb: {}", e),
        }
    }
}

// IntoDatum is just the other way around. You turn the Tensor literal into a pointer to the
// JSONB
// the rust_regtypein is a pgrx macro that tells postgres what SQL type OID corresponds to this Rust type
impl IntoDatum for Tensor {
    fn into_datum(self) -> Option<pg_sys::Datum> {
        let v = match serde_json::to_value(&self) {
            Ok(v) => v,
            Err(e) => pgrx::error!("failed to serialize Tensor to jsonb: {}", e),
        };
        JsonB(v).into_datum()
    }
    fn type_oid() -> pg_sys::Oid {
        rust_regtypein::<Self>()
    }
}

// okay so this part is a bit harder to explain
// On the postgres side you have a Datum (explained above).
// that Datum is inside a NullableDatum struct, which is basically
// {datum : Datum, isnull: bool}. Then, you have a FunctionCallInfo
// struct (aka fcinfo), which contains the call frame for a func. Basically, it's
// {nargs: int, args: [NullableDatum], ...some metadata}. Now, on the Rust side, FcInfo<'fcx>
// is pgrx’s safe(-ish) wrapper around that raw fcinfo pointer.
// 'fcx lifetime is function call context
// Arg<'a, 'fcx> is pgrx’s view of one argument slot, which contains a reference to the
// FcInfo, it's index in the arg array, and a reference to the NullableDatum for that arg

// long story short, ArgAbi<T> is “given an Arg, make me a T”. just calls
// Arg::unbox_arg_using_from_datum() which calls the FromDatum
// basically how to read an argument from Postgres’ calling convention
unsafe impl<'fcx> ArgAbi<'fcx> for Tensor
where
    Self: 'fcx,
{
    unsafe fn unbox_arg_unchecked(arg: pgrx::callconv::Arg<'_, 'fcx>) -> Self {
        // we expecet a non null arg input, so panic if it is null
        arg.unbox_arg_using_from_datum()
            .expect("expect it to be non-NULL")
    }
}

// BoxRet is the opposite, “given a T, put it back into fcinfo as a return value”
// calls IntoDatum and then fcinfo.return_raw_datum(...)
// basically how to write a result back into Postgres’ calling convention
unsafe impl BoxRet for Tensor {
    unsafe fn box_into<'fcx>(
        self,
        fcinfo: &mut pgrx::callconv::FcInfo<'fcx>,
    ) -> pgrx::datum::Datum<'fcx> {
        match self.into_datum() {
            Some(d) => fcinfo.return_raw_datum(d),
            None => fcinfo.return_null(),
        }
    }
}

// this runs when parsing a value literal of our type, like '[[[...]]]'
// Postgres gives us the string literal, the OID of our type, and the typmod.
// basically parse the JSON styled nested lists into our Tensor type
// Look at the
#[pg_extern(immutable, strict, parallel_safe, requires = ["shell_type"])]
fn tensor_input(input: &CStr, _oid: pg_sys::Oid, type_modifier: i32) -> Tensor {
    let s = input.to_str().expect("tensor input must be valid UTF-8");

    // hard coding f64 for now, but later we should add parsing so that you can optionally
    // pass in the datatype, like so
    // tensor(1,2,3,"f64") or tensor(4,5,"i32")
    // easy to implement but I just dont feel like doing that rn (just want MVP)
    let dtype = DataType::Float64;
    let tensor = Tensor::from_literal(s, dtype).unwrap_or_else(|e| pgrx::error!("{}", e));

    if type_modifier != -1 {
        let (expected_ndims, dtype, expected_nelems) = unpack_typmod(type_modifier);
        let expected_buffer_length = dtype.size_of() * expected_nelems as usize;
        if tensor.nelems != expected_nelems {
            pgrx::error!(
                "nelems mismatch, expected {}, found {}",
                expected_nelems,
                tensor.nelems
            );
        }

        if tensor.ndims != expected_ndims {
            pgrx::error!(
                "ndims mismatch, expected {}, found {}",
                expected_ndims,
                tensor.ndims
            );
        }

        if tensor.elem_buffer.len() != expected_buffer_length {
            pgrx::error!(
                "buffer length mismatch, expected {}, found {}", // probably a dtype error?
                expected_ndims,
                tensor.elem_buffer.len()
            );
        }
    }

    tensor
}

// Other way around, called by postgres when our
// Tensor value needs to be printed back out,
// like SELECTing it or casting to text. we just serialize it back into
// our literal format and hand it to Postgres as a CString.
#[pg_extern(immutable, strict, parallel_safe, requires = [ "shell_type" ])]
fn tensor_output(tensor: Tensor) -> CString {
    let string_repr = tensor.to_literal().unwrap();
    CString::new(string_repr).expect("there should be no NUL in the middle")
}

// when you create a table like so CREATE TABLE t (x tensor(2,3,4));
// this function will receive the args as a list of CStrings, so ["2", "3", "4"]
// parse the dim sizes and create the typmod so we can validate tensor entries
// Want to support changin the tensor scalar type like tensor(2,3,4,"f64")
#[pg_extern(immutable, strict, parallel_safe, requires = [ "shell_type" ])]
fn tensor_modifier_input(list: pgrx::datum::Array<&CStr>) -> i32 {
    let dims = list
        .iter()
        .map(|cstr| cstr.unwrap().to_str().unwrap())
        .map(str::trim)
        .map(|t| t.parse::<i32>())
        .collect::<Result<Vec<i32>, _>>()
        .expect("each dimension must be a valid i32");
    let ndims = u8::try_from(dims.len()).expect("too many dimensions (max 15)");
    assert!(ndims <= 0x0F, "too many dimensions (max 15)");
    // hard coding float64 only for now
    let typmod = DataType::Float64;
    pack_typmod(ndims, typmod, &dims)
}

// used to display the type mod
// like in \d or pg_dump. we take the packed i32 typmod, unpack it,
// and format it back into a human-readable string
#[pg_extern(immutable, strict, parallel_safe, requires = [ "shell_type" ])]
fn tensor_modifier_output(type_modifier: i32) -> CString {
    let (ndims, dtype, nelems) = unpack_typmod(type_modifier);
    CString::new(format!(
        "(Tensor ndims={} dtype={} nelems={})",
        ndims, dtype, nelems
    ))
    .expect("no NUL in the middle")
}

// cast a tensor to a tensor, the conversion is meaningless but we need it to
// do the rank/nelems/dtype checks
// At parse time, postgres calls our tensor_input function but with a typmod of -1
// because it does not yet have access to the table's atttypmod attribute which holds the typmod
// for that column

// The actual typmod gets enforced later with a “length conversion cast”.
// In short, parsing a tensor skips validation, but when you insert it into a
// column with a declared typmod, Postgres applies this self-cast (tensor->tensor).
// That’s our chance to validate shape/rank/etc, then return the value unchanged if it's good
#[pgrx::pg_extern(immutable, strict, parallel_safe, requires = ["concrete_type"])]
fn cast_tensor_to_tensor(tensor: Tensor, type_modifier: i32, _explicit: bool) -> Tensor {
    let (expected_ndims, dtype, expected_nelems) = unpack_typmod(type_modifier);
    let expected_buffer_length = dtype.size_of() * expected_nelems as usize;

    if tensor.nelems != expected_nelems {
        pgrx::error!(
            "nelems mismatch, expected {}, found {}",
            expected_nelems,
            tensor.nelems
        );
    }

    if tensor.ndims != expected_ndims {
        pgrx::error!(
            "ndims mismatch, expected {}, found {}",
            expected_ndims,
            tensor.ndims
        );
    }

    if tensor.elem_buffer.len() != expected_buffer_length {
        pgrx::error!(
            "buffer length mismatch, expected {}, found {}", // probably a dtype error?
            expected_ndims,
            tensor.ndims
        );
    }

    tensor
}

extension_sql!(
    r#"
CREATE TYPE vector; -- shell type
"#,
    name = "shell_type",
    bootstrap // declare this extension_sql block as the "bootstrap" block so that it happens first in sql generation
);

extension_sql!(
    r#"
CREATE TYPE tensor (
    INPUT = tensor_input,
    OUTPUT = tensor_output,
    TYPMOD_IN = tensor_modifier_input,
    TYPMOD_OUT = tensor_modifier_output,
    STORAGE = external
);
"#,
    name = "concrete_type",
    creates = [Type(tensor)],
    requires = [
        "shell_type",
        tensor_input,
        tensor_output,
        tensor_modifier_input,
        tensor_modifier_output
    ]
);

extension_sql!(
    r#"
    CREATE CAST (tensor AS tensor)
    WITH FUNCTION cast_tensor_to_tensor(tensor, integer, boolean);
    "#,
    name = "cast_tensor_to_tensor",
    requires = ["concrete_type", cast_tensor_to_tensor]
);
