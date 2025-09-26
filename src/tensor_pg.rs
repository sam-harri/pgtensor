// postgres glue for the tensor_core interface
use crate::tensor_core::{self, Tensor};
use arbitrary_int::traits::Integer;
use arbitrary_int::{u10, u26, u31, u4, TryNewError};
use bitbybit::bitfield;
use core::ffi::CStr;
use pgrx::callconv::{ArgAbi, BoxRet};
use pgrx::datum::{FromDatum, IntoDatum, JsonB};
use pgrx::pg_sys::AsPgCStr;
use pgrx::pgrx_sql_entity_graph::metadata::{
    ArgumentError, Returns, ReturnsError, SqlMapping, SqlTranslatable,
};
use pgrx::prelude::*;
use pgrx::wrappers::rust_regtypein;
use serde_json::Value as JValue;
use std::ffi::CString;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::str::FromStr;

#[bitfield(u31)]
struct Typmod {
    #[bits(27..=30, rw)]
    ndims: u4,
    // TODO: Could add in a third option here to store exact dimensions... not
    // sure how likely that is to be used though, and whether we want to pay the
    // extra bit for that for all other tensors that don't use it.
    #[bit(26, rw)]
    exact_nelems: bool,
    #[bits(0..=25, rw)]
    rest_bits: u26,
}

enum TypemodRest {
    ExactNElems(TypmodRestExactNElems),
    FullHash(u26),
}

#[bitfield(u26)]
struct TypmodRestExactNElems {
    #[bits(10..=25, rw)]
    nelems: u16,
    #[bits(0..=9, rw)]
    hash: u10,
}

impl Typmod {
    fn rest(&self) -> TypemodRest {
        if self.exact_nelems() {
            TypemodRest::ExactNElems(TypmodRestExactNElems::new_with_raw_value(self.rest_bits()))
        } else {
            TypemodRest::FullHash(self.rest_bits())
        }
    }

    // Convert a typmod into its raw encoded form.
    fn pack(&self) -> i32 {
        self.raw_value().as_i32()
    }

    // Try to convert the raw typmod value into a structured Typmod value.
    fn try_unpack(raw: i32) -> Option<Typmod> {
        let raw = u32::try_from(raw).ok()?;
        Some(Typmod::new_with_raw_value(u31::new(raw)))
    }

    fn unpack_unchecked(raw: i32) -> Typmod {
        Typmod::try_unpack(raw).expect("expected a non-empty raw typmod")
    }

    fn validate(&self, tensor: &Tensor) {
        if tensor.ndims != self.ndims().value() {
            pgrx::error!(
                "ndims mismatch, expected {}, found {}",
                self.ndims(),
                tensor.ndims
            );
        }

        let mut hasher = DefaultHasher::new();
        tensor.dims.hash(&mut hasher);
        let hash = hasher.finish();

        match self.rest() {
            TypemodRest::ExactNElems(rest) => {
                if tensor.nelems != rest.nelems().value() as u32 {
                    pgrx::error!(
                        "nelems mismatch, expected {}, found {}",
                        rest.nelems(),
                        tensor.nelems
                    );
                }

                let actual_hash = u10::extract_u64(hash, 0);
                if actual_hash != rest.hash() {
                    pgrx::error!(
                        "dimension hash mismatch, potentially incorrect dimensions, expected {:#x}, found {:#x}",
                        rest.hash(),
                        actual_hash
                    );
                }
            }
            TypemodRest::FullHash(expected_hash) => {
                let actual_hash = u26::extract_u64(hash, 0);
                if actual_hash != expected_hash {
                    pgrx::error!(
                        "dimension hash mismatch, potentially incorrect nelems or dimensions, expected {:#x}, found {:#x}",
                        expected_hash,
                        actual_hash
                    );
                }
            }
        }
    }
}

impl ToString for Typmod {
    fn to_string(&self) -> String {
        match self.rest() {
            TypemodRest::ExactNElems(rest) => {
                format!("ndims={} nelems={}", self.ndims(), rest.nelems())
            }
            TypemodRest::FullHash(_) => format!("ndims={}", self.ndims()),
        }
    }
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
        if is_null {
            return None;
        }
        let bytes: &[u8] = <&[u8]>::from_polymorphic_datum(datum, is_null, typoid)?;

        match serde_cbor::from_slice::<Tensor>(bytes) {
            Ok(t) => Some(t),
            Err(e) => pgrx::error!("failed to deserialize Tensor from CBOR: {}", e),
        }
    }
}

// IntoDatum is just the other way around. You turn the Tensor literal into a pointer to the
// JSONB
// the rust_regtypein is a pgrx macro that tells postgres what SQL type OID corresponds to this Rust type
impl IntoDatum for Tensor {
    fn into_datum(self) -> Option<pg_sys::Datum> {
        let v = match serde_cbor::to_vec(&self) {
            Ok(v) => v,
            Err(e) => pgrx::error!("failed to serialize Tensor to CBOR: {}", e),
        };
        v.into_datum()
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
fn tensor_input(input: &CStr, _oid: pg_sys::Oid, raw_typmod: i32) -> Tensor {
    let s = input.to_str().expect("tensor input must be valid UTF-8");
    let tensor = Tensor::from_str(s).unwrap_or_else(|e| pgrx::error!("{}", e));

    if let Some(typmod) = Typmod::try_unpack(raw_typmod) {
        typmod.validate(&tensor);
    }

    tensor
}

// Other way around, called by postgres when our
// Tensor value needs to be printed back out,
// like SELECTing it or casting to text. we just serialize it back into
// our literal format and hand it to Postgres as a CString.
#[pg_extern(immutable, strict, parallel_safe, requires = [ "shell_type" ])]
fn tensor_output(tensor: Tensor) -> CString {
    let string_repr: String = tensor.into();
    CString::new(string_repr).expect("there should be no NUL in the middle")
}

// when you create a table like so CREATE TABLE t (x tensor(2,3,4));
// this function will receive the args as a list of CStrings, so ["2", "3", "4"]
// parse the dim sizes and create the typmod so we can validate tensor entries
#[pg_extern(immutable, strict, parallel_safe, requires = [ "shell_type" ])]
fn tensor_modifier_input(list: pgrx::datum::Array<&CStr>) -> i32 {
    let mut parts: Vec<&str> = list
        .iter()
        .map(|c| c.unwrap().to_str().unwrap().trim())
        .collect();

    if parts.is_empty() {
        pgrx::error!("tensor(...) requires at least one dimension");
    }

    let dims: Vec<u32> = parts
        .iter()
        .map(|&s| {
            s.parse::<u32>().unwrap_or_else(|_| {
                pgrx::error!(r#"invalid dimension "{s}" (must be a 32-bit integer)"#)
            })
        })
        .collect();

    let ndims = u8::try_from(dims.len())
        .ok()
        .and_then(|ndims| u4::try_new(ndims).ok())
        .unwrap_or_else(|| pgrx::error!("too many dimensions (max 15)"));

    let builder = Typmod::builder().with_ndims(ndims);

    let mut hasher = DefaultHasher::new();
    dims.hash(&mut hasher);
    let hash = hasher.finish();

    let Some(nelems) = dims.iter().fold(Some(1_u16), |acc, dim| {
        acc.and_then(|acc| acc.checked_mul(u16::try_from(*dim).ok()?))
    }) else {
        return builder
            .with_exact_nelems(false)
            .with_rest_bits(u26::extract_u64(hash, 0))
            .build()
            .raw_value()
            .value()
            .cast_signed();
    };

    builder
        .with_exact_nelems(true)
        .with_rest_bits(
            TypmodRestExactNElems::builder()
                .with_nelems(nelems)
                .with_hash(u10::extract_u64(hash, 0))
                .build()
                .raw_value(),
        )
        .build()
        .raw_value()
        .value()
        .cast_signed()
}

// used to display the type mod
// like in \d or pg_dump. we take the packed i32 typmod, unpack it,
// and format it back into a human-readable string
#[pg_extern(immutable, strict, parallel_safe, requires = [ "shell_type" ])]
fn tensor_modifier_output(raw_typmod: i32) -> CString {
    let typmod_string = Typmod::unpack_unchecked(raw_typmod).to_string();
    CString::new(format!("({typmod_string})")).expect("no NUL bytes")
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
fn cast_tensor_to_tensor(tensor: Tensor, raw_typmod: i32, _explicit: bool) -> Tensor {
    Typmod::unpack_unchecked(raw_typmod).validate(&tensor);
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

#[pgrx::pg_extern(immutable, strict, parallel_safe, requires = ["concrete_type"])]
pub fn elemwise_add(t1: Tensor, t2: Tensor) -> Tensor {
    Tensor::elemwise_add(&t1, &t2)
        .map_err(|err| pgrx::error!("{}", err))
        .unwrap()
}
