#![allow(unused)]

mod tensor_pg;
mod tensor_core;
mod onnx_runtime;
mod dynamic_bgworker;

use pgrx::prelude::*;
use std::error::Error;

::pgrx::pg_module_magic!(name, version);

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::{datum::TryFromDatumError, pg_sys::Oid, prelude::*, spi::SpiError};
    use std::error::Error;

    use crate::tensor_core::{self, Tensor};

    #[pg_test]
    fn test_create_table() -> Result<(), Box<dyn Error>> {
        Spi::run(
            "CREATE TABLE t (x tensor(2,3));

            INSERT INTO t VALUES ('[[0,1,2],[2,3,4]]');
            ",
        )?;
        Ok(())
    }

    #[pg_test(
        error = "dimension hash mismatch, potentially incorrect dimensions, expected 0x1c7, found 0x14c"
    )]
    fn test_typmod_hash() -> Result<(), Box<dyn Error>> {
        Spi::run("CREATE TABLE t (x tensor(2,3));")?;
        Spi::run("INSERT INTO t VALUES ('[[0,1],[2,2],[3,4]]');")?;
        Ok(())
    }

    #[pg_test]
    fn test_typmod_repr() -> Result<(), Box<dyn Error>> {
        Spi::run(
            "CREATE TABLE t (x tensor(1));
            ",
        )?;

        let typ = Spi::get_one::<String>(
            r#"
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_attribute a ON a.attrelid = c.oid
            WHERE n.nspname = current_schema()
            AND c.relname = 't'
            AND a.attname = 'x'
            AND a.attnum > 0
            "#,
        )?
        .unwrap();
        assert_eq!(typ, "tensor(ndims=1 nelems=1)");
        Ok(())
    }

    #[pg_test]
    fn test_select() -> Result<(), Box<dyn Error>> {
        let t = Spi::get_one::<tensor_core::Tensor>("SELECT '[1]'::tensor")?.unwrap();
        assert_eq!(t, "[1.0]".parse::<tensor_core::Tensor>()?);
        Ok(())
    }

    #[pg_test]
    fn test_elemwise_addition() -> Result<(), Box<dyn Error>> {
        let t_output: tensor_core::Tensor = Spi::get_one::<tensor_core::Tensor>(
            "SELECT elemwise_add('[[1,2.0],[3,4]]', '[[10,20.0],[30,40]]');",
        )?
        .ok_or("elemwise_add returned NULL")?;

        let t_expected = "[[11.0,22],[33,44]]".parse::<tensor_core::Tensor>()?;
        assert_eq!(t_expected, t_output);

        Ok(())
    }

    #[pg_test]
    fn test_onnx_sigmoid_bgworker_infers() -> Result<(), Box<dyn std::error::Error>> {
        let started: bool = Spi::get_one("SELECT load_model('sigmoid','x','y')")?.unwrap();
        assert!(started, "bgworker failed to start");

        let t_str: String = crate::tensor_core::Tensor::ones(vec![3, 4, 5])?.into();
        let query = format!(
            "SELECT run_inference('sigmoid', '{}')",
            t_str
        );
        let out: crate::tensor_core::Tensor = Spi::get_one(&query)?
        .ok_or("run_inference returned NULL")?;

        assert_eq!(out.dims, vec![3, 4, 5]);
        assert_eq!(out.ndims, 3);
        assert_eq!(out.nelems, 60);
        assert_eq!(out.strides, vec![20, 5, 1]);

        // value check, sigmoid(1) ~= 0.7310585786300049
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

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        let base = concat!(env!("CARGO_MANIFEST_DIR"), "/models");
        std::env::set_var("PGTENSOR_MODELS_DIR", base);
    }

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec!["shared_preload_libraries = 'pgtensor'"]
    }
}
