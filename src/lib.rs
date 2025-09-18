#![allow(unused)]

mod dynamic_bgworker;
mod onnx_runtime;
mod tensor_core;
mod tensor_pg;

use pgrx::prelude::*;
use std::error::Error;

::pgrx::pg_module_magic!(name, version);

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;
    use std::error::Error;

    use crate::tensor_core::{self};

    #[pg_test]
    fn test_create_table() -> Result<(), Box<dyn Error>> {
        Spi::run(
            "CREATE TABLE t (x tensor(2,3));

            INSERT INTO t VALUES ('[[0,1,2],[2,3,4]]');
            ",
        );
        Ok(())
    }

    #[pg_test]
    fn test_typmod_repr() -> Result<(), Box<dyn Error>> {
        Spi::run(
            "CREATE TABLE t (x tensor(1));
            ",
        );

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
    fn test_bgworker_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let started: bool = Spi::get_one("SELECT load_bgworker('alpha')")?.unwrap();
        assert!(started);

        let echoed: crate::tensor_core::Tensor =
            Spi::get_one("SELECT to_bgworker('alpha', '[[1,2],[3,4]]'::tensor)")?
                .ok_or("worker did not respond")?;

        assert_eq!(
            echoed,
            "[[2,3],[4,5]]".parse::<crate::tensor_core::Tensor>()?
        );

        Ok(())
    }
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec!["shared_preload_libraries = 'pgtensor'"]
    }
}
