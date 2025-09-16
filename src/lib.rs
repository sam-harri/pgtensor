#![allow(unused)]

mod tensor_core;
mod tensor_pg;

use pgrx::prelude::*;
use std::error::Error;

::pgrx::pg_module_magic!(name, version);

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use std::error::Error;
    use pgrx::prelude::*;

    use crate::tensor_core::{self, to_literal};

    #[pg_test]
    fn test_create_table() -> Result<(), Box<dyn Error>> {
        Spi::run(
            "CREATE TABLE t (x tensor(2,3));

            INSERT INTO t VALUES ('[[0,1,2],[2,3,4]]');

            SELECT x FROM t;
            "
        );
        Ok(())
    }

    #[pg_test]
    fn test_select()-> Result<(), Box<dyn Error>>  {
        let t = Spi::get_one::<tensor_core::Tensor>("SELECT '[1]'::tensor")?.unwrap();
        assert_eq!(t.ndims, 1);
        assert_eq!(t.dims, Vec::from([1]));
        assert_eq!(t.dtype, tensor_core::DataType::Float64);
        assert_eq!(t.nelems, 1);
        assert_eq!(t.strides, Vec::from([tensor_core::DataType::Float64.size_of() as u32]));
        assert_eq!(t.buffer, (1.0 as f64).to_le_bytes());
        assert_eq!(to_literal(&t), "[1.0]".to_owned());
        Ok(())
    }
}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}
