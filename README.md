# pgtensor

Open-source Postgres extension that adds a `tensor` type with fixed shape and dtype. Built with [pgrx](https://github.com/pgcentralfoundation/pgrx) for Rust.

-   Define columns like `tensor(2,3)`
    
-   Insert with simple literals like `'[[0,1,2],[2,3,4]]'`
    
-   Cast from scalars and arrays, e.g. `'[1]'::tensor`

## Roadmap

- [ ] `load_model(path => text)` function to load ONNX models into background workers for inference  
- [ ] SQL API for running inference against loaded models  
- [ ] Flexible `tensor` column type supporting different lengths and dtypes in the same column


## Install

Requirements: Postgres (13+, ideally 17), Rust, and pgrx.

```sh
# one-time pgrx setup (points pgrx at your local Postgres)
cargo install --locked cargo-pgrx
cargo pgrx init

# build and run the extension into Postgres, interactive through `psql`
cargo pgrx run

# or create installation package
cargo pgrx package
```

If you have multiple Postgres versions, pass `--pg-config` to `cargo pgrx install`.

## Getting started

In each database where you want to use it:

```sql
CREATE EXTENSION pgtensor;
```

Create a table with a tensor column (2×3):

```sql
CREATE TABLE t (x tensor(2,3));
```

Insert and query:

```sql
INSERT INTO t VALUES ('[[0,1,2],[2,3,4]]');
SELECT x FROM t;
```

Cast from a literal:

```sql
SELECT '[1]'::tensor;
```

## Development

Common commands:

```sh
# start a dev Postgres with the extension available
cargo pgrx run

# rebuild & reinstall after changes
cargo pgrx install
```
    

## License

MIT (see `LICENSE`).

---
