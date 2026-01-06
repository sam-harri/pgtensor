# pgtensor

Open-source Postgres extension that adds a `tensor`  type, and an ONNX inference engine using dynamic worker processes. Built with [pgrx](https://github.com/pgcentralfoundation/pgrx) for Rust.

-   Define columns like `tensor(2,3)`
-   Insert with simple literals like `'[[0,1,2],[2,3,4]]'`, and validate shapes
-   Support for f16, f32, f64, i32, and i64 entry datatypes.
-   Various tensor operations accessible in SQL via functions and inline operators (i.e. `+`, `-`, `@`, etc.)
-   Load ONNX models into background processes, and run inference on stored tensors
-   Compatible with Postgres versions 13-17

## Roadmap 
- [ ] Flexible `tensor` column type supporting different lengths and dtypes in the same column
- [ ] Better synchronization, move from 1 lock for the entire shared memory queue to a lock per slot in queue

## API overview

Load up the extention :

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

Run inference : 

First load a model from `/var/lib/postgresql/pgtensor_model`

```sql
SELECT load_model('sigmoid','x','y');
```

where the first arg is the model name, so `sigmoid.onnx`,
and the next 2 args are the input and output tensor names, respectivly

Then run an inference call

```sql
SELECT run_inference('sigmoid', '[1]');
```

using the model name as the first arg, and the tensor you want to run inference on in the second

Which will return the tensor `'[0.7310585786300049]'`

## Install

Requirements: Postgres (13+, ideally 17), Rust, and pgrx.

```sh
# one-time pgrx setup
cargo install --locked cargo-pgrx
cargo pgrx init

# build and run the extension into Postgres, interactive through `psql`
cargo pgrx run
# and optionally specify a version
cargo pgrx run pg13

# You can also install the package using
cargo pgrx package
# for release mode use
cargo pgrx install --release
```

## License

MIT (see `LICENSE`).

---
