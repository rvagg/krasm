# krasm

An experimental & educational WebAssembly runtime implementation in Rust that provides parsing, validation, and execution utilities for WebAssembly modules.

*If you want to join me in exploring WebAssembly by building, hit me up!*

## Features

- **Binary parser** with full section support and validation
- **WAT parser** for the WebAssembly text format
- **Binary encoder** producing spec-compliant `.wasm` from any parsed Module
- **Flat bytecode interpreter** supporting WebAssembly 2.0
- **Full SIMD (v128)** support: all v128 instructions across integer, float, bitwise, comparison, shuffle, conversion, and memory operations
- **WASI preview1** support: fd_read/write, args, environ, proc_exit, filesystem preopens
- **AssemblyScript** compatibility (env.abort with UTF-16 string extraction)
- Disassembler compatible with WABT wasm-objdump format
- Structured-tree interpreter retained for explicit selection and comparison
- Linear memory with bounds checking and page-based growth
- Tables with indirect function calls (call_indirect)
- Cross-module linking via Store-based architecture
- Native `.wast` spec test runner for core and SIMD specification conformance
- Comprehensive test suite: unit, encoder round-trip, dump comparison, wast spec, WASI integration

## CLI

```bash
# Run a WASI module (.wasm or .wat)
krasm run examples/hello.wat
krasm run module.wasm -- arg1 arg2
krasm run module.wasm --dir ./data -- arg1
krasm run module.wasm --engine structured      # select the alternative interpreter

# Compile WAT to binary
krasm compile examples/hello.wat              # produces examples/hello.wasm
krasm compile input.wat -o output.wasm

# Inspect a module
krasm dump module.wasm                        # detailed section info
krasm dump module.wasm --header               # magic and version only
krasm dump module.wasm -d                     # disassemble
```

Build and run via cargo:

```bash
cargo run --bin krasm -- run examples/hello.wat
cargo run --bin krasm -- compile examples/hello.wat
cargo run --bin krasm -- dump module.wasm -d
```

The CLI and library default to the flat bytecode engine. Library callers can
select `EngineKind::Structured` with `Store::set_engine()` before creating an
instance; existing instances retain their engine. Start functions use their
instance's engine, and cross-module calls can bridge both engines.

Instruction budgets count engine operations, not portable WebAssembly fuel.
The same budget can stop at different points on the two engines.

## Project Structure

```
src/parser/         Binary parser, validation, encoding primitives
src/wat/            WAT text format parser (lexer, S-expression, Module builder)
src/encoder.rs      Binary encoder (Module → .wasm)
src/runtime/        Interpreter, Store, WASI implementation
examples/           Example WAT modules
benches/            Criterion benchmarks with WAT modules
tests/              Unit tests, spec test suite, WASI integration tests
fuzz/               Fuzz targets (binary parser, executor, WAT lexer/parser)
```

## Development

The full check requires Node.js/npm, Python 3.9+ and the Rust WASI target:

```bash
rustup target add wasm32-wasip1
```

It includes default, profiling and superinstruction feature tests, offline sequence
analysis tests, and native/both-engine WASI checks for the SIMD CommP example.

```bash
./check.sh              # Format, lint, test (must pass before committing)
cargo test              # Run all tests
cargo bench             # Run benchmarks
```

WAST and WASI integration tests explicitly exercise both engines. Execution
benchmarks compare both through the Store API, with instantiation timed
separately. See [performance notes](docs/PERFORMANCE.md) for the switchover
measurements and bounded fuzzing coverage.

## Experimental superinstructions

The opt-in `superinstructions` Cargo feature fuses scalar sequences in the flat
engine. Default builds and the structured engine remain unfused:

```bash
cargo run --release --features superinstructions --bin krasm -- run module.wasm
```

For unfused instruction profiling and build comparisons, see
`python3 scripts/profile.py --help`.

## License

This project is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for details.
