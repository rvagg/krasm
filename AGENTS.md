# krasm - WebAssembly Runtime (Rust)

Previously "kasm"; renamed to `krasm` for crates.io publication in March 2026.

## Project Intent
- Learning and educational project: code should teach maintainers how WebAssembly works.
- Optimise for excellence over speed: clear abstractions, explicit invariants, useful docs, maintainable tests.
- Prefer simple, local, spec-shaped code. Add abstraction only when it clarifies behaviour or removes real duplication.

## Critical Rules
- Run `./check.sh` after changes. It runs check, fmt, clippy, docs, tests, AssemblyScript builds, and WASI examples.
- Never run `git commit` or `git push`.
- Use Australian English in prose: initialise, analyse, behaviour.
- Follow existing patterns before inventing new ones.

## Architecture
```text
src/parser/              Binary parser, validation, module model
  instruction/           Opcode decode/encode, including SIMD
  structure_builder.rs   Flat instructions -> structured control flow
  structured.rs          Structured function tree
  validate.rs            Module-level validation
  encoding.rs            LEB128, float, vector encoding primitives

src/wat/                 WebAssembly text format parser
  lexer.rs               Tokeniser
  sexpr.rs               S-expression reader
  parser.rs              S-expr -> Module
  error.rs               Span-aware LexError/ParseError

src/wast/                .wast spec test support
  parser.rs              Script commands and assertions
  command.rs             WastCommand, WastValue, WastFloat, etc.
  spectest.rs            spectest imports
  values.rs              Value conversion and NaN-aware comparison

src/runtime/             Interpreter and host integration
  store.rs               Store, global addresses, cross-module calls, host funcs
  instance.rs            Module instantiation and resource address maps
  executor.rs            Primary structured interpreter; resumable external calls
  bytecode.rs            Experimental flat bytecode representation
  compiler.rs            Structured tree -> flat bytecode compiler
  flat_executor.rs       Experimental flat bytecode executor
  imports.rs             Import resolution
  memory.rs/table.rs     Linear memory and tables
  ops/                   Instruction implementations by category
  wasi/                  WASI preview1 plus AssemblyScript env.abort

src/encoder.rs           Module -> .wasm binary encoder
src/main.rs              CLI: run, dump, compile
examples/assemblyscript/ AssemblyScript WASI examples
examples/commp/          Filecoin CommP SIMD/WASI example
benches/                 Criterion benchmarks
fuzz/                    cargo-fuzz targets and dictionaries
```

## Current Status
- WebAssembly 2.0 binary parser, WAT parser, encoder, interpreter, disassembler, and CLI.
- 427+ instructions implemented, including all 236 SIMD instructions.
- Native `.wast` runner passes 148 spec files: 90 core + 58 SIMD, pinned to `wg-2.0` (2025-08-28).
- Test inventory: 877 unit tests, 82 encoder tests, 148 dump tests, 148 wast tests, 23 WASI tests, 20 doctests.
- WASI preview1: 46 imports registered; implemented calls plus NOSYS stubs where appropriate.
- AssemblyScript support: `env.abort` with UTF-16 string extraction.
- No external `wat` crate dependency; use `krasm::wat::parse()`.

## CLI
```bash
krasm run <file> [-- args...]        # Execute WASI module (.wasm or .wat)
krasm run <file> --dir ./data -- ... # Preopen a host directory
krasm dump <file>                    # Module details
krasm dump <file> --header           # Header only
krasm dump <file> -d                 # Disassemble
krasm compile <file.wat> [-o out]    # WAT -> .wasm
```

## Development
```bash
./check.sh                           # Required full check
./check.sh -f                        # Then fuzz parse_module for 60s
./check.sh -f 300                    # Then fuzz for 5 minutes
./check.sh -f 60 -t execute_module   # Fuzz another supported target
cargo test                           # All tests
cargo test <pattern>                 # Focused tests
cargo test -- --nocapture            # Show println output
RUST_BACKTRACE=1 cargo test <pattern>
```

## Adding Instructions
1. Add the opcode to `InstructionKind` in `src/parser/instruction/mod.rs`.
2. Decode it in `src/parser/instruction/decode.rs`.
3. Encode it in `src/parser/instruction/encode.rs` if it must round-trip.
4. Execute it in `src/runtime/executor.rs` or the relevant `src/runtime/ops/*.rs`.
5. Add focused tests, then run `./check.sh`.

## Test Infrastructure
- `tests/wast_tests.rs`: native `.wast` runner for assertions, traps, invalid/malformed/unlinkable/uninstantiable/exhaustion cases.
- `tests/dump_tests.rs`: compares display output with `wasm-objdump` fixtures.
- `tests/encoder_tests.rs`: encode -> parse -> encode stability tests using WAT, hand-built modules, and spec fixtures.
- `tests/wasi_tests.rs`: WASI integration tests using inline WAT.
- `tests/spec/*.json`: base64 wasm + dump fixtures. Regenerate with:
  `WAST2JSON=... WASM_OBJDUMP=... node tests/compile_test.mjs --batch <wast-dir> tests/spec`

## Fuzzing
```bash
cargo install cargo-fuzz
./fuzz/seed_corpus.sh
cargo +nightly fuzz run parse_module -- -dict=fuzz/wasm.dict
cargo +nightly fuzz run execute_module -- -max_total_time=60 -dict=fuzz/wasm.dict
cargo +nightly fuzz run generate_module -- -max_total_time=60
cargo +nightly fuzz run lex_wat -- -max_total_time=60 -dict=fuzz/wat.dict
cargo +nightly fuzz run parse_wat -- -max_total_time=60 -dict=fuzz/wat.dict
```

Targets: `parse_module`, `execute_module`, `generate_module`, `lex_wat`, `parse_wat`.
Commit fuzz source files and dictionaries; ignore `fuzz/target/`, `fuzz/corpus/`, and `fuzz/artifacts/`.

## Benchmarks
```bash
cargo bench --bench execution
cargo bench --bench execution -- noop
cargo bench --bench execution -- --test
cargo bench --bench validation
```

Modules: `noop_loop`, `fib_iterative`, `fib_recursive`, `memcpy`, `primes`.
CommP benchmark input is `benches/commp_bench_500k.bin`; expected hash:
`c1bb8f1985dbf4bf34d06c7190d10a916d228dccd668ba87a10cb1cf0cf3b523`.

## Public API Pointers
- `krasm::wat::parse()` parses WAT to `Module`.
- `krasm::encoder::encode()` encodes `Module` to `.wasm`.
- `Store::create_instance()` instantiates modules.
- `Store::invoke_export()` calls exported functions.
- `Store::wrap()` and `Store::wrap_with_caller()` expose Rust host functions.
- `krasm::wasi::{WasiContext, create_wasi_instance}` builds WASI instances.

## Error Types
- `DecodeError`: binary parser errors with byte positions.
- `ValidationError`: module validation errors.
- `LexError` / `ParseError`: WAT errors with source spans.
- `RuntimeError`: traps, import/export failures, type/resource errors.
- `EncodeError`: binary encoder errors.
