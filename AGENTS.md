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
  flat_executor/         Default engine: untagged stack of 64-bit words; resumable external calls
  bytecode.rs            Flat bytecode ops; depths, arities and locals counted in words
  compiler.rs            Structured tree -> flat bytecode, type-checked against validation
  executor.rs            Structured interpreter on tagged values; the differential reference
  imports.rs             Import resolution
  memory.rs/table.rs     Linear memory and tables
  ops/                   Pure instruction implementations shared by both engines
  trap.rs                One-byte Trap that operations return
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
- Native `.wast` runner passes 148 spec files (90 core + 58 SIMD, pinned to `wg-2.0`, 2025-08-28)
  and 8 regression files, each on both engines.
- Test inventory: 952 unit tests, 82 encoder tests, 148 dump tests, 312 wast runs, 46 WASI tests,
  11 instruction-budget tests, 7 host-boundary tests, 2 validation tests, 14 doctests.
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
4. Give it a signature in `src/parser/instruction/signature.rs`: `fixed_signature`, or
   `context_signature` when its types come from an immediate. Validation and the compiler share it.
5. Implement it as a pure function in `src/runtime/ops/*.rs`, returning `Result<_, Trap>` if it traps.
6. Wire both engines: an arm in `src/runtime/executor.rs`, and an `Op` in `bytecode.rs` lowered in
   `compiler/lower.rs` and dispatched in `flat_executor/mod.rs`.
7. Add focused tests, then run `./check.sh`.

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
cargo +nightly fuzz run -a execute_module -- -max_total_time=60 -dict=fuzz/wasm.dict
cargo +nightly fuzz run -a generate_module -- -max_total_time=60
cargo +nightly fuzz run lex_wat -- -max_total_time=60 -dict=fuzz/wat.dict
cargo +nightly fuzz run parse_wat -- -max_total_time=60 -dict=fuzz/wat.dict
```

Targets: `parse_module`, `execute_module`, `generate_module`, `lex_wat`, `parse_wat`.
`execute_module` and `generate_module` run every export on both engines and fail on any difference
in results or trap kind; `-a` keeps debug assertions, including the flat executor's frame check.
`--features instruction-profile` (as `check.sh -f` runs them) also replays each completed flat call
under exactly the ops its profile counts, which must complete, and one fewer, which must exhaust.
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

## Performance Measurement
- Instructions retired decide mechanical changes; cycles and wall time vary 10-25% with
  code placement, so compare them only across layout groups, never single builds.
- `python3 scripts/profile.py compare --cpu 4 --arm a=BIN_OR_DIR --arm b=...` (perf counters).
- `python3 scripts/profile.py build --seeds 5 [--rev REV]` builds seeded layouts for `compare`.
- Corpus workloads: `--registry target/corpus/registry.json --corpus qualified`.

## Public API Pointers
- `krasm::wat::parse()` and `krasm::parser::parse()` return a `ValidatedModule`.
- `ValidatedModule::new()` validates a hand-built `Module`; `into_inner()` recovers one to edit.
- `krasm::encoder::encode()` encodes `Module` to `.wasm`.
- `Store::create_instance()` instantiates a `&ValidatedModule`; validation is the trust boundary.
- `Store::invoke_export()` calls exported functions.
- `Store::wrap()` and `Store::wrap_with_caller()` expose Rust host functions.
- `krasm::wasi::{WasiContext, create_wasi_instance}` builds WASI instances.

## Error Types
- `DecodeError`: binary parser errors with byte positions.
- `ValidationError`: module validation errors.
- `LexError` / `ParseError`: WAT errors with source spans.
- `RuntimeError`: traps, import/export failures, type/resource errors; `Compile` for
  inconsistencies validation should have rejected.
- `EncodeError`: binary encoder errors.
