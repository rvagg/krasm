# krasm agent guide

krasm is a WebAssembly 2.0 runtime in Rust: binary and text parsers,
validator, encoder, two interpreters, a WASI preview1 host and a CLI. It is
published on crates.io as `krasm`.

This file is for coding agents and the people driving them, including agents
working without a brief. `README.md` is the user manual. This file is the
contract for changing the code. It assumes nothing beyond this repository.

## Telos

One sentence: **a WebAssembly runtime you can learn WebAssembly from, that is
safe to run modules you do not trust.**

krasm primarily exists to as an educational project, exploring how WebAssembly
works and teaching about concepts that arise from implementing a WASM runtime.
The runtime is also meant to be fit for real use one day, so what it teaches
must be true, and nothing it does may be unsafe. Judge every change
against these, in order:

1. **Safety.** Module bytes, WAT source, `.wast` scripts and WASI guests are
   hostile input. No input may crash the host, exhaust its memory, run
   forever without a budget, read or write outside the sandbox, or reach
   undefined behaviour. A panic reachable from input is a bug. A conformance
   bug in validation is a security bug.
2. **Correctness.** Behaviour matches the W3C Working Group 2.0 specification
   at tag `wg-2.0`: core, bulk memory, reference types and SIMD. Spec test
   files run on both engines and every assertion passes. Cite the spec
   section when a decision rests on it.
3. **Clarity.** Code is shaped like the spec it implements, named in the
   spec's terms, and explained where the mechanism is not obvious from the
   code. A reader should finish a module understanding one more piece of
   WebAssembly than when they started. Abstraction is added only when it
   clarifies behaviour or removes real duplication.
4. **Performance.** Always want faster; the long-term aim is to be
   competitive with production interpreters. Measured, never assumed, never
   bought with any of the three above. A fast path that is harder to read
   carries the explanation that makes it readable.

A change that improves safety is welcome on its own merits. A change that
only improves speed must show its numbers. A change that only improves
clarity must leave the numbers alone.

## What krasm is and is not

- **Scope:** WebAssembly 2.0 as `wg-2.0` defines it. The binary format, the
  text format (folded instructions, inline exports, inline `call_indirect`
  types), validation, execution, encoding and disassembly in `wasm-objdump`
  format. WASI preview1 with directory preopens. AssemblyScript's `env.abort`.
- **Not yet:** threads, exception handling, GC, memory64, multi-memory,
  tail calls, extended constant expressions, `module quote` in `.wast`. A
  proposal is adopted whole or not at all: decoder, validator, encoder, text
  format, both engines, disassembler and its spec files, in one body of
  work. Do not land a proposal's opcodes piecemeal.
- **Not, but maybe one day:** a JIT, a browser engine or a WASI preview2 host.
- **Users:** library embedders through `Store`, and the `krasm` CLI. Both
  ship from this crate. The library API is small and should stay so.

## How it works

```text
.wasm bytes --> parser (decode + validate) --> ValidatedModule
.wat source --> wat (lex, sexpr, build Module) --> validate --> ValidatedModule
ValidatedModule --> encoder --> .wasm bytes

Store::create_instance(&ValidatedModule)
   --> compiler: structured tree -> flat bytecode, typed against validation
   --> Instance: memories, tables, globals as addresses into Store resources
Store::invoke_export
   --> flat executor (default): untagged word stack, handler per op, tail-call dispatch
   --> structured executor (oracle): tagged values, rechecks types as it goes
   --> host functions through Caller; imported calls suspend and resume
```

### Validation is the trust boundary

`Store::create_instance` accepts only a `ValidatedModule`. Both parsers
produce one, and `ValidatedModule::new` validates a `Module` built or edited
by hand. Past that point the runtime relies on what validation proved rather
than rechecking it per instruction: the flat engine's bytecode is typed by
the compiler, stack depths and arities are counted in words at compile time,
and stack capacity is checked once per call from the precomputed maximum.

The consequences for any change:

- Anything the executor assumes, validation must guarantee, and the
  compiler must check. The compiler tracks operand types as validation does
  and fails with `RuntimeError::Compile` on any disagreement rather than
  emitting bytecode that relies on a false assumption.
- Calls that cross out of validated code, host functions and foreign
  `call_indirect` targets, are checked by the `Store` at the crossing
  (`tests/host_boundary_tests.rs`).
- The structured engine is the differential oracle. It executes tagged
  values and checks them as it goes; agreement with the flat engine on every
  result and trap is the evidence that the flat engine's trust is earned.
  Keep it correct and agreeing. Do not optimise it or grow it beyond what
  the flat engine does.

### Modules

```text
src/parser/              Binary decoder, validator, module model
  reader.rs              Byte reader: LEB128, floats, vectors, with positions
  encoding.rs            Binary format constants and encoding primitives
  limits.rs              Implementation limits, aligned with V8's
  index_spaces.rs        Types of each index space, imports first
  module.rs              Module, sections, types
  instruction/           Opcode decode/encode, signatures, SIMD under 0xFD
  structure_builder.rs   Flat instructions -> structured control flow
  structured.rs          Structured function tree
  validate.rs            Module-level validation of a built Module

src/wat/                 Text format: cursor, lexer, s-expressions, parser, spans
src/wast/                .wast scripts: commands, spectest imports, NaN-aware comparison
src/encoder.rs           Module -> .wasm

src/runtime/
  store.rs               Store<T>: resources, addresses, cross-module calls, host functions
  instance.rs            Instantiation and resource address maps
  compiler.rs, compiler/ Structured tree -> flat bytecode; lower.rs is the op table,
                         emit.rs the op sequence with branch fixups and run metering
  bytecode.rs            Op, BrTarget, CompiledFunction; depths and arities in words
  flat_executor/         Default engine; words.rs is the untagged word model
  executor.rs            Structured engine, with control.rs, frame.rs, stack.rs
  ops/                   Pure instruction implementations shared by both engines
  trap.rs                One-byte Trap that operations return
  host.rs                WasmType, IntoHostFunc: typed Rust closures as imports
  imports.rs             Import resolution
  memory.rs, table.rs    Linear memory and tables, bounds-checked, no unsafe
  profile.rs             Dispatch counts behind the instruction-profile feature
  wasi/                  WASI preview1 plus AssemblyScript env.abort

src/main.rs              CLI: run, dump, compile
src/bin/superinstruct.rs Instruction-sequence analysis tool, not published
build.rs                 Fixture directory watches; trampoline vs tail-call cfg

tests/                   Integration tests, spec fixtures, regressions
benches/                 Criterion benchmarks and their WAT modules
fuzz/                    cargo-fuzz targets and dictionaries
scripts/                 Profiling, layout builds, counter comparison, corpus prep
examples/                hello.wat, AssemblyScript WASI programs, Filecoin CommP
```

Each module opens with a `//!` comment that explains its mechanism, not its
API: how frames are laid out, why a trap is one byte, what a word holds.
That is the house style and the main teaching surface. `flat_executor/mod.rs`
and `words.rs` are the examples to match.

### The flat engine

Operands and locals live on one shared stack of untagged 64-bit words. A
`v128` takes two words; every other value takes one; every type's zero is
all-zero words. Each op has its own handler; in optimised x86-64 builds a
handler ends by calling the next op's handler in tail position, which
compiles to a jump, so every op dispatches from its own predicted branch.
Elsewhere, and in debug builds, handlers return to a loop, the trampoline.
`KRASM_TRAMPOLINE=1` at build time forces the loop for comparison, and
`scripts/check_dispatch.py` verifies that a release binary really tail-calls.

The instruction budget is charged per run of straight-line ops where control
lands, not per op. Budgets count engine operations, not portable fuel, so
the two engines exhaust at different points.

Imported calls suspend the executor and hand control to the `Store`, which
performs the call and resumes with the results. The executor keeps its stack
across the suspension, so host functions can re-enter the store.

### Where unsafe lives

`unsafe` is confined to the flat executor's word stack (`flat_executor/`)
and the structured engine's instruction slice (`executor.rs`). Linear memory
and tables are fully bounds-checked safe Rust. Every unsafe function states
its contract under `# Safety`; every unsafe block that does more than
discharge such a contract carries a `SAFETY:` comment naming the invariant
it relies on and who upholds it. The flat executor's tests run under Miri
with both Stacked and Tree Borrows (`./check.sh -m`). New unsafe outside
these files needs a reason in the PR and the same evidence.

## Security

Threat model: anyone can hand krasm a module. The embedder trusts the host;
the module is the adversary. Resource exhaustion is in scope.

Standing defences, which changes must preserve:

- `parser/limits.rs` caps every count a module can claim, aligned with V8.
  A new section, count or length read from bytes gets a limit before it gets
  an allocation.
- Every arithmetic step on a guest-supplied value is checked. Pointers and
  lengths from guest `i32`s are validated for sign and range before any
  cast to `usize`; offsets into guest memory are combined with `checked_add`
  and `checked_mul`, never bare `+`.
- Linear memory bounds-checks every access and validates growth against the
  maximum. Tables likewise.
- Execution is bounded three ways: an optional instruction budget per
  `invoke_export`, a byte-limited stack (`Store::set_stack_limit`), and a
  call-depth backstop. Deep recursion in the guest cannot overflow the
  host's stack.
- WASI exposes only preopened directories. Paths are canonicalised against
  the preopen before use. Open file descriptors and iovec counts are capped
  (`WasiContextBuilder`). Unimplemented calls return `NOSYS`, never a
  partial result.
- Every parser and both engines are fuzzed (`fuzz/`). The execution fuzzers
  run every export on both engines and fail on any difference in result or
  trap kind, with debug assertions on.
- Dependencies are few and pinned by `Cargo.lock`. Dependabot groups minor
  and patch bumps; the depsound workflow reviews manifest changes on every
  PR. A new runtime dependency needs a sentence in the PR saying why the
  crate should carry it.

Known gaps, each a welcome contribution:

- `resolve_path` canonicalises then opens, a time-of-check gap. Closing it
  means opening relative to the preopen's descriptor (`openat`).
- `path_open` always follows symlinks; the spec says the final component
  should not when `LOOKUP_SYMLINK_FOLLOW` is absent.

Rules of engagement for a security finding: fix it, add the regression as a
`.wast` file or unit test, run the relevant fuzzer, and say in the PR what
input triggered it and what now stops it. Do not describe exploitation
beyond what the test shows.

## Making a change

1. Read the spec section first (`wg-2.0`, linked under References). For a
   WASI call, read its entry in the preview1 docs.
2. Write the test first and watch it fail. A runtime behaviour gets a
   `.wast` file in `tests/regressions/`; the runner discovers it and runs it
   on both engines. A parser or validator behaviour gets a unit test beside
   the code. A WASI behaviour gets a case in `tests/wasi_tests.rs`.
3. Run the focused test, then `./check.sh`. It must pass before the work is
   called done. It runs fmt, clippy with `-D warnings` on all targets and
   features in debug and release, rustdoc with `-D warnings`, every test
   suite in debug, release and each feature set, the dispatch check, the
   profiling scripts' own tests, the SIMD CommP example and the WASI
   example programs. Prerequisites: stable Rust with the `wasm32-wasip1`
   target, Node.js and npm, Python 3.9+.
4. Both engines, always. An instruction, trap or host-boundary behaviour
   that lands in one engine lands in the other in the same change, with the
   pure operation in `ops/` shared between them.
5. Anything on the execution path: measure, as described under Performance,
   and put the numbers in the PR.
6. Public API change: rustdoc on the item, the `lib.rs` examples if they are
   affected, the README. Rustdoc examples compile as tests.
7. One concern per PR. No reformatting, renaming or tidying of unrelated
   code in the same change.

### Adding an instruction

1. Add the opcode to `InstructionKind` in `src/parser/instruction/mod.rs`.
2. Decode it in `decode.rs`; encode it in `encode.rs` so it round-trips.
3. Give it a signature in `signature.rs`: `fixed_signature`, or
   `context_signature` when its types come from an immediate. Validation and
   the compiler share that one table.
4. Implement it as a pure function in `src/runtime/ops/`, returning
   `Result<_, Trap>` if it can trap.
5. Wire both engines: an arm in `executor.rs`; an `Op` in `bytecode.rs`,
   lowered in `compiler/lower.rs`, with a handler in the `handlers!` list in
   `flat_executor/mod.rs`.
6. Tests: decode, encode round trip, a `.wast` regression covering the trap
   cases, and the disassembly if `wasm-objdump` prints it differently.

## Testing

Everything runs with `cargo test`; `./check.sh` is the full gate.

| Suite | What it proves |
|-------|----------------|
| Unit tests beside the code | Decoder, validator, lexer, parser, ops, executor internals |
| `tests/wast_tests.rs` | Every pinned spec file and every local regression, on both engines |
| `tests/dump_tests.rs` | `Module` display matches `wasm-objdump` output in `tests/spec/*.json` |
| `tests/encoder_tests.rs` | Encode -> parse -> encode is byte-stable |
| `tests/validation_tests.rs` | The decoder's inline validation and `ValidatedModule::new` agree |
| `tests/host_boundary_tests.rs` | Signature and value checks where calls leave validated code |
| `tests/instruction_budget_tests.rs` | Hand-counted budgets complete at N and exhaust at N-1 |
| `tests/instruction_profile_tests.rs` | Dispatch counts, behind `instruction-profile` |
| `tests/wasi_tests.rs` | WASI calls against isolated temporary directories, both engines |
| `fuzz/` | No panics in parsers; engines agree on every result and trap |

Fixtures:

- `tests/spec/wast/` is a verbatim copy of the spec suite at `wg-2.0`. Its
  `README.md` records the pin and how to update it. Do not edit the files;
  update the pin.
- `tests/spec/*.json` hold base64 modules and `wasm-objdump` output,
  generated from `.wast` by `tests/compile_test.mjs` using wabt's
  `wast2json` and `wasm-objdump`. Regenerate rather than hand-edit.
- `tests/regressions/*.wast` are local cases. Each one encodes a bug the
  pinned suite did not catch; name the file for the behaviour.
- `build.rs` watches both fixture directories, so adding a file is enough.

Fuzzing:

```bash
cargo install cargo-fuzz
./fuzz/seed_corpus.sh
./check.sh -f                        # parse_module for 60s after the checks
./check.sh -f 300 -t execute_module  # another target, longer
cargo +nightly fuzz run -a generate_module -- -max_total_time=60
```

Targets: `parse_module`, `execute_module`, `generate_module`, `lex_wat`,
`parse_wat`. `-a` keeps debug assertions, including the flat executor's
frame check. With `--features instruction-profile`, each completed flat call
is also replayed under exactly the budget its profile counts, which must
complete, and one fewer, which must exhaust. A crash found by fuzzing
becomes a regression file before it becomes a fix.

## Performance

Measurement protocol:

- A change that leaves the set of executed handlers unchanged is judged on
  instructions retired. A change that splits, adds or reselects handlers is
  judged on cycles and branch misses.
- Cycles and wall time vary 10 to 25 percent with code placement. Compare
  them only across groups of seeded layouts, never between two single
  builds. `python3 scripts/profile.py build --seeds 5 [--rev REV]` builds
  the layouts; `python3 scripts/profile.py compare --cpu N --arm a=... --arm b=...`
  reads the counters.
- `python3 scripts/prepare_runtimes.py` builds pinned Wasmtime and Wasmi
  under `target/runtimes`; `python3 scripts/profile.py external` compares
  wall time against them. Other runtimes are evidence of what is possible,
  not designs to copy.
- A `perf:` commit or PR body carries before and after numbers, the
  workload, and the machine. Small deltas are noise; do not claim them.

Workloads: `cargo bench --bench execution` runs `noop_loop`,
`fib_iterative`, `fib_recursive`, `memcpy` and `primes` on both engines
through the `Store` API; `cargo bench --bench validation` times parsing.
`scripts/profile_workloads.json` names the CommP and corpus workloads. The
CommP input is 500,000 bytes of `0x42`, generated locally:

```bash
python3 -c "open('benches/commp_bench_500k.bin', 'wb').write(b'\x42' * 500000)"
```

Its SHA-256 is pinned in `examples/commp-simd/build.py`.

Where the time goes, for an agent choosing what to attack: dispatch and the
per-op bookkeeping around it, memory access through the execution context on
every load and store, and the budget and bounds checks that remain per op.
Any of these is fair game under the protocol above. Fusing ops into
superinstructions is not off the table, but it is a design, not a patch;
write the case up before writing code.

## Code style

- Rust 2024 edition, stable toolchain, `rustfmt` at 120 columns, clippy
  clean with `-D warnings` on all targets and features. Fix lint rather than
  allowing it; an `allow` that stays carries a comment saying why the lint is
  wrong here.
- Australian English in prose, comments and messages: initialise, analyse,
  behaviour.
- `use` at the top of the file. No imports inside functions outside test
  modules.
- Name things as the spec names them: `br_table`, `funcref`, `memarg`,
  `elem`. A function implementing spec behaviour cites the section (`spec
  4.4.8`) in its doc comment.
- Comments earn their place and their word count. They describe the code as
  it is now and what a reader needs to know to trust it: the invariant, the
  spec rule, the reason a fast path is shaped the way it is. Never what
  changed, why a line was added or which issue it fixed; that belongs in the
  commit. No step-by-step narration. No "Step 1".
- Module docs (`//!`) explain the mechanism and the design, with a diagram
  where layout matters. Item docs explain the contract. Both build as
  rustdoc with `-D warnings`, so links must resolve.
- Simple language in prose, docs and messages. Short sentences. Plain
  punctuation: no em or en dashes, no emoji, no characters that break
  monospace alignment.
- Follow existing patterns before inventing new ones. Nothing new at the
  repository root. Source in `src/`, tests in `tests/`, tooling in
  `scripts/`.
- Run `cargo fmt` before handing work over; `check.sh` fails on formatting.

## Commits and pull requests

Conventional commits drive release-plz: on every push to `master` it opens
or updates a release PR with the version bump and changelog; merging that PR
tags the release and publishes to crates.io. The commit type is therefore a
release decision: `feat` for new behaviour, `fix` for corrected behaviour,
`perf` for a measured speedup, `!` on the type for anything an embedder must
react to. release-plz sizes the bump from those types and the current `0.x`
version.

- Subject: imperative, lower case, no trailing period, under 72 characters.
  Scopes in use: `runtime`, `flat`, `compiler`, `bytecode`, `ops`,
  `validate`, `parser`, `encoder`, `wat`, `wast`, `wasi`, `store`, `simd`,
  `cli`, `profiling`, `scripts`, `fuzz`, `bench`, `check`, `ci`, `deps`.
- Terse and undecorated. No labels, badges or boilerplate.
- A breaking change says who is affected and what they must do.
- History is linear. Dependency PRs are squash-merged with the title as the
  subject.
- `CHANGELOG.md` and the version in `Cargo.toml` are generated. Never edit
  them.
- Do not change CI, release configuration, dependabot or crate metadata
  unless that is the point of the PR.
- Agents do not commit, push, tag or open PRs on their own. Prepare the
  change and the message; the person driving performs the Git step, or asks
  for it by name.

## Direction

For an agent asked to improve the project without a brief. Good work,
roughly in order of value:

- **Close a known security gap** listed above, with the test that proves
  it closed.
- **Run the fuzzers longer** and turn anything they find into a regression
  file and a fix. Add a seed when a new corner appears.
- **Find a panic reachable from input.** Grep for `unwrap`, `expect`,
  indexing and `as` casts on paths fed by module bytes or guest memory, and
  replace each with a checked path and a test.
- **Teach better.** A module whose `//!` comment states only its API, a
  function whose shape hides the spec rule it implements, a magic number
  without its spec name. Rewrite for the reader who is learning
  WebAssembly, keep behaviour identical, and prove it with the existing
  tests.
- **Make the flat engine faster** under the measurement protocol, one
  mechanism per PR.
- **Shrink the structured engine's surface** without changing what it
  proves: it exists to disagree with the flat engine when the flat engine is
  wrong.

Work to avoid:

- Landing part of a new proposal, or any opcode without both engines, the
  encoder, validation and a spec file.
- Optimising the structured engine, or adding to it beyond what the flat
  engine does.
- `unsafe` for a small win, or anywhere a bounds check costs nothing
  measurable.
- Weakening a limit, a bounds check or a validation rule to make a module
  work. The module is wrong.
- A new dependency where a hundred lines of clear Rust would do.
- Touching the spec fixtures, the release workflow or the dependabot
  configuration as a side effect.
- Renaming or restructuring for taste. The layout teaches; change it only
  when the lesson improves.

## Keep this file true

When a change makes anything here inaccurate (a module moved, a limit or
setter renamed, a check added to `check.sh`, a gap closed), update this file
in the same PR. Describe the project as it is. No history, no change notes.

## References

- WebAssembly 2.0 specification, `wg-2.0`:
  https://www.w3.org/TR/wasm-core-2/ and
  https://webassembly.github.io/spec/core/
- Spec test suite: https://github.com/WebAssembly/spec/tree/wg-2.0/test/core
- WASI preview1:
  https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md
- V8 implementation limits, which `parser/limits.rs` mirrors:
  https://chromium.googlesource.com/v8/v8/+/main/src/wasm/wasm-limits.h
- wabt (`wast2json`, `wasm-objdump`) for fixtures:
  https://github.com/WebAssembly/wabt
- AssemblyScript, for the example programs: https://www.assemblyscript.org/
- Filecoin CommP, the SIMD example's algorithm: `examples/commp/README.md`
- Crate: https://crates.io/crates/krasm
