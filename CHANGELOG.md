# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2](https://github.com/rvagg/krasm/compare/v0.0.1...v0.0.2) - 2026-10-30

### Added

- *(profile)* count branch misses against external runtimes
- *(scripts)* compare against pinned Wasmtime and Wasmi builds
- *(flat)* byte-limited stack and deep recursion
- *(store)* configurable flat stack limit; release stack spikes when idle
- *(flat)* [**breaking**] execute on an untagged word stack with locals in the frame
- *(runtime)* slot-stack executor prototype behind slot-proto feature
- *(profiling)* compare code-layout groups built from seeded lld shuffles
- [**breaking**] make validation the instantiation trust boundary
- *(profiling)* add perf-counter comparison of frozen binaries
- *(profiling)* add pinned six-family WASI corpus and qualification tooling
- *(runtime)* add opt-in scalar superinstructions for the flat engine
- *(profiling)* collect instruction counts and rank fusion candidates
- *(bench)* add commp-wasm, profiling scripts
- make the flat executor the library and CLI default
- *(runtime)* instruction budget for flat executor
- *(runtime)* simd low form and float min/max ops in flat executor
- *(runtime)* simd rounding ops for flat executor
- *(runtime)* float abs & neg ops in flat executor
- *(runtime)* simd dot & q15 product ops in flat executor
- *(runtime)* simd pairwise add for flat executor
- *(runtime)* extended multiplication ops for flat executor
- *(runtime)* simd extension ops for flat executor
- *(runtime)* simd narrowing ops for flat executor
- *(runtime)* simd lane average ops for flat executor
- *(runtime)* simd minmax in flat executor
- feat(runtime) simd saturating arithmetic with the flat executor
- *(runtime)* simd abs, neg, popcnt in the flat executor
- *(runtime)* simd arithmetic ops in the flat executor
- *(runtime)* simd shift operations for flat executor
- *(runtime)* simd float comparison ops for flat executor
- *(runtime)* full simd integer comparison op set in flat executor
- *(runtime)* simd byte comparison operations in flat executor
- *(runtime)* simd lane reduction ops (all_true, bitmask) in flat executor
- *(runtime)* simd splat, shuffle, swizzle in flat executor
- *(runtime)* simd lane memory operations in flat executor
- *(runtime)* simd lane instructions in the flat executor
- *(runtime)* implement v128 short load operations
- *(runtime)* implement v128 load and store operations
- *(runtime)* add flat SIMD constants and bitwise operations
- *(runtime)* flat engine refs, tables, and bulk memory
- *(runtime)* flat engine float batch
- *(cli)* --engine flag for krasm run; named unsupported-op traps
- *(runtime)* flat engine i64 batch
- *(runtime)* flat engine i32 batch; one-line op arm structure
- *(runtime)* flat engine dispatch through the Store
- *(runtime)* call_indirect in flat executor
- *(runtime)* resumable external calls in flat executor
- *(runtime)* flat executor function calls with shared-stack semantics
- *(runtime)* flat executor memory ops and comprehensive test coverage
- *(runtime)* globals support and ExecContext for flat executor
- *(runtime)* flat bytecode compiler and executor with multi-value branches
- *(runtime)* initial PoC of flat bytecode compiler and executor

### Fixed

- *(runtime)* unwind on return, check foreign signatures, support re-entry
- refresh test discovery when regression fixtures change
- correct table indices and construct only the selected engine
- route start functions through the selected engine and Store dispatch
- correct import indices in table.init and function calls
- *(runtime)* typed local defaults and dead-code handling in flat compiler
- *(parser)* include parameters in binary-parsed local_count
- clippy clean on rust 1.96, lint all targets in check.sh

### Other

- *(ci)* publish crates with waxseal
- *(ci)* reset unpublished 0.0.2 release metadata
- *(compiler)* emit ops through an Emitter
- *(ci)* use Waxseal for releases
- update agents context docs
- *(flat)* give ops their exits, weights and targets
- *(flat)* cover branch results at every arity
- *(flat)* build handler errors out of line
- *(flat)* carry pc as a pointer to its slot
- *(flat)* keep each op only in its slot
- *(flat)* dispatch each op from its own handler by tail call
- *(deps)* bump the github-actions-minor-patch group with 2 updates ([#55](https://github.com/rvagg/krasm/pull/55))
- *(deps)* bump dtolnay/rust-toolchain ([#56](https://github.com/rvagg/krasm/pull/56))
- *(deps)* bump dtolnay/rust-toolchain ([#54](https://github.com/rvagg/krasm/pull/54))
- *(deps)* bump the github-actions-minor-patch group across 1 directory with 2 updates ([#53](https://github.com/rvagg/krasm/pull/53))
- *(flat)* stop dispatching block labels
- *(check)* optional Miri run of the flat executor tests
- *(flat)* map call_indirect funcrefs by address; resolve memory once per run
- *(flat)* charge the instruction budget per run; keep hot state in registers
- [**breaking**] hold globals as one entry; trim validator legacy; hide bytecode
- *(compiler)* move the fixed lowering table to compiler/lower.rs
- one definition of the words a value occupies
- remove the slot-stack prototype
- *(fuzz)* compare engines in execution fuzzers; fix two bugs found
- *(bytecode)* give Op an explicit u16 tag
- *(ops)* conversion, memory and SIMD ops as pure functions
- *(ops)* numeric, bitwise and comparison ops as pure functions
- share context-typed signatures; compile with Result
- *(compiler)* track and check operand types
- *(validate)* derive fixed instruction signatures from one table
- [**breaking**] remove the experimental superinstructions feature
- enable fat LTO and a single codegen unit for release
- *(runtime)* isolate superinstruction dispatch
- *(deps)* bump rstest from 0.26.1 to 0.27.0 ([#51](https://github.com/rvagg/krasm/pull/51))
- *(deps)* bump the cargo-minor-patch group with 3 updates ([#50](https://github.com/rvagg/krasm/pull/50))
- *(deps)* bump release-plz/action ([#49](https://github.com/rvagg/krasm/pull/49))
- opcode slice caching
- local.set/set lazy error construction, local.tee peek
- *(runtime)* streamline stack pops and isolate cold error paths
- lazy construction of error strings
- *(runtime)* move segment state and initialisation into Instance
- *(deps)* bump release-plz/action ([#48](https://github.com/rvagg/krasm/pull/48))
- *(deps)* bump the cargo-minor-patch group with 3 updates ([#47](https://github.com/rvagg/krasm/pull/47))
- *(deps)* bump dtolnay/rust-toolchain ([#46](https://github.com/rvagg/krasm/pull/46))
- release v0.0.2 ([#34](https://github.com/rvagg/krasm/pull/34))
- *(deps)* bump base64 from 0.22.1 to 0.23.0 ([#45](https://github.com/rvagg/krasm/pull/45))
- *(deps)* bump the cargo-minor-patch group across 1 directory with 5 updates ([#44](https://github.com/rvagg/krasm/pull/44))
- *(deps)* bump the github-actions-minor-patch group with 2 updates ([#43](https://github.com/rvagg/krasm/pull/43))
- *(deps)* bump the github-actions-minor-patch group with 3 updates ([#42](https://github.com/rvagg/krasm/pull/42))
- *(ci)* update to single depsound-action workflow v0.3 ([#41](https://github.com/rvagg/krasm/pull/41))
- *(deps)* bump release-plz/action ([#39](https://github.com/rvagg/krasm/pull/39))
- *(deps)* bump actions/setup-node from 6.5.0 to 7.0.0 ([#40](https://github.com/rvagg/krasm/pull/40))
- *(deps)* bump regex in the cargo-minor-patch group ([#38](https://github.com/rvagg/krasm/pull/38))
- *(ci)* tweak dependabot, add depsound, use sha refs ([#37](https://github.com/rvagg/krasm/pull/37))
- *(deps)* bump rand from 0.10.1 to 0.10.2 ([#36](https://github.com/rvagg/krasm/pull/36))
- *(deps)* bump actions/cache from 5.0.5 to 6.1.0 ([#35](https://github.com/rvagg/krasm/pull/35))
- *(wast)* KRASM_FLAT=1 runs the spec suite on the flat engine
- *(deps)* bump actions/checkout from 6.0.3 to 7.0.0 ([#33](https://github.com/rvagg/krasm/pull/33))
- *(deps)* bump actions/cache from 5 to 5.0.5 ([#30](https://github.com/rvagg/krasm/pull/30))
- *(deps)* bump rust-lang/crates-io-auth-action from 1.0.4 to 1.0.5 ([#31](https://github.com/rvagg/krasm/pull/31))
- *(deps)* bump release-plz/action from 0.5.129 to 0.5.130 ([#32](https://github.com/rvagg/krasm/pull/32))
- *(deps)* bump regex from 1.12.3 to 1.12.4 ([#27](https://github.com/rvagg/krasm/pull/27))
- *(deps)* bump actions/checkout from 6 to 6.0.3 ([#28](https://github.com/rvagg/krasm/pull/28))
- *(deps)* bump rust-lang/crates-io-auth-action from 1 to 1.0.4 ([#26](https://github.com/rvagg/krasm/pull/26))
- *(deps)* bump release-plz/action from 0.5 to 0.5.129 ([#25](https://github.com/rvagg/krasm/pull/25))
- *(deps)* bump serde_json from 1.0.149 to 1.0.150 ([#23](https://github.com/rvagg/krasm/pull/23))
- update agents file
- *(deps)* bump clap from 4.6.0 to 4.6.1
- *(deps)* bump rand from 0.10.0 to 0.10.1
- *(deps)* bump proptest from 1.10.0 to 1.11.0
- *(runtime)* extract flat executor resource ops to ExecContext methods
- Bump clap from 4.5.60 to 4.6.0 ([#18](https://github.com/rvagg/krasm/pull/18))
- Bump once_cell from 1.21.3 to 1.21.4 ([#17](https://github.com/rvagg/krasm/pull/17))

