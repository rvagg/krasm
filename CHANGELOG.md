# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2](https://github.com/rvagg/krasm/compare/v0.0.1...v0.0.2) - 2026-08-03

### Added

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

- *(runtime)* typed local defaults and dead-code handling in flat compiler
- *(parser)* include parameters in binary-parsed local_count
- clippy clean on rust 1.96, lint all targets in check.sh

### Other

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
