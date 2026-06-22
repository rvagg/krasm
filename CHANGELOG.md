# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2](https://github.com/rvagg/krasm/compare/v0.0.1...v0.0.2) - 2026-06-22

### Added

- *(runtime)* flat executor function calls with shared-stack semantics
- *(runtime)* flat executor memory ops and comprehensive test coverage
- *(runtime)* globals support and ExecContext for flat executor
- *(runtime)* flat bytecode compiler and executor with multi-value branches
- *(runtime)* initial PoC of flat bytecode compiler and executor

### Other

- *(deps)* bump release-plz/action from 0.5 to 0.5.129 ([#25](https://github.com/rvagg/krasm/pull/25))
- *(deps)* bump serde_json from 1.0.149 to 1.0.150 ([#23](https://github.com/rvagg/krasm/pull/23))
- update agents file
- *(deps)* bump clap from 4.6.0 to 4.6.1
- *(deps)* bump rand from 0.10.0 to 0.10.1
- *(deps)* bump proptest from 1.10.0 to 1.11.0
- *(runtime)* extract flat executor resource ops to ExecContext methods
- Bump clap from 4.5.60 to 4.6.0 ([#18](https://github.com/rvagg/krasm/pull/18))
- Bump once_cell from 1.21.3 to 1.21.4 ([#17](https://github.com/rvagg/krasm/pull/17))
