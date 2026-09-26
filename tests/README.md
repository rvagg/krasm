# krasm Test Infrastructure

Tests cover parsing, encoding, disassembly, execution and WASI integration.

## Test Files

- `wast_tests.rs` - Native WAST runner for pinned spec files and local regressions
- `wasi_tests.rs` - WASI integration scenarios
- `encoder_tests.rs` - Binary round trips
- `dump_tests.rs` - Disassembly fixture comparisons
- `compile_test.mjs` - Compiles .wast files to .json format for testing
- `extract_utf8_tests.mjs` - Extracts UTF-8 validation tests to Rust unit tests
- `spec/` - Directory containing compiled test fixtures in JSON format
- `spec/wast/` - Pinned upstream WAST fixtures
- `regressions/` - Local WAST regression cases

## Running Tests

### Standard Tests
```bash
cargo test
```

### Execution Tests

Every WAST file and WASI scenario runs explicitly on both structured and flat
engines, independently of the library default. WASI filesystem cases own isolated
temporary directories so the engine cases can run concurrently.

```bash
cargo test --test wast_tests
cargo test --test wasi_tests
cargo test --test wast_tests -- structured
cargo test --test wast_tests -- flat
```

### UTF-8 Tests Only
```bash
cargo test utf8_validation
```

## Adding New Tests

For runtime regressions, add a `.wast` file to `regressions/`; the native runner
discovers it automatically and runs both engines. Keep upstream fixtures pinned.
For dump fixtures, compile upstream WAST to JSON:

```bash
node compile_test.mjs ../wasm-spec/test/core/testname.wast ./spec/testname.json
```

## UTF-8 Validation Tests

We extract UTF-8 validation tests from the spec and run them as Rust unit tests:

1. Compile UTF-8 test files (already done):
   ```bash
   node compile_test.mjs ../wasm-spec/test/core/utf8-custom-section-id.wast ./spec/utf8-custom-section-id.json
   node compile_test.mjs ../wasm-spec/test/core/utf8-import-field.wast ./spec/utf8-import-field.json
   node compile_test.mjs ../wasm-spec/test/core/utf8-import-module.wast ./spec/utf8-import-module.json
   ```

2. Extract and generate Rust tests:
   ```bash
   node extract_utf8_tests.mjs
   ```

This generates `src/parser/utf8_tests.rs` with 528 test cases.
