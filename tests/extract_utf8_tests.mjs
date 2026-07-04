import { readFile, writeFile } from 'fs/promises'

/**
 * Extract UTF-8 Test Cases from WebAssembly Spec Tests
 * 
 * This script extracts UTF-8 validation test cases from the WebAssembly spec test suite
 * and generates a Rust test file that can be run independently of the main test harness.
 * 
 * Background:
 * - The WebAssembly spec includes UTF-8 validation tests that use `(module binary ...)` syntax
 * - These tests check that invalid UTF-8 sequences are properly rejected by the parser
 * - The tests cover UTF-8 validation in custom sections, import fields, and import modules
 * 
 * Usage:
 * 1. First compile the UTF-8 test files using compile_test.mjs:
 *    ```
 *    node compile_test.mjs ../wasm-spec/test/core/utf8-custom-section-id.wast ./spec/utf8-custom-section-id.json
 *    node compile_test.mjs ../wasm-spec/test/core/utf8-import-field.wast ./spec/utf8-import-field.json
 *    node compile_test.mjs ../wasm-spec/test/core/utf8-import-module.wast ./spec/utf8-import-module.json
 *    ```
 * 
 * 2. Then run this script to extract and generate the Rust tests:
 *    ```
 *    node extract_utf8_tests.mjs
 *    ```
 * 
 * Output:
 * - Generates src/parser/utf8_tests.rs with all extracted test cases
 * - The generated file contains a single test function that runs all UTF-8 validation tests
 * 
 * Note: utf8-invalid-encoding.wast cannot be extracted because it uses `(module quote ...)` syntax
 * which is not supported by WABT's wast2json tool.
 */
async function extractUtf8Tests() {
    const testFiles = [
        'utf8-custom-section-id',
        'utf8-import-field',
        'utf8-import-module',
        'utf8-invalid-encoding'
    ];

    const allTests = [];

    for (const testFile of testFiles) {
        try {
            const jsonPath = `tests/spec/${testFile}.json`;
            const content = await readFile(jsonPath, 'utf8');
            const data = JSON.parse(content);
            
            // Extract each test case
            for (const [index, command] of data.spec.commands.entries()) {
                if (command.type === 'assert_malformed' && command.module_type === 'binary') {
                    const binaryKey = command.filename;
                    const binaryBase64 = data.bin[binaryKey];
                    const binaryBytes = Buffer.from(binaryBase64, 'base64');
                    
                    allTests.push({
                        source: testFile,
                        index: index,
                        line: command.line,
                        expectedError: command.text,
                        binary: Array.from(binaryBytes),
                        binaryHex: binaryBytes.toString('hex'),
                        description: `${testFile}:${command.line}`
                    });
                }
            }
            
            console.log(`Extracted ${data.spec.commands.length} tests from ${testFile}`);
        } catch (e) {
            console.log(`Skipping ${testFile}: ${e.message}`);
        }
    }

    // Write the extracted tests to a Rust file
    let rustCode = `// AUTO-GENERATED FILE - DO NOT EDIT
// Generated from WebAssembly spec UTF-8 tests

#[cfg(test)]
mod tests {
    use crate::parser;

    #[derive(Debug)]
    struct Utf8TestCase {
        name: &'static str,
        binary: &'static [u8],
        expected_error: &'static str,
    }

    const UTF8_TEST_CASES: &[Utf8TestCase] = &[
`;

    for (const test of allTests) {
        rustCode += `        Utf8TestCase {
            name: "${test.description}",
            binary: &[${test.binary.join(', ')}],
            expected_error: "${test.expectedError}",
        },
`;
    }

    rustCode += `    ];

    #[test]
    fn test_utf8_validation() {
        let mut passed = 0;
        let mut failed = 0;
        let module_registry = std::collections::HashMap::new();

        for test_case in UTF8_TEST_CASES {
            let mut reader = parser::reader::Reader::new(test_case.binary.to_vec());
            match parser::parse(&module_registry, test_case.name, &mut reader) {
                Ok(_) => {
                    eprintln!("Test {} should have failed with '{}', but succeeded", 
                             test_case.name, test_case.expected_error);
                    failed += 1;
                },
                Err(e) => {
                    let error_string = e.to_string();
                    // Accept different UTF-8 error messages
                    if error_string.contains(test_case.expected_error) || 
                       (test_case.expected_error.contains("malformed UTF-8") && 
                        error_string.contains("invalid utf-8")) {
                        passed += 1;
                    } else {
                        eprintln!("Test {} failed with '{}', expected '{}'", 
                                 test_case.name, error_string, test_case.expected_error);
                        failed += 1;
                    }
                }
            }
        }

        println!("UTF-8 validation tests: {} passed, {} failed", passed, failed);
        assert_eq!(failed, 0, "{} UTF-8 tests failed", failed);
    }
}
`;

    await writeFile('src/parser/utf8_tests.rs', rustCode);
    console.log(`\nGenerated src/parser/utf8_tests.rs with ${allTests.length} test cases`);
    
    // Also generate a summary report
    const summary = {
        totalTests: allTests.length,
        byFile: {}
    };
    
    for (const test of allTests) {
        if (!summary.byFile[test.source]) {
            summary.byFile[test.source] = 0;
        }
        summary.byFile[test.source]++;
    }
    
    console.log('\nSummary:');
    console.log(JSON.stringify(summary, null, 2));
}

extractUtf8Tests().catch(console.error);