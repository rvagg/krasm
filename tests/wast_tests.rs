#[cfg(test)]
mod tests {
    use krasm::wast::*;
    use krasm::{EngineKind, ImportObject, Module, Store, Value};
    use rstest::rstest;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    /*
     * Native .wast spec test runner.
     *
     * Parses .wast files directly using krasm::wast::parse_script().
     * Executes all assertion types against our runtime.
     */

    struct WastRunner {
        store: Store,
        spectest_imports: ImportObject,
        /// Maps $name identifiers to instance IDs
        named_instances: HashMap<String, usize>,
        /// Maps registered "as" names to instance IDs
        registered_instances: HashMap<String, usize>,
        last_instance: Option<usize>,
        /// Module registry for binary parser (spectest etc.)
        module_registry: HashMap<String, Module>,
    }

    impl WastRunner {
        fn new() -> Self {
            let mut module_registry = HashMap::new();
            module_registry.insert("spectest".to_string(), create_spectest_module());

            let mut store = Store::new();
            // KRASM_FLAT=1 runs the whole suite on the flat bytecode engine,
            // as a differential harness against the structured default.
            if std::env::var_os("KRASM_FLAT").is_some_and(|v| v == "1") {
                store.set_engine(EngineKind::Flat);
            }
            let spectest_imports = create_spectest_imports(&mut store);

            WastRunner {
                store,
                spectest_imports,
                named_instances: HashMap::new(),
                registered_instances: HashMap::new(),
                last_instance: None,
                module_registry,
            }
        }

        /// Parse and validate a WastModule. Both binary and WAT paths perform type-level validation.
        fn parse_module(&self, wast_module: &WastModule) -> Result<Module, String> {
            match wast_module {
                WastModule::Binary(bytes) => {
                    let mut reader = krasm::parser::reader::Reader::new(bytes.clone());
                    krasm::parser::parse(&self.module_registry, "<binary>", &mut reader).map_err(|e| format!("{e}"))
                }
                _ => {
                    let source = wast_module.to_wat_source().ok_or_else(|| "no WAT source".to_string())?;
                    krasm::wat::parse(&source).map_err(|e| format!("{e}"))
                }
            }
        }

        /// Parse and instantiate a module, returning the instance ID.
        fn instantiate_module(&mut self, wast_module: &WastModule, name: Option<&str>) -> Result<usize, String> {
            let module = self.parse_module(wast_module)?;

            let instance_id = self
                .store
                .create_instance(Arc::new(module), Some(&self.spectest_imports))
                .map_err(|e| format!("{e}"))?;

            if let Some(n) = name {
                self.named_instances.insert(n.to_string(), instance_id);
            }
            self.last_instance = Some(instance_id);
            Ok(instance_id)
        }

        /// Register a module's exports under a given name for imports.
        fn register_module(&mut self, as_name: &str, module_name: Option<&str>) {
            let instance_id = if let Some(name) = module_name {
                *self
                    .named_instances
                    .get(name)
                    .unwrap_or_else(|| panic!("module ${name} not found for register"))
            } else {
                self.last_instance.expect("no module to register")
            };

            self.registered_instances.insert(as_name.to_string(), instance_id);

            self.store
                .register_exports(instance_id, as_name, &mut self.spectest_imports)
                .unwrap_or_else(|e| panic!("register_exports failed: {e}"));
        }

        /// Resolve which instance an action targets.
        fn resolve_instance(&self, module_name: &Option<String>) -> usize {
            if let Some(name) = module_name {
                // Try named instances first, then registered
                if let Some(&id) = self.named_instances.get(name) {
                    return id;
                }
                if let Some(&id) = self.registered_instances.get(name) {
                    return id;
                }
                panic!("module ${name} not found");
            }
            self.last_instance.expect("no module available for action")
        }

        /// Execute an action, returning the result values.
        fn execute_action(&mut self, action: &WastAction) -> Result<Vec<Value>, String> {
            match action {
                WastAction::Invoke { module, name, args } => {
                    let instance_id = self.resolve_instance(module);
                    let runtime_args = convert_args(args)?;
                    self.store
                        .invoke_export(instance_id, name, runtime_args, None)
                        .map_err(|e| format!("{e}"))
                }
                WastAction::Get { module, name } => {
                    let instance_id = self.resolve_instance(module);
                    let value = self
                        .store
                        .get_global_export(instance_id, name)
                        .map_err(|e| format!("{e}"))?;
                    Ok(vec![value])
                }
            }
        }
    }

    /// Check whether an actual error message matches a spec-expected message.
    ///
    /// Uses case-insensitive substring matching: the expected message may appear
    /// within the actual message. For cases where krasm uses genuinely different
    /// terminology, a curated equivalence table provides explicit mappings.
    /// Each mapping has a justification comment.
    fn error_message_matches(actual: &str, expected: &str) -> bool {
        let actual_lower = actual.to_lowercase();
        let expected_lower = expected.to_lowercase();

        // Case-insensitive substring: spec message found within our error
        if actual_lower.contains(&expected_lower) {
            return true;
        }

        // Each equivalence maps a spec message to krasm alternatives with justification.
        // Entries are grouped by category. Within each entry, the spec message is
        // matched case-insensitively (exact or substring), then each krasm alternative
        // is checked as a substring of our actual error message (also case-insensitive).
        let equivalences: &[(&str, &[&str])] = &[
            // ================================================================
            // "unexpected token" — spec catch-all for WAT syntax errors.
            // Our parser provides specific diagnostics for each condition.
            // ================================================================
            //
            // Missing or wrong-type literal value in const expression,
            // e.g. `(i32.const)` or `(f32.const nan:canonical)` in i32 context
            (
                "unexpected token",
                &["expected i32", "expected i64", "expected f32", "expected f64"],
            ),
            // Wrong token shape: list where atom expected, missing lane index,
            // wrong literal type (float where integer needed), wrong token for
            // value type, misordered param/result, missing v128.const lane type
            (
                "unexpected token",
                &[
                    "expected atom",
                    "expected lane index",
                    "expected integer, found",
                    "expected value type",
                    "result before parameter",
                    "v128.const requires lane type",
                ],
            ),
            // Keywords (type/param/result) in instruction position inside
            // control block or function bodies — not valid instructions
            (
                "unexpected token",
                &[
                    "unknown instruction: type",
                    "unknown instruction: param",
                    "unknown instruction: result",
                ],
            ),
            // Conflicting type annotations on control blocks (e.g. duplicate
            // `(type $t)` or `(type $t) (result i32)` that disagree): spec
            // catches at parse time, our validator catches as type conflict
            ("unexpected token", &["type mismatch"]),
            // ================================================================
            // "unknown operator" — spec catch-all for malformed tokens/operands.
            // Our parser identifies the specific lexing or parsing failure.
            // ================================================================
            //
            // Unknown instruction keyword (e.g. obsolete `anyfunc`)
            ("unknown operator", &["unknown instruction", "expected value type"]),
            // Malformed number literals: incomplete hex, bad hex float
            ("unknown operator", &["expected hex digits after", "invalid hex float"]),
            // Malformed instruction operands (alignment/offset)
            ("unknown operator", &["invalid align", "invalid offset"]),
            // Leading-underscore numbers, dotted numbers, invalid nan syntax —
            // lexer tokenises as Keyword, parser reports type expectation failure
            ("unknown operator", &["found keyword"]),
            // ================================================================
            // Synonyms: identical condition, different wording
            // ================================================================
            ("undefined element", &["uninitialized element"]),
            ("duplicate memory", &["multiple memories"]),
            ("import after memory", &["multiple memories"]),
            ("unknown import", &["unknown export"]),
            ("incompatible import type", &["type mismatch"]),
            // ================================================================
            // Subsumption: we validly don't distinguish these sub-categories
            // ================================================================
            ("invalid result arity", &["type mismatch"]),
            ("inline function type", &["type mismatch"]),
            // ================================================================
            // More specific: krasm reports a more informative message
            // ================================================================
            ("data segment does not fit", &["out of bounds"]),
            ("elements segment does not fit", &["out of bounds"]),
            ("i32 constant out of range", &["constant out of range"]),
            ("i32 constant", &["invalid offset"]),
            (
                "constant out of range",
                &["i64 out of range", "expected i64, found float"],
            ),
            ("unknown label", &["br_table requires at least one label"]),
            // ================================================================
            // Binary format: duplicate code sections — spec detects as
            // section ordering violation, we detect as count mismatch
            // because the first code section's count != function count
            // ================================================================
            (
                "unexpected content after last section",
                &["function and code section have inconsistent lengths"],
            ),
            // ================================================================
            // Binary format: oversize alignment exponent (2^32+) in memop
            // flags — spec treats as binary malformation, we report at
            // validation as alignment exceeding natural width
            // ================================================================
            ("malformed memop flags", &["alignment must not be larger than natural"]),
            // ================================================================
            // SIMD-specific
            // ================================================================
            // Shuffle with non-integer lane values: extra tokens remain after
            // parsing 16 integer lanes (e.g. `15.0` → `15` consumed, `.0` left)
            (
                "malformed lane index",
                &["lane index out of range", "unexpected token after expression"],
            ),
            // Wrong lane count: parser fails on missing or wrong-type lane value
            (
                "wrong number of lane literals",
                &[
                    "expected v128 lane value",
                    "expected integer, found",
                    "expected f32",
                    "expected f64",
                ],
            ),
            // ================================================================
            // Global/element init expressions
            // ================================================================
            (
                "type mismatch",
                &["expected init expression", "element init expression missing ref.func"],
            ),
            (
                "constant expression required",
                &["type mismatch", "element init expression missing ref.func"],
            ),
        ];
        for (spec_msg, our_msgs) in equivalences {
            if expected_lower == *spec_msg || expected_lower.contains(spec_msg) {
                for our_msg in *our_msgs {
                    if actual_lower.contains(our_msg) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn run_wast_file(path: &PathBuf) {
        let file_name = path.file_name().unwrap().to_str().unwrap();

        let source = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

        let script = parse_script(&source).unwrap_or_else(|e| panic!("failed to parse {file_name}: {e}"));

        let mut runner = WastRunner::new();
        let mut stats = Stats::default();
        let mut failures: Vec<String> = Vec::new();
        let mut module_broken = false; // Set when a module fails to instantiate

        for command in &script.commands {
            let line = command_line(command);
            match command {
                WastCommand::Module { name, module, .. } => match runner.instantiate_module(module, name.as_deref()) {
                    Ok(_) => {
                        stats.modules += 1;
                        module_broken = false;
                    }
                    Err(e) => {
                        failures.push(format!("{file_name}:{line}: module instantiation failed: {e}"));
                        module_broken = true;
                    }
                },

                WastCommand::Register {
                    as_name, module_name, ..
                } => {
                    if !module_broken {
                        runner.register_module(as_name, module_name.as_deref());
                        stats.registers += 1;
                    }
                }

                WastCommand::Action { action, .. } => {
                    if module_broken {
                        stats.skipped += 1;
                        continue;
                    }
                    match runner.execute_action(action) {
                        Ok(_) => stats.actions += 1,
                        Err(e) => failures.push(format!("{file_name}:{line}: action failed: {e}")),
                    }
                }

                WastCommand::AssertReturn { action, expected, .. } => {
                    if module_broken {
                        stats.skipped += 1;
                        continue;
                    }
                    match runner.execute_action(action) {
                        Ok(results) => match match_results(&results, expected) {
                            Ok(()) => stats.assert_return += 1,
                            Err(msg) => {
                                failures.push(format!("{file_name}:{line}: assert_return: {msg}"));
                            }
                        },
                        Err(e) => failures.push(format!("{file_name}:{line}: assert_return invoke failed: {e}")),
                    }
                }

                WastCommand::AssertTrap { action, message, .. } => {
                    if module_broken {
                        stats.skipped += 1;
                        continue;
                    }
                    match runner.execute_action(action) {
                        Err(e) => {
                            let e_str = e.to_string();
                            if error_message_matches(&e_str, message) {
                                stats.assert_trap += 1;
                            } else {
                                failures.push(format!(
                                    "{file_name}:{line}: assert_trap: expected '{message}', got '{e_str}'"
                                ));
                            }
                        }
                        Ok(results) => {
                            failures.push(format!(
                                "{file_name}:{line}: assert_trap: expected trap '{message}', got {results:?}"
                            ));
                        }
                    }
                }

                WastCommand::AssertModuleTrap { module, message, .. } => {
                    match runner.instantiate_module(module, None) {
                        Err(e) => {
                            let e_str = e.to_string();
                            if error_message_matches(&e_str, message) {
                                stats.assert_trap += 1;
                            } else {
                                failures.push(format!(
                                    "{file_name}:{line}: assert_trap (module): expected '{message}', got '{e_str}'"
                                ));
                            }
                        }
                        Ok(_) => {
                            failures.push(format!(
                                "{file_name}:{line}: assert_trap (module): expected trap '{message}'"
                            ));
                        }
                    }
                }

                WastCommand::AssertInvalid { module, message, .. } => match runner.parse_module(module) {
                    Err(e) => {
                        if error_message_matches(&e, message) {
                            stats.assert_invalid += 1;
                        } else {
                            failures.push(format!(
                                "{file_name}:{line}: assert_invalid: expected '{message}', got '{e}'"
                            ));
                        }
                    }
                    Ok(_module) => {
                        failures.push(format!(
                            "{file_name}:{line}: assert_invalid: expected error '{message}', but validated OK"
                        ));
                    }
                },

                WastCommand::AssertMalformed { module, message, .. } => match runner.parse_module(module) {
                    Err(e) => {
                        if error_message_matches(&e, message) {
                            stats.assert_malformed += 1;
                        } else {
                            failures.push(format!(
                                "{file_name}:{line}: assert_malformed: expected '{message}', got '{e}'"
                            ));
                        }
                    }
                    Ok(_) => {
                        failures.push(format!(
                            "{file_name}:{line}: assert_malformed: expected error '{message}', but parsed OK"
                        ));
                    }
                },

                WastCommand::AssertUnlinkable { module, message, .. } => {
                    match runner.instantiate_module(module, None) {
                        Err(e) => {
                            let e_str = e.to_string();
                            if error_message_matches(&e_str, message) {
                                stats.assert_unlinkable += 1;
                            } else {
                                failures.push(format!(
                                    "{file_name}:{line}: assert_unlinkable: expected '{message}', got '{e_str}'"
                                ));
                            }
                        }
                        Ok(_) => {
                            failures.push(format!(
                                "{file_name}:{line}: assert_unlinkable: expected error '{message}'"
                            ));
                        }
                    }
                }

                WastCommand::AssertUninstantiable { module, message, .. } => {
                    match runner.instantiate_module(module, None) {
                        Err(e) => {
                            let e_str = e.to_string();
                            if error_message_matches(&e_str, message) {
                                stats.assert_uninstantiable += 1;
                            } else {
                                failures.push(format!(
                                    "{file_name}:{line}: assert_uninstantiable: expected '{message}', got '{e_str}'"
                                ));
                            }
                        }
                        Ok(_) => {
                            failures.push(format!(
                                "{file_name}:{line}: assert_uninstantiable: expected error '{message}'"
                            ));
                        }
                    }
                }

                WastCommand::AssertExhaustion { action, message, .. } => {
                    if module_broken {
                        stats.skipped += 1;
                        continue;
                    }
                    match runner.execute_action(action) {
                        Err(e) => {
                            let e_str = e.to_string();
                            if error_message_matches(&e_str, message) {
                                stats.assert_exhaustion += 1;
                            } else {
                                failures.push(format!(
                                    "{file_name}:{line}: assert_exhaustion: expected '{message}', got '{e_str}'"
                                ));
                            }
                        }
                        Ok(results) => {
                            failures.push(format!(
                                "{file_name}:{line}: assert_exhaustion: expected '{message}', got {results:?}"
                            ));
                        }
                    }
                }
            }
        }

        if failures.is_empty() {
            let skipped_msg = if stats.skipped > 0 {
                format!(" ({} skipped due to broken module)", stats.skipped)
            } else {
                String::new()
            };
            println!(
                "PASS {file_name}: {} modules, {} assert_return, {} assert_trap, \
                 {} assert_invalid, {} assert_malformed, {} assert_unlinkable, \
                 {} assert_uninstantiable, {} assert_exhaustion, \
                 {} registers, {} actions{skipped_msg}",
                stats.modules,
                stats.assert_return,
                stats.assert_trap,
                stats.assert_invalid,
                stats.assert_malformed,
                stats.assert_unlinkable,
                stats.assert_uninstantiable,
                stats.assert_exhaustion,
                stats.registers,
                stats.actions,
            );
        } else {
            let total_passed = stats.modules
                + stats.assert_return
                + stats.assert_trap
                + stats.assert_invalid
                + stats.assert_malformed
                + stats.assert_unlinkable
                + stats.assert_uninstantiable
                + stats.assert_exhaustion;
            let skipped_msg = if stats.skipped > 0 {
                format!(", {} skipped", stats.skipped)
            } else {
                String::new()
            };
            panic!(
                "FAIL {file_name}: {total_passed} passed, {} failed{skipped_msg}\n  {}",
                failures.len(),
                failures.join("\n  ")
            );
        }
    }

    fn command_line(cmd: &WastCommand) -> usize {
        match cmd {
            WastCommand::Module { span, .. }
            | WastCommand::Register { span, .. }
            | WastCommand::Action { span, .. }
            | WastCommand::AssertReturn { span, .. }
            | WastCommand::AssertTrap { span, .. }
            | WastCommand::AssertModuleTrap { span, .. }
            | WastCommand::AssertInvalid { span, .. }
            | WastCommand::AssertMalformed { span, .. }
            | WastCommand::AssertUnlinkable { span, .. }
            | WastCommand::AssertUninstantiable { span, .. }
            | WastCommand::AssertExhaustion { span, .. } => span.line as usize,
        }
    }

    #[derive(Default)]
    struct Stats {
        modules: usize,
        assert_return: usize,
        assert_trap: usize,
        assert_invalid: usize,
        assert_malformed: usize,
        assert_unlinkable: usize,
        assert_uninstantiable: usize,
        assert_exhaustion: usize,
        registers: usize,
        actions: usize,
        skipped: usize,
    }

    #[rstest]
    fn test_core_wast(#[files("tests/spec/wast/*.wast")] path: PathBuf) {
        run_wast_file(&path);
    }

    #[rstest]
    fn test_simd_wast(#[files("tests/spec/wast/simd/*.wast")] path: PathBuf) {
        run_wast_file(&path);
    }
}
