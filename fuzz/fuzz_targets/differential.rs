//! Differential execution: run a module's exports on both engines and require
//! the same outcome from each.
//!
//! The structured engine executes tagged values and checks their types as it
//! goes; the flat engine trusts validation and the typed compiler. Agreement
//! on every result and trap is evidence that the flat engine's assumptions
//! hold, including the shared instruction signature tables.
//!
//! With `instruction-profile`, each flat call is also checked against the
//! budget it should need: exactly the ops the profile counts.

use krasm::parser::module::{ExportIndex, ValueType};
use krasm::{EngineKind, RuntimeError, Store, ValidatedModule, Value};

/// Instruction budget per call. The engines count different units, so they
/// exhaust at different points; comparison stops at the first exhaustion.
const BUDGET: u64 = 100_000;

/// Invoke every exported function on both engines with `args(params)` and
/// panic on any difference in results or trap kinds.
pub fn compare_engines(module: &ValidatedModule, mut args: impl FnMut(&[ValueType]) -> Vec<Value>) {
    let mut stores = [EngineKind::Structured, EngineKind::Flat].map(|engine| {
        let mut store = Store::new();
        store.set_engine(engine);
        store
    });
    let [structured, flat] = &mut stores;
    let (structured_id, flat_id) = match (
        structured.create_instance(module, None),
        flat.create_instance(module, None),
    ) {
        (Ok(s), Ok(f)) => (s, f),
        (Err(s), Err(f)) => return assert_same_trap(&s, &f, "instantiation"),
        (s, f) => panic!("engines disagree on instantiation: structured {s:?}, flat {f:?}"),
    };
    #[cfg(feature = "instruction-profile")]
    let mut oracle = budget::Oracle::new(module);

    for export in &module.exports.exports {
        let ExportIndex::Function(func_idx) = export.index else {
            continue;
        };
        let Some(func_type) = module.get_function_type_by_idx(func_idx) else {
            continue;
        };
        let args = args(&func_type.parameters);
        let s = structured.invoke_export(structured_id, &export.name, args.clone(), Some(BUDGET));
        #[cfg(feature = "instruction-profile")]
        let before = budget::executed(flat, flat_id);
        let f = flat.invoke_export(flat_id, &export.name, args.clone(), Some(BUDGET));
        #[cfg(feature = "instruction-profile")]
        if !exhausted(&f) {
            oracle.check(&export.name, args, &f, budget::executed(flat, flat_id) - before);
        }
        match (&s, &f) {
            // Exhaustion depends on how each engine counts; state may differ after it.
            _ if exhausted(&s) || exhausted(&f) => return,
            (Ok(s), Ok(f)) => assert!(
                s.len() == f.len() && s.iter().zip(f).all(|(s, f)| same_value(s, f)),
                "engines disagree on {}: structured {s:?}, flat {f:?}",
                export.name
            ),
            (Err(s), Err(f)) => assert_same_trap(s, f, &export.name),
            _ => panic!("engines disagree on {}: structured {s:?}, flat {f:?}", export.name),
        }
    }
}

fn exhausted(outcome: &Result<Vec<Value>, RuntimeError>) -> bool {
    matches!(
        outcome,
        Err(RuntimeError::InstructionBudgetExhausted | RuntimeError::CallStackOverflow)
    )
}

/// Traps agree on kind; message wording may differ between engines.
fn assert_same_trap(s: &RuntimeError, f: &RuntimeError, what: &str) {
    assert!(
        std::mem::discriminant(s) == std::mem::discriminant(f),
        "engines trap differently in {what}: structured {s}, flat {f}"
    );
}

/// Equal values, comparing floats by bits so NaN payloads must match too.
fn same_value(s: &Value, f: &Value) -> bool {
    match (s, f) {
        (Value::F32(s), Value::F32(f)) => s.to_bits() == f.to_bits(),
        (Value::F64(s), Value::F64(f)) => s.to_bits() == f.to_bits(),
        _ => s == f,
    }
}

#[cfg(feature = "instruction-profile")]
mod budget {
    use krasm::{EngineKind, RuntimeError, Store, ValidatedModule, Value};

    /// Budget the instance has spent: each op's count times its weight.
    pub fn executed(store: &Store, id: usize) -> u64 {
        let instance = store.get_instance(id).expect("instance exists");
        let profile = instance.instruction_profile().expect("flat instances are profiled");
        (profile.functions.iter())
            .flat_map(|f| f.ops.iter().zip(f.counts))
            .map(|(op, count)| u64::from(op.weight()) * count)
            .sum()
    }

    /// Replays the calls made so far on fresh flat instances, under exact
    /// budgets, so each replay starts from the state the original call saw.
    pub struct Oracle<'m> {
        module: &'m ValidatedModule,
        calls: Vec<(String, Vec<Value>)>,
    }

    impl<'m> Oracle<'m> {
        pub fn new(module: &'m ValidatedModule) -> Self {
            Oracle {
                module,
                calls: Vec::new(),
            }
        }

        /// The state after the calls so far, with no budget applied.
        fn replay(&self) -> (Store, usize) {
            let mut store = Store::new();
            store.set_engine(EngineKind::Flat);
            let id = store.create_instance(self.module, None).expect("instantiated before");
            for (name, args) in &self.calls {
                let _ = store.invoke_export(id, name, args.clone(), None);
            }
            (store, id)
        }

        /// A call that executed `count` ops exhausts under one less: every
        /// charge pays for ops before they run. One that completed also
        /// completes again under exactly `count`.
        pub fn check(&mut self, name: &str, args: Vec<Value>, outcome: &Result<Vec<Value>, RuntimeError>, count: u64) {
            if let Some(short_budget) = count.checked_sub(1) {
                let (mut store, id) = self.replay();
                let short = store.invoke_export(id, name, args.clone(), Some(short_budget));
                assert!(
                    matches!(short, Err(RuntimeError::InstructionBudgetExhausted)),
                    "{name} under {short_budget} gave {short:?}, expected exhaustion"
                );
            }
            if let Ok(expected) = outcome {
                let (mut store, id) = self.replay();
                match store.invoke_export(id, name, args.clone(), Some(count)) {
                    Ok(results) => assert!(
                        results.len() == expected.len()
                            && results.iter().zip(expected).all(|(a, b)| super::same_value(a, b)),
                        "{name} returned {results:?} under its exact budget {count}, {expected:?} without"
                    ),
                    Err(error) => panic!("{name} failed under its exact budget {count}: {error}"),
                }
            }
            self.calls.push((name.to_string(), args));
        }
    }
}
