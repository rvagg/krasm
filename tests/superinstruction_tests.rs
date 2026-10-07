#![cfg(feature = "superinstructions")]

use krasm::{EngineKind, Store, Value};
use std::sync::Arc;

#[test]
fn scalar_families_match_wasm_arithmetic_at_boundaries() {
    let values = [i32::MIN, i32::MAX, -1, 0, 1, 31, 32, 33];
    for op in [
        "add", "sub", "mul", "and", "or", "xor", "shl", "shr_s", "shr_u", "rotl", "rotr",
    ] {
        for rhs in values {
            let source = format!(
                "(module (func (export \"run\") (param i32 i32) (result i32 i32 i32)
                    local.get 0 local.get 1 i32.{op}
                    local.get 0 i32.const {rhs} i32.{op}
                    local.get 0 local.get 0 i32.{op}))"
            );
            let module = Arc::new(krasm::wat::parse(&source).unwrap());
            let mut reference = Store::new();
            reference.set_engine(EngineKind::Structured);
            let reference_id = reference.create_instance(Arc::clone(&module), None).unwrap();
            let mut fused = Store::new();
            let fused_id = fused.create_instance(module, None).unwrap();
            for lhs in values {
                let args = vec![Value::I32(lhs), Value::I32(rhs)];
                let expected = reference
                    .invoke_export(reference_id, "run", args.clone(), None)
                    .unwrap();
                let actual = fused.invoke_export(fused_id, "run", args, None).unwrap();
                assert_eq!(actual, expected, "i32.{op}: lhs={lhs}, rhs={rhs}");
            }
        }
    }
}

#[test]
fn trailing_tee_preserves_results_and_aliased_local_operands() {
    let values = [i32::MIN, i32::MAX, -1, 0, 1, 31, 32, 33];
    for op in [
        "add", "sub", "mul", "and", "or", "xor", "shl", "shr_s", "shr_u", "rotl", "rotr",
    ] {
        for destination in 0..3 {
            for rhs in values {
                let source = format!(
                    "(module (func (export \"run\") (param i32 i32) (result i32 i32 i32 i32 i32)
                        (local i32)
                        local.get 0 i32.const {rhs} i32.{op} local.tee {destination}
                        local.get 0 local.get 1 i32.{op} local.tee {destination}
                        local.get 0 local.get 1 local.get 2))"
                );
                let module = Arc::new(krasm::wat::parse(&source).unwrap());
                let mut reference = Store::new();
                reference.set_engine(EngineKind::Structured);
                let reference_id = reference.create_instance(Arc::clone(&module), None).unwrap();
                let mut fused = Store::new();
                let fused_id = fused.create_instance(module, None).unwrap();
                for lhs in values {
                    let args = vec![Value::I32(lhs), Value::I32(rhs)];
                    let expected = reference
                        .invoke_export(reference_id, "run", args.clone(), None)
                        .unwrap();
                    let actual = fused.invoke_export(fused_id, "run", args, None).unwrap();
                    assert_eq!(actual, expected, "i32.{op}: lhs={lhs}, rhs={rhs}, tee={destination}");
                }
            }
        }
    }
}

#[test]
fn local_store_preserves_stack_values_and_aliased_locals() {
    let values = [i32::MIN, i32::MAX, -1, 0, 1];
    for destination in [0, 1] {
        for value in values {
            let source = format!(
                "(module (func (export \"run\") (param i32 i32) (result i32 i32 i32)
                    i32.const 99
                    local.get 0 i32.const {value} i32.add local.set {destination}
                    local.get 0 local.get 1))"
            );
            let module = Arc::new(krasm::wat::parse(&source).unwrap());
            let mut reference = Store::new();
            reference.set_engine(EngineKind::Structured);
            let reference_id = reference.create_instance(Arc::clone(&module), None).unwrap();
            let mut fused = Store::new();
            let fused_id = fused.create_instance(module, None).unwrap();
            for lhs in values {
                let args = vec![Value::I32(lhs), Value::I32(17)];
                let expected = reference
                    .invoke_export(reference_id, "run", args.clone(), None)
                    .unwrap();
                assert_eq!(expected[0], Value::I32(99));
                assert_eq!(expected[1 + destination as usize], Value::I32(lhs.wrapping_add(value)));
                assert_eq!(fused.invoke_export(fused_id, "run", args, None).unwrap(), expected);
            }
        }
    }
}

#[cfg(feature = "instruction-profile")]
#[test]
fn fused_profiles_count_dispatches_but_budgets_charge_constituents() {
    use krasm::RuntimeError;
    use krasm::runtime::bytecode::Op;
    use krasm::runtime::superinstructions::{I32Operation, SuperInstruction};

    let mut store = Store::new();
    let id = store
        .create_instance(
            Arc::new(
                krasm::wat::parse(
                    "(module (func (export \"run\") (param i32) (result i32 i32) (local i32)
                local.get 0 i32.const 1 i32.add local.tee 1 local.get 1))",
                )
                .unwrap(),
            ),
            None,
        )
        .unwrap();
    for budget in 0..6 {
        assert!(matches!(
            store.invoke_export(id, "run", vec![Value::I32(41)], Some(budget)),
            Err(RuntimeError::InstructionBudgetExhausted)
        ));
    }
    assert_eq!(
        store.invoke_export(id, "run", vec![Value::I32(41)], Some(6)).unwrap(),
        vec![Value::I32(42), Value::I32(42)]
    );
    let profile = store.get_instance(id).unwrap().instruction_profile().unwrap();
    assert_eq!(profile.op_format_version, 2);
    assert!(matches!(
        profile.functions[0].ops[0],
        Op::Super(SuperInstruction::I32LocalConst {
            local: 0,
            value: 1,
            operation: I32Operation::Add,
        })
    ));
    assert_eq!(profile.functions[0].counts, &[6, 3, 2, 1]);
}
