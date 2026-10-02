#![cfg(feature = "instruction-profile")]

use krasm::runtime::bytecode::Op;
use krasm::runtime::imports::ImportObject;
use krasm::{EngineKind, Store, Value};
use std::sync::Arc;

fn instantiate(store: &mut Store, source: &str, imports: Option<&ImportObject>) -> usize {
    store
        .create_instance(Arc::new(krasm::wat::parse(source).unwrap()), imports)
        .unwrap()
}

#[test]
fn loops_and_cold_branches_have_exact_counts() {
    let mut store = Store::new();
    let id = instantiate(
        &mut store,
        r#"(module
      (func (export "run") (param i32) (result i32)
        (loop $again
          local.get 0 i32.const 1 i32.sub local.tee 0 br_if $again)
        local.get 0
        if (result i32) i32.const 99 else i32.const 7 end))"#,
        None,
    );
    assert_eq!(
        store.invoke_export(id, "run", vec![Value::I32(3)], None).unwrap(),
        vec![Value::I32(7)]
    );
    let profile = store.get_instance(id).unwrap().instruction_profile().unwrap();
    let function = &profile.functions[0];
    let count = |predicate: fn(&Op) -> bool| -> u64 {
        function
            .ops
            .iter()
            .zip(function.counts)
            .filter(|(op, _)| predicate(op))
            .map(|(_, count)| count)
            .sum()
    };
    assert_eq!(
        count(|op| match op {
            #[cfg(feature = "superinstructions")]
            Op::Super(krasm::runtime::superinstructions::SuperInstruction::I32LocalConst {
                operation: krasm::runtime::superinstructions::I32Operation::Sub,
                ..
            }) => true,
            _ => matches!(op, Op::I32Sub),
        }),
        3
    );
    assert_eq!(count(|op| matches!(op, Op::I32Const(99))), 0);
    assert_eq!(count(|op| matches!(op, Op::I32Const(7))), 1);
    assert_eq!(count(|op| matches!(op, Op::BrIf { .. })), 4);
}

#[test]
fn recursion_host_resumes_and_repeated_invocations_accumulate() {
    let mut store = Store::new();
    let host = store.wrap(|x: i32| -> i32 { x + 1 });
    let mut imports = ImportObject::new();
    imports.add_function("env", "inc", host);
    let id = instantiate(
        &mut store,
        r#"(module
      (import "env" "inc" (func $inc (param i32) (result i32)))
      (func $recurse (export "run") (param i32) (result i32)
        local.get 0
        if (result i32)
          local.get 0 i32.const 1 i32.sub call $recurse call $inc
        else i32.const 0 end))"#,
        Some(&imports),
    );
    for _ in 0..2 {
        assert_eq!(
            store.invoke_export(id, "run", vec![Value::I32(3)], None).unwrap(),
            vec![Value::I32(3)]
        );
    }
    let profile = store.get_instance(id).unwrap().instruction_profile().unwrap();
    let function = &profile.functions[0];
    assert_eq!(function.function_index, 1);
    let calls: Vec<_> = function
        .ops
        .iter()
        .zip(function.counts)
        .filter_map(|(op, count)| {
            if let Op::Call { func_idx } = op {
                Some((*func_idx, *count))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(calls, vec![(1, 6), (0, 6)]);
    assert_eq!(function.counts[0], 8);
    assert_eq!(*function.counts.last().unwrap(), 8);
}

#[test]
fn trap_and_budget_stop_counts_without_erasing_history() {
    let mut store = Store::new();
    let id = instantiate(
        &mut store,
        r#"(module
      (func (export "run") (param i32) (result i32)
        i32.const 10 local.get 0 i32.div_u i32.const 1 i32.add))"#,
        None,
    );
    assert!(store.invoke_export(id, "run", vec![Value::I32(0)], None).is_err());
    assert_eq!(
        store.get_instance(id).unwrap().instruction_profile().unwrap().functions[0].counts,
        &[1, 1, 1, 0, 0, 0]
    );
    assert!(store.invoke_export(id, "run", vec![Value::I32(2)], Some(2)).is_err());
    assert_eq!(
        store.get_instance(id).unwrap().instruction_profile().unwrap().functions[0].counts,
        &[2, 2, 1, 0, 0, 0]
    );
    assert_eq!(
        store.invoke_export(id, "run", vec![Value::I32(2)], None).unwrap(),
        vec![Value::I32(6)]
    );
    assert_eq!(
        store.get_instance(id).unwrap().instruction_profile().unwrap().functions[0].counts,
        &[3, 3, 2, 1, 1, 1]
    );
}

#[test]
fn start_and_cross_module_calls_belong_to_their_instances() {
    let mut store = Store::new();
    let callee = instantiate(
        &mut store,
        r#"(module
      (func (export "value") (result i32) i32.const 41))"#,
        None,
    );
    let addr = store.get_instance(callee).unwrap().get_function_addr("value").unwrap();
    let mut imports = ImportObject::new();
    imports.add_function("other", "value", addr);
    let caller = instantiate(
        &mut store,
        r#"(module
      (import "other" "value" (func $value (result i32)))
      (func $init call $value drop)
      (start $init)
      (func (export "run") (result i32) call $value i32.const 1 i32.add))"#,
        Some(&imports),
    );
    assert_eq!(
        store.invoke_export(caller, "run", vec![], None).unwrap(),
        vec![Value::I32(42)]
    );
    let a = store.get_instance(callee).unwrap().instruction_profile().unwrap();
    assert_eq!(a.functions[0].counts, &[2, 2]);
    let b = store.get_instance(caller).unwrap().instruction_profile().unwrap();
    assert!(b.functions[0].is_start);
    assert_eq!(b.functions[0].function_index, 1);
    assert_eq!(b.functions[0].counts, &[1, 1, 1]);
    assert_eq!(b.functions[1].counts, &[1, 1, 1, 1]);
}

#[test]
fn profile_preserves_float_bits_and_memory_immediates() {
    let mut store = Store::new();
    let id = instantiate(
        &mut store,
        r#"(module (memory 1)
      (func (export "run")
        f32.const nan:0x12345 drop f64.const -0 drop
        i32.const 0 i32.load offset=12 align=2 drop))"#,
        None,
    );
    let profile = store.get_instance(id).unwrap().instruction_profile().unwrap();
    let json = serde_json::to_value(&profile).unwrap();
    let ops = json["functions"][0]["ops"].as_array().unwrap();
    assert_eq!(ops[0]["immediates"], 0x7f812345u32);
    assert_eq!(ops[2]["immediates"], 0x8000000000000000u64);
    assert_eq!(ops[5]["immediates"], serde_json::json!({"align": 1, "offset": 12}));
    assert!(profile.functions[0].counts.iter().all(|count| *count == 0));
    let mut structured = Store::new();
    structured.set_engine(EngineKind::Structured);
    let id = instantiate(&mut structured, "(module)", None);
    assert!(structured.get_instance(id).unwrap().instruction_profile().is_none());
}
