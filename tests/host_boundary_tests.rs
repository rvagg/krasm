//! Calls that cross out of validated code: foreign `call_indirect` targets and
//! host functions. Neither side of these calls is covered by module validation,
//! so the Store checks signatures and values at the crossing.

use krasm::parser::module::{FunctionType, ValueType};
use krasm::runtime::store::FunctionInstance;
use krasm::{EngineKind, ImportObject, RuntimeError, Store, Value};
use rstest::rstest;
use std::sync::Arc;

/// Instantiate `provider` (exporting "table"), then `caller` importing it.
fn link(engine: EngineKind, store: &mut Store, imports: &mut ImportObject, provider: &str, caller: &str) -> usize {
    store.set_engine(engine);
    let provider = krasm::wat::parse(provider).unwrap();
    let provider_id = store.create_instance(Arc::new(provider), Some(imports)).unwrap();
    store.register_exports(provider_id, "provider", imports).unwrap();
    let caller = krasm::wat::parse(caller).unwrap();
    store.create_instance(Arc::new(caller), Some(imports)).unwrap()
}

const CALLER: &str = r#"(module
  (import "provider" "table" (table 2 funcref))
  (type $to_i32 (func (param i32) (result i32)))
  (func (export "call") (param i32) (result i32)
    (call_indirect (type $to_i32) (i32.const 41) (local.get 0))))"#;

#[rstest]
fn foreign_wasm_callee_signature_is_checked(#[values(EngineKind::Structured, EngineKind::Flat)] engine: EngineKind) {
    let provider = r#"(module
      (table (export "table") 2 funcref)
      (func $matching (param i32) (result i32) (i32.add (local.get 0) (i32.const 1)))
      (func $wider (param i32) (result i64) (i64.const 7))
      (elem (i32.const 0) $matching $wider))"#;
    let mut store = Store::new();
    let mut imports = ImportObject::new();
    let id = link(engine, &mut store, &mut imports, provider, CALLER);

    assert_eq!(
        store.invoke_export(id, "call", vec![Value::I32(0)], None).unwrap(),
        vec![Value::I32(42)]
    );
    let error = store.invoke_export(id, "call", vec![Value::I32(1)], None).unwrap_err();
    assert!(
        matches!(error, RuntimeError::IndirectCallTypeMismatch { .. }),
        "{error:?}"
    );
    // The trap leaves the caller usable.
    assert_eq!(
        store.invoke_export(id, "call", vec![Value::I32(0)], None).unwrap(),
        vec![Value::I32(42)]
    );
}

#[rstest]
fn foreign_host_callee_signature_is_checked(#[values(EngineKind::Structured, EngineKind::Flat)] engine: EngineKind) {
    // The provider imports two host functions and places them in its table;
    // the caller reaches them only through the shared table.
    let provider = r#"(module
      (import "env" "matching" (func $matching (param i32) (result i32)))
      (import "env" "nullary" (func $nullary (result i32)))
      (table (export "table") 2 funcref)
      (elem (i32.const 0) $matching $nullary))"#;
    let mut store = Store::new();
    let matching = store.wrap(|x: i32| -> i32 { x + 1 });
    let nullary = store.wrap(|| -> i32 { 9 });
    let mut imports = ImportObject::new();
    imports.add_function("env", "matching", matching);
    imports.add_function("env", "nullary", nullary);
    let id = link(engine, &mut store, &mut imports, provider, CALLER);

    assert_eq!(
        store.invoke_export(id, "call", vec![Value::I32(0)], None).unwrap(),
        vec![Value::I32(42)]
    );
    let error = store.invoke_export(id, "call", vec![Value::I32(1)], None).unwrap_err();
    assert!(
        matches!(error, RuntimeError::IndirectCallTypeMismatch { .. }),
        "{error:?}"
    );
}

#[test]
fn host_arguments_are_checked() {
    let mut store = Store::new();
    let addr = store.wrap(|x: i32| -> i32 { x });
    for args in [vec![], vec![Value::I64(1)], vec![Value::I32(1), Value::I32(2)]] {
        let error = store.execute(addr, args).unwrap_err();
        assert!(matches!(error, RuntimeError::TypeMismatch { .. }), "{error:?}");
    }
    assert_eq!(store.execute(addr, vec![Value::I32(3)]).unwrap(), vec![Value::I32(3)]);
}

#[rstest]
fn host_results_are_checked(#[values(EngineKind::Structured, EngineKind::Flat)] engine: EngineKind) {
    let mut store = Store::new();
    store.set_engine(engine);
    // A raw host function whose results disagree with its declared signature.
    let liar = store.allocate_function(FunctionInstance::Host {
        func_type: FunctionType {
            parameters: vec![],
            return_types: vec![ValueType::I32],
        },
        func: Box::new(|_, _| Ok(vec![Value::I64(1)])),
    });
    let mut imports = ImportObject::new();
    imports.add_function("env", "liar", liar);
    let module = krasm::wat::parse(
        r#"(module
          (import "env" "liar" (func $liar (result i32)))
          (func (export "call") (result i32) (call $liar)))"#,
    )
    .unwrap();
    let id = store.create_instance(Arc::new(module), Some(&imports)).unwrap();
    let error = store.invoke_export(id, "call", vec![], None).unwrap_err();
    assert!(matches!(error, RuntimeError::TypeMismatch { .. }), "{error:?}");
}
