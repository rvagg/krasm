//! The flat engine's instruction budget: each straight-line run of bytecode
//! ops is charged in full where control enters it.
//!
//! Expected counts are derived by hand from the compiled bytecode, one unit
//! per op including the final `end`. A call completes under a
//! budget exactly when its count fits, so each count is checked from both
//! sides: the count completes and one less exhausts.

use krasm::{EngineKind, ImportObject, RuntimeError, Store, Value};

fn instantiate<T: 'static>(store: &mut Store<T>, source: &str, imports: Option<&ImportObject>) -> usize {
    store
        .create_instance(&krasm::wat::parse(source).unwrap(), imports)
        .unwrap()
}

fn flat(source: &str) -> (Store, usize) {
    let mut store = Store::new();
    store.set_engine(EngineKind::Flat);
    let id = instantiate(&mut store, source, None);
    (store, id)
}

fn exhausted(result: Result<Vec<Value>, RuntimeError>) -> bool {
    matches!(result, Err(RuntimeError::InstructionBudgetExhausted))
}

/// `name(args)` returns `expected` under a budget of exactly `count`, and
/// exhausts under `count - 1`.
fn assert_count(store: &mut Store, id: usize, name: &str, args: &[Value], expected: &[Value], count: u64) {
    let result = store.invoke_export(id, name, args.to_vec(), Some(count));
    assert_eq!(result.unwrap(), expected, "{name}{args:?} under {count}");
    let short = store.invoke_export(id, name, args.to_vec(), Some(count - 1));
    assert!(exhausted(short), "{name}{args:?} should exhaust under {}", count - 1);
}

#[test]
fn loops_cost_their_entry_and_each_iteration() {
    // Entry run: the exit test (4). Each iteration falls
    // through the br_if (5) and branches back to the test (4); leaving
    // lands on the final local.get and end (2).
    let (mut store, id) = flat(include_str!("../benches/modules/noop_loop.wat"));
    for n in [0, 1, 2, 10, 1000] {
        assert_count(
            &mut store,
            id,
            "run",
            &[Value::I32(n)],
            &[Value::I32(n)],
            6 + 9 * n as u64,
        );
    }

    // n = 0 returns after the entry run (4) and the early return (2).
    // Otherwise: entry (4), setup to the first exit test (10), n - 1
    // iterations of 13 + 4, and the exit (2).
    let (mut store, id) = flat(include_str!("../benches/modules/fib_iterative.wat"));
    assert_count(&mut store, id, "fib", &[Value::I32(0)], &[Value::I32(0)], 6);
    for (n, fib) in [(1, 1), (2, 1), (10, 55)] {
        assert_count(
            &mut store,
            id,
            "fib",
            &[Value::I32(n)],
            &[Value::I32(fib)],
            17 * n as u64 - 1,
        );
    }
}

#[test]
fn each_branch_charges_the_run_it_lands_in() {
    let (mut store, id) = flat(
        "(module
          (func (export \"skip\") (param i32) (result i32)
            (block (br_if 0 (local.get 0)) (nop) (nop) (nop))
            (i32.const 1))
          (func (export \"choose\") (param i32) (result i32)
            (if (result i32) (local.get 0) (then (nop) (i32.const 1)) (else (i32.const 2))))
          (func (export \"pick\") (param i32) (result i32)
            (block $c
              (block $b
                (block $a (br_table $a $b $c (local.get 0)))
                (nop) (nop) (return (i32.const 10)))
              (return (i32.const 20)))
            (nop) (i32.const 30)))",
    );
    // Entry: local.get, br_if (2). Taken: const, end (2). Not taken: three
    // nops, const, end (5).
    assert_count(&mut store, id, "skip", &[Value::I32(1)], &[Value::I32(1)], 4);
    assert_count(&mut store, id, "skip", &[Value::I32(0)], &[Value::I32(1)], 7);
    // Entry: local.get, i32.eqz, br_if (3). Then: nop, const, br (3) and the
    // end it lands on (1). Else: const, end (2).
    assert_count(&mut store, id, "choose", &[Value::I32(1)], &[Value::I32(1)], 7);
    assert_count(&mut store, id, "choose", &[Value::I32(0)], &[Value::I32(2)], 5);
    // Entry: local.get, br_table (2), then the landing run:
    // $a nop, nop, const, return (4); $b const, return (2); $c and the
    // default nop, const, end (3).
    for (index, result, count) in [(0, 10, 6), (1, 20, 4), (2, 30, 5), (7, 30, 5), (-1, 30, 5)] {
        assert_count(
            &mut store,
            id,
            "pick",
            &[Value::I32(index)],
            &[Value::I32(result)],
            count,
        );
    }
}

#[test]
fn calls_charge_the_callee_and_the_caller_prepays_its_tail() {
    let (mut store, id) = flat(
        "(module
          (type $unit (func))
          (table 1 funcref)
          (elem (i32.const 0) $callee)
          (func $callee (nop) (nop) (nop))
          (func (export \"direct\") (result i32) (call $callee) (i32.const 1))
          (func (export \"indirect\") (result i32) (call_indirect (type $unit) (i32.const 0)) (i32.const 1))
          (func $down (export \"down\") (param i32) (result i32)
            (if (result i32) (local.get 0)
              (then (i32.add (call $down (i32.sub (local.get 0) (i32.const 1))) (i32.const 1)))
              (else (i32.const 0)))))",
    );
    // Caller: call, const, end (3); callee: three nops, end (4).
    assert_count(&mut store, id, "direct", &[], &[Value::I32(1)], 7);
    // Under 3 the caller's run is paid, and the callee cannot start.
    assert!(exhausted(store.invoke_export(id, "direct", vec![], Some(3))));
    // Caller: const, call_indirect, const, end (4).
    assert_count(&mut store, id, "indirect", &[], &[Value::I32(1)], 8);

    // Entry: local.get, i32.eqz, br_if (3). Recursing falls through into
    // local.get, const, sub, call, const, add, br (7), then lands on the end
    // (1) after the callee returns. The base case lands on const, end (2).
    for n in [0u64, 1, 5, 100] {
        let count = 3 * (n + 1) + 8 * n + 2;
        assert_count(
            &mut store,
            id,
            "down",
            &[Value::I32(n as i32)],
            &[Value::I32(n as i32)],
            count,
        );
    }
}

#[test]
fn a_run_the_budget_cannot_cover_never_starts() {
    let (mut store, id) = flat(
        "(module
          (global $g (export \"g\") (mut i32) (i32.const 0))
          (func (export \"set\") (global.set $g (i32.const 1)) (nop) (nop))
          (func (export \"div\") (param i32) (result i32)
            (i32.div_u (i32.const 1) (local.get 0)) (nop)))",
    );
    // const, global.set, nop, nop, end: one run of 5.
    assert!(exhausted(store.invoke_export(id, "set", vec![], Some(4))));
    assert_eq!(store.get_global_export(id, "g").unwrap(), Value::I32(0));
    store.invoke_export(id, "set", vec![], Some(5)).unwrap();
    assert_eq!(store.get_global_export(id, "g").unwrap(), Value::I32(1));

    // A run that would trap partway costs its whole length: short of it,
    // exhaustion comes first.
    assert!(matches!(
        store.invoke_export(id, "div", vec![Value::I32(0)], Some(5)),
        Err(RuntimeError::DivisionByZero)
    ));
    assert!(exhausted(store.invoke_export(id, "div", vec![Value::I32(0)], Some(4))));
}

#[test]
fn waiting_callers_have_paid_for_their_tails() {
    let (mut store, id) = flat(
        "(module
          (global $g (export \"g\") (mut i32) (i32.const 0))
          (func $r (export \"r\") (param i32)
            (if (local.get 0) (then (call $r (i32.sub (local.get 0) (i32.const 1)))))
            (global.set $g (i32.add (global.get $g) (i32.const 1)))))",
    );
    // Entry: local.get, i32.eqz, br_if (3). Recursing falls through
    // local.get, const, sub, call and the tail global.get, const, add,
    // global.set, end (9); the base case lands on the tail (5).
    let count = 12 * 10 + 8;
    // One short, the deepest call cannot pay for its tail. Every caller paid
    // for its own on the way down, so no global.set has run.
    assert!(exhausted(store.invoke_export(
        id,
        "r",
        vec![Value::I32(10)],
        Some(count - 1)
    )));
    assert_eq!(store.get_global_export(id, "g").unwrap(), Value::I32(0));
    store.invoke_export(id, "r", vec![Value::I32(10)], Some(count)).unwrap();
    assert_eq!(store.get_global_export(id, "g").unwrap(), Value::I32(11));
}

#[test]
fn unreachable_ends_its_run() {
    // unreachable alone: the nops after the block are not charged.
    let (mut store, id) = flat("(module (func (export \"trap\") (block (unreachable)) (nop) (nop) (nop)))");
    assert!(matches!(
        store.invoke_export(id, "trap", vec![], Some(1)),
        Err(RuntimeError::Trap(message)) if message == "unreachable"
    ));
    assert!(exhausted(store.invoke_export(id, "trap", vec![], Some(0))));
}

#[test]
fn budget_extremes() {
    let (mut store, id) = flat("(module (func (export \"run\") (result i32) (i32.const 7)))");
    assert!(exhausted(store.invoke_export(id, "run", vec![], Some(0))));
    for budget in [None, Some(u64::MAX)] {
        assert_eq!(
            store.invoke_export(id, "run", vec![], budget).unwrap(),
            vec![Value::I32(7)]
        );
    }
}

#[test]
fn infinite_loops_exhaust() {
    let (mut store, id) = flat(
        "(module
          (func (export \"br\") (loop (br 0)))
          (func (export \"br_if\") (loop (br_if 0 (i32.const 1))))
          (func (export \"br_table\") (loop (br_table 0 0 (i32.const 1))))
          (func $f (export \"call\") (call $f)))",
    );
    for name in ["br", "br_if", "br_table"] {
        assert!(exhausted(store.invoke_export(id, name, vec![], Some(10_000))), "{name}");
    }
    // Each call costs 2 (call, end), so the depth limit comes first.
    assert!(matches!(
        store.invoke_export(id, "call", vec![], Some(u64::MAX)),
        Err(RuntimeError::CallStackOverflow)
    ));
    assert!(exhausted(store.invoke_export(id, "call", vec![], Some(10_000))));
}

#[test]
fn an_unknown_export_leaves_no_budget_behind() {
    let (mut store, id) = flat("(module (func (export \"run\") (result i32) (i32.const 7)))");
    assert!(store.invoke_export(id, "missing", vec![], Some(0)).is_err());
    let addr = store.get_instance(id).unwrap().get_function_addr("run").unwrap();
    assert_eq!(store.execute(addr, vec![]).unwrap(), vec![Value::I32(7)]);
}

#[test]
fn host_calls_sit_inside_a_prepaid_run() {
    let mut store = Store::with_data(0u32);
    store.set_engine(EngineKind::Flat);
    let tick = store.wrap_with_caller(|caller: &mut krasm::Caller<'_, u32>| {
        *caller.data_mut() += 1;
    });
    let mut imports = ImportObject::new();
    imports.add_function("env", "tick", tick);
    let id = instantiate(
        &mut store,
        "(module
          (import \"env\" \"tick\" (func $tick))
          (func (export \"run\") (result i32) (call $tick) (call $tick) (i32.const 7)))",
        Some(&imports),
    );
    // call, call, const, end: one run of 4, so a short budget calls nothing.
    assert!(exhausted(store.invoke_export(id, "run", vec![], Some(3))));
    assert_eq!(*store.data(), 0);
    assert_eq!(
        store.invoke_export(id, "run", vec![], Some(4)).unwrap(),
        vec![Value::I32(7)]
    );
    assert_eq!(*store.data(), 2);
}

#[test]
fn a_re_entered_activation_shares_the_budget() {
    // $a calls through its table into $b, which calls $a.inc twice. The
    // nested activations of $a draw on the budget of the outer one.
    let mut store = Store::new();
    store.set_engine(EngineKind::Flat);
    let a = instantiate(
        &mut store,
        "(module
          (type $unary (func (param i32) (result i32)))
          (table (export \"table\") 1 funcref)
          (global $calls (export \"calls\") (mut i32) (i32.const 0))
          (func (export \"inc\") (param i32) (result i32)
            (global.set $calls (i32.add (global.get $calls) (i32.const 1)))
            (i32.add (local.get 0) (i32.const 1)))
          (func (export \"run\") (param i32) (result i32)
            (i32.add (i32.const 1000) (call_indirect (type $unary) (local.get 0) (i32.const 0)))))",
        None,
    );
    let mut imports = ImportObject::new();
    store.register_exports(a, "a", &mut imports).unwrap();
    instantiate(
        &mut store,
        "(module
          (import \"a\" \"table\" (table 1 funcref))
          (import \"a\" \"inc\" (func $inc (param i32) (result i32)))
          (func $twice (param i32) (result i32) (call $inc (call $inc (local.get 0))))
          (elem (i32.const 0) $twice))",
        Some(&imports),
    );
    let calls = |store: &Store| store.get_global_export(a, "calls").unwrap();

    // run: const, local.get, const, call_indirect, add, end (6); each inc:
    // global.get, const, add, global.set, local.get, const, add, end (8).
    assert_count(&mut store, a, "run", &[Value::I32(5)], &[Value::I32(1007)], 22);
    // The first call completed both incs; the short one ran only the first.
    assert_eq!(calls(&store), Value::I32(3));
    assert_eq!(
        store.invoke_export(a, "run", vec![Value::I32(1)], None).unwrap(),
        vec![Value::I32(1003)]
    );
    assert_eq!(calls(&store), Value::I32(5));
}
