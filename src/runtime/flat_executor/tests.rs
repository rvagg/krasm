//! Flat executor unit tests. Mechanical per-op coverage lives in the
//! wast suite (`KRASM_FLAT=1 cargo test --test wast_tests`); these tests
//! cover the executor machinery itself.

use super::*;
use crate::parser::module::ValueType;
use crate::runtime::compiler;
use crate::wat;

/// Shorthand for building a FunctionType in test contexts.
fn ftype(params: &[ValueType], results: &[ValueType]) -> FunctionType {
    FunctionType {
        parameters: params.to_vec(),
        return_types: results.to_vec(),
    }
}

fn compile_wat(source: &str) -> Vec<CompiledFunction> {
    let module = wat::parse(source).expect("WAT parse failed");
    compiler::compile_module(&module)
}

fn compile_and_run(source: &str, args: &[Value]) -> Vec<Value> {
    let funcs = compile_wat(source);
    execute_flat(&funcs, 0, args, None).expect("execution failed")
}

fn expect_trap(source: &str, expected_msg: &str) {
    let funcs = compile_wat(source);
    let err = execute_flat(&funcs, 0, &[], None).unwrap_err();
    assert!(
        err.to_string().contains(expected_msg),
        "expected '{expected_msg}', got: {err}"
    );
}

#[test]
fn simple_add() {
    let result = compile_and_run(
        "(module (func (param i32 i32) (result i32) local.get 0 local.get 1 i32.add))",
        &[Value::I32(3), Value::I32(4)],
    );
    assert_eq!(result, vec![Value::I32(7)]);
}

#[test]
fn simple_const() {
    let result = compile_and_run("(module (func (result i32) i32.const 42))", &[]);
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn noop_loop_1000() {
    let result = compile_and_run(
        include_str!("../../../benches/modules/noop_loop.wat"),
        &[Value::I32(1000)],
    );
    assert_eq!(result, vec![Value::I32(1000)]);
}

#[test]
fn fib_0() {
    let result = compile_and_run(
        include_str!("../../../benches/modules/fib_iterative.wat"),
        &[Value::I32(0)],
    );
    assert_eq!(result, vec![Value::I32(0)]);
}

#[test]
fn fib_1() {
    let result = compile_and_run(
        include_str!("../../../benches/modules/fib_iterative.wat"),
        &[Value::I32(1)],
    );
    assert_eq!(result, vec![Value::I32(1)]);
}

#[test]
fn fib_10() {
    let result = compile_and_run(
        include_str!("../../../benches/modules/fib_iterative.wat"),
        &[Value::I32(10)],
    );
    assert_eq!(result, vec![Value::I32(55)]);
}

#[test]
fn fib_20() {
    let result = compile_and_run(
        include_str!("../../../benches/modules/fib_iterative.wat"),
        &[Value::I32(20)],
    );
    assert_eq!(result, vec![Value::I32(6765)]);
}

#[test]
fn fib_46() {
    let result = compile_and_run(
        include_str!("../../../benches/modules/fib_iterative.wat"),
        &[Value::I32(46)],
    );
    assert_eq!(result, vec![Value::I32(1836311903)]);
}

#[test]
fn if_then_return() {
    let result = compile_and_run(
        "(module (func (param i32) (result i32)
            (if (i32.eqz (local.get 0))
                (then (return (i32.const 99))))
            (i32.const 0)))",
        &[Value::I32(0)],
    );
    assert_eq!(result, vec![Value::I32(99)]);

    let result = compile_and_run(
        "(module (func (param i32) (result i32)
            (if (i32.eqz (local.get 0))
                (then (return (i32.const 99))))
            (i32.const 0)))",
        &[Value::I32(1)],
    );
    assert_eq!(result, vec![Value::I32(0)]);
}

#[test]
fn if_then_else() {
    let result = compile_and_run(
        "(module (func (param i32) (result i32)
            (if (result i32) (local.get 0)
                (then (i32.const 10))
                (else (i32.const 20)))))",
        &[Value::I32(1)],
    );
    assert_eq!(result, vec![Value::I32(10)]);

    let result = compile_and_run(
        "(module (func (param i32) (result i32)
            (if (result i32) (local.get 0)
                (then (i32.const 10))
                (else (i32.const 20)))))",
        &[Value::I32(0)],
    );
    assert_eq!(result, vec![Value::I32(20)]);
}

#[test]
fn block_br() {
    let result = compile_and_run("(module (func (result i32) (block (br 0)) (i32.const 42)))", &[]);
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn nested_block_br() {
    let result = compile_and_run(
        "(module (func (result i32)
            (block (block (br 1)) (unreachable))
            (i32.const 7)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(7)]);
}

#[test]
fn block_result_br() {
    // Block with result: br carries the value
    let result = compile_and_run(
        "(module (func (result i32)
            (block (result i32)
                (i32.const 42)
                (br 0))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn block_result_br_with_garbage() {
    // Branch must keep 1 result, discard extra values
    let result = compile_and_run(
        "(module (func (result i32)
            (block (result i32)
                (i32.const 99)
                (i32.const 42)
                (br 0))))",
        &[],
    );
    // br 0 keeps top 1 (42), discards 99
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn br_table_first() {
    let result = compile_and_run(
        "(module (func (param i32) (result i32)
            (block $a (result i32)
                (block $b (result i32)
                    (i32.const 10)
                    (local.get 0)
                    (br_table 0 1 0)))))",
        &[Value::I32(0)],
    );
    // index 0 -> label 0 (inner block $b), exits with 10
    assert_eq!(result, vec![Value::I32(10)]);
}

#[test]
fn br_table_second() {
    let result = compile_and_run(
        "(module (func (param i32) (result i32)
            (block $a (result i32)
                (block $b (result i32)
                    (i32.const 10)
                    (local.get 0)
                    (br_table 0 1 0)))))",
        &[Value::I32(1)],
    );
    // index 1 -> label 1 (outer block $a), exits with 10
    assert_eq!(result, vec![Value::I32(10)]);
}

#[test]
fn br_table_default() {
    let result = compile_and_run(
        "(module (func (param i32) (result i32)
            (block $a (result i32)
                (block $b (result i32)
                    (i32.const 10)
                    (local.get 0)
                    (br_table 0 1 0)))))",
        &[Value::I32(99)],
    );
    // index 99 out of bounds -> default (label 0, inner block)
    assert_eq!(result, vec![Value::I32(10)]);
}

#[test]
fn global_get_set() {
    let source = "(module
        (global $g (mut i32) (i32.const 10))
        (func (result i32)
            (global.set $g (i32.add (global.get $g) (i32.const 5)))
            (global.get $g)))";
    let funcs = compile_wat(source);
    let compiled = funcs.into_iter().next().unwrap();

    // Set up resources with one global initialised to 10
    let mut resources = Resources {
        memories: Vec::new(),
        tables: Vec::new(),
        globals: vec![Value::I32(10)],
    };
    let global_addrs = vec![GlobalAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &global_addrs,
        memory_addrs: &[],
        table_addrs: &[],
        types: &[],
        functions: &[],
        num_imported: 0,
        segments: &mut segments,
        data_segments: &[],
    };

    let result = execute_flat(&[compiled], 0, &[], Some(&mut ctx)).expect("execution failed");
    assert_eq!(result, vec![Value::I32(15)]);
    // Global should be mutated in resources
    assert_eq!(ctx.resources.globals[0], Value::I32(15));
}

#[test]
fn memory_store_load() {
    let source = "(module
        (memory 1)
        (func (result i32)
            (i32.store (i32.const 0) (i32.const 42))
            (i32.load (i32.const 0))))";
    let funcs = compile_wat(source);
    let compiled = funcs.into_iter().next().unwrap();

    let mut resources = Resources {
        memories: vec![crate::runtime::memory::Memory::new(1, Some(1)).unwrap()],
        tables: Vec::new(),
        globals: Vec::new(),
    };
    let memory_addrs = vec![MemoryAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &memory_addrs,
        table_addrs: &[],
        types: &[],
        functions: &[],
        num_imported: 0,
        segments: &mut segments,
        data_segments: &[],
    };

    let result = execute_flat(&[compiled], 0, &[], Some(&mut ctx)).expect("execution failed");
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn memory_load8_store8() {
    let source = "(module
        (memory 1)
        (func (result i32)
            (i32.store8 (i32.const 0) (i32.const 255))
            (i32.load8_u (i32.const 0))))";
    let funcs = compile_wat(source);
    let compiled = funcs.into_iter().next().unwrap();

    let mut resources = Resources {
        memories: vec![crate::runtime::memory::Memory::new(1, Some(1)).unwrap()],
        tables: Vec::new(),
        globals: Vec::new(),
    };
    let memory_addrs = vec![MemoryAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &memory_addrs,
        table_addrs: &[],
        types: &[],
        functions: &[],
        num_imported: 0,
        segments: &mut segments,
        data_segments: &[],
    };

    let result = execute_flat(&[compiled], 0, &[], Some(&mut ctx)).expect("execution failed");
    assert_eq!(result, vec![Value::I32(255)]);
}

#[test]
fn memory_grow_and_size() {
    let source = "(module
        (memory 1)
        (func (result i32 i32)
            (memory.grow (i32.const 2))
            (memory.size)))";
    let funcs = compile_wat(source);
    let compiled = funcs.into_iter().next().unwrap();

    let mut resources = Resources {
        memories: vec![crate::runtime::memory::Memory::new(1, None).unwrap()],
        tables: Vec::new(),
        globals: Vec::new(),
    };
    let memory_addrs = vec![MemoryAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &memory_addrs,
        table_addrs: &[],
        types: &[],
        functions: &[],
        num_imported: 0,
        segments: &mut segments,
        data_segments: &[],
    };

    let result = execute_flat(&[compiled], 0, &[], Some(&mut ctx)).expect("execution failed");
    // memory.grow returns old size (1), memory.size returns new size (3)
    assert_eq!(result, vec![Value::I32(1), Value::I32(3)]);
}

// ================================================================
// Block tests
// ================================================================

#[test]
fn block_empty() {
    let result = compile_and_run("(module (func (result i32) (block) (i32.const 42)))", &[]);
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn block_with_value_fallthrough() {
    // Block produces a value by falling through (no br)
    let result = compile_and_run(
        "(module (func (result i32)
            (block (result i32) (i32.const 42))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn block_br_with_value() {
    // br carries the value past unreachable code
    let result = compile_and_run(
        "(module (func (result i32)
            (block (result i32)
                (i32.const 42)
                (br 0)
                (unreachable))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn block_nested_three_levels() {
    // br 2 from three levels deep exits the outermost block
    let result = compile_and_run(
        "(module (func (result i32)
            (block (result i32)
                (block
                    (block
                        (i32.const 7)
                        (br 2)))
                (unreachable))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(7)]);
}

// ================================================================
// Loop tests
// ================================================================

#[test]
fn loop_simple_fallthrough() {
    // Loop that doesn't branch back, just falls through
    let result = compile_and_run(
        "(module (func (result i32)
            (loop (nop))
            (i32.const 42)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn loop_with_counter() {
    // Count from 0 to 5 using a loop
    let result = compile_and_run(
        "(module (func (result i32)
            (local $i i32)
            (loop $l
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br_if $l (i32.lt_u (local.get $i) (i32.const 5))))
            (local.get $i)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(5)]);
}

#[test]
fn loop_break_via_outer_block() {
    // Loop exits via br to an enclosing block when counter reaches 3
    let result = compile_and_run(
        "(module (func (result i32)
            (local $i i32)
            (block $done (result i32)
                (loop $l
                    (local.set $i (i32.add (local.get $i) (i32.const 1)))
                    (if (i32.ge_u (local.get $i) (i32.const 3))
                        (then (local.get $i) (br $done)))
                    (br $l))
                (unreachable))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(3)]);
}

// ================================================================
// If/else tests
// ================================================================

#[test]
fn if_true_no_else() {
    let result = compile_and_run(
        "(module (func (result i32)
            (if (i32.const 1) (then (nop)))
            (i32.const 42)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn if_false_no_else() {
    let result = compile_and_run(
        "(module (func (result i32)
            (if (i32.const 0) (then (unreachable)))
            (i32.const 42)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn if_with_value() {
    let result = compile_and_run(
        "(module (func (result i32)
            (if (result i32) (i32.const 1)
                (then (i32.const 42))
                (else (i32.const 99)))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn nested_if() {
    let result = compile_and_run(
        "(module (func (result i32)
            (if (result i32) (i32.const 1)
                (then
                    (if (result i32) (i32.const 0)
                        (then (i32.const 11))
                        (else (i32.const 22))))
                (else (i32.const 33)))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(22)]);
}

#[test]
fn if_br_out() {
    // br from inside if exits the if's block
    let result = compile_and_run(
        "(module (func (result i32)
            (if (result i32) (i32.const 1)
                (then
                    (i32.const 42)
                    (br 0)
                    (unreachable))
                (else (i32.const 99)))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

// ================================================================
// Return tests
// ================================================================

#[test]
fn return_simple() {
    let result = compile_and_run(
        "(module (func (result i32)
            (return (i32.const 42))
            (unreachable)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn return_no_value() {
    let result = compile_and_run("(module (func (return)))", &[]);
    assert_eq!(result, vec![]);
}

#[test]
fn return_multiple_values() {
    let result = compile_and_run(
        "(module (func (result i32 i32)
            (return (i32.const 1) (i32.const 2))
            (unreachable)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(1), Value::I32(2)]);
}

#[test]
fn return_from_block() {
    let result = compile_and_run(
        "(module (func (result i32)
            (block (return (i32.const 42)))
            (unreachable)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn return_from_nested_blocks() {
    let result = compile_and_run(
        "(module (func (result i32)
            (block (block (return (i32.const 42))))
            (unreachable)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn return_from_if() {
    let result = compile_and_run(
        "(module (func (result i32)
            (if (i32.const 1)
                (then (return (i32.const 42))))
            (unreachable)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn return_from_else() {
    let result = compile_and_run(
        "(module (func (result i32)
            (if (i32.const 0)
                (then (unreachable))
                (else (return (i32.const 42))))
            (unreachable)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn return_from_loop() {
    let result = compile_and_run(
        "(module (func (result i32)
            (loop (return (i32.const 42)) (br 0))
            (unreachable)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

// ================================================================
// Unreachable tests
// ================================================================

#[test]
fn unreachable_traps() {
    expect_trap("(module (func (unreachable)))", "unreachable");
}

#[test]
fn unreachable_in_block() {
    expect_trap("(module (func (block (unreachable))))", "unreachable");
}

#[test]
fn unreachable_in_if_taken() {
    expect_trap("(module (func (if (i32.const 1) (then (unreachable)))))", "unreachable");
}

#[test]
fn unreachable_in_else_taken() {
    expect_trap(
        "(module (func (if (i32.const 0) (then (nop)) (else (unreachable)))))",
        "unreachable",
    );
}

// ================================================================
// br_table edge cases
// ================================================================

#[test]
fn br_table_negative_index() {
    // Negative i32 is a large u32, should use default
    let result = compile_and_run(
        "(module (func (result i32)
            (block (result i32)
                (i32.const 42)
                (i32.const -1)
                (br_table 0 0))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn br_table_very_large_index() {
    let result = compile_and_run(
        "(module (func (result i32)
            (block (result i32)
                (i32.const 42)
                (i32.const 1000000)
                (br_table 0 0))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn br_table_three_way() {
    // Three nested blocks, br_table selects which to exit
    for (index, expected) in [(0, 100), (1, 200), (2, 300), (99, 300)] {
        let result = compile_and_run(
            &format!(
                "(module (func (result i32)
                    (block $c
                        (block $b
                            (block $a
                                (i32.const {index})
                                (br_table 0 1 2 2))
                            (return (i32.const 100)))
                        (return (i32.const 200)))
                    (i32.const 300)))"
            ),
            &[],
        );
        assert_eq!(result, vec![Value::I32(expected)], "br_table index {index}");
    }
}

#[test]
fn br_table_single_label() {
    let result = compile_and_run(
        "(module (func (result i32)
            (block (result i32)
                (i32.const 42)
                (i32.const 0)
                (br_table 0))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

// ================================================================
// br_if edge cases
// ================================================================

#[test]
fn br_if_with_value() {
    // br_if taken carries block result
    let result = compile_and_run(
        "(module (func (result i32)
            (block (result i32)
                (i32.const 42)
                (i32.const 1)
                (br_if 0)
                (unreachable))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn br_if_not_taken() {
    // br_if not taken, execution continues
    let result = compile_and_run(
        "(module (func (result i32)
            (block (result i32)
                (i32.const 99)
                (i32.const 0)
                (br_if 0)
                (drop)
                (i32.const 42))))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

// ================================================================
// Drop test
// ================================================================

#[test]
fn drop_value() {
    let result = compile_and_run(
        "(module (func (result i32)
            (i32.const 99)
            (drop)
            (i32.const 42)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

// ================================================================
// Local variable edge cases
// ================================================================

#[test]
fn local_tee() {
    let result = compile_and_run(
        "(module (func (result i32)
            (local $x i32)
            (local.tee $x (i32.const 42))
            (drop)
            (local.get $x)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn locals_default_to_zero() {
    let result = compile_and_run(
        "(module (func (result i32)
            (local $x i32)
            (local.get $x)))",
        &[],
    );
    assert_eq!(result, vec![Value::I32(0)]);
}

// ================================================================
// Memory edge cases
// ================================================================

#[test]
fn memory_load_with_offset() {
    let source = "(module
        (memory 1)
        (func (result i32)
            (i32.store (i32.const 4) (i32.const 99))
            (i32.load offset=4 (i32.const 0))))";
    let funcs = compile_wat(source);
    let compiled = funcs.into_iter().next().unwrap();

    let mut resources = Resources {
        memories: vec![crate::runtime::memory::Memory::new(1, Some(1)).unwrap()],
        tables: Vec::new(),
        globals: Vec::new(),
    };
    let memory_addrs = vec![MemoryAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &memory_addrs,
        table_addrs: &[],
        types: &[],
        functions: &[],
        num_imported: 0,
        segments: &mut segments,
        data_segments: &[],
    };

    let result = execute_flat(&[compiled], 0, &[], Some(&mut ctx)).expect("execution failed");
    assert_eq!(result, vec![Value::I32(99)]);
}

#[test]
fn memory_fill_and_load() {
    let source = "(module
        (memory 1)
        (func (result i32)
            (memory.fill (i32.const 0) (i32.const 0xAB) (i32.const 4))
            (i32.load (i32.const 0))))";
    let funcs = compile_wat(source);
    let compiled = funcs.into_iter().next().unwrap();

    let mut resources = Resources {
        memories: vec![crate::runtime::memory::Memory::new(1, Some(1)).unwrap()],
        tables: Vec::new(),
        globals: Vec::new(),
    };
    let memory_addrs = vec![MemoryAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &memory_addrs,
        table_addrs: &[],
        types: &[],
        functions: &[],
        num_imported: 0,
        segments: &mut segments,
        data_segments: &[],
    };

    let result = execute_flat(&[compiled], 0, &[], Some(&mut ctx)).expect("execution failed");
    // 4 bytes of 0xAB = 0xABABABAB
    assert_eq!(result, vec![Value::I32(0xABABABABu32 as i32)]);
}

#[test]
fn memory_copy_and_load() {
    let source = "(module
        (memory 1)
        (func (result i32)
            (i32.store (i32.const 0) (i32.const 12345))
            (memory.copy (i32.const 100) (i32.const 0) (i32.const 4))
            (i32.load (i32.const 100))))";
    let funcs = compile_wat(source);
    let compiled = funcs.into_iter().next().unwrap();

    let mut resources = Resources {
        memories: vec![crate::runtime::memory::Memory::new(1, Some(1)).unwrap()],
        tables: Vec::new(),
        globals: Vec::new(),
    };
    let memory_addrs = vec![MemoryAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &memory_addrs,
        table_addrs: &[],
        types: &[],
        functions: &[],
        num_imported: 0,
        segments: &mut segments,
        data_segments: &[],
    };

    let result = execute_flat(&[compiled], 0, &[], Some(&mut ctx)).expect("execution failed");
    assert_eq!(result, vec![Value::I32(12345)]);
}

// ================================================================
// Compiler stack_depth verification
// ================================================================

#[test]
fn compiler_block_stack_depth() {
    // Verify the compiler produces correct stack_depth for a block with result.
    // The block has result i32, so br should carry arity=1.
    // Before the block, we push a "base" value. The block's stack_depth
    // should be 1 (the base value is below the block).
    use crate::runtime::bytecode::Op;

    let source = "(module (func (result i32)
        (i32.const 0)
        (drop)
        (block (result i32)
            (i32.const 42)
            (br 0))))";
    let funcs = compile_wat(source);
    let compiled = &funcs[0];

    // Find the Br op and check its metadata
    let br = compiled
        .ops
        .iter()
        .find(|op| matches!(op, Op::Br { arity, .. } if *arity > 0))
        .expect("should have a Br with arity > 0");
    if let Op::Br { arity, stack_depth, .. } = br {
        assert_eq!(*arity, 1, "block result arity");
        // After i32.const 0 + drop, stack depth is 0 at block entry
        assert_eq!(*stack_depth, 0, "stack depth at block entry");
    }
}

// ================================================================
// Function call tests
// ================================================================

#[test]
fn call_simple() {
    // $add is func 0, $main is func 1
    let funcs = compile_wat(
        "(module
            (func $add (param i32 i32) (result i32)
                (i32.add (local.get 0) (local.get 1)))
            (func $main (result i32)
                (call $add (i32.const 3) (i32.const 4))))",
    );
    let result = execute_flat(&funcs, 1, &[], None).expect("execution failed");
    assert_eq!(result, vec![Value::I32(7)]);
}

#[test]
fn call_recursive_fib() {
    let funcs = compile_wat(include_str!("../../../benches/modules/fib_recursive.wat"));
    for (n, expected) in [(0, 0), (1, 1), (2, 1), (10, 55), (20, 6765)] {
        let result = execute_flat(&funcs, 0, &[Value::I32(n)], None).expect("execution failed");
        assert_eq!(result, vec![Value::I32(expected)], "fib({n})");
    }
}

#[test]
fn call_multiple_functions() {
    // $double=0, $inc=1, $main=2
    let funcs = compile_wat(
        "(module
            (func $double (param i32) (result i32)
                (i32.mul (local.get 0) (i32.const 2)))
            (func $inc (param i32) (result i32)
                (i32.add (local.get 0) (i32.const 1)))
            (func $main (result i32)
                (call $inc (call $double (i32.const 5)))))",
    );
    // double(5) = 10, inc(10) = 11
    let result = execute_flat(&funcs, 2, &[], None).expect("execution failed");
    assert_eq!(result, vec![Value::I32(11)]);
}

#[test]
fn call_stack_overflow() {
    let funcs = compile_wat("(module (func $inf (call $inf)))");
    let err = execute_flat(&funcs, 0, &[], None).unwrap_err();
    assert!(
        err.to_string().contains("call stack"),
        "expected call stack overflow, got: {err}"
    );
}

#[test]
fn call_preserves_caller_locals() {
    let funcs = compile_wat(
        "(module
            (func $noop)
            (func $main (result i32)
                (local $x i32)
                (local.set $x (i32.const 42))
                (call $noop)
                (local.get $x)))",
    );
    let result = execute_flat(&funcs, 1, &[], None).expect("execution failed");
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn call_multiple_return_values() {
    let (funcs, result) = {
        let funcs = compile_wat(
            "(module
                (func $pair (result i32 i32)
                    (i32.const 10) (i32.const 20))
                (func $main (result i32)
                    (call $pair)
                    (i32.add)))",
        );
        let result = execute_flat(&funcs, 1, &[], None).expect("execution failed");
        (funcs, result)
    };
    let _ = funcs;
    assert_eq!(result, vec![Value::I32(30)]);
}

#[test]
fn call_results_survive_on_stack() {
    // Two calls in an expression: first call's result must survive
    // while the second call executes (tests stack_base correctness)
    let funcs = compile_wat(
        "(module
            (func $id (param i32) (result i32) (local.get 0))
            (func $main (result i32)
                (i32.add
                    (call $id (i32.const 100))
                    (call $id (i32.const 7)))))",
    );
    let result = execute_flat(&funcs, 1, &[], None).expect("execution failed");
    assert_eq!(result, vec![Value::I32(107)]);
}

#[test]
fn call_inside_loop() {
    // Call inside a loop with a branch -- stack_base and branch cleanup
    // must coexist correctly
    let funcs = compile_wat(
        "(module
            (func $inc (param i32) (result i32)
                (i32.add (local.get 0) (i32.const 1)))
            (func $main (result i32)
                (local $i i32)
                (loop $l
                    (local.set $i (call $inc (local.get $i)))
                    (br_if $l (i32.lt_u (local.get $i) (i32.const 5))))
                (local.get $i)))",
    );
    let result = execute_flat(&funcs, 1, &[], None).expect("execution failed");
    assert_eq!(result, vec![Value::I32(5)]);
}

#[test]
fn call_inside_block_with_br() {
    // Call result used as block result via br
    let funcs = compile_wat(
        "(module
            (func $const42 (result i32) (i32.const 42))
            (func $main (result i32)
                (block (result i32)
                    (call $const42)
                    (br 0)
                    (unreachable))))",
    );
    let result = execute_flat(&funcs, 1, &[], None).expect("execution failed");
    assert_eq!(result, vec![Value::I32(42)]);
}

// -- Imported calls (suspend/resume) --

/// Fresh Resources with no memories, tables, or globals.
fn empty_resources() -> Resources {
    Resources {
        memories: Vec::new(),
        tables: Vec::new(),
        globals: Vec::new(),
    }
}

/// Unwrap a NeedsExternalCall outcome into its request.
fn expect_external_call(outcome: ExecutionOutcome) -> ExternalCallRequest {
    match outcome {
        ExecutionOutcome::NeedsExternalCall(request) => request,
        ExecutionOutcome::Complete(values) => {
            panic!("expected external call, got Complete({values:?})")
        }
    }
}

/// Unwrap a Complete outcome into its results.
fn expect_complete(outcome: ExecutionOutcome) -> Vec<Value> {
    match outcome {
        ExecutionOutcome::Complete(values) => values,
        ExecutionOutcome::NeedsExternalCall(request) => {
            panic!("expected completion, got external call to {:?}", request.func_addr)
        }
    }
}

#[test]
fn imported_call_suspends_and_resumes() {
    // run(n) = mul2(n) + 1, where mul2 is imported
    let funcs = compile_wat(
        "(module
            (import \"env\" \"mul2\" (func $mul2 (param i32) (result i32)))
            (func $run (param i32) (result i32)
                (i32.add (call $mul2 (local.get 0)) (i32.const 1))))",
    );

    let mut resources = empty_resources();
    let types = [ftype(&[ValueType::I32], &[ValueType::I32])];
    let functions = [FuncEntry {
        addr: FuncAddr(7),
        type_idx: 0,
    }];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &[],
        table_addrs: &[],
        types: &types,
        functions: &functions,
        num_imported: 1,
        segments: &mut segments,
        data_segments: &[],
    };

    let mut executor = FlatExecutor::new();
    let outcome = executor
        .invoke(&funcs, 0, &[Value::I32(21)], Some(&mut ctx))
        .expect("invoke failed");

    // Suspended at the import with the popped argument
    let request = expect_external_call(outcome);
    assert_eq!(request.func_addr.0, 7);
    assert_eq!(request.args, vec![Value::I32(21)]);

    // The Store would call mul2(21) = 42; resume with that result
    let outcome = executor
        .resume_with_results(&funcs, vec![Value::I32(42)], Some(&mut ctx))
        .expect("resume failed");
    assert_eq!(expect_complete(outcome), vec![Value::I32(43)]);
}

#[test]
fn imported_call_from_nested_frame() {
    // Suspension happens two frames deep; the internal call stack and
    // both frames' locals must survive across the external call.
    let funcs = compile_wat(
        "(module
            (import \"env\" \"get\" (func $get (result i32)))
            (func $helper (result i32)
                (i32.add (call $get) (i32.const 10)))
            (func $run (param i32) (result i32)
                (i32.add (call $helper) (local.get 0))))",
    );

    let mut resources = empty_resources();
    let types = [ftype(&[], &[ValueType::I32])];
    let functions = [FuncEntry {
        addr: FuncAddr(0),
        type_idx: 0,
    }];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &[],
        table_addrs: &[],
        types: &types,
        functions: &functions,
        num_imported: 1,
        segments: &mut segments,
        data_segments: &[],
    };

    let mut executor = FlatExecutor::new();
    let outcome = executor
        .invoke(&funcs, 1, &[Value::I32(100)], Some(&mut ctx))
        .expect("invoke failed");
    let request = expect_external_call(outcome);
    assert!(request.args.is_empty());

    let outcome = executor
        .resume_with_results(&funcs, vec![Value::I32(5)], Some(&mut ctx))
        .expect("resume failed");
    // get() = 5, helper() = 15, run(100) = 115
    assert_eq!(expect_complete(outcome), vec![Value::I32(115)]);
}

#[test]
fn two_sequential_imported_calls() {
    let funcs = compile_wat(
        "(module
            (import \"env\" \"get\" (func $get (result i32)))
            (func $run (result i32)
                (i32.add (call $get) (call $get))))",
    );

    let mut resources = empty_resources();
    let types = [ftype(&[], &[ValueType::I32])];
    let functions = [FuncEntry {
        addr: FuncAddr(3),
        type_idx: 0,
    }];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &[],
        table_addrs: &[],
        types: &types,
        functions: &functions,
        num_imported: 1,
        segments: &mut segments,
        data_segments: &[],
    };

    let mut executor = FlatExecutor::new();
    let outcome = executor.invoke(&funcs, 0, &[], Some(&mut ctx)).expect("invoke failed");
    expect_external_call(outcome);

    let outcome = executor
        .resume_with_results(&funcs, vec![Value::I32(3)], Some(&mut ctx))
        .expect("first resume failed");
    expect_external_call(outcome);

    let outcome = executor
        .resume_with_results(&funcs, vec![Value::I32(4)], Some(&mut ctx))
        .expect("second resume failed");
    assert_eq!(expect_complete(outcome), vec![Value::I32(7)]);
}

#[test]
fn resume_without_suspension_traps() {
    let mut executor = FlatExecutor::new();
    let err = executor.resume_with_results(&[], vec![], None).unwrap_err();
    assert!(err.to_string().contains("resume called without saved execution state"));
}

#[test]
fn resume_with_wrong_result_count_traps() {
    let funcs = compile_wat(
        "(module
            (import \"env\" \"get\" (func $get (result i32)))
            (func $run (result i32) (call $get)))",
    );

    let mut resources = empty_resources();
    let types = [ftype(&[], &[ValueType::I32])];
    let functions = [FuncEntry {
        addr: FuncAddr(0),
        type_idx: 0,
    }];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &[],
        table_addrs: &[],
        types: &types,
        functions: &functions,
        num_imported: 1,
        segments: &mut segments,
        data_segments: &[],
    };

    let mut executor = FlatExecutor::new();
    let outcome = executor.invoke(&funcs, 0, &[], Some(&mut ctx)).expect("invoke failed");
    expect_external_call(outcome);

    // The import declares one result; resuming with two must trap
    let err = executor
        .resume_with_results(&funcs, vec![Value::I32(1), Value::I32(2)], Some(&mut ctx))
        .unwrap_err();
    assert!(
        err.to_string().contains("returned 2 values, expected 1"),
        "unexpected error: {err}"
    );
}

#[test]
fn execute_flat_rejects_imported_call() {
    let funcs = compile_wat(
        "(module
            (import \"env\" \"get\" (func $get (result i32)))
            (func $run (result i32) (call $get)))",
    );

    let mut resources = empty_resources();
    let types = [ftype(&[], &[ValueType::I32])];
    let functions = [FuncEntry {
        addr: FuncAddr(0),
        type_idx: 0,
    }];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &[],
        table_addrs: &[],
        types: &types,
        functions: &functions,
        num_imported: 1,
        segments: &mut segments,
        data_segments: &[],
    };

    let err = execute_flat(&funcs, 0, &[], Some(&mut ctx)).unwrap_err();
    assert!(err.to_string().contains("requires Store dispatch"));
}

// -- call_indirect --

use crate::parser::module::{ExternalKind, Limits, RefType};
use crate::runtime::table::Table;

/// Parse and compile a module, returning everything a call_indirect
/// context needs: compiled functions, the module's type section, and a
/// FuncEntry per module-level function with `FuncAddr(i)` assigned in
/// module index order (imports first), plus the import count.
fn compile_with_metadata(source: &str) -> (Vec<CompiledFunction>, Vec<FunctionType>, Vec<FuncEntry>, usize) {
    let module = wat::parse(source).expect("WAT parse failed");
    let types = module.types.types.clone();

    let mut entries = Vec::new();
    for imp in &module.imports.imports {
        if let ExternalKind::Function(type_idx) = imp.external_kind {
            entries.push(FuncEntry {
                addr: FuncAddr(entries.len()),
                type_idx,
            });
        }
    }
    let num_imported = entries.len();
    for func in &module.functions.functions {
        entries.push(FuncEntry {
            addr: FuncAddr(entries.len()),
            type_idx: func.ftype_index,
        });
    }

    (compiler::compile_module(&module), types, entries, num_imported)
}

/// A funcref table populated with the given entries (None = null slot).
fn make_table(entries: &[Option<usize>]) -> Table {
    let limits = Limits {
        min: entries.len() as u32,
        max: Some(entries.len() as u32),
    };
    let mut table = Table::new(RefType::FuncRef, limits).expect("table creation failed");
    for (i, entry) in entries.iter().enumerate() {
        if let Some(addr) = entry {
            table
                .set(i as u32, Some(Value::FuncRef(Some(FuncAddr(*addr)))))
                .expect("table set failed");
        }
    }
    table
}

const DISPATCH_WAT: &str = "(module
    (type $binop (func (param i32 i32) (result i32)))
    (table 3 funcref)
    (func $add (type $binop) (i32.add (local.get 0) (local.get 1)))
    (func $sub (type $binop) (i32.sub (local.get 0) (local.get 1)))
    (func $dispatch (param i32 i32 i32) (result i32)
        (call_indirect (type $binop) (local.get 1) (local.get 2) (local.get 0))))";

#[test]
fn call_indirect_dispatches_by_table_index() {
    let (funcs, types, entries, num_imported) = compile_with_metadata(DISPATCH_WAT);
    let mut resources = empty_resources();
    resources.tables.push(make_table(&[Some(0), Some(1)]));
    let table_addrs = [TableAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &[],
        table_addrs: &table_addrs,
        types: &types,
        functions: &entries,
        num_imported,
        segments: &mut segments,
        data_segments: &[],
    };

    // Entry 0 is $add, entry 1 is $sub; $dispatch is compiled func 2
    let args = [Value::I32(0), Value::I32(10), Value::I32(4)];
    let result = execute_flat(&funcs, 2, &args, Some(&mut ctx)).expect("execution failed");
    assert_eq!(result, vec![Value::I32(14)]);

    let args = [Value::I32(1), Value::I32(10), Value::I32(4)];
    let result = execute_flat(&funcs, 2, &args, Some(&mut ctx)).expect("execution failed");
    assert_eq!(result, vec![Value::I32(6)]);
}

#[test]
fn call_indirect_type_mismatch_traps() {
    let (funcs, types, entries, num_imported) = compile_with_metadata(
        "(module
            (type $binop (func (param i32 i32) (result i32)))
            (type $unop (func (param i32) (result i32)))
            (table 1 funcref)
            (func $neg (type $unop) (i32.sub (i32.const 0) (local.get 0)))
            (func $main (param i32 i32) (result i32)
                (call_indirect (type $binop) (local.get 0) (local.get 1) (i32.const 0))))",
    );
    let mut resources = empty_resources();
    resources.tables.push(make_table(&[Some(0)]));
    let table_addrs = [TableAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &[],
        table_addrs: &table_addrs,
        types: &types,
        functions: &entries,
        num_imported,
        segments: &mut segments,
        data_segments: &[],
    };

    let err = execute_flat(&funcs, 1, &[Value::I32(1), Value::I32(2)], Some(&mut ctx)).unwrap_err();
    assert!(
        err.to_string().contains("indirect call type mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn call_indirect_null_entry_traps() {
    let (funcs, types, entries, num_imported) = compile_with_metadata(DISPATCH_WAT);
    let mut resources = empty_resources();
    resources.tables.push(make_table(&[None]));
    let table_addrs = [TableAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &[],
        table_addrs: &table_addrs,
        types: &types,
        functions: &entries,
        num_imported,
        segments: &mut segments,
        data_segments: &[],
    };

    let args = [Value::I32(0), Value::I32(1), Value::I32(2)];
    let err = execute_flat(&funcs, 2, &args, Some(&mut ctx)).unwrap_err();
    assert!(
        err.to_string().contains("uninitialized element"),
        "unexpected error: {err}"
    );
}

#[test]
fn call_indirect_out_of_bounds_traps() {
    let (funcs, types, entries, num_imported) = compile_with_metadata(DISPATCH_WAT);
    let mut resources = empty_resources();
    resources.tables.push(make_table(&[Some(0)]));
    let table_addrs = [TableAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &[],
        table_addrs: &table_addrs,
        types: &types,
        functions: &entries,
        num_imported,
        segments: &mut segments,
        data_segments: &[],
    };

    let args = [Value::I32(5), Value::I32(1), Value::I32(2)];
    let err = execute_flat(&funcs, 2, &args, Some(&mut ctx)).unwrap_err();
    assert!(
        err.to_string().contains("uninitialized element"),
        "unexpected error: {err}"
    );
}

#[test]
fn call_indirect_to_import_suspends() {
    let (funcs, types, entries, num_imported) = compile_with_metadata(
        "(module
            (type $unop (func (param i32) (result i32)))
            (import \"env\" \"ext\" (func $ext (type $unop)))
            (table 1 funcref)
            (func $main (param i32) (result i32)
                (call_indirect (type $unop) (local.get 0) (i32.const 0))))",
    );
    assert_eq!(num_imported, 1);
    let mut resources = empty_resources();
    // Table slot 0 holds the import's address
    resources.tables.push(make_table(&[Some(0)]));
    let table_addrs = [TableAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &[],
        table_addrs: &table_addrs,
        types: &types,
        functions: &entries,
        num_imported,
        segments: &mut segments,
        data_segments: &[],
    };

    let mut executor = FlatExecutor::new();
    let outcome = executor
        .invoke(&funcs, 0, &[Value::I32(21)], Some(&mut ctx))
        .expect("invoke failed");
    let request = expect_external_call(outcome);
    assert_eq!(request.func_addr.0, 0);
    assert_eq!(request.args, vec![Value::I32(21)]);

    let outcome = executor
        .resume_with_results(&funcs, vec![Value::I32(42)], Some(&mut ctx))
        .expect("resume failed");
    assert_eq!(expect_complete(outcome), vec![Value::I32(42)]);
}

#[test]
fn call_indirect_foreign_funcref_suspends() {
    // A funcref whose address is not in this module's function list:
    // dispatch goes to the Store, which owns the type check.
    let (funcs, types, entries, num_imported) = compile_with_metadata(DISPATCH_WAT);
    let mut resources = empty_resources();
    resources.tables.push(make_table(&[Some(99)]));
    let table_addrs = [TableAddr(0)];
    let mut segments = SegmentState::default();
    let mut ctx = ExecContext {
        resources: &mut resources,
        global_addrs: &[],
        memory_addrs: &[],
        table_addrs: &table_addrs,
        types: &types,
        functions: &entries,
        num_imported,
        segments: &mut segments,
        data_segments: &[],
    };

    let mut executor = FlatExecutor::new();
    let args = [Value::I32(0), Value::I32(10), Value::I32(4)];
    let outcome = executor
        .invoke(&funcs, 2, &args, Some(&mut ctx))
        .expect("invoke failed");
    let request = expect_external_call(outcome);
    assert_eq!(request.func_addr.0, 99);
    // Arguments popped per the expected $binop signature
    assert_eq!(request.args, vec![Value::I32(10), Value::I32(4)]);

    let outcome = executor
        .resume_with_results(&funcs, vec![Value::I32(14)], Some(&mut ctx))
        .expect("resume failed");
    assert_eq!(expect_complete(outcome), vec![Value::I32(14)]);
}
