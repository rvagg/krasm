//! Flat bytecode executor.
//!
//! Executes a `CompiledFunction` by walking its `Vec<Op>` with a program
//! counter. Branch targets are pre-resolved absolute indices, so there is no
//! context stack, no label stack, and no multi-level dispatch. The executor
//! reuses the existing `Stack` and `Value` types.
//!
//! Instruction implementations are delegated to the `ops` module where
//! possible, keeping the dispatch loop thin.

use super::RuntimeError;
use super::bytecode::{CompiledFunction, Op};
use super::ops;
use super::stack::Stack;
use super::value::Value;

/// Perform stack cleanup for a branch: keep `arity` values from the top,
/// discard everything down to `stack_depth`, push the kept values back.
/// When arity is 0, this is a simple truncate.
fn branch_cleanup(stack: &mut Stack, arity: u16, stack_depth: u32) -> Result<(), RuntimeError> {
    if arity == 0 {
        stack.truncate(stack_depth as usize);
        return Ok(());
    }
    let mut kept = Vec::with_capacity(arity as usize);
    for _ in 0..arity {
        kept.push(stack.pop()?);
    }
    stack.truncate(stack_depth as usize);
    for v in kept.into_iter().rev() {
        stack.push(v);
    }
    Ok(())
}

/// Execute a compiled function with the given arguments.
///
/// Returns the function's result values.
pub fn execute_flat(func: &CompiledFunction, args: &[Value]) -> Result<Vec<Value>, RuntimeError> {
    let mut stack = Stack::new();

    // Initialise locals: parameters first, then zero-initialised locals.
    let mut locals = Vec::with_capacity(func.local_count as usize);
    for arg in args {
        locals.push(*arg);
    }
    // Remaining locals default to i32(0). A full implementation would use
    // the declared local types; for the spike, i32(0) covers all cases
    // since the benchmarks only use i32 locals.
    while locals.len() < func.local_count as usize {
        locals.push(Value::I32(0));
    }

    let ops_slice = &func.ops;
    let mut pc: usize = 0;

    loop {
        if pc >= ops_slice.len() {
            break;
        }

        match &ops_slice[pc] {
            // -- Constants --
            Op::I32Const(v) => {
                ops::numeric::i32_const(&mut stack, *v)?;
                pc += 1;
            }

            // -- Arithmetic --
            Op::I32Add => {
                ops::numeric::i32_add(&mut stack)?;
                pc += 1;
            }
            Op::I32Sub => {
                ops::numeric::i32_sub(&mut stack)?;
                pc += 1;
            }
            Op::I32Mul => {
                ops::numeric::i32_mul(&mut stack)?;
                pc += 1;
            }

            // -- Comparison --
            Op::I32Eqz => {
                ops::comparison::i32_eqz(&mut stack)?;
                pc += 1;
            }
            Op::I32Eq => {
                ops::comparison::i32_eq(&mut stack)?;
                pc += 1;
            }
            Op::I32Ne => {
                ops::comparison::i32_ne(&mut stack)?;
                pc += 1;
            }
            Op::I32LtS => {
                ops::comparison::i32_lt_s(&mut stack)?;
                pc += 1;
            }
            Op::I32LtU => {
                ops::comparison::i32_lt_u(&mut stack)?;
                pc += 1;
            }
            Op::I32GtS => {
                ops::comparison::i32_gt_s(&mut stack)?;
                pc += 1;
            }
            Op::I32GtU => {
                ops::comparison::i32_gt_u(&mut stack)?;
                pc += 1;
            }
            Op::I32LeS => {
                ops::comparison::i32_le_s(&mut stack)?;
                pc += 1;
            }
            Op::I32LeU => {
                ops::comparison::i32_le_u(&mut stack)?;
                pc += 1;
            }
            Op::I32GeS => {
                ops::comparison::i32_ge_s(&mut stack)?;
                pc += 1;
            }
            Op::I32GeU => {
                ops::comparison::i32_ge_u(&mut stack)?;
                pc += 1;
            }

            // -- Local variables --
            // These interact with the locals array directly; no ops function.
            Op::LocalGet { index } => {
                let val = locals
                    .get(*index as usize)
                    .copied()
                    .ok_or(RuntimeError::LocalIndexOutOfBounds(*index))?;
                stack.push(val);
                pc += 1;
            }
            Op::LocalSet { index } => {
                let val = stack.pop()?;
                let slot = locals
                    .get_mut(*index as usize)
                    .ok_or(RuntimeError::LocalIndexOutOfBounds(*index))?;
                *slot = val;
                pc += 1;
            }
            Op::LocalTee { index } => {
                let val = stack.pop()?;
                let slot = locals
                    .get_mut(*index as usize)
                    .ok_or(RuntimeError::LocalIndexOutOfBounds(*index))?;
                *slot = val;
                stack.push(val);
                pc += 1;
            }

            // -- Control flow --
            // These mutate pc directly; no ops function.
            Op::Br {
                target,
                arity,
                stack_depth,
            } => {
                branch_cleanup(&mut stack, *arity, *stack_depth)?;
                pc = *target as usize;
            }
            Op::BrIf {
                target,
                arity,
                stack_depth,
            } => {
                let cond = stack.pop_i32()?;
                if cond != 0 {
                    branch_cleanup(&mut stack, *arity, *stack_depth)?;
                    pc = *target as usize;
                } else {
                    pc += 1;
                }
            }
            Op::BrTable { targets, default } => {
                let index = stack.pop_i32()? as u32;
                let target = if (index as usize) < targets.len() {
                    &targets[index as usize]
                } else {
                    default
                };
                branch_cleanup(&mut stack, target.arity, target.stack_depth)?;
                pc = target.pc as usize;
            }
            Op::Return | Op::End => {
                break;
            }
            Op::Nop | Op::Label { .. } => {
                pc += 1;
            }
            Op::Drop => {
                ops::parametric::drop(&mut stack)?;
                pc += 1;
            }
            Op::Unreachable => {
                return Err(RuntimeError::Trap("unreachable".to_string()));
            }
        }
    }

    // Collect return values from the stack
    let mut results = Vec::with_capacity(func.result_count as usize);
    for _ in 0..func.result_count {
        results.push(stack.pop()?);
    }
    results.reverse();
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::compiler;
    use crate::wat;

    fn compile_and_run(source: &str, args: &[Value]) -> Vec<Value> {
        let module = wat::parse(source).expect("WAT parse failed");
        let func = &module.code.code[0];
        let ftype_idx = module.functions.functions[0].ftype_index;
        let ftype = module.types.get(ftype_idx).expect("type not found");
        let compiled = compiler::compile(&func.body, ftype.parameters.len() as u32, &module.types.types);
        execute_flat(&compiled, args).expect("execution failed")
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
    fn noop_loop_10() {
        let result = compile_and_run(include_str!("../../benches/modules/noop_loop.wat"), &[Value::I32(10)]);
        assert_eq!(result, vec![Value::I32(10)]);
    }

    #[test]
    fn noop_loop_1000() {
        let result = compile_and_run(include_str!("../../benches/modules/noop_loop.wat"), &[Value::I32(1000)]);
        assert_eq!(result, vec![Value::I32(1000)]);
    }

    #[test]
    fn fib_0() {
        let result = compile_and_run(
            include_str!("../../benches/modules/fib_iterative.wat"),
            &[Value::I32(0)],
        );
        assert_eq!(result, vec![Value::I32(0)]);
    }

    #[test]
    fn fib_1() {
        let result = compile_and_run(
            include_str!("../../benches/modules/fib_iterative.wat"),
            &[Value::I32(1)],
        );
        assert_eq!(result, vec![Value::I32(1)]);
    }

    #[test]
    fn fib_10() {
        let result = compile_and_run(
            include_str!("../../benches/modules/fib_iterative.wat"),
            &[Value::I32(10)],
        );
        assert_eq!(result, vec![Value::I32(55)]);
    }

    #[test]
    fn fib_20() {
        let result = compile_and_run(
            include_str!("../../benches/modules/fib_iterative.wat"),
            &[Value::I32(20)],
        );
        assert_eq!(result, vec![Value::I32(6765)]);
    }

    #[test]
    fn fib_46() {
        let result = compile_and_run(
            include_str!("../../benches/modules/fib_iterative.wat"),
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
}
