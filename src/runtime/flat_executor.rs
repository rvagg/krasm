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
use super::store::{GlobalAddr, MemoryAddr, Resources};
use super::value::Value;

/// Perform stack cleanup for a branch: keep `arity` values from the top,
/// discard everything down to `stack_depth`, push the kept values back.
/// When arity is 0, this is a simple truncate.
fn branch_cleanup(stack: &mut Stack, arity: u16, abs_depth: usize) -> Result<(), RuntimeError> {
    if arity == 0 {
        stack.truncate(abs_depth);
        return Ok(());
    }
    let mut kept = Vec::with_capacity(arity as usize);
    for _ in 0..arity {
        kept.push(stack.pop()?);
    }
    stack.truncate(abs_depth);
    for v in kept.into_iter().rev() {
        stack.push(v);
    }
    Ok(())
}

/// Execution context providing access to store resources and address maps.
/// Passed to `execute_flat` when the function uses globals or memory.
pub struct ExecContext<'a> {
    pub resources: &'a mut Resources,
    pub global_addrs: &'a [GlobalAddr],
    pub memory_addrs: &'a [MemoryAddr],
    /// Number of imported functions. `Op::Call { func_idx }` values below
    /// this are imports (not yet supported); values at or above index into
    /// the `funcs` slice passed to `execute_flat`.
    pub num_imported: u32,
}

const MAX_CALL_DEPTH: usize = 1000;

/// Saved caller state for the call stack.
struct CallFrame {
    /// Index into the `funcs` slice (the function to resume).
    func_idx: usize,
    /// Program counter to resume at in the caller.
    pc: usize,
    /// Caller's stack base (for resolving branch stack_depth).
    stack_base: usize,
    /// Caller's local variables.
    locals: Vec<Value>,
}

impl ExecContext<'_> {
    /// Resolve module-local memory index 0 to the memory instance.
    fn memory(&self) -> Result<&super::memory::Memory, RuntimeError> {
        let addr = self
            .memory_addrs
            .first()
            .ok_or_else(|| RuntimeError::MemoryError("no memory".to_string()))?;
        self.resources
            .memories
            .get(addr.0)
            .ok_or_else(|| RuntimeError::MemoryError("invalid memory address".to_string()))
    }

    /// Resolve module-local memory index 0 to a mutable memory instance.
    fn memory_mut(&mut self) -> Result<&mut super::memory::Memory, RuntimeError> {
        let addr = self
            .memory_addrs
            .first()
            .ok_or_else(|| RuntimeError::MemoryError("no memory".to_string()))?;
        self.resources
            .memories
            .get_mut(addr.0)
            .ok_or_else(|| RuntimeError::MemoryError("invalid memory address".to_string()))
    }

    fn global_get(&self, stack: &mut Stack, index: u32) -> Result<(), RuntimeError> {
        let addr = self
            .global_addrs
            .get(index as usize)
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(index))?;
        let value = self
            .resources
            .globals
            .get(addr.0)
            .copied()
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(index))?;
        stack.push(value);
        Ok(())
    }

    fn global_set(&mut self, stack: &mut Stack, index: u32) -> Result<(), RuntimeError> {
        let val = stack.pop()?;
        let addr = self
            .global_addrs
            .get(index as usize)
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(index))?;
        let slot = self
            .resources
            .globals
            .get_mut(addr.0)
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(index))?;
        *slot = val;
        Ok(())
    }
}

/// Require a mutable reference to the execution context, or trap.
macro_rules! require_ctx {
    ($ctx:expr) => {
        $ctx.as_mut()
            .ok_or_else(|| RuntimeError::Trap("operation requires execution context".to_string()))?
    };
}

/// Build locals for the entry function from explicit arguments.
fn init_locals(func: &CompiledFunction, args: &[Value]) -> Vec<Value> {
    let mut locals = Vec::with_capacity(func.local_count as usize);
    locals.extend_from_slice(args);
    locals.resize(func.local_count as usize, Value::I32(0));
    locals
}

/// Build locals for a callee by popping arguments from the stack.
fn init_locals_from_stack(callee: &CompiledFunction, stack: &mut Stack) -> Result<Vec<Value>, RuntimeError> {
    let mut args = Vec::with_capacity(callee.param_count as usize);
    for _ in 0..callee.param_count {
        args.push(stack.pop()?);
    }
    args.reverse();
    let mut locals = Vec::with_capacity(callee.local_count as usize);
    locals.extend(args);
    locals.resize(callee.local_count as usize, Value::I32(0));
    Ok(locals)
}

/// Execute a compiled function with the given arguments.
///
/// `funcs` is the slice of all compiled local functions for the module.
/// `func_idx` is the index into `funcs` of the entry function.
/// `ctx` provides access to globals, memory, and import count. Pass `None`
/// for pure computation with no calls, globals, or memory.
pub fn execute_flat(
    funcs: &[CompiledFunction],
    func_idx: usize,
    args: &[Value],
    mut ctx: Option<&mut ExecContext<'_>>,
) -> Result<Vec<Value>, RuntimeError> {
    let func = &funcs[func_idx];
    let mut stack = Stack::new();

    // Initialise locals: parameters first, then zero-initialised locals.
    let mut locals = init_locals(func, args);

    let mut current_func_idx: usize = func_idx;
    let mut pc: usize = 0;
    let mut call_stack: Vec<CallFrame> = Vec::new();
    // Stack base for the current function. Branch stack_depth values are
    // offsets from this base, since the physical stack is shared across
    // all active call frames.
    let mut stack_base: usize = 0;

    loop {
        let ops_slice = &funcs[current_func_idx].ops;
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

            // -- Global variables --
            Op::GlobalGet { index } => {
                require_ctx!(ctx).global_get(&mut stack, *index)?;
                pc += 1;
            }
            Op::GlobalSet { index } => {
                require_ctx!(ctx).global_set(&mut stack, *index)?;
                pc += 1;
            }

            // -- Memory --
            Op::I32Load(m) => {
                ops::memory::i32_load(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I32Load8S(m) => {
                ops::memory::i32_load8_s(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I32Load8U(m) => {
                ops::memory::i32_load8_u(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I32Load16S(m) => {
                ops::memory::i32_load16_s(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I32Load16U(m) => {
                ops::memory::i32_load16_u(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I64Load(m) => {
                ops::memory::i64_load(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I64Load8S(m) => {
                ops::memory::i64_load8_s(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I64Load8U(m) => {
                ops::memory::i64_load8_u(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I64Load16S(m) => {
                ops::memory::i64_load16_s(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I64Load16U(m) => {
                ops::memory::i64_load16_u(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I64Load32S(m) => {
                ops::memory::i64_load32_s(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I64Load32U(m) => {
                ops::memory::i64_load32_u(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::F32Load(m) => {
                ops::memory::f32_load(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::F64Load(m) => {
                ops::memory::f64_load(&mut stack, require_ctx!(ctx).memory()?, m)?;
                pc += 1;
            }
            Op::I32Store(m) => {
                ops::memory::i32_store(&mut stack, require_ctx!(ctx).memory_mut()?, m)?;
                pc += 1;
            }
            Op::I32Store8(m) => {
                ops::memory::i32_store8(&mut stack, require_ctx!(ctx).memory_mut()?, m)?;
                pc += 1;
            }
            Op::I32Store16(m) => {
                ops::memory::i32_store16(&mut stack, require_ctx!(ctx).memory_mut()?, m)?;
                pc += 1;
            }
            Op::I64Store(m) => {
                ops::memory::i64_store(&mut stack, require_ctx!(ctx).memory_mut()?, m)?;
                pc += 1;
            }
            Op::I64Store8(m) => {
                ops::memory::i64_store8(&mut stack, require_ctx!(ctx).memory_mut()?, m)?;
                pc += 1;
            }
            Op::I64Store16(m) => {
                ops::memory::i64_store16(&mut stack, require_ctx!(ctx).memory_mut()?, m)?;
                pc += 1;
            }
            Op::I64Store32(m) => {
                ops::memory::i64_store32(&mut stack, require_ctx!(ctx).memory_mut()?, m)?;
                pc += 1;
            }
            Op::F32Store(m) => {
                ops::memory::f32_store(&mut stack, require_ctx!(ctx).memory_mut()?, m)?;
                pc += 1;
            }
            Op::F64Store(m) => {
                ops::memory::f64_store(&mut stack, require_ctx!(ctx).memory_mut()?, m)?;
                pc += 1;
            }
            Op::MemorySize => {
                ops::memory::memory_size(&mut stack, require_ctx!(ctx).memory()?)?;
                pc += 1;
            }
            Op::MemoryGrow => {
                ops::memory::memory_grow(&mut stack, require_ctx!(ctx).memory_mut()?)?;
                pc += 1;
            }
            Op::MemoryCopy => {
                ops::memory::memory_copy(&mut stack, require_ctx!(ctx).memory_mut()?)?;
                pc += 1;
            }
            Op::MemoryFill => {
                ops::memory::memory_fill(&mut stack, require_ctx!(ctx).memory_mut()?)?;
                pc += 1;
            }

            // -- Control flow --
            // These mutate pc directly; no ops function.
            Op::Br {
                target,
                arity,
                stack_depth,
            } => {
                branch_cleanup(&mut stack, *arity, stack_base + *stack_depth as usize)?;
                pc = *target as usize;
            }
            Op::BrIf {
                target,
                arity,
                stack_depth,
            } => {
                let cond = stack.pop_i32()?;
                if cond != 0 {
                    branch_cleanup(&mut stack, *arity, stack_base + *stack_depth as usize)?;
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
                branch_cleanup(&mut stack, target.arity, stack_base + target.stack_depth as usize)?;
                pc = target.pc as usize;
            }
            Op::Call { func_idx } => {
                let num_imported = ctx.as_ref().map(|c| c.num_imported).unwrap_or(0);
                let local_idx = (*func_idx as usize)
                    .checked_sub(num_imported as usize)
                    .ok_or_else(|| RuntimeError::Trap("imported function calls not yet supported".to_string()))?;
                if local_idx >= funcs.len() {
                    return Err(RuntimeError::FunctionIndexOutOfBounds(*func_idx));
                }
                if call_stack.len() >= MAX_CALL_DEPTH {
                    return Err(RuntimeError::CallStackOverflow);
                }

                let callee = &funcs[local_idx];
                let callee_locals = init_locals_from_stack(callee, &mut stack)?;

                let new_stack_base = stack.len();
                call_stack.push(CallFrame {
                    func_idx: current_func_idx,
                    pc: pc + 1,
                    stack_base,
                    locals: std::mem::replace(&mut locals, callee_locals),
                });
                current_func_idx = local_idx;
                stack_base = new_stack_base;
                pc = 0;
            }
            Op::Return | Op::End => {
                if let Some(frame) = call_stack.pop() {
                    current_func_idx = frame.func_idx;
                    pc = frame.pc;
                    stack_base = frame.stack_base;
                    locals = frame.locals;
                } else {
                    break;
                }
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
        let mut ctx = ExecContext {
            resources: &mut resources,
            global_addrs: &global_addrs,
            memory_addrs: &[],
            num_imported: 0,
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
        let mut ctx = ExecContext {
            resources: &mut resources,
            global_addrs: &[],
            memory_addrs: &memory_addrs,
            num_imported: 0,
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
        let mut ctx = ExecContext {
            resources: &mut resources,
            global_addrs: &[],
            memory_addrs: &memory_addrs,
            num_imported: 0,
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
        let mut ctx = ExecContext {
            resources: &mut resources,
            global_addrs: &[],
            memory_addrs: &memory_addrs,
            num_imported: 0,
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
        let mut ctx = ExecContext {
            resources: &mut resources,
            global_addrs: &[],
            memory_addrs: &memory_addrs,
            num_imported: 0,
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
        let mut ctx = ExecContext {
            resources: &mut resources,
            global_addrs: &[],
            memory_addrs: &memory_addrs,
            num_imported: 0,
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
        let mut ctx = ExecContext {
            resources: &mut resources,
            global_addrs: &[],
            memory_addrs: &memory_addrs,
            num_imported: 0,
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
        let funcs = compile_wat(include_str!("../../benches/modules/fib_recursive.wat"));
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
}
