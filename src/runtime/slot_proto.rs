//! Prototype slot-stack executor, built only with the `slot-proto` feature.
//!
//! Untagged 8-byte slots, the stack pointer, frame pointer and program counter
//! held in locals, pure operations inlined into each arm, and stack capacity
//! checked once per call from `max_stack_words`. It runs only modules without
//! imports, using integer operations, locals, branches and direct calls, all
//! of whose values occupy one slot.
//!
//! Frame layout, growing upwards:
//!
//! ```text
//! fp -> [params][declared locals][operands ...] <- sp
//!                                ^ base: branch depths count from here
//! ```

use super::RuntimeError;
use super::bytecode::{CompiledFunction, Op};
use super::compiler::compile_module;
use super::ops::{bitwise, comparison, numeric};
use crate::parser::ValidatedModule;
use crate::parser::module::ExportIndex;

const STACK_SLOTS: usize = 1 << 20;
const MAX_CALL_DEPTH: usize = 1000;

/// Where a caller resumes.
struct Frame {
    func: usize,
    pc: usize,
    fp: *mut u64,
    base: *mut u64,
}

/// Run the exported function `name` with integer arguments, returning its
/// results as raw slots.
pub fn invoke(module: &ValidatedModule, name: &str, args: &[u64]) -> Result<Vec<u64>, RuntimeError> {
    if module.imports.function_count() != 0 {
        return Err(RuntimeError::Trap("slot prototype: imports unsupported".into()));
    }
    let func = module
        .exports
        .exports
        .iter()
        .find_map(|export| match export.index {
            ExportIndex::Function(idx) if export.name == name => Some(idx as usize),
            _ => None,
        })
        .ok_or_else(|| RuntimeError::UnknownExport(name.into()))?;
    let funcs = compile_module(module)?;
    if let Some(op) = funcs.iter().flat_map(|f| &f.ops).find(|op| !supported(op)) {
        return Err(RuntimeError::UnimplementedInstruction(format!("slot prototype: {op}")));
    }
    let mut slots = vec![0u64; STACK_SLOTS];
    // SAFETY: `run` checks capacity on entry to every function.
    unsafe { run(&funcs, func, args, &mut slots) }
}

/// Whether the loop in `run` implements `op`.
fn supported(op: &Op) -> bool {
    use Op::*;
    matches!(
        op,
        I32Const(_)
            | I64Const(_)
            | LocalGet { .. }
            | LocalSet { .. }
            | LocalTee { .. }
            | Drop
            | Select
            | I32Add
            | I32Sub
            | I32Mul
            | I32DivS
            | I32DivU
            | I32RemS
            | I32RemU
            | I32And
            | I32Or
            | I32Xor
            | I32Shl
            | I32ShrS
            | I32ShrU
            | I32Eqz
            | I32Eq
            | I32Ne
            | I32LtS
            | I32LtU
            | I32GtS
            | I32GtU
            | I32LeS
            | I32LeU
            | I32GeS
            | I32GeU
            | I64Add
            | I64Sub
            | I64Mul
            | I64Eq
            | I64LtS
            | Label { .. }
            | Nop
            | Br { .. }
            | BrIf { .. }
            | Call { .. }
            | Return
            | End
    )
}

#[inline(always)]
unsafe fn get_i32(p: *mut u64) -> i32 {
    unsafe { *p as u32 as i32 }
}

#[inline(always)]
unsafe fn put_i32(p: *mut u64, v: i32) {
    unsafe { *p = v as u32 as u64 }
}

#[inline(always)]
unsafe fn get_i64(p: *mut u64) -> i64 {
    unsafe { *p as i64 }
}

#[inline(always)]
unsafe fn put_i64(p: *mut u64, v: i64) {
    unsafe { *p = v as u64 }
}

/// Reserve a callee's locals and operands above `fp`, or fail if they would
/// pass the end of the slot stack. Returns the operand base.
#[inline(always)]
unsafe fn enter(func: &CompiledFunction, fp: *mut u64, end: *mut u64) -> Result<*mut u64, RuntimeError> {
    unsafe {
        let params = func.param_words as usize;
        let locals = (func.local_words - func.param_words) as usize;
        if (end.offset_from(fp) as usize) < params + locals + func.max_stack_words as usize {
            return Err(RuntimeError::CallStackOverflow);
        }
        // Every supported local type's zero is all-zero bits.
        fp.add(params).write_bytes(0, locals);
        Ok(fp.add(params + locals))
    }
}

unsafe fn run(
    funcs: &[CompiledFunction],
    entry: usize,
    args: &[u64],
    slots: &mut [u64],
) -> Result<Vec<u64>, RuntimeError> {
    unsafe {
        let start = slots.as_mut_ptr();
        let end = start.add(slots.len());
        let mut frames: Vec<Frame> = Vec::new();

        let mut func = entry;
        let mut ops = funcs[func].ops.as_slice();
        let mut pc = 0usize;
        let mut fp = start;
        if args.len() != funcs[func].param_words as usize {
            return Err(RuntimeError::Trap("slot prototype: wrong argument count".into()));
        }
        fp.copy_from_nonoverlapping(args.as_ptr(), args.len());
        let mut base = enter(&funcs[func], fp, end)?;
        let mut sp = base;

        macro_rules! binary {
            ($get:ident, $put:ident, $f:path) => {{
                sp = sp.sub(1);
                $put(sp.sub(1), $f($get(sp.sub(1)), $get(sp)));
                pc += 1;
            }};
        }
        macro_rules! try_binary {
            ($get:ident, $put:ident, $f:path) => {{
                sp = sp.sub(1);
                $put(sp.sub(1), $f($get(sp.sub(1)), $get(sp))?);
                pc += 1;
            }};
        }
        macro_rules! compare {
            ($get:ident, $f:path) => {{
                sp = sp.sub(1);
                put_i32(sp.sub(1), $f($get(sp.sub(1)), $get(sp)));
                pc += 1;
            }};
        }
        // Move the top `arity` slots down to `base + depth` and jump.
        macro_rules! branch {
            ($target:expr, $arity:expr, $depth:expr) => {{
                let dest = base.add($depth as usize);
                let arity = $arity as usize;
                dest.copy_from(sp.sub(arity), arity);
                sp = dest.add(arity);
                pc = $target as usize;
            }};
        }

        loop {
            match ops.get_unchecked(pc) {
                Op::I32Const(v) => {
                    put_i32(sp, *v);
                    sp = sp.add(1);
                    pc += 1;
                }
                Op::I64Const(v) => {
                    put_i64(sp, *v);
                    sp = sp.add(1);
                    pc += 1;
                }
                Op::LocalGet { index } => {
                    *sp = *fp.add(*index as usize);
                    sp = sp.add(1);
                    pc += 1;
                }
                Op::LocalSet { index } => {
                    sp = sp.sub(1);
                    *fp.add(*index as usize) = *sp;
                    pc += 1;
                }
                Op::LocalTee { index } => {
                    *fp.add(*index as usize) = *sp.sub(1);
                    pc += 1;
                }
                Op::Drop => {
                    sp = sp.sub(1);
                    pc += 1;
                }
                Op::Select => {
                    sp = sp.sub(2);
                    if get_i32(sp.add(1)) == 0 {
                        *sp.sub(1) = *sp;
                    }
                    pc += 1;
                }

                Op::I32Add => binary!(get_i32, put_i32, numeric::i32_add),
                Op::I32Sub => binary!(get_i32, put_i32, numeric::i32_sub),
                Op::I32Mul => binary!(get_i32, put_i32, numeric::i32_mul),
                Op::I32DivS => try_binary!(get_i32, put_i32, numeric::i32_div_s),
                Op::I32DivU => try_binary!(get_i32, put_i32, numeric::i32_div_u),
                Op::I32RemS => try_binary!(get_i32, put_i32, numeric::i32_rem_s),
                Op::I32RemU => try_binary!(get_i32, put_i32, numeric::i32_rem_u),
                Op::I32And => binary!(get_i32, put_i32, bitwise::i32_and),
                Op::I32Or => binary!(get_i32, put_i32, bitwise::i32_or),
                Op::I32Xor => binary!(get_i32, put_i32, bitwise::i32_xor),
                Op::I32Shl => binary!(get_i32, put_i32, bitwise::i32_shl),
                Op::I32ShrS => binary!(get_i32, put_i32, bitwise::i32_shr_s),
                Op::I32ShrU => binary!(get_i32, put_i32, bitwise::i32_shr_u),
                Op::I32Eqz => {
                    put_i32(sp.sub(1), comparison::i32_eqz(get_i32(sp.sub(1))));
                    pc += 1;
                }
                Op::I32Eq => compare!(get_i32, comparison::i32_eq),
                Op::I32Ne => compare!(get_i32, comparison::i32_ne),
                Op::I32LtS => compare!(get_i32, comparison::i32_lt_s),
                Op::I32LtU => compare!(get_i32, comparison::i32_lt_u),
                Op::I32GtS => compare!(get_i32, comparison::i32_gt_s),
                Op::I32GtU => compare!(get_i32, comparison::i32_gt_u),
                Op::I32LeS => compare!(get_i32, comparison::i32_le_s),
                Op::I32LeU => compare!(get_i32, comparison::i32_le_u),
                Op::I32GeS => compare!(get_i32, comparison::i32_ge_s),
                Op::I32GeU => compare!(get_i32, comparison::i32_ge_u),
                Op::I64Add => binary!(get_i64, put_i64, numeric::i64_add),
                Op::I64Sub => binary!(get_i64, put_i64, numeric::i64_sub),
                Op::I64Mul => binary!(get_i64, put_i64, numeric::i64_mul),
                Op::I64Eq => compare!(get_i64, comparison::i64_eq),
                Op::I64LtS => compare!(get_i64, comparison::i64_lt_s),

                Op::Label { .. } | Op::Nop => pc += 1,
                Op::Br {
                    target,
                    arity,
                    stack_depth,
                } => branch!(*target, *arity, *stack_depth),
                Op::BrIf {
                    target,
                    arity,
                    stack_depth,
                } => {
                    sp = sp.sub(1);
                    if get_i32(sp) != 0 {
                        branch!(*target, *arity, *stack_depth);
                    } else {
                        pc += 1;
                    }
                }
                Op::Call { func_idx } => {
                    if frames.len() >= MAX_CALL_DEPTH {
                        return Err(RuntimeError::CallStackOverflow);
                    }
                    frames.push(Frame {
                        func,
                        pc: pc + 1,
                        fp,
                        base,
                    });
                    func = *func_idx as usize;
                    let callee = &funcs[func];
                    fp = sp.sub(callee.param_words as usize);
                    base = enter(callee, fp, end)?;
                    sp = base;
                    ops = callee.ops.as_slice();
                    pc = 0;
                }
                Op::Return | Op::End => {
                    let results = funcs[func].result_words as usize;
                    fp.copy_from(sp.sub(results), results);
                    sp = fp.add(results);
                    let Some(caller) = frames.pop() else {
                        return Ok(std::slice::from_raw_parts(fp, results).to_vec());
                    };
                    func = caller.func;
                    pc = caller.pc;
                    fp = caller.fp;
                    base = caller.base;
                    ops = funcs[func].ops.as_slice();
                }
                // SAFETY: `invoke` rejects functions with any other op. A
                // complete executor matches every op, so this arm stands in
                // for exhaustiveness.
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }
}
