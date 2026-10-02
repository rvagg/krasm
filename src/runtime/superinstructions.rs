//! Optional scalar superinstructions: recognition, PC relocation and execution.
//!
//! The compiler lowers wasm normally, then fuses complete straight-line sequences.
//! No branch or label endpoint may enter a sequence's interior. Operands stay inline
//! and execution pushes only the result; no temporary operand-stack traffic is needed.
//!
//! Keep new families here, behind the single `Op::Super` executor hook. Each family
//! must preserve operand order, stack effect, traps and unfused instruction budgets.

use super::bytecode::Op;
use super::stack::Stack;
use super::{RuntimeError, Value};
use std::fmt;

/// Pure, non-trapping i32 binary operations. Division and remainder are excluded.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "instruction-profile", derive(serde::Serialize))]
pub enum I32Operation {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    ShrS,
    ShrU,
    Rotl,
    Rotr,
}

impl I32Operation {
    fn from_op(op: &Op) -> Option<Self> {
        Some(match op {
            Op::I32Add => Self::Add,
            Op::I32Sub => Self::Sub,
            Op::I32Mul => Self::Mul,
            Op::I32And => Self::And,
            Op::I32Or => Self::Or,
            Op::I32Xor => Self::Xor,
            Op::I32Shl => Self::Shl,
            Op::I32ShrS => Self::ShrS,
            Op::I32ShrU => Self::ShrU,
            Op::I32Rotl => Self::Rotl,
            Op::I32Rotr => Self::Rotr,
            _ => return None,
        })
    }

    fn apply(self, lhs: i32, rhs: i32) -> i32 {
        match self {
            Self::Add => lhs.wrapping_add(rhs),
            Self::Sub => lhs.wrapping_sub(rhs),
            Self::Mul => lhs.wrapping_mul(rhs),
            Self::And => lhs & rhs,
            Self::Or => lhs | rhs,
            Self::Xor => lhs ^ rhs,
            Self::Shl => lhs.wrapping_shl(rhs as u32),
            Self::ShrS => lhs.wrapping_shr(rhs as u32),
            Self::ShrU => (lhs as u32).wrapping_shr(rhs as u32) as i32,
            Self::Rotl => lhs.rotate_left(rhs as u32),
            Self::Rotr => lhs.rotate_right(rhs as u32),
        }
    }
}

/// Fused bytecode families, not WebAssembly instructions.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "instruction-profile", derive(serde::Serialize))]
#[cfg_attr(feature = "instruction-profile", serde(tag = "kind"))]
pub enum SuperInstruction {
    /// `local.get local; i32.const value; i32.<operation>` (net stack effect +1).
    I32LocalConst {
        local: u32,
        value: i32,
        operation: I32Operation,
    },
    /// `local.get lhs; local.get rhs; i32.<operation>` (net stack effect +1).
    I32LocalLocal {
        lhs: u32,
        rhs: u32,
        operation: I32Operation,
    },
}

impl SuperInstruction {
    fn recognise(ops: &[Op]) -> Option<Self> {
        match ops {
            [Op::LocalGet { index }, Op::I32Const(value), op] => Some(Self::I32LocalConst {
                local: *index,
                value: *value,
                operation: I32Operation::from_op(op)?,
            }),
            [Op::LocalGet { index: lhs }, Op::LocalGet { index: rhs }, op] => Some(Self::I32LocalLocal {
                lhs: *lhs,
                rhs: *rhs,
                operation: I32Operation::from_op(op)?,
            }),
            _ => None,
        }
    }

    /// The dispatcher has already charged the first constituent instruction.
    #[inline]
    pub(super) fn execute(
        self,
        locals: &[Value],
        stack: &mut Stack,
        budget: &mut Option<u64>,
    ) -> Result<(), RuntimeError> {
        let (lhs_index, operation) = match self {
            Self::I32LocalConst { local, operation, .. } => (local, operation),
            Self::I32LocalLocal { lhs, operation, .. } => (lhs, operation),
        };
        let lhs = get_local(locals, lhs_index)?;
        let rhs = if let Some(remaining) = budget {
            charge(remaining)?;
            let rhs = self.rhs(locals)?;
            charge(remaining)?;
            rhs
        } else {
            self.rhs(locals)?
        };
        // Binary wasm operations pop the right operand before the left.
        let rhs = as_i32(rhs)?;
        let lhs = as_i32(lhs)?;
        stack.push(Value::I32(operation.apply(lhs, rhs)));
        Ok(())
    }

    fn rhs(self, locals: &[Value]) -> Result<Value, RuntimeError> {
        match self {
            Self::I32LocalConst { value, .. } => Ok(Value::I32(value)),
            Self::I32LocalLocal { rhs, .. } => get_local(locals, rhs),
        }
    }
}

fn get_local(locals: &[Value], index: u32) -> Result<Value, RuntimeError> {
    locals
        .get(index as usize)
        .copied()
        .ok_or(RuntimeError::LocalIndexOutOfBounds(index))
}

fn as_i32(value: Value) -> Result<i32, RuntimeError> {
    match value {
        Value::I32(value) => Ok(value),
        value => Err(type_mismatch(value)),
    }
}

#[cold]
fn type_mismatch(value: Value) -> RuntimeError {
    RuntimeError::TypeMismatch {
        expected: "I32".into(),
        actual: format!("{:?}", value.typ()),
    }
}

fn charge(remaining: &mut u64) -> Result<(), RuntimeError> {
    if *remaining == 0 {
        return Err(RuntimeError::InstructionBudgetExhausted);
    }
    *remaining -= 1;
    Ok(())
}

impl fmt::Display for SuperInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32LocalConst {
                local,
                value,
                operation,
            } => {
                write!(f, "super.i32.local_const.{operation:?} local={local} value={value}")
            }
            Self::I32LocalLocal { lhs, rhs, operation } => {
                write!(f, "super.i32.local_local.{operation:?} lhs={lhs} rhs={rhs}")
            }
        }
    }
}

/// Compact compiler-produced bytecode in place, then relocate every absolute PC.
/// The one-past-end position is mapped too, for label endpoints.
pub(super) fn fuse(ops: &mut Vec<Op>) {
    // Avoid scratch allocations for functions with no fusion candidates.
    if !ops
        .windows(3)
        .any(|window| SuperInstruction::recognise(window).is_some())
    {
        return;
    }
    // Protect every PC that a branch or label can refer to.
    let mut entries = vec![false; ops.len() + 1];
    for op in ops.iter() {
        match op {
            Op::Br { target, .. } | Op::BrIf { target, .. } => entries[*target as usize] = true,
            Op::BrTable { targets, default } => {
                for target in targets.iter().chain(std::iter::once(default)) {
                    entries[target.pc as usize] = true;
                }
            }
            Op::Label { end_target } => entries[*end_target as usize] = true,
            _ => {}
        }
    }

    // Compact towards the front: read tracks original PCs, write tracks fused PCs.
    // Protected entries ensure no reference needs a swallowed interior PC.
    let mut relocated = vec![0; ops.len() + 1];
    let mut read = 0;
    let mut write = 0;
    while read < ops.len() {
        relocated[read] = write as u32;
        let fused = ops.get(read..read + 3).and_then(SuperInstruction::recognise);
        if let Some(fused) = fused.filter(|_| !entries[read + 1] && !entries[read + 2]) {
            ops[write] = Op::Super(fused);
            read += 3;
        } else {
            // Move owned immediates, such as branch tables, without cloning them.
            ops.swap(write, read);
            read += 1;
        }
        write += 1;
    }
    // Once the map is complete, relocate all references, including one-past-end.
    relocated[ops.len()] = write as u32;
    ops.truncate(write);
    for op in ops {
        match op {
            Op::Br { target, .. } | Op::BrIf { target, .. } => *target = relocated[*target as usize],
            Op::BrTable { targets, default } => {
                for target in targets.iter_mut().chain(std::iter::once(default)) {
                    target.pc = relocated[target.pc as usize];
                }
            }
            Op::Label { end_target } => *end_target = relocated[*end_target as usize],
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ExecutionOutcome;
    use crate::runtime::bytecode::{BrTarget, CompiledFunction};
    use crate::runtime::flat_executor::FlatExecutor;

    fn run(ops: Vec<Op>, args: &[Value], budget: Option<u64>) -> Result<Vec<Value>, RuntimeError> {
        let function = CompiledFunction {
            ops,
            local_defaults: vec![],
            param_count: args.len() as u32,
            result_count: 1,
        };
        let mut executor = FlatExecutor::new();
        executor.set_instruction_budget(budget);
        match executor.invoke(&[function], 0, args, None)? {
            ExecutionOutcome::Complete(values) => Ok(values),
            ExecutionOutcome::NeedsExternalCall(_) => panic!("unexpected external call"),
        }
    }

    #[test]
    fn branch_entries_inside_candidates_are_not_swallowed() {
        for interior in [1, 2] {
            // Br, taken BrIf, BrTable entry and BrTable default.
            for branch in 0..4 {
                let mut ops = vec![
                    Op::LocalGet { index: 0 },
                    Op::I32Const(1),
                    Op::I32Add,
                    Op::Drop,
                    Op::I32Const(10),
                ];
                if interior == 2 {
                    ops.push(Op::I32Const(5));
                }
                if branch != 0 {
                    ops.push(Op::I32Const(if branch == 2 { 0 } else { 1 }));
                }
                let target = BrTarget {
                    pc: (ops.len() + 1 + interior) as u32,
                    arity: interior as u16,
                    stack_depth: 0,
                };
                ops.push(match branch {
                    0 => Op::Br {
                        target: target.pc,
                        arity: target.arity,
                        stack_depth: 0,
                    },
                    1 => Op::BrIf {
                        target: target.pc,
                        arity: target.arity,
                        stack_depth: 0,
                    },
                    _ => Op::BrTable {
                        targets: vec![target],
                        default: target,
                    },
                });
                ops.extend([Op::LocalGet { index: 0 }, Op::I32Const(5), Op::I32Add, Op::End]);
                let expected = run(ops.clone(), &[Value::I32(100)], None).unwrap();
                assert_eq!(expected, vec![Value::I32(15)]);
                fuse(&mut ops);
                assert_eq!(run(ops, &[Value::I32(100)], None).unwrap(), expected);
            }
        }
    }

    #[test]
    fn label_endpoints_relocate_without_entering_fused_interiors() {
        let mut ops = vec![
            Op::Label { end_target: 5 },
            Op::LocalGet { index: 0 },
            Op::I32Const(1),
            Op::I32Add,
            Op::LocalGet { index: 0 },
            Op::I32Const(2),
            Op::I32Add,
            Op::Label { end_target: 9 },
            Op::End,
        ];
        fuse(&mut ops);
        let Op::Label { end_target } = ops[0] else {
            panic!("missing label")
        };
        assert!(matches!(ops[end_target as usize], Op::I32Const(2)));
        let Op::Label { end_target } = ops[ops.len() - 2] else {
            panic!("missing label")
        };
        assert_eq!(end_target as usize, ops.len());
    }

    #[test]
    fn budgets_and_operand_errors_preserve_unfused_order() {
        for (ops, args) in [
            (
                vec![Op::LocalGet { index: 0 }, Op::I32Const(1), Op::I32Add, Op::End],
                vec![Value::I32(i32::MAX)],
            ),
            (
                vec![
                    Op::LocalGet { index: 0 },
                    Op::LocalGet { index: 1 },
                    Op::I32Sub,
                    Op::End,
                ],
                vec![Value::I32(9), Value::I32(2)],
            ),
            (
                vec![
                    Op::LocalGet { index: 0 },
                    Op::LocalGet { index: 1 },
                    Op::I32Add,
                    Op::End,
                ],
                vec![Value::F64(1.0), Value::I64(2)],
            ),
            (
                vec![Op::LocalGet { index: 1 }, Op::I32Const(1), Op::I32Add, Op::End],
                vec![Value::I32(1)],
            ),
            (
                vec![
                    Op::LocalGet { index: 0 },
                    Op::LocalGet { index: 1 },
                    Op::I32Add,
                    Op::End,
                ],
                vec![Value::F64(1.0)],
            ),
        ] {
            let mut fused = ops.clone();
            fuse(&mut fused);
            for budget in [None, Some(0), Some(1), Some(2), Some(3), Some(4)] {
                let expected = run(ops.clone(), &args, budget);
                let actual = run(fused.clone(), &args, budget);
                match (actual, expected) {
                    (Ok(actual), Ok(expected)) => assert_eq!(actual, expected),
                    (Err(RuntimeError::InstructionBudgetExhausted), Err(RuntimeError::InstructionBudgetExhausted)) => {}
                    (
                        Err(RuntimeError::LocalIndexOutOfBounds(actual)),
                        Err(RuntimeError::LocalIndexOutOfBounds(expected)),
                    ) => assert_eq!(actual, expected),
                    (
                        Err(RuntimeError::TypeMismatch {
                            expected: ae,
                            actual: aa,
                        }),
                        Err(RuntimeError::TypeMismatch {
                            expected: ee,
                            actual: ea,
                        }),
                    ) => assert_eq!((ae, aa), (ee, ea)),
                    (actual, expected) => panic!("budget {budget:?}: {actual:?} != {expected:?}"),
                }
            }
        }
    }
}
