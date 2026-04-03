//! Flat bytecode representation for the WebAssembly interpreter.
//!
//! A linear sequence of `Op` values with pre-resolved branch targets. Each
//! function compiles to a `CompiledFunction` containing a `Vec<Op>` that the
//! flat executor (`flat_executor.rs`) walks with a program counter. Branch
//! targets are absolute indices into the bytecode array.

use std::fmt;

/// Branch target with stack cleanup metadata.
#[derive(Debug, Clone, Copy)]
pub struct BrTarget {
    /// Absolute index in the bytecode array.
    pub pc: u32,
    /// Number of values to keep across the branch.
    pub arity: u16,
    /// Stack depth to restore to (below the kept values).
    pub stack_depth: u32,
}

/// A single operation in the flat bytecode.
///
/// Each variant carries its immediates inline. Branch targets are absolute
/// indices into the `Vec<Op>`, resolved at compile time.
#[derive(Debug, Clone)]
pub enum Op {
    // -- Constants --
    I32Const(i32),

    // -- Arithmetic --
    I32Add,
    I32Sub,
    I32Mul,

    // -- Comparison --
    I32Eqz,
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,

    // -- Local variables --
    /// Push the value of local `index` onto the stack.
    LocalGet {
        index: u32,
    },
    /// Pop the stack and store into local `index`.
    LocalSet {
        index: u32,
    },
    /// Copy top of stack into local `index` (value stays on stack).
    LocalTee {
        index: u32,
    },

    // -- Control flow --
    // Branch ops carry stack cleanup metadata: `arity` is the number of
    // values to keep (block results or loop params), `stack_depth` is
    // the stack depth to restore to before pushing kept values back.
    // For branches that don't cross block boundaries (e.g. compiler-internal
    // skip-then jumps), arity=0 and the cleanup is a no-op.
    /// Unconditional jump to `target` (absolute index in bytecode).
    Br {
        target: u32,
        arity: u16,
        stack_depth: u32,
    },
    /// Pop i32; if non-zero, jump to `target`.
    BrIf {
        target: u32,
        arity: u16,
        stack_depth: u32,
    },
    /// Pop i32 index; jump to `targets[index]` or `default` if out of bounds.
    /// All targets share the same arity (spec validation ensures this).
    BrTable {
        targets: Vec<BrTarget>,
        default: BrTarget,
    },
    /// Return from the current function.
    Return,
    /// No operation. Used as a placeholder (e.g. after block/loop markers
    /// that have been compiled away).
    Nop,
    /// End of function. The executor stops when it reaches this.
    End,

    // -- Block bookkeeping --
    /// Marks the start of a block's scope. At runtime this is a no-op; it
    /// exists so that bytecode dumps show the control flow structure, and
    /// future compilation tiers can identify basic block boundaries.
    /// `end_target` points to the instruction after the block's End.
    Label {
        end_target: u32,
    },

    /// Unreachable trap.
    Unreachable,

    /// Drop top of stack.
    Drop,
}

impl Op {
    /// Net stack effect: how many values this op pushes minus how many it pops.
    /// Used by the compiler to track stack depth during emission.
    pub fn stack_delta(&self) -> i32 {
        match self {
            Op::I32Const(_) => 1,
            Op::I32Add | Op::I32Sub | Op::I32Mul => -1,
            Op::I32Eqz => 0,
            Op::I32Eq | Op::I32Ne => -1,
            Op::I32LtS | Op::I32LtU | Op::I32GtS | Op::I32GtU => -1,
            Op::I32LeS | Op::I32LeU | Op::I32GeS | Op::I32GeU => -1,
            Op::LocalGet { .. } => 1,
            Op::LocalSet { .. } => -1,
            Op::LocalTee { .. } => 0,
            Op::Br { .. } => 0,      // unreachable after, depth irrelevant
            Op::BrIf { .. } => -1,   // pops condition
            Op::BrTable { .. } => -1, // pops index
            Op::Return | Op::End | Op::Unreachable => 0,
            Op::Nop | Op::Label { .. } => 0,
            Op::Drop => -1,
        }
    }
}

/// A compiled function ready for flat execution.
#[derive(Debug, Clone)]
pub struct CompiledFunction {
    /// The flat bytecode for this function.
    pub ops: Vec<Op>,
    /// Number of locals (including parameters).
    pub local_count: u32,
    /// Number of parameters (first N locals).
    pub param_count: u32,
    /// Number of return values.
    pub result_count: u32,
}

impl fmt::Display for CompiledFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "CompiledFunction(params={}, locals={}, results={})",
            self.param_count,
            self.local_count - self.param_count,
            self.result_count,
        )?;
        for (i, op) in self.ops.iter().enumerate() {
            writeln!(f, "  {i:4}: {op}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::I32Const(v) => write!(f, "i32.const {v}"),
            Op::I32Add => write!(f, "i32.add"),
            Op::I32Sub => write!(f, "i32.sub"),
            Op::I32Mul => write!(f, "i32.mul"),
            Op::I32Eqz => write!(f, "i32.eqz"),
            Op::I32Eq => write!(f, "i32.eq"),
            Op::I32Ne => write!(f, "i32.ne"),
            Op::I32LtS => write!(f, "i32.lt_s"),
            Op::I32LtU => write!(f, "i32.lt_u"),
            Op::I32GtS => write!(f, "i32.gt_s"),
            Op::I32GtU => write!(f, "i32.gt_u"),
            Op::I32LeS => write!(f, "i32.le_s"),
            Op::I32LeU => write!(f, "i32.le_u"),
            Op::I32GeS => write!(f, "i32.ge_s"),
            Op::I32GeU => write!(f, "i32.ge_u"),
            Op::LocalGet { index } => write!(f, "local.get {index}"),
            Op::LocalSet { index } => write!(f, "local.set {index}"),
            Op::LocalTee { index } => write!(f, "local.tee {index}"),
            Op::Br {
                target,
                arity,
                stack_depth,
            } => {
                if *arity == 0 {
                    write!(f, "br -> {target}")
                } else {
                    write!(f, "br -> {target} (stack {stack_depth}, keep {arity} values)")
                }
            }
            Op::BrIf {
                target,
                arity,
                stack_depth,
            } => {
                if *arity == 0 {
                    write!(f, "br_if -> {target}")
                } else {
                    write!(f, "br_if -> {target} (stack {stack_depth}, keep {arity} values)")
                }
            }
            Op::BrTable { targets, default } => {
                write!(f, "br_table [")?;
                for (i, t) in targets.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", t.pc)?;
                }
                write!(f, "] default -> {}", default.pc)
            }
            Op::Return => write!(f, "return"),
            Op::Nop => write!(f, "nop"),
            Op::End => write!(f, "end"),
            Op::Label { end_target } => write!(f, "label (end -> {end_target})"),
            Op::Unreachable => write!(f, "unreachable"),
            Op::Drop => write!(f, "drop"),
        }
    }
}
