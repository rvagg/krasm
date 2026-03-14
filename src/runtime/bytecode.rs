//! Flat bytecode representation for the WebAssembly interpreter.
//!
//! A linear sequence of `Op` values with pre-resolved branch targets. Each
//! function compiles to a `CompiledFunction` containing a `Vec<Op>` that the
//! flat executor (`flat_executor.rs`) walks with a program counter. Branch
//! targets are absolute indices into the bytecode array.

use std::fmt;

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
    /// Unconditional jump to `target` (absolute index in bytecode).
    Br {
        target: u32,
    },
    /// Pop i32; if non-zero, jump to `target`.
    BrIf {
        target: u32,
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
            Op::Br { target } => write!(f, "br -> {target}"),
            Op::BrIf { target } => write!(f, "br_if -> {target}"),
            Op::Return => write!(f, "return"),
            Op::Nop => write!(f, "nop"),
            Op::End => write!(f, "end"),
            Op::Label { end_target } => write!(f, "label (end -> {end_target})"),
            Op::Unreachable => write!(f, "unreachable"),
            Op::Drop => write!(f, "drop"),
        }
    }
}
