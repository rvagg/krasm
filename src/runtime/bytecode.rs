//! Flat bytecode representation for the WebAssembly interpreter.
//!
//! A linear sequence of `Op` values with pre-resolved branch targets. Each
//! function compiles to a `CompiledFunction` containing a `Vec<Op>` that the
//! flat executor (`flat_executor.rs`) walks with a program counter. Branch
//! targets are absolute indices into the bytecode array.

use crate::parser::instruction::MemArg;
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

    // -- Global variables --
    /// Push the value of global at `index` (module-local) onto the stack.
    GlobalGet {
        index: u32,
    },
    /// Pop the stack and store into global at `index` (module-local).
    GlobalSet {
        index: u32,
    },

    // -- Memory --
    // Load ops pop an i32 address, apply offset from MemArg, push the loaded value.
    I32Load(MemArg),
    I32Load8S(MemArg),
    I32Load8U(MemArg),
    I32Load16S(MemArg),
    I32Load16U(MemArg),
    I64Load(MemArg),
    I64Load8S(MemArg),
    I64Load8U(MemArg),
    I64Load16S(MemArg),
    I64Load16U(MemArg),
    I64Load32S(MemArg),
    I64Load32U(MemArg),
    F32Load(MemArg),
    F64Load(MemArg),
    // Store ops pop a value and an i32 address, apply offset, write to memory.
    I32Store(MemArg),
    I32Store8(MemArg),
    I32Store16(MemArg),
    I64Store(MemArg),
    I64Store8(MemArg),
    I64Store16(MemArg),
    I64Store32(MemArg),
    F32Store(MemArg),
    F64Store(MemArg),
    MemorySize,
    MemoryGrow,
    MemoryCopy,
    MemoryFill,

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
            Op::GlobalGet { .. } => 1,
            Op::GlobalSet { .. } => -1,
            // Loads: pop addr, push value = 0
            Op::I32Load(_) | Op::I32Load8S(_) | Op::I32Load8U(_) => 0,
            Op::I32Load16S(_) | Op::I32Load16U(_) => 0,
            Op::I64Load(_) | Op::I64Load8S(_) | Op::I64Load8U(_) => 0,
            Op::I64Load16S(_) | Op::I64Load16U(_) => 0,
            Op::I64Load32S(_) | Op::I64Load32U(_) => 0,
            Op::F32Load(_) | Op::F64Load(_) => 0,
            // Stores: pop value + addr = -2
            Op::I32Store(_) | Op::I32Store8(_) | Op::I32Store16(_) => -2,
            Op::I64Store(_) | Op::I64Store8(_) | Op::I64Store16(_) | Op::I64Store32(_) => -2,
            Op::F32Store(_) | Op::F64Store(_) => -2,
            Op::MemorySize => 1,      // push page count
            Op::MemoryGrow => 0,      // pop pages, push old size
            Op::MemoryCopy => -3,     // pop dest, src, len
            Op::MemoryFill => -3,     // pop dest, val, len
            Op::Br { .. } => 0,       // unreachable after, depth irrelevant
            Op::BrIf { .. } => -1,    // pops condition
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
            Op::GlobalGet { index } => write!(f, "global.get {index}"),
            Op::GlobalSet { index } => write!(f, "global.set {index}"),
            Op::I32Load(m) => write!(f, "i32.load offset={}", m.offset),
            Op::I32Load8S(m) => write!(f, "i32.load8_s offset={}", m.offset),
            Op::I32Load8U(m) => write!(f, "i32.load8_u offset={}", m.offset),
            Op::I32Load16S(m) => write!(f, "i32.load16_s offset={}", m.offset),
            Op::I32Load16U(m) => write!(f, "i32.load16_u offset={}", m.offset),
            Op::I64Load(m) => write!(f, "i64.load offset={}", m.offset),
            Op::I64Load8S(m) => write!(f, "i64.load8_s offset={}", m.offset),
            Op::I64Load8U(m) => write!(f, "i64.load8_u offset={}", m.offset),
            Op::I64Load16S(m) => write!(f, "i64.load16_s offset={}", m.offset),
            Op::I64Load16U(m) => write!(f, "i64.load16_u offset={}", m.offset),
            Op::I64Load32S(m) => write!(f, "i64.load32_s offset={}", m.offset),
            Op::I64Load32U(m) => write!(f, "i64.load32_u offset={}", m.offset),
            Op::F32Load(m) => write!(f, "f32.load offset={}", m.offset),
            Op::F64Load(m) => write!(f, "f64.load offset={}", m.offset),
            Op::I32Store(m) => write!(f, "i32.store offset={}", m.offset),
            Op::I32Store8(m) => write!(f, "i32.store8 offset={}", m.offset),
            Op::I32Store16(m) => write!(f, "i32.store16 offset={}", m.offset),
            Op::I64Store(m) => write!(f, "i64.store offset={}", m.offset),
            Op::I64Store8(m) => write!(f, "i64.store8 offset={}", m.offset),
            Op::I64Store16(m) => write!(f, "i64.store16 offset={}", m.offset),
            Op::I64Store32(m) => write!(f, "i64.store32 offset={}", m.offset),
            Op::F32Store(m) => write!(f, "f32.store offset={}", m.offset),
            Op::F64Store(m) => write!(f, "f64.store offset={}", m.offset),
            Op::MemorySize => write!(f, "memory.size"),
            Op::MemoryGrow => write!(f, "memory.grow"),
            Op::MemoryCopy => write!(f, "memory.copy"),
            Op::MemoryFill => write!(f, "memory.fill"),
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
