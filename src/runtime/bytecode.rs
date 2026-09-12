//! Flat bytecode representation for the WebAssembly interpreter.
//!
//! A linear sequence of `Op` values with pre-resolved branch targets. Each
//! function compiles to a `CompiledFunction` containing a `Vec<Op>` that the
//! flat executor (`flat_executor.rs`) walks with a program counter. Branch
//! targets are absolute indices into the bytecode array.

use super::value::Value;
use crate::parser::instruction::MemArg;
use crate::parser::module::ValueType;
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
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),

    // -- SIMD --
    V128Const([u8; 16]),
    V128Not,
    V128And,
    V128AndNot,
    V128Or,
    V128Xor,
    V128Bitselect,
    V128AnyTrue,
    // Lane reductions replace one v128 with one i32.
    I8x16AllTrue,
    I8x16Bitmask,
    I16x8AllTrue,
    I16x8Bitmask,
    I32x4AllTrue,
    I32x4Bitmask,
    I64x2AllTrue,
    I64x2Bitmask,
    // Scalar splats consume operand-stack values; V128Load*Splat below reads memory.
    I8x16Splat,
    I16x8Splat,
    I32x4Splat,
    I64x2Splat,
    F32x4Splat,
    F64x2Splat,
    // Shuffle indices are fixed validated immediates; swizzle indices are runtime values.
    I8x16Shuffle([u8; 16]),
    I8x16Swizzle,
    // Lane indices are validated immediates, not operand-stack values.
    // Extract: v128 -> scalar, stack delta 0.
    I8x16ExtractLaneS(u8),
    I8x16ExtractLaneU(u8),
    I16x8ExtractLaneS(u8),
    I16x8ExtractLaneU(u8),
    I32x4ExtractLane(u8),
    I64x2ExtractLane(u8),
    F32x4ExtractLane(u8),
    F64x2ExtractLane(u8),
    // Replace: v128 + scalar -> v128, stack delta -1.
    I8x16ReplaceLane(u8),
    I16x8ReplaceLane(u8),
    I32x4ReplaceLane(u8),
    I64x2ReplaceLane(u8),
    F32x4ReplaceLane(u8),
    F64x2ReplaceLane(u8),

    // -- i32 arithmetic --
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32Clz,
    I32Ctz,
    I32Popcnt,

    // -- i32 bitwise --
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Rotl,
    I32Rotr,

    // -- i32 sign extension --
    I32Extend8S,
    I32Extend16S,

    // -- i64 (arithmetic, bitwise, comparison, extension) --
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64Clz,
    I64Ctz,
    I64Popcnt,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rotl,
    I64Rotr,
    I64Eqz,
    I64Eq,
    I64Ne,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,
    I64Extend8S,
    I64Extend16S,
    I64Extend32S,
    I64ExtendI32S,
    I64ExtendI32U,
    I32WrapI64,

    // -- f32/f64 (arithmetic, comparison) and conversions --
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Min,
    F32Max,
    F32Copysign,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Min,
    F64Max,
    F64Copysign,
    F32Abs,
    F32Neg,
    F32Sqrt,
    F32Ceil,
    F32Floor,
    F32Trunc,
    F32Nearest,
    F64Abs,
    F64Neg,
    F64Sqrt,
    F64Ceil,
    F64Floor,
    F64Trunc,
    F64Nearest,
    F32Eq,
    F32Ne,
    F32Lt,
    F32Gt,
    F32Le,
    F32Ge,
    F64Eq,
    F64Ne,
    F64Lt,
    F64Gt,
    F64Le,
    F64Ge,
    F32ConvertI32S,
    F32ConvertI32U,
    F32ConvertI64S,
    F32ConvertI64U,
    F64ConvertI32S,
    F64ConvertI32U,
    F64ConvertI64S,
    F64ConvertI64U,
    F32DemoteF64,
    F64PromoteF32,
    I32TruncF32S,
    I32TruncF32U,
    I32TruncF64S,
    I32TruncF64U,
    I64TruncF32S,
    I64TruncF32U,
    I64TruncF64S,
    I64TruncF64U,
    I32TruncSatF32S,
    I32TruncSatF32U,
    I32TruncSatF64S,
    I32TruncSatF64U,
    I64TruncSatF32S,
    I64TruncSatF32U,
    I64TruncSatF64S,
    I64TruncSatF64U,
    I32ReinterpretF32,
    I64ReinterpretF64,
    F32ReinterpretI32,
    F64ReinterpretI64,

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
    V128Load(MemArg),
    // These read less than 16 bytes but each produces a full v128.
    V128Load8x8S(MemArg),
    V128Load8x8U(MemArg),
    V128Load16x4S(MemArg),
    V128Load16x4U(MemArg),
    V128Load32x2S(MemArg),
    V128Load32x2U(MemArg),
    V128Load8Splat(MemArg),
    V128Load16Splat(MemArg),
    V128Load32Splat(MemArg),
    V128Load64Splat(MemArg),
    V128Load32Zero(MemArg),
    V128Load64Zero(MemArg),
    // Lane loads consume an existing v128 and replace only the selected lane.
    V128Load8Lane {
        memarg: MemArg,
        lane: u8,
    },
    V128Load16Lane {
        memarg: MemArg,
        lane: u8,
    },
    V128Load32Lane {
        memarg: MemArg,
        lane: u8,
    },
    V128Load64Lane {
        memarg: MemArg,
        lane: u8,
    },
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
    V128Store(MemArg),
    V128Store8Lane {
        memarg: MemArg,
        lane: u8,
    },
    V128Store16Lane {
        memarg: MemArg,
        lane: u8,
    },
    V128Store32Lane {
        memarg: MemArg,
        lane: u8,
    },
    V128Store64Lane {
        memarg: MemArg,
        lane: u8,
    },
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
    /// Call function at module-level `func_idx`. The flat executor adjusts
    /// by import count to index into the compiled functions slice.
    Call {
        func_idx: u32,
    },
    /// Pop an i32 element index, look up a funcref in table `table_idx`,
    /// check its signature against type `type_idx`, and call it. The callee
    /// may be local, imported (suspends execution), or foreign (a funcref
    /// from another module; also suspends, with the type check deferred to
    /// the Store).
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
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
    /// Placeholder for an instruction the flat compiler does not support
    /// yet. Traps with the instruction's mnemonic, so a failing program
    /// names exactly what is missing.
    Unsupported(&'static str),

    /// Drop top of stack.
    Drop,
    /// Pop an i32 condition and two values; push the first if the condition
    /// is non-zero, the second otherwise. Typed and untyped select compile
    /// to the same op (the type annotation is validation-only).
    Select,

    // -- References --
    /// Push a null reference of the given type (FuncRef or ExternRef).
    RefNull(ValueType),
    /// Pop a reference; push 1 if null, 0 otherwise.
    RefIsNull,
    /// Push a funcref to module-level function `func_idx`.
    RefFunc {
        func_idx: u32,
    },

    // -- Tables and bulk memory --
    TableGet {
        table_idx: u32,
    },
    TableSet {
        table_idx: u32,
    },
    TableSize {
        table_idx: u32,
    },
    TableGrow {
        table_idx: u32,
    },
    TableFill {
        table_idx: u32,
    },
    TableCopy {
        dst_table: u32,
        src_table: u32,
    },
    TableInit {
        elem_idx: u32,
        table_idx: u32,
    },
    ElemDrop {
        elem_idx: u32,
    },
    MemoryInit {
        data_idx: u32,
    },
    DataDrop {
        data_idx: u32,
    },
}

impl Op {
    /// Net stack effect: how many values this op pushes minus how many it pops.
    /// Used by the compiler to track stack depth during emission.
    pub fn stack_delta(&self) -> i32 {
        match self {
            Op::I32Const(_) | Op::I64Const(_) | Op::F32Const(_) | Op::F64Const(_) | Op::V128Const(_) => 1,
            Op::I32Add | Op::I32Sub | Op::I32Mul => -1,
            Op::I32DivS | Op::I32DivU | Op::I32RemS | Op::I32RemU => -1,
            Op::I32And | Op::I32Or | Op::I32Xor => -1,
            Op::I32Shl | Op::I32ShrS | Op::I32ShrU | Op::I32Rotl | Op::I32Rotr => -1,
            Op::I32Clz | Op::I32Ctz | Op::I32Popcnt => 0,
            Op::I32Extend8S | Op::I32Extend16S => 0,
            Op::I64Add | Op::I64Sub | Op::I64Mul | Op::I64DivS => -1,
            Op::I64DivU | Op::I64RemS | Op::I64RemU | Op::I64And => -1,
            Op::I64Or | Op::I64Xor | Op::I64Shl | Op::I64ShrS => -1,
            Op::I64ShrU | Op::I64Rotl | Op::I64Rotr | Op::I64Eq => -1,
            Op::I64Ne | Op::I64LtS | Op::I64LtU | Op::I64GtS => -1,
            Op::I64GtU | Op::I64LeS | Op::I64LeU | Op::I64GeS => -1,
            Op::I64GeU => -1,
            Op::I64Clz | Op::I64Ctz | Op::I64Popcnt | Op::I64Eqz => 0,
            Op::I64Extend8S | Op::I64Extend16S | Op::I64Extend32S | Op::I64ExtendI32S => 0,
            Op::I64ExtendI32U | Op::I32WrapI64 => 0,
            Op::F32Add | Op::F32Sub | Op::F32Mul | Op::F32Div => -1,
            Op::F32Min | Op::F32Max | Op::F32Copysign | Op::F64Add => -1,
            Op::F64Sub | Op::F64Mul | Op::F64Div | Op::F64Min => -1,
            Op::F64Max | Op::F64Copysign | Op::F32Eq | Op::F32Ne => -1,
            Op::F32Lt | Op::F32Gt | Op::F32Le | Op::F32Ge => -1,
            Op::F64Eq | Op::F64Ne | Op::F64Lt | Op::F64Gt => -1,
            Op::F64Le | Op::F64Ge => -1,
            Op::F32Abs | Op::F32Neg | Op::F32Sqrt | Op::F32Ceil => 0,
            Op::F32Floor | Op::F32Trunc | Op::F32Nearest | Op::F64Abs => 0,
            Op::F64Neg | Op::F64Sqrt | Op::F64Ceil | Op::F64Floor => 0,
            Op::F64Trunc | Op::F64Nearest | Op::F32ConvertI32S | Op::F32ConvertI32U => 0,
            Op::F32ConvertI64S | Op::F32ConvertI64U | Op::F64ConvertI32S | Op::F64ConvertI32U => 0,
            Op::F64ConvertI64S | Op::F64ConvertI64U | Op::F32DemoteF64 | Op::F64PromoteF32 => 0,
            Op::I32TruncF32S | Op::I32TruncF32U | Op::I32TruncF64S | Op::I32TruncF64U => 0,
            Op::I64TruncF32S | Op::I64TruncF32U | Op::I64TruncF64S | Op::I64TruncF64U => 0,
            Op::I32TruncSatF32S | Op::I32TruncSatF32U | Op::I32TruncSatF64S | Op::I32TruncSatF64U => 0,
            Op::I64TruncSatF32S | Op::I64TruncSatF32U | Op::I64TruncSatF64S | Op::I64TruncSatF64U => 0,
            Op::I32ReinterpretF32 | Op::I64ReinterpretF64 | Op::F32ReinterpretI32 | Op::F64ReinterpretI64 => 0,
            Op::V128Not
            | Op::V128AnyTrue
            | Op::I8x16AllTrue
            | Op::I8x16Bitmask
            | Op::I16x8AllTrue
            | Op::I16x8Bitmask
            | Op::I32x4AllTrue
            | Op::I32x4Bitmask
            | Op::I64x2AllTrue
            | Op::I64x2Bitmask => 0,
            Op::I8x16Splat | Op::I16x8Splat | Op::I32x4Splat | Op::I64x2Splat | Op::F32x4Splat | Op::F64x2Splat => 0,
            Op::I8x16Shuffle(_) | Op::I8x16Swizzle => -1,
            Op::I8x16ExtractLaneS(_)
            | Op::I8x16ExtractLaneU(_)
            | Op::I16x8ExtractLaneS(_)
            | Op::I16x8ExtractLaneU(_)
            | Op::I32x4ExtractLane(_)
            | Op::I64x2ExtractLane(_)
            | Op::F32x4ExtractLane(_)
            | Op::F64x2ExtractLane(_) => 0,
            Op::I8x16ReplaceLane(_)
            | Op::I16x8ReplaceLane(_)
            | Op::I32x4ReplaceLane(_)
            | Op::I64x2ReplaceLane(_)
            | Op::F32x4ReplaceLane(_)
            | Op::F64x2ReplaceLane(_) => -1,
            Op::V128And | Op::V128AndNot | Op::V128Or | Op::V128Xor => -1,
            Op::V128Bitselect => -2,
            Op::Select => -2, // pop condition + one branch, keep the other
            Op::RefNull(_) | Op::RefFunc { .. } | Op::TableSize { .. } => 1,
            Op::RefIsNull | Op::TableGet { .. } => 0, // pop one, push one
            Op::TableSet { .. } => -2,
            Op::TableGrow { .. } => -1, // pop init + delta, push old size
            Op::TableFill { .. } | Op::TableCopy { .. } | Op::TableInit { .. } => -3,
            Op::MemoryInit { .. } => -3,
            Op::ElemDrop { .. } | Op::DataDrop { .. } => 0,
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
            Op::F32Load(_)
            | Op::F64Load(_)
            | Op::V128Load(_)
            | Op::V128Load8x8S(_)
            | Op::V128Load8x8U(_)
            | Op::V128Load16x4S(_)
            | Op::V128Load16x4U(_)
            | Op::V128Load32x2S(_)
            | Op::V128Load32x2U(_)
            | Op::V128Load8Splat(_)
            | Op::V128Load16Splat(_)
            | Op::V128Load32Splat(_)
            | Op::V128Load64Splat(_)
            | Op::V128Load32Zero(_)
            | Op::V128Load64Zero(_) => 0,
            // Lane loads: pop vector + addr, push updated vector = -1
            Op::V128Load8Lane { .. }
            | Op::V128Load16Lane { .. }
            | Op::V128Load32Lane { .. }
            | Op::V128Load64Lane { .. } => -1,
            // Stores: pop value + addr = -2
            Op::I32Store(_) | Op::I32Store8(_) | Op::I32Store16(_) => -2,
            Op::I64Store(_) | Op::I64Store8(_) | Op::I64Store16(_) | Op::I64Store32(_) => -2,
            Op::F32Store(_)
            | Op::F64Store(_)
            | Op::V128Store(_)
            | Op::V128Store8Lane { .. }
            | Op::V128Store16Lane { .. }
            | Op::V128Store32Lane { .. }
            | Op::V128Store64Lane { .. } => -2,
            Op::MemorySize => 1,          // push page count
            Op::MemoryGrow => 0,          // pop pages, push old size
            Op::MemoryCopy => -3,         // pop dest, src, len
            Op::MemoryFill => -3,         // pop dest, val, len
            Op::Call { .. } => 0,         // variable: handled separately in compiler
            Op::CallIndirect { .. } => 0, // variable: handled separately in compiler
            Op::Br { .. } => 0,           // unreachable after, depth irrelevant
            Op::BrIf { .. } => -1,        // pops condition
            Op::BrTable { .. } => -1,     // pops index
            Op::Return | Op::End | Op::Unreachable | Op::Unsupported(_) => 0,
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
    /// Typed zero values for the declared (non-parameter) locals, expanded
    /// at compile time. The executor copies these when building a frame, so
    /// an untouched local reads as the correct type's zero, per spec.
    pub local_defaults: Vec<Value>,
    /// Number of parameters (first N locals).
    pub param_count: u32,
    /// Number of return values.
    pub result_count: u32,
}

impl CompiledFunction {
    /// Total locals: parameters plus declared locals.
    pub fn local_count(&self) -> usize {
        self.param_count as usize + self.local_defaults.len()
    }
}

impl fmt::Display for CompiledFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "CompiledFunction(params={}, locals={}, results={})",
            self.param_count,
            self.local_defaults.len(),
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
            Op::I64Const(v) => write!(f, "i64.const {v}"),
            Op::F32Const(v) => write!(f, "f32.const {v}"),
            Op::F64Const(v) => write!(f, "f64.const {v}"),
            Op::V128Const(value) => {
                write!(f, "v128.const")?;
                for i in 0..4 {
                    let lane = u32::from_le_bytes([value[i * 4], value[i * 4 + 1], value[i * 4 + 2], value[i * 4 + 3]]);
                    write!(f, " 0x{lane:08x}")?;
                }
                Ok(())
            }
            Op::I32Add => write!(f, "i32.add"),
            Op::I32Sub => write!(f, "i32.sub"),
            Op::I32Mul => write!(f, "i32.mul"),
            Op::I32DivS => write!(f, "i32.div_s"),
            Op::I32DivU => write!(f, "i32.div_u"),
            Op::I32RemS => write!(f, "i32.rem_s"),
            Op::I32RemU => write!(f, "i32.rem_u"),
            Op::I32Clz => write!(f, "i32.clz"),
            Op::I32Ctz => write!(f, "i32.ctz"),
            Op::I32Popcnt => write!(f, "i32.popcnt"),
            Op::I32And => write!(f, "i32.and"),
            Op::I32Or => write!(f, "i32.or"),
            Op::I32Xor => write!(f, "i32.xor"),
            Op::I32Shl => write!(f, "i32.shl"),
            Op::I32ShrS => write!(f, "i32.shr_s"),
            Op::I32ShrU => write!(f, "i32.shr_u"),
            Op::I32Rotl => write!(f, "i32.rotl"),
            Op::I32Rotr => write!(f, "i32.rotr"),
            Op::I32Extend8S => write!(f, "i32.extend8_s"),
            Op::I32Extend16S => write!(f, "i32.extend16_s"),
            Op::I64Add => write!(f, "i64.add"),
            Op::I64Sub => write!(f, "i64.sub"),
            Op::I64Mul => write!(f, "i64.mul"),
            Op::I64DivS => write!(f, "i64.div_s"),
            Op::I64DivU => write!(f, "i64.div_u"),
            Op::I64RemS => write!(f, "i64.rem_s"),
            Op::I64RemU => write!(f, "i64.rem_u"),
            Op::I64Clz => write!(f, "i64.clz"),
            Op::I64Ctz => write!(f, "i64.ctz"),
            Op::I64Popcnt => write!(f, "i64.popcnt"),
            Op::I64And => write!(f, "i64.and"),
            Op::I64Or => write!(f, "i64.or"),
            Op::I64Xor => write!(f, "i64.xor"),
            Op::I64Shl => write!(f, "i64.shl"),
            Op::I64ShrS => write!(f, "i64.shr_s"),
            Op::I64ShrU => write!(f, "i64.shr_u"),
            Op::I64Rotl => write!(f, "i64.rotl"),
            Op::I64Rotr => write!(f, "i64.rotr"),
            Op::I64Eqz => write!(f, "i64.eqz"),
            Op::I64Eq => write!(f, "i64.eq"),
            Op::I64Ne => write!(f, "i64.ne"),
            Op::I64LtS => write!(f, "i64.lt_s"),
            Op::I64LtU => write!(f, "i64.lt_u"),
            Op::I64GtS => write!(f, "i64.gt_s"),
            Op::I64GtU => write!(f, "i64.gt_u"),
            Op::I64LeS => write!(f, "i64.le_s"),
            Op::I64LeU => write!(f, "i64.le_u"),
            Op::I64GeS => write!(f, "i64.ge_s"),
            Op::I64GeU => write!(f, "i64.ge_u"),
            Op::I64Extend8S => write!(f, "i64.extend8_s"),
            Op::I64Extend16S => write!(f, "i64.extend16_s"),
            Op::I64Extend32S => write!(f, "i64.extend32_s"),
            Op::I64ExtendI32S => write!(f, "i64.extend_i32_s"),
            Op::I64ExtendI32U => write!(f, "i64.extend_i32_u"),
            Op::I32WrapI64 => write!(f, "i32.wrap_i64"),
            Op::F32Add => write!(f, "f32.add"),
            Op::F32Sub => write!(f, "f32.sub"),
            Op::F32Mul => write!(f, "f32.mul"),
            Op::F32Div => write!(f, "f32.div"),
            Op::F32Min => write!(f, "f32.min"),
            Op::F32Max => write!(f, "f32.max"),
            Op::F32Copysign => write!(f, "f32.copysign"),
            Op::F64Add => write!(f, "f64.add"),
            Op::F64Sub => write!(f, "f64.sub"),
            Op::F64Mul => write!(f, "f64.mul"),
            Op::F64Div => write!(f, "f64.div"),
            Op::F64Min => write!(f, "f64.min"),
            Op::F64Max => write!(f, "f64.max"),
            Op::F64Copysign => write!(f, "f64.copysign"),
            Op::F32Abs => write!(f, "f32.abs"),
            Op::F32Neg => write!(f, "f32.neg"),
            Op::F32Sqrt => write!(f, "f32.sqrt"),
            Op::F32Ceil => write!(f, "f32.ceil"),
            Op::F32Floor => write!(f, "f32.floor"),
            Op::F32Trunc => write!(f, "f32.trunc"),
            Op::F32Nearest => write!(f, "f32.nearest"),
            Op::F64Abs => write!(f, "f64.abs"),
            Op::F64Neg => write!(f, "f64.neg"),
            Op::F64Sqrt => write!(f, "f64.sqrt"),
            Op::F64Ceil => write!(f, "f64.ceil"),
            Op::F64Floor => write!(f, "f64.floor"),
            Op::F64Trunc => write!(f, "f64.trunc"),
            Op::F64Nearest => write!(f, "f64.nearest"),
            Op::F32Eq => write!(f, "f32.eq"),
            Op::F32Ne => write!(f, "f32.ne"),
            Op::F32Lt => write!(f, "f32.lt"),
            Op::F32Gt => write!(f, "f32.gt"),
            Op::F32Le => write!(f, "f32.le"),
            Op::F32Ge => write!(f, "f32.ge"),
            Op::F64Eq => write!(f, "f64.eq"),
            Op::F64Ne => write!(f, "f64.ne"),
            Op::F64Lt => write!(f, "f64.lt"),
            Op::F64Gt => write!(f, "f64.gt"),
            Op::F64Le => write!(f, "f64.le"),
            Op::F64Ge => write!(f, "f64.ge"),
            Op::F32ConvertI32S => write!(f, "f32.convert_i32_s"),
            Op::F32ConvertI32U => write!(f, "f32.convert_i32_u"),
            Op::F32ConvertI64S => write!(f, "f32.convert_i64_s"),
            Op::F32ConvertI64U => write!(f, "f32.convert_i64_u"),
            Op::F64ConvertI32S => write!(f, "f64.convert_i32_s"),
            Op::F64ConvertI32U => write!(f, "f64.convert_i32_u"),
            Op::F64ConvertI64S => write!(f, "f64.convert_i64_s"),
            Op::F64ConvertI64U => write!(f, "f64.convert_i64_u"),
            Op::F32DemoteF64 => write!(f, "f32.demote_f64"),
            Op::F64PromoteF32 => write!(f, "f64.promote_f32"),
            Op::I32TruncF32S => write!(f, "i32.trunc_f32_s"),
            Op::I32TruncF32U => write!(f, "i32.trunc_f32_u"),
            Op::I32TruncF64S => write!(f, "i32.trunc_f64_s"),
            Op::I32TruncF64U => write!(f, "i32.trunc_f64_u"),
            Op::I64TruncF32S => write!(f, "i64.trunc_f32_s"),
            Op::I64TruncF32U => write!(f, "i64.trunc_f32_u"),
            Op::I64TruncF64S => write!(f, "i64.trunc_f64_s"),
            Op::I64TruncF64U => write!(f, "i64.trunc_f64_u"),
            Op::I32TruncSatF32S => write!(f, "i32.trunc_sat_f32_s"),
            Op::I32TruncSatF32U => write!(f, "i32.trunc_sat_f32_u"),
            Op::I32TruncSatF64S => write!(f, "i32.trunc_sat_f64_s"),
            Op::I32TruncSatF64U => write!(f, "i32.trunc_sat_f64_u"),
            Op::I64TruncSatF32S => write!(f, "i64.trunc_sat_f32_s"),
            Op::I64TruncSatF32U => write!(f, "i64.trunc_sat_f32_u"),
            Op::I64TruncSatF64S => write!(f, "i64.trunc_sat_f64_s"),
            Op::I64TruncSatF64U => write!(f, "i64.trunc_sat_f64_u"),
            Op::I32ReinterpretF32 => write!(f, "i32.reinterpret_f32"),
            Op::I64ReinterpretF64 => write!(f, "i64.reinterpret_f64"),
            Op::F32ReinterpretI32 => write!(f, "f32.reinterpret_i32"),
            Op::F64ReinterpretI64 => write!(f, "f64.reinterpret_i64"),
            Op::V128Not => write!(f, "v128.not"),
            Op::V128And => write!(f, "v128.and"),
            Op::V128AndNot => write!(f, "v128.andnot"),
            Op::V128Or => write!(f, "v128.or"),
            Op::V128Xor => write!(f, "v128.xor"),
            Op::V128Bitselect => write!(f, "v128.bitselect"),
            Op::V128AnyTrue => write!(f, "v128.any_true"),
            Op::I8x16AllTrue => write!(f, "i8x16.all_true"),
            Op::I8x16Bitmask => write!(f, "i8x16.bitmask"),
            Op::I16x8AllTrue => write!(f, "i16x8.all_true"),
            Op::I16x8Bitmask => write!(f, "i16x8.bitmask"),
            Op::I32x4AllTrue => write!(f, "i32x4.all_true"),
            Op::I32x4Bitmask => write!(f, "i32x4.bitmask"),
            Op::I64x2AllTrue => write!(f, "i64x2.all_true"),
            Op::I64x2Bitmask => write!(f, "i64x2.bitmask"),
            Op::I8x16Splat => write!(f, "i8x16.splat"),
            Op::I16x8Splat => write!(f, "i16x8.splat"),
            Op::I32x4Splat => write!(f, "i32x4.splat"),
            Op::I64x2Splat => write!(f, "i64x2.splat"),
            Op::F32x4Splat => write!(f, "f32x4.splat"),
            Op::F64x2Splat => write!(f, "f64x2.splat"),
            Op::I8x16Shuffle(lanes) => {
                write!(f, "i8x16.shuffle")?;
                for i in 0..4 {
                    let lane = u32::from_le_bytes([lanes[i * 4], lanes[i * 4 + 1], lanes[i * 4 + 2], lanes[i * 4 + 3]]);
                    write!(f, " 0x{lane:08x}")?;
                }
                Ok(())
            }
            Op::I8x16Swizzle => write!(f, "i8x16.swizzle"),
            Op::I8x16ExtractLaneS(lane) => write!(f, "i8x16.extract_lane_s {lane}"),
            Op::I8x16ExtractLaneU(lane) => write!(f, "i8x16.extract_lane_u {lane}"),
            Op::I16x8ExtractLaneS(lane) => write!(f, "i16x8.extract_lane_s {lane}"),
            Op::I16x8ExtractLaneU(lane) => write!(f, "i16x8.extract_lane_u {lane}"),
            Op::I32x4ExtractLane(lane) => write!(f, "i32x4.extract_lane {lane}"),
            Op::I64x2ExtractLane(lane) => write!(f, "i64x2.extract_lane {lane}"),
            Op::F32x4ExtractLane(lane) => write!(f, "f32x4.extract_lane {lane}"),
            Op::F64x2ExtractLane(lane) => write!(f, "f64x2.extract_lane {lane}"),
            Op::I8x16ReplaceLane(lane) => write!(f, "i8x16.replace_lane {lane}"),
            Op::I16x8ReplaceLane(lane) => write!(f, "i16x8.replace_lane {lane}"),
            Op::I32x4ReplaceLane(lane) => write!(f, "i32x4.replace_lane {lane}"),
            Op::I64x2ReplaceLane(lane) => write!(f, "i64x2.replace_lane {lane}"),
            Op::F32x4ReplaceLane(lane) => write!(f, "f32x4.replace_lane {lane}"),
            Op::F64x2ReplaceLane(lane) => write!(f, "f64x2.replace_lane {lane}"),
            Op::Select => write!(f, "select"),
            Op::RefNull(t) => write!(f, "ref.null {t}"),
            Op::RefIsNull => write!(f, "ref.is_null"),
            Op::RefFunc { func_idx } => write!(f, "ref.func {func_idx}"),
            Op::TableGet { table_idx } => write!(f, "table.get {table_idx}"),
            Op::TableSet { table_idx } => write!(f, "table.set {table_idx}"),
            Op::TableSize { table_idx } => write!(f, "table.size {table_idx}"),
            Op::TableGrow { table_idx } => write!(f, "table.grow {table_idx}"),
            Op::TableFill { table_idx } => write!(f, "table.fill {table_idx}"),
            Op::TableCopy { dst_table, src_table } => write!(f, "table.copy {dst_table} {src_table}"),
            Op::TableInit { elem_idx, table_idx } => write!(f, "table.init {elem_idx} {table_idx}"),
            Op::ElemDrop { elem_idx } => write!(f, "elem.drop {elem_idx}"),
            Op::MemoryInit { data_idx } => write!(f, "memory.init {data_idx}"),
            Op::DataDrop { data_idx } => write!(f, "data.drop {data_idx}"),
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
            Op::V128Load(m) => write!(f, "v128.load offset={}", m.offset),
            Op::V128Load8x8S(m) => write!(f, "v128.load8x8_s offset={}", m.offset),
            Op::V128Load8x8U(m) => write!(f, "v128.load8x8_u offset={}", m.offset),
            Op::V128Load16x4S(m) => write!(f, "v128.load16x4_s offset={}", m.offset),
            Op::V128Load16x4U(m) => write!(f, "v128.load16x4_u offset={}", m.offset),
            Op::V128Load32x2S(m) => write!(f, "v128.load32x2_s offset={}", m.offset),
            Op::V128Load32x2U(m) => write!(f, "v128.load32x2_u offset={}", m.offset),
            Op::V128Load8Splat(m) => write!(f, "v128.load8_splat offset={}", m.offset),
            Op::V128Load16Splat(m) => write!(f, "v128.load16_splat offset={}", m.offset),
            Op::V128Load32Splat(m) => write!(f, "v128.load32_splat offset={}", m.offset),
            Op::V128Load64Splat(m) => write!(f, "v128.load64_splat offset={}", m.offset),
            Op::V128Load32Zero(m) => write!(f, "v128.load32_zero offset={}", m.offset),
            Op::V128Load64Zero(m) => write!(f, "v128.load64_zero offset={}", m.offset),
            Op::V128Load8Lane { memarg, lane } => {
                write!(f, "v128.load8_lane offset={} {lane}", memarg.offset)
            }
            Op::V128Load16Lane { memarg, lane } => {
                write!(f, "v128.load16_lane offset={} {lane}", memarg.offset)
            }
            Op::V128Load32Lane { memarg, lane } => {
                write!(f, "v128.load32_lane offset={} {lane}", memarg.offset)
            }
            Op::V128Load64Lane { memarg, lane } => {
                write!(f, "v128.load64_lane offset={} {lane}", memarg.offset)
            }
            Op::I32Store(m) => write!(f, "i32.store offset={}", m.offset),
            Op::I32Store8(m) => write!(f, "i32.store8 offset={}", m.offset),
            Op::I32Store16(m) => write!(f, "i32.store16 offset={}", m.offset),
            Op::I64Store(m) => write!(f, "i64.store offset={}", m.offset),
            Op::I64Store8(m) => write!(f, "i64.store8 offset={}", m.offset),
            Op::I64Store16(m) => write!(f, "i64.store16 offset={}", m.offset),
            Op::I64Store32(m) => write!(f, "i64.store32 offset={}", m.offset),
            Op::F32Store(m) => write!(f, "f32.store offset={}", m.offset),
            Op::F64Store(m) => write!(f, "f64.store offset={}", m.offset),
            Op::V128Store(m) => write!(f, "v128.store offset={}", m.offset),
            Op::V128Store8Lane { memarg, lane } => {
                write!(f, "v128.store8_lane offset={} {lane}", memarg.offset)
            }
            Op::V128Store16Lane { memarg, lane } => {
                write!(f, "v128.store16_lane offset={} {lane}", memarg.offset)
            }
            Op::V128Store32Lane { memarg, lane } => {
                write!(f, "v128.store32_lane offset={} {lane}", memarg.offset)
            }
            Op::V128Store64Lane { memarg, lane } => {
                write!(f, "v128.store64_lane offset={} {lane}", memarg.offset)
            }
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
            Op::Call { func_idx } => write!(f, "call {func_idx}"),
            Op::CallIndirect { type_idx, table_idx } => {
                write!(f, "call_indirect (type {type_idx}) (table {table_idx})")
            }
            Op::Return => write!(f, "return"),
            Op::Nop => write!(f, "nop"),
            Op::End => write!(f, "end"),
            Op::Label { end_target } => write!(f, "label (end -> {end_target})"),
            Op::Unreachable => write!(f, "unreachable"),
            Op::Unsupported(name) => write!(f, "unsupported <{name}>"),
            Op::Drop => write!(f, "drop"),
        }
    }
}
