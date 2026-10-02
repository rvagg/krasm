//! WebAssembly instruction representation and decoding
//!
//! This module defines the instruction types, decoding logic, and validation
//! for WebAssembly bytecode instructions.

pub mod decode;
pub mod encode;
pub mod streaming_decode;

pub use decode::{DecodeError, InstructionIterator, ParseType};
pub use streaming_decode::{VecCollector, decode_function, decode_with_processor};

use super::module::ValueType;
use fhex::ToHex;
use std::fmt;

/// Decode a constant expression (used in data segments, globals, etc)
pub fn decode_constant_expression(
    reader: &mut super::reader::Reader,
    imports: &super::module::ImportSection,
    return_type: ValueType,
) -> Result<Vec<Instruction>, DecodeError> {
    let mut validator = super::validate::ConstantExpressionValidator::new(imports, return_type);
    let mut collector = VecCollector::new();
    decode_with_processor(&mut validator, &mut collector, ParseType::ReadTillEnd, reader)?;
    Ok(collector.into_instructions())
}

/// Decode a constant expression that may contain ref.func instructions
pub fn decode_constant_expression_with_ref_func(
    reader: &mut super::reader::Reader,
    imports: &super::module::ImportSection,
    return_type: ValueType,
    total_functions: u32,
) -> Result<Vec<Instruction>, DecodeError> {
    let mut validator =
        super::validate::ConstantExpressionValidator::new(imports, return_type).with_function_count(total_functions);
    let mut collector = VecCollector::new();
    decode_with_processor(&mut validator, &mut collector, ParseType::ReadTillEnd, reader)?;
    Ok(collector.into_instructions())
}

/// Memory argument for memory access instructions
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "instruction-profile", derive(serde::Serialize))]
pub struct MemArg {
    /// Memory alignment (as power of 2)
    pub align: u32,
    /// Memory offset
    pub offset: u32,
}

/// Block type for structured control instructions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockType {
    /// Empty block type (no parameters or results)
    Empty,
    /// Single value type result
    Value(ValueType),
    /// Function type index for multi-value blocks
    FuncType(u32),
}

/// Position information for an instruction in the binary
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ByteRange {
    pub offset: usize,
    pub length: usize,
}

/// Main instruction representation
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    pub kind: InstructionKind,
    pub position: ByteRange,
    /// Original bytes for instructions with non-canonical encodings (e.g., 0xFC prefix)
    /// This preserves the exact byte representation for byte-perfect output
    pub original_bytes: Vec<u8>,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

/// Algebraic data type representing all WebAssembly instructions
#[derive(Debug, Clone, PartialEq)]
pub enum InstructionKind {
    // Control instructions
    Unreachable,
    Nop,
    Block { block_type: BlockType },
    Loop { block_type: BlockType },
    If { block_type: BlockType },
    Else,
    End,
    Br { label_idx: u32 },
    BrIf { label_idx: u32 },
    BrTable { labels: Vec<u32>, default: u32 },
    Return,
    Call { func_idx: u32 },
    CallIndirect { type_idx: u32, table_idx: u32 },

    // Reference instructions
    RefNull { ref_type: ValueType },
    RefIsNull,
    RefFunc { func_idx: u32 },

    // Parametric instructions
    Drop,
    Select,
    SelectTyped { val_types: Vec<ValueType> },

    // Variable instructions
    LocalGet { local_idx: u32 },
    LocalSet { local_idx: u32 },
    LocalTee { local_idx: u32 },
    GlobalGet { global_idx: u32 },
    GlobalSet { global_idx: u32 },

    // Table instructions
    TableGet { table_idx: u32 },
    TableSet { table_idx: u32 },
    TableInit { elem_idx: u32, table_idx: u32 },
    ElemDrop { elem_idx: u32 },
    TableCopy { dst_table: u32, src_table: u32 },
    TableGrow { table_idx: u32 },
    TableSize { table_idx: u32 },
    TableFill { table_idx: u32 },

    // Memory instructions
    I32Load { memarg: MemArg },
    I64Load { memarg: MemArg },
    F32Load { memarg: MemArg },
    F64Load { memarg: MemArg },
    I32Load8S { memarg: MemArg },
    I32Load8U { memarg: MemArg },
    I32Load16S { memarg: MemArg },
    I32Load16U { memarg: MemArg },
    I64Load8S { memarg: MemArg },
    I64Load8U { memarg: MemArg },
    I64Load16S { memarg: MemArg },
    I64Load16U { memarg: MemArg },
    I64Load32S { memarg: MemArg },
    I64Load32U { memarg: MemArg },
    I32Store { memarg: MemArg },
    I64Store { memarg: MemArg },
    F32Store { memarg: MemArg },
    F64Store { memarg: MemArg },
    I32Store8 { memarg: MemArg },
    I32Store16 { memarg: MemArg },
    I64Store8 { memarg: MemArg },
    I64Store16 { memarg: MemArg },
    I64Store32 { memarg: MemArg },
    MemorySize,
    MemoryGrow,
    MemoryInit { data_idx: u32 },
    DataDrop { data_idx: u32 },
    MemoryCopy,
    MemoryFill,

    // Numeric instructions - Constants
    I32Const { value: i32 },
    I64Const { value: i64 },
    F32Const { value: f32 },
    F64Const { value: f64 },

    // Numeric instructions - i32 operations
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
    I32Clz,
    I32Ctz,
    I32Popcnt,
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Rotl,
    I32Rotr,

    // Numeric instructions - i64 operations
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
    I64Clz,
    I64Ctz,
    I64Popcnt,
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rotl,
    I64Rotr,

    // Numeric instructions - f32 operations
    F32Eq,
    F32Ne,
    F32Lt,
    F32Gt,
    F32Le,
    F32Ge,
    F32Abs,
    F32Neg,
    F32Ceil,
    F32Floor,
    F32Trunc,
    F32Nearest,
    F32Sqrt,
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Min,
    F32Max,
    F32Copysign,

    // Numeric instructions - f64 operations
    F64Eq,
    F64Ne,
    F64Lt,
    F64Gt,
    F64Le,
    F64Ge,
    F64Abs,
    F64Neg,
    F64Ceil,
    F64Floor,
    F64Trunc,
    F64Nearest,
    F64Sqrt,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Min,
    F64Max,
    F64Copysign,

    // Numeric instructions - Conversions
    I32WrapI64,
    I32TruncF32S,
    I32TruncF32U,
    I32TruncF64S,
    I32TruncF64U,
    I64ExtendI32S,
    I64ExtendI32U,
    I64TruncF32S,
    I64TruncF32U,
    I64TruncF64S,
    I64TruncF64U,
    F32ConvertI32S,
    F32ConvertI32U,
    F32ConvertI64S,
    F32ConvertI64U,
    F32DemoteF64,
    F64ConvertI32S,
    F64ConvertI32U,
    F64ConvertI64S,
    F64ConvertI64U,
    F64PromoteF32,
    I32ReinterpretF32,
    I64ReinterpretF64,
    F32ReinterpretI32,
    F64ReinterpretI64,

    // Sign extension operations
    I32Extend8S,
    I32Extend16S,
    I64Extend8S,
    I64Extend16S,
    I64Extend32S,

    // Saturating truncation operations (0xFC prefix)
    I32TruncSatF32S,
    I32TruncSatF32U,
    I32TruncSatF64S,
    I32TruncSatF64U,
    I64TruncSatF32S,
    I64TruncSatF32U,
    I64TruncSatF64S,
    I64TruncSatF64U,

    // SIMD instructions (0xFD prefix) — delegated to SimdOp
    Simd(SimdOp),
}

/// SIMD instruction opcodes (0xFD prefix, ~236 sub-opcodes)
#[derive(Debug, Clone, PartialEq)]
pub enum SimdOp {
    // Memory operations
    V128Load { memarg: MemArg },        // 0x00
    V128Load8x8S { memarg: MemArg },    // 0x01
    V128Load8x8U { memarg: MemArg },    // 0x02
    V128Load16x4S { memarg: MemArg },   // 0x03
    V128Load16x4U { memarg: MemArg },   // 0x04
    V128Load32x2S { memarg: MemArg },   // 0x05
    V128Load32x2U { memarg: MemArg },   // 0x06
    V128Load8Splat { memarg: MemArg },  // 0x07
    V128Load16Splat { memarg: MemArg }, // 0x08
    V128Load32Splat { memarg: MemArg }, // 0x09
    V128Load64Splat { memarg: MemArg }, // 0x0A
    V128Store { memarg: MemArg },       // 0x0B

    // Constant and shuffle
    V128Const { value: [u8; 16] },    // 0x0C
    I8x16Shuffle { lanes: [u8; 16] }, // 0x0D

    // Splat
    I8x16Swizzle, // 0x0E
    I8x16Splat,   // 0x0F
    I16x8Splat,   // 0x10
    I32x4Splat,   // 0x11
    I64x2Splat,   // 0x12
    F32x4Splat,   // 0x13
    F64x2Splat,   // 0x14

    // Lane extraction/replacement
    I8x16ExtractLaneS { lane: u8 }, // 0x15
    I8x16ExtractLaneU { lane: u8 }, // 0x16
    I8x16ReplaceLane { lane: u8 },  // 0x17
    I16x8ExtractLaneS { lane: u8 }, // 0x18
    I16x8ExtractLaneU { lane: u8 }, // 0x19
    I16x8ReplaceLane { lane: u8 },  // 0x1A
    I32x4ExtractLane { lane: u8 },  // 0x1B
    I32x4ReplaceLane { lane: u8 },  // 0x1C
    I64x2ExtractLane { lane: u8 },  // 0x1D
    I64x2ReplaceLane { lane: u8 },  // 0x1E
    F32x4ExtractLane { lane: u8 },  // 0x1F
    F32x4ReplaceLane { lane: u8 },  // 0x20
    F64x2ExtractLane { lane: u8 },  // 0x21
    F64x2ReplaceLane { lane: u8 },  // 0x22

    // i8x16 comparisons
    I8x16Eq,  // 0x23
    I8x16Ne,  // 0x24
    I8x16LtS, // 0x25
    I8x16LtU, // 0x26
    I8x16GtS, // 0x27
    I8x16GtU, // 0x28
    I8x16LeS, // 0x29
    I8x16LeU, // 0x2A
    I8x16GeS, // 0x2B
    I8x16GeU, // 0x2C

    // i16x8 comparisons
    I16x8Eq,  // 0x2D
    I16x8Ne,  // 0x2E
    I16x8LtS, // 0x2F
    I16x8LtU, // 0x30
    I16x8GtS, // 0x31
    I16x8GtU, // 0x32
    I16x8LeS, // 0x33
    I16x8LeU, // 0x34
    I16x8GeS, // 0x35
    I16x8GeU, // 0x36

    // i32x4 comparisons
    I32x4Eq,  // 0x37
    I32x4Ne,  // 0x38
    I32x4LtS, // 0x39
    I32x4LtU, // 0x3A
    I32x4GtS, // 0x3B
    I32x4GtU, // 0x3C
    I32x4LeS, // 0x3D
    I32x4LeU, // 0x3E
    I32x4GeS, // 0x3F
    I32x4GeU, // 0x40

    // f32x4 comparisons
    F32x4Eq, // 0x41
    F32x4Ne, // 0x42
    F32x4Lt, // 0x43
    F32x4Gt, // 0x44
    F32x4Le, // 0x45
    F32x4Ge, // 0x46

    // f64x2 comparisons
    F64x2Eq, // 0x47
    F64x2Ne, // 0x48
    F64x2Lt, // 0x49
    F64x2Gt, // 0x4A
    F64x2Le, // 0x4B
    F64x2Ge, // 0x4C

    // v128 bitwise
    V128Not,       // 0x4D
    V128And,       // 0x4E
    V128AndNot,    // 0x4F
    V128Or,        // 0x50
    V128Xor,       // 0x51
    V128Bitselect, // 0x52
    V128AnyTrue,   // 0x53

    // Load/store lane
    V128Load8Lane { memarg: MemArg, lane: u8 },   // 0x54
    V128Load16Lane { memarg: MemArg, lane: u8 },  // 0x55
    V128Load32Lane { memarg: MemArg, lane: u8 },  // 0x56
    V128Load64Lane { memarg: MemArg, lane: u8 },  // 0x57
    V128Store8Lane { memarg: MemArg, lane: u8 },  // 0x58
    V128Store16Lane { memarg: MemArg, lane: u8 }, // 0x59
    V128Store32Lane { memarg: MemArg, lane: u8 }, // 0x5A
    V128Store64Lane { memarg: MemArg, lane: u8 }, // 0x5B

    // Load zero
    V128Load32Zero { memarg: MemArg }, // 0x5C
    V128Load64Zero { memarg: MemArg }, // 0x5D

    // Conversions (placed between load-zero and i8x16 ops in opcode order)
    F32x4DemoteF64x2Zero, // 0x5E
    F64x2PromoteLowF32x4, // 0x5F

    // i8x16 operations (0x60-0x7B)
    // f32x4 and f64x2 rounding ops are interleaved here per the spec —
    // they were added later and assigned to gaps in this opcode range.
    I8x16Abs,          // 0x60
    I8x16Neg,          // 0x61
    I8x16Popcnt,       // 0x62
    I8x16AllTrue,      // 0x63
    I8x16Bitmask,      // 0x64
    I8x16NarrowI16x8S, // 0x65
    I8x16NarrowI16x8U, // 0x66
    F32x4Ceil,         // 0x67
    F32x4Floor,        // 0x68
    F32x4Trunc,        // 0x69
    F32x4Nearest,      // 0x6A
    I8x16Shl,          // 0x6B
    I8x16ShrS,         // 0x6C
    I8x16ShrU,         // 0x6D
    I8x16Add,          // 0x6E
    I8x16AddSatS,      // 0x6F
    I8x16AddSatU,      // 0x70
    I8x16Sub,          // 0x71
    I8x16SubSatS,      // 0x72
    I8x16SubSatU,      // 0x73
    F64x2Ceil,         // 0x74
    F64x2Floor,        // 0x75
    I8x16MinS,         // 0x76
    I8x16MinU,         // 0x77
    I8x16MaxS,         // 0x78
    I8x16MaxU,         // 0x79
    F64x2Trunc,        // 0x7A
    I8x16AvgrU,        // 0x7B

    // Pairwise extension
    I16x8ExtAddPairwiseI8x16S, // 0x7C
    I16x8ExtAddPairwiseI8x16U, // 0x7D
    I32x4ExtAddPairwiseI16x8S, // 0x7E
    I32x4ExtAddPairwiseI16x8U, // 0x7F

    // i16x8 operations (0x80-0x9F, with f64x2.nearest at 0x94)
    I16x8Abs,              // 0x80
    I16x8Neg,              // 0x81
    I16x8Q15MulrSatS,      // 0x82
    I16x8AllTrue,          // 0x83
    I16x8Bitmask,          // 0x84
    I16x8NarrowI32x4S,     // 0x85
    I16x8NarrowI32x4U,     // 0x86
    I16x8ExtendLowI8x16S,  // 0x87
    I16x8ExtendHighI8x16S, // 0x88
    I16x8ExtendLowI8x16U,  // 0x89
    I16x8ExtendHighI8x16U, // 0x8A
    I16x8Shl,              // 0x8B
    I16x8ShrS,             // 0x8C
    I16x8ShrU,             // 0x8D
    I16x8Add,              // 0x8E
    I16x8AddSatS,          // 0x8F
    I16x8AddSatU,          // 0x90
    I16x8Sub,              // 0x91
    I16x8SubSatS,          // 0x92
    I16x8SubSatU,          // 0x93
    F64x2Nearest,          // 0x94
    I16x8Mul,              // 0x95
    I16x8MinS,             // 0x96
    I16x8MinU,             // 0x97
    I16x8MaxS,             // 0x98
    I16x8MaxU,             // 0x99
    // 0x9A reserved
    I16x8AvgrU,            // 0x9B
    I16x8ExtMulLowI8x16S,  // 0x9C
    I16x8ExtMulHighI8x16S, // 0x9D
    I16x8ExtMulLowI8x16U,  // 0x9E
    I16x8ExtMulHighI8x16U, // 0x9F

    // i32x4 operations
    I32x4Abs, // 0xA0
    I32x4Neg, // 0xA1
    // 0xA2 reserved
    I32x4AllTrue, // 0xA3
    I32x4Bitmask, // 0xA4
    // 0xA5-0xA6 reserved
    I32x4ExtendLowI16x8S,  // 0xA7
    I32x4ExtendHighI16x8S, // 0xA8
    I32x4ExtendLowI16x8U,  // 0xA9
    I32x4ExtendHighI16x8U, // 0xAA
    I32x4Shl,              // 0xAB
    I32x4ShrS,             // 0xAC
    I32x4ShrU,             // 0xAD
    I32x4Add,              // 0xAE
    // 0xAF-0xB0 reserved
    I32x4Sub, // 0xB1
    // 0xB2-0xB4 reserved
    I32x4Mul,       // 0xB5
    I32x4MinS,      // 0xB6
    I32x4MinU,      // 0xB7
    I32x4MaxS,      // 0xB8
    I32x4MaxU,      // 0xB9
    I32x4DotI16x8S, // 0xBA
    // 0xBB reserved
    I32x4ExtMulLowI16x8S,  // 0xBC
    I32x4ExtMulHighI16x8S, // 0xBD
    I32x4ExtMulLowI16x8U,  // 0xBE
    I32x4ExtMulHighI16x8U, // 0xBF

    // i64x2 operations
    I64x2Abs, // 0xC0
    I64x2Neg, // 0xC1
    // 0xC2 reserved
    I64x2AllTrue, // 0xC3
    I64x2Bitmask, // 0xC4
    // 0xC5-0xC6 reserved
    I64x2ExtendLowI32x4S,  // 0xC7
    I64x2ExtendHighI32x4S, // 0xC8
    I64x2ExtendLowI32x4U,  // 0xC9
    I64x2ExtendHighI32x4U, // 0xCA
    I64x2Shl,              // 0xCB
    I64x2ShrS,             // 0xCC
    I64x2ShrU,             // 0xCD
    I64x2Add,              // 0xCE
    // 0xCF-0xD0 reserved
    I64x2Sub, // 0xD1
    // 0xD2-0xD4 reserved
    I64x2Mul,              // 0xD5
    I64x2Eq,               // 0xD6
    I64x2Ne,               // 0xD7
    I64x2LtS,              // 0xD8
    I64x2GtS,              // 0xD9
    I64x2LeS,              // 0xDA
    I64x2GeS,              // 0xDB
    I64x2ExtMulLowI32x4S,  // 0xDC
    I64x2ExtMulHighI32x4S, // 0xDD
    I64x2ExtMulLowI32x4U,  // 0xDE
    I64x2ExtMulHighI32x4U, // 0xDF

    // f32x4 operations
    F32x4Abs, // 0xE0
    F32x4Neg, // 0xE1
    // 0xE2 reserved
    F32x4Sqrt, // 0xE3
    F32x4Add,  // 0xE4
    F32x4Sub,  // 0xE5
    F32x4Mul,  // 0xE6
    F32x4Div,  // 0xE7
    F32x4Min,  // 0xE8
    F32x4Max,  // 0xE9
    F32x4PMin, // 0xEA
    F32x4PMax, // 0xEB

    // f64x2 operations
    F64x2Abs, // 0xEC
    F64x2Neg, // 0xED
    // 0xEE reserved
    F64x2Sqrt, // 0xEF
    F64x2Add,  // 0xF0
    F64x2Sub,  // 0xF1
    F64x2Mul,  // 0xF2
    F64x2Div,  // 0xF3
    F64x2Min,  // 0xF4
    F64x2Max,  // 0xF5
    F64x2PMin, // 0xF6
    F64x2PMax, // 0xF7

    // Truncation/conversion
    I32x4TruncSatF32x4S,     // 0xF8
    I32x4TruncSatF32x4U,     // 0xF9
    F32x4ConvertI32x4S,      // 0xFA
    F32x4ConvertI32x4U,      // 0xFB
    I32x4TruncSatF64x2SZero, // 0xFC
    I32x4TruncSatF64x2UZero, // 0xFD
    F64x2ConvertLowI32x4S,   // 0xFE
    F64x2ConvertLowI32x4U,   // 0xFF
}

impl InstructionKind {
    /// Format the instruction with module context
    pub fn format_with_context(&self, module: &crate::parser::module::Module) -> String {
        use InstructionKind::*;

        let base = self.to_string();

        // Add function/table names for call instructions
        match self {
            Call { func_idx } => {
                if let Some(name) = module.get_function_name(*func_idx) {
                    format!("{base} <{name}>")
                } else {
                    base
                }
            }
            CallIndirect { type_idx, table_idx } => {
                let table_part = if let Some(name) = module.get_table_name(*table_idx) {
                    format!(" <{name}>")
                } else {
                    String::new()
                };
                format!("call_indirect {table_idx}{table_part} (type {type_idx})")
            }
            GlobalGet { global_idx } => {
                if let Some(name) = module.get_global_name(*global_idx) {
                    format!("{base} <{name}>")
                } else {
                    base
                }
            }
            GlobalSet { global_idx } => {
                if let Some(name) = module.get_global_name(*global_idx) {
                    format!("{base} <{name}>")
                } else {
                    base
                }
            }
            _ => base,
        }
    }

    /// Get the mnemonic for this instruction
    pub fn mnemonic(&self) -> &'static str {
        use InstructionKind::*;
        match self {
            // Control
            Unreachable => "unreachable",
            Nop => "nop",
            Block { .. } => "block",
            Loop { .. } => "loop",
            If { .. } => "if",
            Else => "else",
            End => "end",
            Br { .. } => "br",
            BrIf { .. } => "br_if",
            BrTable { .. } => "br_table",
            Return => "return",
            Call { .. } => "call",
            CallIndirect { .. } => "call_indirect",

            // Reference
            RefNull { .. } => "ref.null",
            RefIsNull => "ref.is_null",
            RefFunc { .. } => "ref.func",

            // Parametric
            Drop => "drop",
            Select => "select",
            SelectTyped { .. } => "select",

            // Variable
            LocalGet { .. } => "local.get",
            LocalSet { .. } => "local.set",
            LocalTee { .. } => "local.tee",
            GlobalGet { .. } => "global.get",
            GlobalSet { .. } => "global.set",

            // Table
            TableGet { .. } => "table.get",
            TableSet { .. } => "table.set",
            TableInit { .. } => "table.init",
            ElemDrop { .. } => "elem.drop",
            TableCopy { .. } => "table.copy",
            TableGrow { .. } => "table.grow",
            TableSize { .. } => "table.size",
            TableFill { .. } => "table.fill",

            // Memory
            I32Load { .. } => "i32.load",
            I64Load { .. } => "i64.load",
            F32Load { .. } => "f32.load",
            F64Load { .. } => "f64.load",
            I32Load8S { .. } => "i32.load8_s",
            I32Load8U { .. } => "i32.load8_u",
            I32Load16S { .. } => "i32.load16_s",
            I32Load16U { .. } => "i32.load16_u",
            I64Load8S { .. } => "i64.load8_s",
            I64Load8U { .. } => "i64.load8_u",
            I64Load16S { .. } => "i64.load16_s",
            I64Load16U { .. } => "i64.load16_u",
            I64Load32S { .. } => "i64.load32_s",
            I64Load32U { .. } => "i64.load32_u",
            I32Store { .. } => "i32.store",
            I64Store { .. } => "i64.store",
            F32Store { .. } => "f32.store",
            F64Store { .. } => "f64.store",
            I32Store8 { .. } => "i32.store8",
            I32Store16 { .. } => "i32.store16",
            I64Store8 { .. } => "i64.store8",
            I64Store16 { .. } => "i64.store16",
            I64Store32 { .. } => "i64.store32",
            MemorySize => "memory.size",
            MemoryGrow => "memory.grow",
            MemoryInit { .. } => "memory.init",
            DataDrop { .. } => "data.drop",
            MemoryCopy => "memory.copy",
            MemoryFill => "memory.fill",

            // Numeric
            I32Const { .. } => "i32.const",
            I64Const { .. } => "i64.const",
            F32Const { .. } => "f32.const",
            F64Const { .. } => "f64.const",
            I32Eqz => "i32.eqz",
            I32Eq => "i32.eq",
            I32Ne => "i32.ne",
            I32LtS => "i32.lt_s",
            I32LtU => "i32.lt_u",
            I32GtS => "i32.gt_s",
            I32GtU => "i32.gt_u",
            I32LeS => "i32.le_s",
            I32LeU => "i32.le_u",
            I32GeS => "i32.ge_s",
            I32GeU => "i32.ge_u",
            I64Eqz => "i64.eqz",
            I64Eq => "i64.eq",
            I64Ne => "i64.ne",
            I64LtS => "i64.lt_s",
            I64LtU => "i64.lt_u",
            I64GtS => "i64.gt_s",
            I64GtU => "i64.gt_u",
            I64LeS => "i64.le_s",
            I64LeU => "i64.le_u",
            I64GeS => "i64.ge_s",
            I64GeU => "i64.ge_u",
            F32Eq => "f32.eq",
            F32Ne => "f32.ne",
            F32Lt => "f32.lt",
            F32Gt => "f32.gt",
            F32Le => "f32.le",
            F32Ge => "f32.ge",
            F64Eq => "f64.eq",
            F64Ne => "f64.ne",
            F64Lt => "f64.lt",
            F64Gt => "f64.gt",
            F64Le => "f64.le",
            F64Ge => "f64.ge",
            I32Clz => "i32.clz",
            I32Ctz => "i32.ctz",
            I32Popcnt => "i32.popcnt",
            I32Add => "i32.add",
            I32Sub => "i32.sub",
            I32Mul => "i32.mul",
            I32DivS => "i32.div_s",
            I32DivU => "i32.div_u",
            I32RemS => "i32.rem_s",
            I32RemU => "i32.rem_u",
            I32And => "i32.and",
            I32Or => "i32.or",
            I32Xor => "i32.xor",
            I32Shl => "i32.shl",
            I32ShrS => "i32.shr_s",
            I32ShrU => "i32.shr_u",
            I32Rotl => "i32.rotl",
            I32Rotr => "i32.rotr",
            I64Clz => "i64.clz",
            I64Ctz => "i64.ctz",
            I64Popcnt => "i64.popcnt",
            I64Add => "i64.add",
            I64Sub => "i64.sub",
            I64Mul => "i64.mul",
            I64DivS => "i64.div_s",
            I64DivU => "i64.div_u",
            I64RemS => "i64.rem_s",
            I64RemU => "i64.rem_u",
            I64And => "i64.and",
            I64Or => "i64.or",
            I64Xor => "i64.xor",
            I64Shl => "i64.shl",
            I64ShrS => "i64.shr_s",
            I64ShrU => "i64.shr_u",
            I64Rotl => "i64.rotl",
            I64Rotr => "i64.rotr",
            F32Abs => "f32.abs",
            F32Neg => "f32.neg",
            F32Ceil => "f32.ceil",
            F32Floor => "f32.floor",
            F32Trunc => "f32.trunc",
            F32Nearest => "f32.nearest",
            F32Sqrt => "f32.sqrt",
            F32Add => "f32.add",
            F32Sub => "f32.sub",
            F32Mul => "f32.mul",
            F32Div => "f32.div",
            F32Min => "f32.min",
            F32Max => "f32.max",
            F32Copysign => "f32.copysign",
            F64Abs => "f64.abs",
            F64Neg => "f64.neg",
            F64Ceil => "f64.ceil",
            F64Floor => "f64.floor",
            F64Trunc => "f64.trunc",
            F64Nearest => "f64.nearest",
            F64Sqrt => "f64.sqrt",
            F64Add => "f64.add",
            F64Sub => "f64.sub",
            F64Mul => "f64.mul",
            F64Div => "f64.div",
            F64Min => "f64.min",
            F64Max => "f64.max",
            F64Copysign => "f64.copysign",
            I32WrapI64 => "i32.wrap_i64",
            I32TruncF32S => "i32.trunc_f32_s",
            I32TruncF32U => "i32.trunc_f32_u",
            I32TruncF64S => "i32.trunc_f64_s",
            I32TruncF64U => "i32.trunc_f64_u",
            I64ExtendI32S => "i64.extend_i32_s",
            I64ExtendI32U => "i64.extend_i32_u",
            I64TruncF32S => "i64.trunc_f32_s",
            I64TruncF32U => "i64.trunc_f32_u",
            I64TruncF64S => "i64.trunc_f64_s",
            I64TruncF64U => "i64.trunc_f64_u",
            F32ConvertI32S => "f32.convert_i32_s",
            F32ConvertI32U => "f32.convert_i32_u",
            F32ConvertI64S => "f32.convert_i64_s",
            F32ConvertI64U => "f32.convert_i64_u",
            F32DemoteF64 => "f32.demote_f64",
            F64ConvertI32S => "f64.convert_i32_s",
            F64ConvertI32U => "f64.convert_i32_u",
            F64ConvertI64S => "f64.convert_i64_s",
            F64ConvertI64U => "f64.convert_i64_u",
            F64PromoteF32 => "f64.promote_f32",
            I32ReinterpretF32 => "i32.reinterpret_f32",
            I64ReinterpretF64 => "i64.reinterpret_f64",
            F32ReinterpretI32 => "f32.reinterpret_i32",
            F64ReinterpretI64 => "f64.reinterpret_i64",
            I32Extend8S => "i32.extend8_s",
            I32Extend16S => "i32.extend16_s",
            I64Extend8S => "i64.extend8_s",
            I64Extend16S => "i64.extend16_s",
            I64Extend32S => "i64.extend32_s",
            I32TruncSatF32S => "i32.trunc_sat_f32_s",
            I32TruncSatF32U => "i32.trunc_sat_f32_u",
            I32TruncSatF64S => "i32.trunc_sat_f64_s",
            I32TruncSatF64U => "i32.trunc_sat_f64_u",
            I64TruncSatF32S => "i64.trunc_sat_f32_s",
            I64TruncSatF32U => "i64.trunc_sat_f32_u",
            I64TruncSatF64S => "i64.trunc_sat_f64_s",
            I64TruncSatF64U => "i64.trunc_sat_f64_u",
            Simd(op) => op.mnemonic(),
        }
    }
}

impl SimdOp {
    /// Get the mnemonic for this SIMD instruction
    pub fn mnemonic(&self) -> &'static str {
        use SimdOp::*;
        match self {
            V128Load { .. } => "v128.load",
            V128Load8x8S { .. } => "v128.load8x8_s",
            V128Load8x8U { .. } => "v128.load8x8_u",
            V128Load16x4S { .. } => "v128.load16x4_s",
            V128Load16x4U { .. } => "v128.load16x4_u",
            V128Load32x2S { .. } => "v128.load32x2_s",
            V128Load32x2U { .. } => "v128.load32x2_u",
            V128Load8Splat { .. } => "v128.load8_splat",
            V128Load16Splat { .. } => "v128.load16_splat",
            V128Load32Splat { .. } => "v128.load32_splat",
            V128Load64Splat { .. } => "v128.load64_splat",
            V128Store { .. } => "v128.store",
            V128Const { .. } => "v128.const",
            I8x16Shuffle { .. } => "i8x16.shuffle",
            I8x16Swizzle => "i8x16.swizzle",
            I8x16Splat => "i8x16.splat",
            I16x8Splat => "i16x8.splat",
            I32x4Splat => "i32x4.splat",
            I64x2Splat => "i64x2.splat",
            F32x4Splat => "f32x4.splat",
            F64x2Splat => "f64x2.splat",
            I8x16ExtractLaneS { .. } => "i8x16.extract_lane_s",
            I8x16ExtractLaneU { .. } => "i8x16.extract_lane_u",
            I8x16ReplaceLane { .. } => "i8x16.replace_lane",
            I16x8ExtractLaneS { .. } => "i16x8.extract_lane_s",
            I16x8ExtractLaneU { .. } => "i16x8.extract_lane_u",
            I16x8ReplaceLane { .. } => "i16x8.replace_lane",
            I32x4ExtractLane { .. } => "i32x4.extract_lane",
            I32x4ReplaceLane { .. } => "i32x4.replace_lane",
            I64x2ExtractLane { .. } => "i64x2.extract_lane",
            I64x2ReplaceLane { .. } => "i64x2.replace_lane",
            F32x4ExtractLane { .. } => "f32x4.extract_lane",
            F32x4ReplaceLane { .. } => "f32x4.replace_lane",
            F64x2ExtractLane { .. } => "f64x2.extract_lane",
            F64x2ReplaceLane { .. } => "f64x2.replace_lane",
            I8x16Eq => "i8x16.eq",
            I8x16Ne => "i8x16.ne",
            I8x16LtS => "i8x16.lt_s",
            I8x16LtU => "i8x16.lt_u",
            I8x16GtS => "i8x16.gt_s",
            I8x16GtU => "i8x16.gt_u",
            I8x16LeS => "i8x16.le_s",
            I8x16LeU => "i8x16.le_u",
            I8x16GeS => "i8x16.ge_s",
            I8x16GeU => "i8x16.ge_u",
            I16x8Eq => "i16x8.eq",
            I16x8Ne => "i16x8.ne",
            I16x8LtS => "i16x8.lt_s",
            I16x8LtU => "i16x8.lt_u",
            I16x8GtS => "i16x8.gt_s",
            I16x8GtU => "i16x8.gt_u",
            I16x8LeS => "i16x8.le_s",
            I16x8LeU => "i16x8.le_u",
            I16x8GeS => "i16x8.ge_s",
            I16x8GeU => "i16x8.ge_u",
            I32x4Eq => "i32x4.eq",
            I32x4Ne => "i32x4.ne",
            I32x4LtS => "i32x4.lt_s",
            I32x4LtU => "i32x4.lt_u",
            I32x4GtS => "i32x4.gt_s",
            I32x4GtU => "i32x4.gt_u",
            I32x4LeS => "i32x4.le_s",
            I32x4LeU => "i32x4.le_u",
            I32x4GeS => "i32x4.ge_s",
            I32x4GeU => "i32x4.ge_u",
            F32x4Eq => "f32x4.eq",
            F32x4Ne => "f32x4.ne",
            F32x4Lt => "f32x4.lt",
            F32x4Gt => "f32x4.gt",
            F32x4Le => "f32x4.le",
            F32x4Ge => "f32x4.ge",
            F64x2Eq => "f64x2.eq",
            F64x2Ne => "f64x2.ne",
            F64x2Lt => "f64x2.lt",
            F64x2Gt => "f64x2.gt",
            F64x2Le => "f64x2.le",
            F64x2Ge => "f64x2.ge",
            V128Not => "v128.not",
            V128And => "v128.and",
            V128AndNot => "v128.andnot",
            V128Or => "v128.or",
            V128Xor => "v128.xor",
            V128Bitselect => "v128.bitselect",
            V128AnyTrue => "v128.any_true",
            V128Load8Lane { .. } => "v128.load8_lane",
            V128Load16Lane { .. } => "v128.load16_lane",
            V128Load32Lane { .. } => "v128.load32_lane",
            V128Load64Lane { .. } => "v128.load64_lane",
            V128Store8Lane { .. } => "v128.store8_lane",
            V128Store16Lane { .. } => "v128.store16_lane",
            V128Store32Lane { .. } => "v128.store32_lane",
            V128Store64Lane { .. } => "v128.store64_lane",
            V128Load32Zero { .. } => "v128.load32_zero",
            V128Load64Zero { .. } => "v128.load64_zero",
            F32x4DemoteF64x2Zero => "f32x4.demote_f64x2_zero",
            F64x2PromoteLowF32x4 => "f64x2.promote_low_f32x4",
            I8x16Abs => "i8x16.abs",
            I8x16Neg => "i8x16.neg",
            I8x16Popcnt => "i8x16.popcnt",
            I8x16AllTrue => "i8x16.all_true",
            I8x16Bitmask => "i8x16.bitmask",
            I8x16NarrowI16x8S => "i8x16.narrow_i16x8_s",
            I8x16NarrowI16x8U => "i8x16.narrow_i16x8_u",
            F32x4Ceil => "f32x4.ceil",
            F32x4Floor => "f32x4.floor",
            F32x4Trunc => "f32x4.trunc",
            F32x4Nearest => "f32x4.nearest",
            I8x16Shl => "i8x16.shl",
            I8x16ShrS => "i8x16.shr_s",
            I8x16ShrU => "i8x16.shr_u",
            I8x16Add => "i8x16.add",
            I8x16AddSatS => "i8x16.add_sat_s",
            I8x16AddSatU => "i8x16.add_sat_u",
            I8x16Sub => "i8x16.sub",
            I8x16SubSatS => "i8x16.sub_sat_s",
            I8x16SubSatU => "i8x16.sub_sat_u",
            F64x2Ceil => "f64x2.ceil",
            F64x2Floor => "f64x2.floor",
            I8x16MinS => "i8x16.min_s",
            I8x16MinU => "i8x16.min_u",
            I8x16MaxS => "i8x16.max_s",
            I8x16MaxU => "i8x16.max_u",
            F64x2Trunc => "f64x2.trunc",
            I8x16AvgrU => "i8x16.avgr_u",
            I16x8ExtAddPairwiseI8x16S => "i16x8.extadd_pairwise_i8x16_s",
            I16x8ExtAddPairwiseI8x16U => "i16x8.extadd_pairwise_i8x16_u",
            I32x4ExtAddPairwiseI16x8S => "i32x4.extadd_pairwise_i16x8_s",
            I32x4ExtAddPairwiseI16x8U => "i32x4.extadd_pairwise_i16x8_u",
            I16x8Abs => "i16x8.abs",
            I16x8Neg => "i16x8.neg",
            I16x8Q15MulrSatS => "i16x8.q15mulr_sat_s",
            I16x8AllTrue => "i16x8.all_true",
            I16x8Bitmask => "i16x8.bitmask",
            I16x8NarrowI32x4S => "i16x8.narrow_i32x4_s",
            I16x8NarrowI32x4U => "i16x8.narrow_i32x4_u",
            I16x8ExtendLowI8x16S => "i16x8.extend_low_i8x16_s",
            I16x8ExtendHighI8x16S => "i16x8.extend_high_i8x16_s",
            I16x8ExtendLowI8x16U => "i16x8.extend_low_i8x16_u",
            I16x8ExtendHighI8x16U => "i16x8.extend_high_i8x16_u",
            I16x8Shl => "i16x8.shl",
            I16x8ShrS => "i16x8.shr_s",
            I16x8ShrU => "i16x8.shr_u",
            I16x8Add => "i16x8.add",
            I16x8AddSatS => "i16x8.add_sat_s",
            I16x8AddSatU => "i16x8.add_sat_u",
            I16x8Sub => "i16x8.sub",
            I16x8SubSatS => "i16x8.sub_sat_s",
            I16x8SubSatU => "i16x8.sub_sat_u",
            F64x2Nearest => "f64x2.nearest",
            I16x8Mul => "i16x8.mul",
            I16x8MinS => "i16x8.min_s",
            I16x8MinU => "i16x8.min_u",
            I16x8MaxS => "i16x8.max_s",
            I16x8MaxU => "i16x8.max_u",
            I16x8AvgrU => "i16x8.avgr_u",
            I16x8ExtMulLowI8x16S => "i16x8.extmul_low_i8x16_s",
            I16x8ExtMulHighI8x16S => "i16x8.extmul_high_i8x16_s",
            I16x8ExtMulLowI8x16U => "i16x8.extmul_low_i8x16_u",
            I16x8ExtMulHighI8x16U => "i16x8.extmul_high_i8x16_u",
            I32x4Abs => "i32x4.abs",
            I32x4Neg => "i32x4.neg",
            I32x4AllTrue => "i32x4.all_true",
            I32x4Bitmask => "i32x4.bitmask",
            I32x4ExtendLowI16x8S => "i32x4.extend_low_i16x8_s",
            I32x4ExtendHighI16x8S => "i32x4.extend_high_i16x8_s",
            I32x4ExtendLowI16x8U => "i32x4.extend_low_i16x8_u",
            I32x4ExtendHighI16x8U => "i32x4.extend_high_i16x8_u",
            I32x4Shl => "i32x4.shl",
            I32x4ShrS => "i32x4.shr_s",
            I32x4ShrU => "i32x4.shr_u",
            I32x4Add => "i32x4.add",
            I32x4Sub => "i32x4.sub",
            I32x4Mul => "i32x4.mul",
            I32x4MinS => "i32x4.min_s",
            I32x4MinU => "i32x4.min_u",
            I32x4MaxS => "i32x4.max_s",
            I32x4MaxU => "i32x4.max_u",
            I32x4DotI16x8S => "i32x4.dot_i16x8_s",
            I32x4ExtMulLowI16x8S => "i32x4.extmul_low_i16x8_s",
            I32x4ExtMulHighI16x8S => "i32x4.extmul_high_i16x8_s",
            I32x4ExtMulLowI16x8U => "i32x4.extmul_low_i16x8_u",
            I32x4ExtMulHighI16x8U => "i32x4.extmul_high_i16x8_u",
            I64x2Abs => "i64x2.abs",
            I64x2Neg => "i64x2.neg",
            I64x2AllTrue => "i64x2.all_true",
            I64x2Bitmask => "i64x2.bitmask",
            I64x2ExtendLowI32x4S => "i64x2.extend_low_i32x4_s",
            I64x2ExtendHighI32x4S => "i64x2.extend_high_i32x4_s",
            I64x2ExtendLowI32x4U => "i64x2.extend_low_i32x4_u",
            I64x2ExtendHighI32x4U => "i64x2.extend_high_i32x4_u",
            I64x2Shl => "i64x2.shl",
            I64x2ShrS => "i64x2.shr_s",
            I64x2ShrU => "i64x2.shr_u",
            I64x2Add => "i64x2.add",
            I64x2Sub => "i64x2.sub",
            I64x2Mul => "i64x2.mul",
            I64x2Eq => "i64x2.eq",
            I64x2Ne => "i64x2.ne",
            I64x2LtS => "i64x2.lt_s",
            I64x2GtS => "i64x2.gt_s",
            I64x2LeS => "i64x2.le_s",
            I64x2GeS => "i64x2.ge_s",
            I64x2ExtMulLowI32x4S => "i64x2.extmul_low_i32x4_s",
            I64x2ExtMulHighI32x4S => "i64x2.extmul_high_i32x4_s",
            I64x2ExtMulLowI32x4U => "i64x2.extmul_low_i32x4_u",
            I64x2ExtMulHighI32x4U => "i64x2.extmul_high_i32x4_u",
            F32x4Abs => "f32x4.abs",
            F32x4Neg => "f32x4.neg",
            F32x4Sqrt => "f32x4.sqrt",
            F32x4Add => "f32x4.add",
            F32x4Sub => "f32x4.sub",
            F32x4Mul => "f32x4.mul",
            F32x4Div => "f32x4.div",
            F32x4Min => "f32x4.min",
            F32x4Max => "f32x4.max",
            F32x4PMin => "f32x4.pmin",
            F32x4PMax => "f32x4.pmax",
            F64x2Abs => "f64x2.abs",
            F64x2Neg => "f64x2.neg",
            F64x2Sqrt => "f64x2.sqrt",
            F64x2Add => "f64x2.add",
            F64x2Sub => "f64x2.sub",
            F64x2Mul => "f64x2.mul",
            F64x2Div => "f64x2.div",
            F64x2Min => "f64x2.min",
            F64x2Max => "f64x2.max",
            F64x2PMin => "f64x2.pmin",
            F64x2PMax => "f64x2.pmax",
            I32x4TruncSatF32x4S => "i32x4.trunc_sat_f32x4_s",
            I32x4TruncSatF32x4U => "i32x4.trunc_sat_f32x4_u",
            F32x4ConvertI32x4S => "f32x4.convert_i32x4_s",
            F32x4ConvertI32x4U => "f32x4.convert_i32x4_u",
            I32x4TruncSatF64x2SZero => "i32x4.trunc_sat_f64x2_s_zero",
            I32x4TruncSatF64x2UZero => "i32x4.trunc_sat_f64x2_u_zero",
            F64x2ConvertLowI32x4S => "f64x2.convert_low_i32x4_s",
            F64x2ConvertLowI32x4U => "f64x2.convert_low_i32x4_u",
        }
    }

    /// Get the SIMD sub-opcode (LEB128-encoded after the 0xFD prefix byte)
    pub fn subopcode(&self) -> u32 {
        use SimdOp::*;
        match self {
            V128Load { .. } => 0x00,
            V128Load8x8S { .. } => 0x01,
            V128Load8x8U { .. } => 0x02,
            V128Load16x4S { .. } => 0x03,
            V128Load16x4U { .. } => 0x04,
            V128Load32x2S { .. } => 0x05,
            V128Load32x2U { .. } => 0x06,
            V128Load8Splat { .. } => 0x07,
            V128Load16Splat { .. } => 0x08,
            V128Load32Splat { .. } => 0x09,
            V128Load64Splat { .. } => 0x0A,
            V128Store { .. } => 0x0B,
            V128Const { .. } => 0x0C,
            I8x16Shuffle { .. } => 0x0D,
            I8x16Swizzle => 0x0E,
            I8x16Splat => 0x0F,
            I16x8Splat => 0x10,
            I32x4Splat => 0x11,
            I64x2Splat => 0x12,
            F32x4Splat => 0x13,
            F64x2Splat => 0x14,
            I8x16ExtractLaneS { .. } => 0x15,
            I8x16ExtractLaneU { .. } => 0x16,
            I8x16ReplaceLane { .. } => 0x17,
            I16x8ExtractLaneS { .. } => 0x18,
            I16x8ExtractLaneU { .. } => 0x19,
            I16x8ReplaceLane { .. } => 0x1A,
            I32x4ExtractLane { .. } => 0x1B,
            I32x4ReplaceLane { .. } => 0x1C,
            I64x2ExtractLane { .. } => 0x1D,
            I64x2ReplaceLane { .. } => 0x1E,
            F32x4ExtractLane { .. } => 0x1F,
            F32x4ReplaceLane { .. } => 0x20,
            F64x2ExtractLane { .. } => 0x21,
            F64x2ReplaceLane { .. } => 0x22,
            I8x16Eq => 0x23,
            I8x16Ne => 0x24,
            I8x16LtS => 0x25,
            I8x16LtU => 0x26,
            I8x16GtS => 0x27,
            I8x16GtU => 0x28,
            I8x16LeS => 0x29,
            I8x16LeU => 0x2A,
            I8x16GeS => 0x2B,
            I8x16GeU => 0x2C,
            I16x8Eq => 0x2D,
            I16x8Ne => 0x2E,
            I16x8LtS => 0x2F,
            I16x8LtU => 0x30,
            I16x8GtS => 0x31,
            I16x8GtU => 0x32,
            I16x8LeS => 0x33,
            I16x8LeU => 0x34,
            I16x8GeS => 0x35,
            I16x8GeU => 0x36,
            I32x4Eq => 0x37,
            I32x4Ne => 0x38,
            I32x4LtS => 0x39,
            I32x4LtU => 0x3A,
            I32x4GtS => 0x3B,
            I32x4GtU => 0x3C,
            I32x4LeS => 0x3D,
            I32x4LeU => 0x3E,
            I32x4GeS => 0x3F,
            I32x4GeU => 0x40,
            F32x4Eq => 0x41,
            F32x4Ne => 0x42,
            F32x4Lt => 0x43,
            F32x4Gt => 0x44,
            F32x4Le => 0x45,
            F32x4Ge => 0x46,
            F64x2Eq => 0x47,
            F64x2Ne => 0x48,
            F64x2Lt => 0x49,
            F64x2Gt => 0x4A,
            F64x2Le => 0x4B,
            F64x2Ge => 0x4C,
            V128Not => 0x4D,
            V128And => 0x4E,
            V128AndNot => 0x4F,
            V128Or => 0x50,
            V128Xor => 0x51,
            V128Bitselect => 0x52,
            V128AnyTrue => 0x53,
            V128Load8Lane { .. } => 0x54,
            V128Load16Lane { .. } => 0x55,
            V128Load32Lane { .. } => 0x56,
            V128Load64Lane { .. } => 0x57,
            V128Store8Lane { .. } => 0x58,
            V128Store16Lane { .. } => 0x59,
            V128Store32Lane { .. } => 0x5A,
            V128Store64Lane { .. } => 0x5B,
            V128Load32Zero { .. } => 0x5C,
            V128Load64Zero { .. } => 0x5D,
            F32x4DemoteF64x2Zero => 0x5E,
            F64x2PromoteLowF32x4 => 0x5F,
            I8x16Abs => 0x60,
            I8x16Neg => 0x61,
            I8x16Popcnt => 0x62,
            I8x16AllTrue => 0x63,
            I8x16Bitmask => 0x64,
            I8x16NarrowI16x8S => 0x65,
            I8x16NarrowI16x8U => 0x66,
            F32x4Ceil => 0x67,
            F32x4Floor => 0x68,
            F32x4Trunc => 0x69,
            F32x4Nearest => 0x6A,
            I8x16Shl => 0x6B,
            I8x16ShrS => 0x6C,
            I8x16ShrU => 0x6D,
            I8x16Add => 0x6E,
            I8x16AddSatS => 0x6F,
            I8x16AddSatU => 0x70,
            I8x16Sub => 0x71,
            I8x16SubSatS => 0x72,
            I8x16SubSatU => 0x73,
            F64x2Ceil => 0x74,
            F64x2Floor => 0x75,
            I8x16MinS => 0x76,
            I8x16MinU => 0x77,
            I8x16MaxS => 0x78,
            I8x16MaxU => 0x79,
            F64x2Trunc => 0x7A,
            I8x16AvgrU => 0x7B,
            I16x8ExtAddPairwiseI8x16S => 0x7C,
            I16x8ExtAddPairwiseI8x16U => 0x7D,
            I32x4ExtAddPairwiseI16x8S => 0x7E,
            I32x4ExtAddPairwiseI16x8U => 0x7F,
            I16x8Abs => 0x80,
            I16x8Neg => 0x81,
            I16x8Q15MulrSatS => 0x82,
            I16x8AllTrue => 0x83,
            I16x8Bitmask => 0x84,
            I16x8NarrowI32x4S => 0x85,
            I16x8NarrowI32x4U => 0x86,
            I16x8ExtendLowI8x16S => 0x87,
            I16x8ExtendHighI8x16S => 0x88,
            I16x8ExtendLowI8x16U => 0x89,
            I16x8ExtendHighI8x16U => 0x8A,
            I16x8Shl => 0x8B,
            I16x8ShrS => 0x8C,
            I16x8ShrU => 0x8D,
            I16x8Add => 0x8E,
            I16x8AddSatS => 0x8F,
            I16x8AddSatU => 0x90,
            I16x8Sub => 0x91,
            I16x8SubSatS => 0x92,
            I16x8SubSatU => 0x93,
            F64x2Nearest => 0x94,
            I16x8Mul => 0x95,
            I16x8MinS => 0x96,
            I16x8MinU => 0x97,
            I16x8MaxS => 0x98,
            I16x8MaxU => 0x99,
            I16x8AvgrU => 0x9B,
            I16x8ExtMulLowI8x16S => 0x9C,
            I16x8ExtMulHighI8x16S => 0x9D,
            I16x8ExtMulLowI8x16U => 0x9E,
            I16x8ExtMulHighI8x16U => 0x9F,
            I32x4Abs => 0xA0,
            I32x4Neg => 0xA1,
            I32x4AllTrue => 0xA3,
            I32x4Bitmask => 0xA4,
            I32x4ExtendLowI16x8S => 0xA7,
            I32x4ExtendHighI16x8S => 0xA8,
            I32x4ExtendLowI16x8U => 0xA9,
            I32x4ExtendHighI16x8U => 0xAA,
            I32x4Shl => 0xAB,
            I32x4ShrS => 0xAC,
            I32x4ShrU => 0xAD,
            I32x4Add => 0xAE,
            I32x4Sub => 0xB1,
            I32x4Mul => 0xB5,
            I32x4MinS => 0xB6,
            I32x4MinU => 0xB7,
            I32x4MaxS => 0xB8,
            I32x4MaxU => 0xB9,
            I32x4DotI16x8S => 0xBA,
            I32x4ExtMulLowI16x8S => 0xBC,
            I32x4ExtMulHighI16x8S => 0xBD,
            I32x4ExtMulLowI16x8U => 0xBE,
            I32x4ExtMulHighI16x8U => 0xBF,
            I64x2Abs => 0xC0,
            I64x2Neg => 0xC1,
            I64x2AllTrue => 0xC3,
            I64x2Bitmask => 0xC4,
            I64x2ExtendLowI32x4S => 0xC7,
            I64x2ExtendHighI32x4S => 0xC8,
            I64x2ExtendLowI32x4U => 0xC9,
            I64x2ExtendHighI32x4U => 0xCA,
            I64x2Shl => 0xCB,
            I64x2ShrS => 0xCC,
            I64x2ShrU => 0xCD,
            I64x2Add => 0xCE,
            I64x2Sub => 0xD1,
            I64x2Mul => 0xD5,
            I64x2Eq => 0xD6,
            I64x2Ne => 0xD7,
            I64x2LtS => 0xD8,
            I64x2GtS => 0xD9,
            I64x2LeS => 0xDA,
            I64x2GeS => 0xDB,
            I64x2ExtMulLowI32x4S => 0xDC,
            I64x2ExtMulHighI32x4S => 0xDD,
            I64x2ExtMulLowI32x4U => 0xDE,
            I64x2ExtMulHighI32x4U => 0xDF,
            F32x4Abs => 0xE0,
            F32x4Neg => 0xE1,
            F32x4Sqrt => 0xE3,
            F32x4Add => 0xE4,
            F32x4Sub => 0xE5,
            F32x4Mul => 0xE6,
            F32x4Div => 0xE7,
            F32x4Min => 0xE8,
            F32x4Max => 0xE9,
            F32x4PMin => 0xEA,
            F32x4PMax => 0xEB,
            F64x2Abs => 0xEC,
            F64x2Neg => 0xED,
            F64x2Sqrt => 0xEF,
            F64x2Add => 0xF0,
            F64x2Sub => 0xF1,
            F64x2Mul => 0xF2,
            F64x2Div => 0xF3,
            F64x2Min => 0xF4,
            F64x2Max => 0xF5,
            F64x2PMin => 0xF6,
            F64x2PMax => 0xF7,
            I32x4TruncSatF32x4S => 0xF8,
            I32x4TruncSatF32x4U => 0xF9,
            F32x4ConvertI32x4S => 0xFA,
            F32x4ConvertI32x4U => 0xFB,
            I32x4TruncSatF64x2SZero => 0xFC,
            I32x4TruncSatF64x2UZero => 0xFD,
            F64x2ConvertLowI32x4S => 0xFE,
            F64x2ConvertLowI32x4U => 0xFF,
        }
    }

    /// Encode instruction-specific operands (after the prefix + subopcode)
    pub fn encode_operands(&self, bytes: &mut Vec<u8>) {
        use crate::parser::encoding;
        use SimdOp::*;
        match self {
            // Memory operations with memarg
            V128Load { memarg }
            | V128Load8x8S { memarg }
            | V128Load8x8U { memarg }
            | V128Load16x4S { memarg }
            | V128Load16x4U { memarg }
            | V128Load32x2S { memarg }
            | V128Load32x2U { memarg }
            | V128Load8Splat { memarg }
            | V128Load16Splat { memarg }
            | V128Load32Splat { memarg }
            | V128Load64Splat { memarg }
            | V128Store { memarg }
            | V128Load32Zero { memarg }
            | V128Load64Zero { memarg } => {
                encoding::write_vu32(bytes, memarg.align);
                encoding::write_vu32(bytes, memarg.offset);
            }
            // Load/store lane: memarg + lane
            V128Load8Lane { memarg, lane }
            | V128Load16Lane { memarg, lane }
            | V128Load32Lane { memarg, lane }
            | V128Load64Lane { memarg, lane }
            | V128Store8Lane { memarg, lane }
            | V128Store16Lane { memarg, lane }
            | V128Store32Lane { memarg, lane }
            | V128Store64Lane { memarg, lane } => {
                encoding::write_vu32(bytes, memarg.align);
                encoding::write_vu32(bytes, memarg.offset);
                bytes.push(*lane);
            }
            // V128 const: 16 raw bytes
            V128Const { value } => {
                bytes.extend(value);
            }
            // Shuffle: 16 lane indices
            I8x16Shuffle { lanes } => {
                bytes.extend(lanes);
            }
            // Lane extraction/replacement: single lane byte
            I8x16ExtractLaneS { lane }
            | I8x16ExtractLaneU { lane }
            | I8x16ReplaceLane { lane }
            | I16x8ExtractLaneS { lane }
            | I16x8ExtractLaneU { lane }
            | I16x8ReplaceLane { lane }
            | I32x4ExtractLane { lane }
            | I32x4ReplaceLane { lane }
            | I64x2ExtractLane { lane }
            | I64x2ReplaceLane { lane }
            | F32x4ExtractLane { lane }
            | F32x4ReplaceLane { lane }
            | F64x2ExtractLane { lane }
            | F64x2ReplaceLane { lane } => {
                bytes.push(*lane);
            }
            // All other SIMD ops have no additional operands
            _ => {}
        }
    }
}

impl fmt::Display for SimdOp {
    /// Formats only the operands (mnemonic is written by InstructionKind::fmt)
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use SimdOp::*;
        match self {
            V128Const { value } => {
                // Display as 4 i32 lanes (little-endian), matching wabt format
                for i in 0..4 {
                    let lane = u32::from_le_bytes([value[i * 4], value[i * 4 + 1], value[i * 4 + 2], value[i * 4 + 3]]);
                    write!(f, " 0x{lane:08x}")?;
                }
                Ok(())
            }
            I8x16Shuffle { lanes } => {
                // Display as 4 i32 lanes in hex, matching wabt format
                for i in 0..4 {
                    let lane = u32::from_le_bytes([lanes[i * 4], lanes[i * 4 + 1], lanes[i * 4 + 2], lanes[i * 4 + 3]]);
                    write!(f, " 0x{lane:08x}")?;
                }
                Ok(())
            }
            // Lane operations
            I8x16ExtractLaneS { lane }
            | I8x16ExtractLaneU { lane }
            | I8x16ReplaceLane { lane }
            | I16x8ExtractLaneS { lane }
            | I16x8ExtractLaneU { lane }
            | I16x8ReplaceLane { lane }
            | I32x4ExtractLane { lane }
            | I32x4ReplaceLane { lane }
            | I64x2ExtractLane { lane }
            | I64x2ReplaceLane { lane }
            | F32x4ExtractLane { lane }
            | F32x4ReplaceLane { lane }
            | F64x2ExtractLane { lane }
            | F64x2ReplaceLane { lane } => write!(f, " {lane}"),
            // Memory operations with memarg
            V128Load { memarg }
            | V128Load8x8S { memarg }
            | V128Load8x8U { memarg }
            | V128Load16x4S { memarg }
            | V128Load16x4U { memarg }
            | V128Load32x2S { memarg }
            | V128Load32x2U { memarg }
            | V128Load8Splat { memarg }
            | V128Load16Splat { memarg }
            | V128Load32Splat { memarg }
            | V128Load64Splat { memarg }
            | V128Store { memarg }
            | V128Load32Zero { memarg }
            | V128Load64Zero { memarg } => write!(f, " {} {}", memarg.align, memarg.offset),
            // Load/store lane: memarg + lane
            V128Load8Lane { memarg, lane }
            | V128Load16Lane { memarg, lane }
            | V128Load32Lane { memarg, lane }
            | V128Load64Lane { memarg, lane }
            | V128Store8Lane { memarg, lane }
            | V128Store16Lane { memarg, lane }
            | V128Store32Lane { memarg, lane }
            | V128Store64Lane { memarg, lane } => {
                write!(f, " {} {} {lane}", memarg.align, memarg.offset)
            }
            // All other SIMD ops have no operands
            _ => Ok(()),
        }
    }
}

impl fmt::Display for InstructionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InstructionKind::*;

        write!(f, "{}", self.mnemonic())?;

        match self {
            // Instructions with operands
            Block { block_type } | Loop { block_type } | If { block_type } => {
                write!(f, "{block_type}")
            }
            Br { label_idx } | BrIf { label_idx } => write!(f, " {label_idx}"),
            BrTable { labels, default } => {
                for label in labels {
                    write!(f, " {label}")?;
                }
                write!(f, " {default}")
            }
            Call { func_idx } | RefFunc { func_idx } => write!(f, " {func_idx}"),
            CallIndirect { type_idx, table_idx } => write!(f, " {table_idx} (type {type_idx})"),
            RefNull { ref_type } => {
                // Use shortened names for ref types to match WABT output
                let type_name = match ref_type {
                    ValueType::FuncRef => "func",
                    ValueType::ExternRef => "extern",
                    _ => return write!(f, " {ref_type}"),
                };
                write!(f, " {type_name}")
            }
            SelectTyped { val_types } => {
                // Display typed select with just the first type
                if let Some(vt) = val_types.first() {
                    write!(f, " {vt}")
                } else {
                    Ok(())
                }
            }
            LocalGet { local_idx } | LocalSet { local_idx } | LocalTee { local_idx } => {
                write!(f, " {local_idx}")
            }
            GlobalGet { global_idx } | GlobalSet { global_idx } => write!(f, " {global_idx}"),
            TableGet { table_idx }
            | TableSet { table_idx }
            | TableGrow { table_idx }
            | TableSize { table_idx }
            | TableFill { table_idx } => write!(f, " {table_idx}"),
            TableInit { elem_idx, table_idx } => write!(f, " {elem_idx} {table_idx}"),
            ElemDrop { elem_idx } => write!(f, " {elem_idx}"),
            TableCopy { dst_table, src_table } => write!(f, " {dst_table} {src_table}"),
            MemoryInit { data_idx } => write!(f, " {data_idx} 0"),
            DataDrop { data_idx } => write!(f, " {data_idx}"),
            MemoryCopy => write!(f, " 0 0"),
            MemoryFill => write!(f, " 0"),
            MemorySize | MemoryGrow => write!(f, " 0"),
            I32Const { value } => {
                // Always display as unsigned to match test expectations
                write!(f, " {}", *value as u32)
            }
            I64Const { value } => {
                // Always display as unsigned to match test expectations
                write!(f, " {}", *value as u64)
            }
            F32Const { value } => write!(f, " {}", value.to_hex()),
            F64Const { value } => write!(f, " {}", value.to_hex()),
            Simd(op) => op.fmt(f),
            // Memory instructions with memarg
            I32Load { memarg }
            | I64Load { memarg }
            | F32Load { memarg }
            | F64Load { memarg }
            | I32Load8S { memarg }
            | I32Load8U { memarg }
            | I32Load16S { memarg }
            | I32Load16U { memarg }
            | I64Load8S { memarg }
            | I64Load8U { memarg }
            | I64Load16S { memarg }
            | I64Load16U { memarg }
            | I64Load32S { memarg }
            | I64Load32U { memarg }
            | I32Store { memarg }
            | I64Store { memarg }
            | F32Store { memarg }
            | F64Store { memarg }
            | I32Store8 { memarg }
            | I32Store16 { memarg }
            | I64Store8 { memarg }
            | I64Store16 { memarg }
            | I64Store32 { memarg } => {
                // Always display align and offset values (align as log2)
                write!(f, " {} {}", memarg.align, memarg.offset)
            }
            // Instructions with no operands
            _ => Ok(()),
        }
    }
}

impl fmt::Display for BlockType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockType::Empty => Ok(()),
            BlockType::Value(vt) => write!(f, " {vt}"),
            BlockType::FuncType(idx) => write!(f, " type[{idx}]"),
        }
    }
}
