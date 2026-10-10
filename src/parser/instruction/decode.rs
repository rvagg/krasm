//! Instruction decoding from binary format

use super::super::limits;
use super::{BlockType, ByteRange, Instruction, InstructionKind, MemArg, SimdOp};
use crate::parser::encoding;
use crate::parser::module::{RefType, ValueType};
use crate::parser::reader::Reader;
use InstructionKind::*;
use std::io;

/// Error type for instruction decoding
#[derive(Debug)]
pub enum DecodeError {
    Io(io::Error),
    UnknownOpcode(u8),
    UnknownInstruction(u8, u32), // For multi-byte opcodes (0xFC prefix for saturating/memory ops, 0xFD for SIMD)
    InvalidValueType(String),
    InvalidBlockType,
    Validation(crate::parser::validate::ValidationError),
}

impl From<io::Error> for DecodeError {
    fn from(e: io::Error) -> Self {
        DecodeError::Io(e)
    }
}

impl From<crate::parser::validate::ValidationError> for DecodeError {
    fn from(e: crate::parser::validate::ValidationError) -> Self {
        DecodeError::Validation(e)
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Io(e) => write!(f, "io error: {e}"),
            DecodeError::UnknownOpcode(_op) => write!(f, "illegal opcode"),
            DecodeError::UnknownInstruction(prefix, sub) => {
                write!(f, "unknown instruction: {prefix:#04x} {sub:#04x}")
            }
            DecodeError::InvalidValueType(s) => write!(f, "invalid value type: {s}"),
            DecodeError::InvalidBlockType => write!(f, "invalid block type"),
            DecodeError::Validation(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<String> for DecodeError {
    fn from(s: String) -> Self {
        DecodeError::InvalidValueType(s)
    }
}

impl MemArg {
    /// Decode a memory argument from the reader
    pub fn decode(reader: &mut Reader) -> Result<Self, DecodeError> {
        let align = reader.read_vu32()?;
        let offset = reader.read_vu32()?;
        Ok(MemArg { align, offset })
    }
}

impl BlockType {
    /// Decode a block type from the reader
    pub fn decode(reader: &mut Reader) -> Result<Self, DecodeError> {
        // Peek at the next byte to determine block type
        let b = reader.read_byte()?;
        if b == encoding::BLOCK_TYPE_EMPTY {
            Ok(BlockType::Empty)
        } else if ValueType::is_value_type_byte(b) {
            // Parse as value type
            match ValueType::decode(b) {
                Ok(vt) => Ok(BlockType::Value(vt)),
                Err(e) => Err(DecodeError::InvalidValueType(e)),
            }
        } else {
            // Not a value type, read as signed LEB128 with first byte already consumed
            let mut first_byte = Some(b);
            let type_idx = crate::parser::reader::read_vs33(&mut || match first_byte.take() {
                Some(byte) => Ok(byte),
                None => reader.read_byte(),
            })? as u32;
            Ok(BlockType::FuncType(type_idx))
        }
    }
}

impl InstructionKind {
    /// Decode an instruction from the reader
    pub fn decode(reader: &mut Reader) -> Result<Self, DecodeError> {
        let opcode = reader.read_byte()?;

        match opcode {
            // Control instructions (0x00-0x11)
            0x00 => Ok(Unreachable),
            0x01 => Ok(Nop),
            0x02 => Ok(Block {
                block_type: BlockType::decode(reader)?,
            }),
            0x03 => Ok(Loop {
                block_type: BlockType::decode(reader)?,
            }),
            0x04 => Ok(If {
                block_type: BlockType::decode(reader)?,
            }),
            0x05 => Ok(Else),
            0x0B => Ok(End),
            0x0C => Ok(Br {
                label_idx: reader.read_vu32()?,
            }),
            0x0D => Ok(BrIf {
                label_idx: reader.read_vu32()?,
            }),
            0x0E => {
                // BrTable
                let count = reader.read_vu32()?;
                if count > limits::MAX_BR_TABLE_LABELS {
                    return Err(DecodeError::Io(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "br_table label count exceeds implementation limit",
                    )));
                }
                reader.validate_item_count(count)?;
                let mut labels = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    labels.push(reader.read_vu32()?);
                }
                let default = reader.read_vu32()?;
                Ok(BrTable { labels, default })
            }
            0x0F => Ok(Return),
            0x10 => Ok(Call {
                func_idx: reader.read_vu32()?,
            }),
            0x11 => Ok(CallIndirect {
                type_idx: reader.read_vu32()?,
                table_idx: reader.read_vu32()?,
            }),

            // Reference instructions (0xD0-0xD2)
            // The immediate is a reference type; number types are malformed here.
            0xD0 => {
                let ref_type = RefType::from_wire_byte(reader.read_byte()?)?.into();
                Ok(RefNull { ref_type })
            }
            0xD1 => Ok(RefIsNull),
            0xD2 => Ok(RefFunc {
                func_idx: reader.read_vu32()?,
            }),

            // Parametric instructions (0x1A-0x1C)
            0x1A => Ok(Drop),
            0x1B => Ok(Select),
            0x1C => {
                // SelectTyped
                let count = reader.read_vu32()?;
                if count > limits::MAX_SELECT_TYPED_VALUES {
                    return Err(DecodeError::Io(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "select typed value count exceeds implementation limit",
                    )));
                }
                reader.validate_item_count(count)?;
                let mut val_types = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    val_types.push(ValueType::decode(reader.read_byte()?)?);
                }
                Ok(SelectTyped { val_types })
            }

            // Variable instructions (0x20-0x24)
            0x20 => Ok(LocalGet {
                local_idx: reader.read_vu32()?,
            }),
            0x21 => Ok(LocalSet {
                local_idx: reader.read_vu32()?,
            }),
            0x22 => Ok(LocalTee {
                local_idx: reader.read_vu32()?,
            }),
            0x23 => Ok(GlobalGet {
                global_idx: reader.read_vu32()?,
            }),
            0x24 => Ok(GlobalSet {
                global_idx: reader.read_vu32()?,
            }),

            // Table instructions (0x25-0x26)
            0x25 => Ok(TableGet {
                table_idx: reader.read_vu32()?,
            }),
            0x26 => Ok(TableSet {
                table_idx: reader.read_vu32()?,
            }),

            // Memory instructions (0x28-0x40)
            0x28 => Ok(I32Load {
                memarg: MemArg::decode(reader)?,
            }),
            0x29 => Ok(I64Load {
                memarg: MemArg::decode(reader)?,
            }),
            0x2A => Ok(F32Load {
                memarg: MemArg::decode(reader)?,
            }),
            0x2B => Ok(F64Load {
                memarg: MemArg::decode(reader)?,
            }),
            0x2C => Ok(I32Load8S {
                memarg: MemArg::decode(reader)?,
            }),
            0x2D => Ok(I32Load8U {
                memarg: MemArg::decode(reader)?,
            }),
            0x2E => Ok(I32Load16S {
                memarg: MemArg::decode(reader)?,
            }),
            0x2F => Ok(I32Load16U {
                memarg: MemArg::decode(reader)?,
            }),
            0x30 => Ok(I64Load8S {
                memarg: MemArg::decode(reader)?,
            }),
            0x31 => Ok(I64Load8U {
                memarg: MemArg::decode(reader)?,
            }),
            0x32 => Ok(I64Load16S {
                memarg: MemArg::decode(reader)?,
            }),
            0x33 => Ok(I64Load16U {
                memarg: MemArg::decode(reader)?,
            }),
            0x34 => Ok(I64Load32S {
                memarg: MemArg::decode(reader)?,
            }),
            0x35 => Ok(I64Load32U {
                memarg: MemArg::decode(reader)?,
            }),
            0x36 => Ok(I32Store {
                memarg: MemArg::decode(reader)?,
            }),
            0x37 => Ok(I64Store {
                memarg: MemArg::decode(reader)?,
            }),
            0x38 => Ok(F32Store {
                memarg: MemArg::decode(reader)?,
            }),
            0x39 => Ok(F64Store {
                memarg: MemArg::decode(reader)?,
            }),
            0x3A => Ok(I32Store8 {
                memarg: MemArg::decode(reader)?,
            }),
            0x3B => Ok(I32Store16 {
                memarg: MemArg::decode(reader)?,
            }),
            0x3C => Ok(I64Store8 {
                memarg: MemArg::decode(reader)?,
            }),
            0x3D => Ok(I64Store16 {
                memarg: MemArg::decode(reader)?,
            }),
            0x3E => Ok(I64Store32 {
                memarg: MemArg::decode(reader)?,
            }),
            // Memory size/grow instructions (0x3F-0x40) are handled specially in the instruction
            // iterator because they require validating a reserved byte that must be 0x00.
            // We shouldn't reach here in normal parsing, but include for completeness.
            0x3F => Err(DecodeError::InvalidValueType(
                "0x3F requires special decoding".to_string(),
            )),
            0x40 => Err(DecodeError::InvalidValueType(
                "0x40 requires special decoding".to_string(),
            )),

            // Numeric instructions (0x41-0xC4)
            0x41 => Ok(I32Const {
                value: reader.read_vs32()?,
            }),
            0x42 => Ok(I64Const {
                value: reader.read_vs64()?,
            }),
            0x43 => Ok(F32Const {
                value: reader.read_f32()?,
            }),
            0x44 => Ok(F64Const {
                value: reader.read_f64()?,
            }),

            // i32 operations
            0x45 => Ok(I32Eqz),
            0x46 => Ok(I32Eq),
            0x47 => Ok(I32Ne),
            0x48 => Ok(I32LtS),
            0x49 => Ok(I32LtU),
            0x4A => Ok(I32GtS),
            0x4B => Ok(I32GtU),
            0x4C => Ok(I32LeS),
            0x4D => Ok(I32LeU),
            0x4E => Ok(I32GeS),
            0x4F => Ok(I32GeU),

            // i64 operations
            0x50 => Ok(I64Eqz),
            0x51 => Ok(I64Eq),
            0x52 => Ok(I64Ne),
            0x53 => Ok(I64LtS),
            0x54 => Ok(I64LtU),
            0x55 => Ok(I64GtS),
            0x56 => Ok(I64GtU),
            0x57 => Ok(I64LeS),
            0x58 => Ok(I64LeU),
            0x59 => Ok(I64GeS),
            0x5A => Ok(I64GeU),

            // f32 operations
            0x5B => Ok(F32Eq),
            0x5C => Ok(F32Ne),
            0x5D => Ok(F32Lt),
            0x5E => Ok(F32Gt),
            0x5F => Ok(F32Le),
            0x60 => Ok(F32Ge),

            // f64 operations
            0x61 => Ok(F64Eq),
            0x62 => Ok(F64Ne),
            0x63 => Ok(F64Lt),
            0x64 => Ok(F64Gt),
            0x65 => Ok(F64Le),
            0x66 => Ok(F64Ge),

            // i32 arithmetic
            0x67 => Ok(I32Clz),
            0x68 => Ok(I32Ctz),
            0x69 => Ok(I32Popcnt),
            0x6A => Ok(I32Add),
            0x6B => Ok(I32Sub),
            0x6C => Ok(I32Mul),
            0x6D => Ok(I32DivS),
            0x6E => Ok(I32DivU),
            0x6F => Ok(I32RemS),
            0x70 => Ok(I32RemU),
            0x71 => Ok(I32And),
            0x72 => Ok(I32Or),
            0x73 => Ok(I32Xor),
            0x74 => Ok(I32Shl),
            0x75 => Ok(I32ShrS),
            0x76 => Ok(I32ShrU),
            0x77 => Ok(I32Rotl),
            0x78 => Ok(I32Rotr),

            // i64 arithmetic
            0x79 => Ok(I64Clz),
            0x7A => Ok(I64Ctz),
            0x7B => Ok(I64Popcnt),
            0x7C => Ok(I64Add),
            0x7D => Ok(I64Sub),
            0x7E => Ok(I64Mul),
            0x7F => Ok(I64DivS),
            0x80 => Ok(I64DivU),
            0x81 => Ok(I64RemS),
            0x82 => Ok(I64RemU),
            0x83 => Ok(I64And),
            0x84 => Ok(I64Or),
            0x85 => Ok(I64Xor),
            0x86 => Ok(I64Shl),
            0x87 => Ok(I64ShrS),
            0x88 => Ok(I64ShrU),
            0x89 => Ok(I64Rotl),
            0x8A => Ok(I64Rotr),

            // f32 arithmetic
            0x8B => Ok(F32Abs),
            0x8C => Ok(F32Neg),
            0x8D => Ok(F32Ceil),
            0x8E => Ok(F32Floor),
            0x8F => Ok(F32Trunc),
            0x90 => Ok(F32Nearest),
            0x91 => Ok(F32Sqrt),
            0x92 => Ok(F32Add),
            0x93 => Ok(F32Sub),
            0x94 => Ok(F32Mul),
            0x95 => Ok(F32Div),
            0x96 => Ok(F32Min),
            0x97 => Ok(F32Max),
            0x98 => Ok(F32Copysign),

            // f64 arithmetic
            0x99 => Ok(F64Abs),
            0x9A => Ok(F64Neg),
            0x9B => Ok(F64Ceil),
            0x9C => Ok(F64Floor),
            0x9D => Ok(F64Trunc),
            0x9E => Ok(F64Nearest),
            0x9F => Ok(F64Sqrt),
            0xA0 => Ok(F64Add),
            0xA1 => Ok(F64Sub),
            0xA2 => Ok(F64Mul),
            0xA3 => Ok(F64Div),
            0xA4 => Ok(F64Min),
            0xA5 => Ok(F64Max),
            0xA6 => Ok(F64Copysign),

            // Conversions
            0xA7 => Ok(I32WrapI64),
            0xA8 => Ok(I32TruncF32S),
            0xA9 => Ok(I32TruncF32U),
            0xAA => Ok(I32TruncF64S),
            0xAB => Ok(I32TruncF64U),
            0xAC => Ok(I64ExtendI32S),
            0xAD => Ok(I64ExtendI32U),
            0xAE => Ok(I64TruncF32S),
            0xAF => Ok(I64TruncF32U),
            0xB0 => Ok(I64TruncF64S),
            0xB1 => Ok(I64TruncF64U),
            0xB2 => Ok(F32ConvertI32S),
            0xB3 => Ok(F32ConvertI32U),
            0xB4 => Ok(F32ConvertI64S),
            0xB5 => Ok(F32ConvertI64U),
            0xB6 => Ok(F32DemoteF64),
            0xB7 => Ok(F64ConvertI32S),
            0xB8 => Ok(F64ConvertI32U),
            0xB9 => Ok(F64ConvertI64S),
            0xBA => Ok(F64ConvertI64U),
            0xBB => Ok(F64PromoteF32),
            0xBC => Ok(I32ReinterpretF32),
            0xBD => Ok(I64ReinterpretF64),
            0xBE => Ok(F32ReinterpretI32),
            0xBF => Ok(F64ReinterpretI64),

            // Sign extension
            0xC0 => Ok(I32Extend8S),
            0xC1 => Ok(I32Extend16S),
            0xC2 => Ok(I64Extend8S),
            0xC3 => Ok(I64Extend16S),
            0xC4 => Ok(I64Extend32S),

            // Multi-byte instructions with 0xFC/0xFD prefixes are handled specially in the
            // instruction iterator to properly capture all bytes including prefix and subopcode.
            // We shouldn't reach here in normal parsing, but include for completeness.
            0xFC => Err(DecodeError::InvalidValueType(
                "0xFC prefix requires special decoding".to_string(),
            )),
            0xFD => Err(DecodeError::InvalidValueType(
                "0xFD prefix requires special decoding".to_string(),
            )),

            _ => Err(DecodeError::UnknownOpcode(opcode)),
        }
    }
}

/// Decode 0xFC prefix instructions
fn decode_0xfc(subopcode: u32, reader: &mut Reader) -> Result<InstructionKind, DecodeError> {
    match subopcode {
        // Saturating truncation instructions
        0x00 => Ok(I32TruncSatF32S),
        0x01 => Ok(I32TruncSatF32U),
        0x02 => Ok(I32TruncSatF64S),
        0x03 => Ok(I32TruncSatF64U),
        0x04 => Ok(I64TruncSatF32S),
        0x05 => Ok(I64TruncSatF32U),
        0x06 => Ok(I64TruncSatF64S),
        0x07 => Ok(I64TruncSatF64U),

        // Memory instructions
        0x08 => {
            let data_idx = reader.read_vu32()?;
            let reserved = reader.read_byte()?;
            if reserved != 0x00 {
                return Err(DecodeError::InvalidValueType(format!(
                    "memory.init reserved byte must be 0x00, got 0x{reserved:02X}"
                )));
            }
            Ok(MemoryInit { data_idx })
        }
        0x09 => Ok(DataDrop {
            data_idx: reader.read_vu32()?,
        }),
        0x0A => {
            let reserved1 = reader.read_byte()?;
            let reserved2 = reader.read_byte()?;
            if reserved1 != 0x00 || reserved2 != 0x00 {
                return Err(DecodeError::InvalidValueType(format!(
                    "memory.copy reserved bytes must be 0x00, got 0x{reserved1:02X} 0x{reserved2:02X}"
                )));
            }
            Ok(MemoryCopy)
        }
        0x0B => {
            let reserved = reader.read_byte()?;
            if reserved != 0x00 {
                return Err(DecodeError::InvalidValueType(format!(
                    "memory.fill reserved byte must be 0x00, got 0x{reserved:02X}"
                )));
            }
            Ok(MemoryFill)
        }

        // Table instructions
        0x0C => Ok(TableInit {
            elem_idx: reader.read_vu32()?,
            table_idx: reader.read_vu32()?,
        }),
        0x0D => Ok(ElemDrop {
            elem_idx: reader.read_vu32()?,
        }),
        0x0E => Ok(TableCopy {
            dst_table: reader.read_vu32()?,
            src_table: reader.read_vu32()?,
        }),
        0x0F => Ok(TableGrow {
            table_idx: reader.read_vu32()?,
        }),
        0x10 => Ok(TableSize {
            table_idx: reader.read_vu32()?,
        }),
        0x11 => Ok(TableFill {
            table_idx: reader.read_vu32()?,
        }),

        _ => Err(DecodeError::UnknownInstruction(0xFC, subopcode)),
    }
}

/// Decode 0xFD prefix instructions (SIMD)
fn decode_0xfd(subopcode: u32, reader: &mut Reader) -> Result<InstructionKind, DecodeError> {
    let op = match subopcode {
        // Memory operations with MemArg
        0x00 => SimdOp::V128Load {
            memarg: MemArg::decode(reader)?,
        },
        0x01 => SimdOp::V128Load8x8S {
            memarg: MemArg::decode(reader)?,
        },
        0x02 => SimdOp::V128Load8x8U {
            memarg: MemArg::decode(reader)?,
        },
        0x03 => SimdOp::V128Load16x4S {
            memarg: MemArg::decode(reader)?,
        },
        0x04 => SimdOp::V128Load16x4U {
            memarg: MemArg::decode(reader)?,
        },
        0x05 => SimdOp::V128Load32x2S {
            memarg: MemArg::decode(reader)?,
        },
        0x06 => SimdOp::V128Load32x2U {
            memarg: MemArg::decode(reader)?,
        },
        0x07 => SimdOp::V128Load8Splat {
            memarg: MemArg::decode(reader)?,
        },
        0x08 => SimdOp::V128Load16Splat {
            memarg: MemArg::decode(reader)?,
        },
        0x09 => SimdOp::V128Load32Splat {
            memarg: MemArg::decode(reader)?,
        },
        0x0A => SimdOp::V128Load64Splat {
            memarg: MemArg::decode(reader)?,
        },
        0x0B => SimdOp::V128Store {
            memarg: MemArg::decode(reader)?,
        },

        // V128Const: 16 raw bytes
        0x0C => {
            let mut value = [0u8; 16];
            for byte in &mut value {
                *byte = reader.read_byte()?;
            }
            SimdOp::V128Const { value }
        }

        // I8x16Shuffle: 16 lane bytes
        0x0D => {
            let mut lanes = [0u8; 16];
            for lane in &mut lanes {
                *lane = reader.read_byte()?;
            }
            SimdOp::I8x16Shuffle { lanes }
        }

        // Splat and swizzle (no operands)
        0x0E => SimdOp::I8x16Swizzle,
        0x0F => SimdOp::I8x16Splat,
        0x10 => SimdOp::I16x8Splat,
        0x11 => SimdOp::I32x4Splat,
        0x12 => SimdOp::I64x2Splat,
        0x13 => SimdOp::F32x4Splat,
        0x14 => SimdOp::F64x2Splat,

        // Lane extract/replace (1 lane byte)
        0x15 => SimdOp::I8x16ExtractLaneS {
            lane: reader.read_byte()?,
        },
        0x16 => SimdOp::I8x16ExtractLaneU {
            lane: reader.read_byte()?,
        },
        0x17 => SimdOp::I8x16ReplaceLane {
            lane: reader.read_byte()?,
        },
        0x18 => SimdOp::I16x8ExtractLaneS {
            lane: reader.read_byte()?,
        },
        0x19 => SimdOp::I16x8ExtractLaneU {
            lane: reader.read_byte()?,
        },
        0x1A => SimdOp::I16x8ReplaceLane {
            lane: reader.read_byte()?,
        },
        0x1B => SimdOp::I32x4ExtractLane {
            lane: reader.read_byte()?,
        },
        0x1C => SimdOp::I32x4ReplaceLane {
            lane: reader.read_byte()?,
        },
        0x1D => SimdOp::I64x2ExtractLane {
            lane: reader.read_byte()?,
        },
        0x1E => SimdOp::I64x2ReplaceLane {
            lane: reader.read_byte()?,
        },
        0x1F => SimdOp::F32x4ExtractLane {
            lane: reader.read_byte()?,
        },
        0x20 => SimdOp::F32x4ReplaceLane {
            lane: reader.read_byte()?,
        },
        0x21 => SimdOp::F64x2ExtractLane {
            lane: reader.read_byte()?,
        },
        0x22 => SimdOp::F64x2ReplaceLane {
            lane: reader.read_byte()?,
        },

        // i8x16 comparisons
        0x23 => SimdOp::I8x16Eq,
        0x24 => SimdOp::I8x16Ne,
        0x25 => SimdOp::I8x16LtS,
        0x26 => SimdOp::I8x16LtU,
        0x27 => SimdOp::I8x16GtS,
        0x28 => SimdOp::I8x16GtU,
        0x29 => SimdOp::I8x16LeS,
        0x2A => SimdOp::I8x16LeU,
        0x2B => SimdOp::I8x16GeS,
        0x2C => SimdOp::I8x16GeU,

        // i16x8 comparisons
        0x2D => SimdOp::I16x8Eq,
        0x2E => SimdOp::I16x8Ne,
        0x2F => SimdOp::I16x8LtS,
        0x30 => SimdOp::I16x8LtU,
        0x31 => SimdOp::I16x8GtS,
        0x32 => SimdOp::I16x8GtU,
        0x33 => SimdOp::I16x8LeS,
        0x34 => SimdOp::I16x8LeU,
        0x35 => SimdOp::I16x8GeS,
        0x36 => SimdOp::I16x8GeU,

        // i32x4 comparisons
        0x37 => SimdOp::I32x4Eq,
        0x38 => SimdOp::I32x4Ne,
        0x39 => SimdOp::I32x4LtS,
        0x3A => SimdOp::I32x4LtU,
        0x3B => SimdOp::I32x4GtS,
        0x3C => SimdOp::I32x4GtU,
        0x3D => SimdOp::I32x4LeS,
        0x3E => SimdOp::I32x4LeU,
        0x3F => SimdOp::I32x4GeS,
        0x40 => SimdOp::I32x4GeU,

        // f32x4 comparisons
        0x41 => SimdOp::F32x4Eq,
        0x42 => SimdOp::F32x4Ne,
        0x43 => SimdOp::F32x4Lt,
        0x44 => SimdOp::F32x4Gt,
        0x45 => SimdOp::F32x4Le,
        0x46 => SimdOp::F32x4Ge,

        // f64x2 comparisons
        0x47 => SimdOp::F64x2Eq,
        0x48 => SimdOp::F64x2Ne,
        0x49 => SimdOp::F64x2Lt,
        0x4A => SimdOp::F64x2Gt,
        0x4B => SimdOp::F64x2Le,
        0x4C => SimdOp::F64x2Ge,

        // v128 bitwise
        0x4D => SimdOp::V128Not,
        0x4E => SimdOp::V128And,
        0x4F => SimdOp::V128AndNot,
        0x50 => SimdOp::V128Or,
        0x51 => SimdOp::V128Xor,
        0x52 => SimdOp::V128Bitselect,
        0x53 => SimdOp::V128AnyTrue,

        // Load/store lane (MemArg + 1 lane byte)
        0x54 => SimdOp::V128Load8Lane {
            memarg: MemArg::decode(reader)?,
            lane: reader.read_byte()?,
        },
        0x55 => SimdOp::V128Load16Lane {
            memarg: MemArg::decode(reader)?,
            lane: reader.read_byte()?,
        },
        0x56 => SimdOp::V128Load32Lane {
            memarg: MemArg::decode(reader)?,
            lane: reader.read_byte()?,
        },
        0x57 => SimdOp::V128Load64Lane {
            memarg: MemArg::decode(reader)?,
            lane: reader.read_byte()?,
        },
        0x58 => SimdOp::V128Store8Lane {
            memarg: MemArg::decode(reader)?,
            lane: reader.read_byte()?,
        },
        0x59 => SimdOp::V128Store16Lane {
            memarg: MemArg::decode(reader)?,
            lane: reader.read_byte()?,
        },
        0x5A => SimdOp::V128Store32Lane {
            memarg: MemArg::decode(reader)?,
            lane: reader.read_byte()?,
        },
        0x5B => SimdOp::V128Store64Lane {
            memarg: MemArg::decode(reader)?,
            lane: reader.read_byte()?,
        },

        // Load zero
        0x5C => SimdOp::V128Load32Zero {
            memarg: MemArg::decode(reader)?,
        },
        0x5D => SimdOp::V128Load64Zero {
            memarg: MemArg::decode(reader)?,
        },

        // Conversions
        0x5E => SimdOp::F32x4DemoteF64x2Zero,
        0x5F => SimdOp::F64x2PromoteLowF32x4,

        // i8x16 operations (0x60-0x7B)
        // f32x4 and f64x2 rounding ops are interleaved here per the spec —
        // they were added later and assigned to gaps in this opcode range.
        0x60 => SimdOp::I8x16Abs,
        0x61 => SimdOp::I8x16Neg,
        0x62 => SimdOp::I8x16Popcnt,
        0x63 => SimdOp::I8x16AllTrue,
        0x64 => SimdOp::I8x16Bitmask,
        0x65 => SimdOp::I8x16NarrowI16x8S,
        0x66 => SimdOp::I8x16NarrowI16x8U,
        0x67 => SimdOp::F32x4Ceil,
        0x68 => SimdOp::F32x4Floor,
        0x69 => SimdOp::F32x4Trunc,
        0x6A => SimdOp::F32x4Nearest,
        0x6B => SimdOp::I8x16Shl,
        0x6C => SimdOp::I8x16ShrS,
        0x6D => SimdOp::I8x16ShrU,
        0x6E => SimdOp::I8x16Add,
        0x6F => SimdOp::I8x16AddSatS,
        0x70 => SimdOp::I8x16AddSatU,
        0x71 => SimdOp::I8x16Sub,
        0x72 => SimdOp::I8x16SubSatS,
        0x73 => SimdOp::I8x16SubSatU,
        0x74 => SimdOp::F64x2Ceil,
        0x75 => SimdOp::F64x2Floor,
        0x76 => SimdOp::I8x16MinS,
        0x77 => SimdOp::I8x16MinU,
        0x78 => SimdOp::I8x16MaxS,
        0x79 => SimdOp::I8x16MaxU,
        0x7A => SimdOp::F64x2Trunc,
        0x7B => SimdOp::I8x16AvgrU,

        // Pairwise extension
        0x7C => SimdOp::I16x8ExtAddPairwiseI8x16S,
        0x7D => SimdOp::I16x8ExtAddPairwiseI8x16U,
        0x7E => SimdOp::I32x4ExtAddPairwiseI16x8S,
        0x7F => SimdOp::I32x4ExtAddPairwiseI16x8U,

        // i16x8 operations (0x80-0x9B, with f64x2.nearest at 0x94)
        0x80 => SimdOp::I16x8Abs,
        0x81 => SimdOp::I16x8Neg,
        0x82 => SimdOp::I16x8Q15MulrSatS,
        0x83 => SimdOp::I16x8AllTrue,
        0x84 => SimdOp::I16x8Bitmask,
        0x85 => SimdOp::I16x8NarrowI32x4S,
        0x86 => SimdOp::I16x8NarrowI32x4U,
        0x87 => SimdOp::I16x8ExtendLowI8x16S,
        0x88 => SimdOp::I16x8ExtendHighI8x16S,
        0x89 => SimdOp::I16x8ExtendLowI8x16U,
        0x8A => SimdOp::I16x8ExtendHighI8x16U,
        0x8B => SimdOp::I16x8Shl,
        0x8C => SimdOp::I16x8ShrS,
        0x8D => SimdOp::I16x8ShrU,
        0x8E => SimdOp::I16x8Add,
        0x8F => SimdOp::I16x8AddSatS,
        0x90 => SimdOp::I16x8AddSatU,
        0x91 => SimdOp::I16x8Sub,
        0x92 => SimdOp::I16x8SubSatS,
        0x93 => SimdOp::I16x8SubSatU,
        0x94 => SimdOp::F64x2Nearest,
        0x95 => SimdOp::I16x8Mul,
        0x96 => SimdOp::I16x8MinS,
        0x97 => SimdOp::I16x8MinU,
        0x98 => SimdOp::I16x8MaxS,
        0x99 => SimdOp::I16x8MaxU,
        // 0x9A reserved
        0x9B => SimdOp::I16x8AvgrU,
        0x9C => SimdOp::I16x8ExtMulLowI8x16S,
        0x9D => SimdOp::I16x8ExtMulHighI8x16S,
        0x9E => SimdOp::I16x8ExtMulLowI8x16U,
        0x9F => SimdOp::I16x8ExtMulHighI8x16U,

        // i32x4 operations
        0xA0 => SimdOp::I32x4Abs,
        0xA1 => SimdOp::I32x4Neg,
        // 0xA2 reserved
        0xA3 => SimdOp::I32x4AllTrue,
        0xA4 => SimdOp::I32x4Bitmask,
        // 0xA5-0xA6 reserved
        0xA7 => SimdOp::I32x4ExtendLowI16x8S,
        0xA8 => SimdOp::I32x4ExtendHighI16x8S,
        0xA9 => SimdOp::I32x4ExtendLowI16x8U,
        0xAA => SimdOp::I32x4ExtendHighI16x8U,
        0xAB => SimdOp::I32x4Shl,
        0xAC => SimdOp::I32x4ShrS,
        0xAD => SimdOp::I32x4ShrU,
        0xAE => SimdOp::I32x4Add,
        // 0xAF-0xB0 reserved
        0xB1 => SimdOp::I32x4Sub,
        // 0xB2-0xB4 reserved
        0xB5 => SimdOp::I32x4Mul,
        0xB6 => SimdOp::I32x4MinS,
        0xB7 => SimdOp::I32x4MinU,
        0xB8 => SimdOp::I32x4MaxS,
        0xB9 => SimdOp::I32x4MaxU,
        0xBA => SimdOp::I32x4DotI16x8S,
        // 0xBB reserved
        0xBC => SimdOp::I32x4ExtMulLowI16x8S,
        0xBD => SimdOp::I32x4ExtMulHighI16x8S,
        0xBE => SimdOp::I32x4ExtMulLowI16x8U,
        0xBF => SimdOp::I32x4ExtMulHighI16x8U,

        // i64x2 operations
        0xC0 => SimdOp::I64x2Abs,
        0xC1 => SimdOp::I64x2Neg,
        // 0xC2 reserved
        0xC3 => SimdOp::I64x2AllTrue,
        0xC4 => SimdOp::I64x2Bitmask,
        // 0xC5-0xC6 reserved
        0xC7 => SimdOp::I64x2ExtendLowI32x4S,
        0xC8 => SimdOp::I64x2ExtendHighI32x4S,
        0xC9 => SimdOp::I64x2ExtendLowI32x4U,
        0xCA => SimdOp::I64x2ExtendHighI32x4U,
        0xCB => SimdOp::I64x2Shl,
        0xCC => SimdOp::I64x2ShrS,
        0xCD => SimdOp::I64x2ShrU,
        0xCE => SimdOp::I64x2Add,
        // 0xCF-0xD0 reserved
        0xD1 => SimdOp::I64x2Sub,
        // 0xD2-0xD4 reserved
        0xD5 => SimdOp::I64x2Mul,
        0xD6 => SimdOp::I64x2Eq,
        0xD7 => SimdOp::I64x2Ne,
        0xD8 => SimdOp::I64x2LtS,
        0xD9 => SimdOp::I64x2GtS,
        0xDA => SimdOp::I64x2LeS,
        0xDB => SimdOp::I64x2GeS,
        0xDC => SimdOp::I64x2ExtMulLowI32x4S,
        0xDD => SimdOp::I64x2ExtMulHighI32x4S,
        0xDE => SimdOp::I64x2ExtMulLowI32x4U,
        0xDF => SimdOp::I64x2ExtMulHighI32x4U,

        // f32x4 operations
        0xE0 => SimdOp::F32x4Abs,
        0xE1 => SimdOp::F32x4Neg,
        // 0xE2 reserved
        0xE3 => SimdOp::F32x4Sqrt,
        0xE4 => SimdOp::F32x4Add,
        0xE5 => SimdOp::F32x4Sub,
        0xE6 => SimdOp::F32x4Mul,
        0xE7 => SimdOp::F32x4Div,
        0xE8 => SimdOp::F32x4Min,
        0xE9 => SimdOp::F32x4Max,
        0xEA => SimdOp::F32x4PMin,
        0xEB => SimdOp::F32x4PMax,

        // f64x2 operations
        0xEC => SimdOp::F64x2Abs,
        0xED => SimdOp::F64x2Neg,
        // 0xEE reserved
        0xEF => SimdOp::F64x2Sqrt,
        0xF0 => SimdOp::F64x2Add,
        0xF1 => SimdOp::F64x2Sub,
        0xF2 => SimdOp::F64x2Mul,
        0xF3 => SimdOp::F64x2Div,
        0xF4 => SimdOp::F64x2Min,
        0xF5 => SimdOp::F64x2Max,
        0xF6 => SimdOp::F64x2PMin,
        0xF7 => SimdOp::F64x2PMax,

        // Truncation/conversion
        0xF8 => SimdOp::I32x4TruncSatF32x4S,
        0xF9 => SimdOp::I32x4TruncSatF32x4U,
        0xFA => SimdOp::F32x4ConvertI32x4S,
        0xFB => SimdOp::F32x4ConvertI32x4U,
        0xFC => SimdOp::I32x4TruncSatF64x2SZero,
        0xFD => SimdOp::I32x4TruncSatF64x2UZero,
        0xFE => SimdOp::F64x2ConvertLowI32x4S,
        0xFF => SimdOp::F64x2ConvertLowI32x4U,

        _ => return Err(DecodeError::UnknownInstruction(0xFD, subopcode)),
    };
    Ok(InstructionKind::Simd(op))
}

/// Iterator for decoding instructions from a byte stream
pub struct InstructionIterator<'a> {
    reader: &'a mut Reader,
    parse_type: ParseType,
    ended: bool,
}

/// Parsing mode for instruction iterator
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParseType {
    /// Read all instructions until end of stream
    ReadAll,
    /// Read instructions until an End instruction is encountered
    ReadTillEnd,
}

impl<'a> InstructionIterator<'a> {
    /// Create a new instruction iterator
    pub fn new(reader: &'a mut Reader, parse_type: ParseType) -> Self {
        InstructionIterator {
            reader,
            parse_type,
            ended: false,
        }
    }
}

impl<'a> Iterator for InstructionIterator<'a> {
    type Item = Result<Instruction, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ended {
            return None;
        }

        // Check if there are more bytes to read
        if !self.reader.has_at_least(1) {
            self.ended = true;
            return None;
        }

        let pos = self.reader.pos();

        // First read the opcode to check for prefix instructions
        let opcode = match self.reader.read_byte() {
            Ok(b) => b,
            Err(e) => {
                self.ended = true;
                return Some(Err(DecodeError::Io(e)));
            }
        };

        // Handle special prefix opcodes that need byte preservation
        let (kind, original_bytes) = match opcode {
            0xFC => {
                let decode_start_pos = pos; // Start from the 0xFC byte
                let subopcode = match self.reader.read_vu32() {
                    Ok(result) => result,
                    Err(e) => {
                        self.ended = true;
                        return Some(Err(DecodeError::Io(e)));
                    }
                };
                match decode_0xfc(subopcode, self.reader) {
                    Ok(kind) => {
                        let decode_end_pos = self.reader.pos();
                        let original = self.reader.bytes[decode_start_pos..decode_end_pos].to_vec();
                        (kind, original)
                    }
                    Err(e) => {
                        self.ended = true;
                        return Some(Err(e));
                    }
                }
            }
            0xFD => {
                let decode_start_pos = pos; // Start from the 0xFD byte
                let subopcode = match self.reader.read_vu32() {
                    Ok(result) => result,
                    Err(e) => {
                        self.ended = true;
                        return Some(Err(DecodeError::Io(e)));
                    }
                };
                match decode_0xfd(subopcode, self.reader) {
                    Ok(kind) => {
                        let decode_end_pos = self.reader.pos();
                        let original = self.reader.bytes[decode_start_pos..decode_end_pos].to_vec();
                        (kind, original)
                    }
                    Err(e) => {
                        self.ended = true;
                        return Some(Err(e));
                    }
                }
            }
            0x3F => {
                // memory.size with reserved byte
                let reserved = match self.reader.read_byte() {
                    Ok(b) => b,
                    Err(e) => {
                        self.ended = true;
                        return Some(Err(DecodeError::Io(e)));
                    }
                };
                if reserved != 0x00 {
                    self.ended = true;
                    return Some(Err(DecodeError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "zero byte expected",
                    ))));
                }
                let original = vec![0x3F, reserved];
                (InstructionKind::MemorySize, original)
            }
            0x40 => {
                // memory.grow with reserved byte
                let reserved = match self.reader.read_byte() {
                    Ok(b) => b,
                    Err(e) => {
                        self.ended = true;
                        return Some(Err(DecodeError::Io(e)));
                    }
                };
                if reserved != 0x00 {
                    self.ended = true;
                    return Some(Err(DecodeError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "zero byte expected",
                    ))));
                }
                let original = vec![0x40, reserved];
                (InstructionKind::MemoryGrow, original)
            }
            _ => {
                // Reset position and decode normally, capturing all bytes
                let reset_pos = pos;
                if self.reader.set_pos(reset_pos).is_err() {
                    self.ended = true;
                    return Some(Err(DecodeError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "failed to reset position",
                    ))));
                }
                let start_decode_pos = self.reader.pos();
                match InstructionKind::decode(self.reader) {
                    Ok(kind) => {
                        let end_decode_pos = self.reader.pos();
                        let original = self.reader.bytes[start_decode_pos..end_decode_pos].to_vec();
                        (kind, original)
                    }
                    Err(e) => {
                        self.ended = true;
                        return Some(Err(e));
                    }
                }
            }
        };

        let end_pos = self.reader.pos();
        let instruction = Instruction {
            kind: kind.clone(),
            position: ByteRange {
                offset: pos,
                length: end_pos - pos,
            },
            original_bytes,
        };

        // Check if we should stop parsing
        self.ended = match self.parse_type {
            ParseType::ReadAll => !self.reader.has_at_least(1),
            ParseType::ReadTillEnd => matches!(kind, InstructionKind::End),
        };

        Some(Ok(instruction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns every SimdOp variant with dummy operands.
    fn all_simd_variants() -> Vec<SimdOp> {
        // Compile-time exhaustiveness guard: adding a variant to SimdOp without
        // updating this match causes a compile error. No wildcard arm — every
        // variant must be listed explicitly.
        #[allow(unused)]
        fn _assert_exhaustive(op: &SimdOp) {
            match op {
                // Memory operations (memarg / memarg+lane)
                SimdOp::V128Load { .. } | SimdOp::V128Load8x8S { .. }
                | SimdOp::V128Load8x8U { .. } | SimdOp::V128Load16x4S { .. }
                | SimdOp::V128Load16x4U { .. } | SimdOp::V128Load32x2S { .. }
                | SimdOp::V128Load32x2U { .. } | SimdOp::V128Load8Splat { .. }
                | SimdOp::V128Load16Splat { .. } | SimdOp::V128Load32Splat { .. }
                | SimdOp::V128Load64Splat { .. } | SimdOp::V128Store { .. }
                | SimdOp::V128Load32Zero { .. } | SimdOp::V128Load64Zero { .. }
                | SimdOp::V128Load8Lane { .. } | SimdOp::V128Load16Lane { .. }
                | SimdOp::V128Load32Lane { .. } | SimdOp::V128Load64Lane { .. }
                | SimdOp::V128Store8Lane { .. } | SimdOp::V128Store16Lane { .. }
                | SimdOp::V128Store32Lane { .. } | SimdOp::V128Store64Lane { .. }
                // Const, shuffle, lane ops
                | SimdOp::V128Const { .. } | SimdOp::I8x16Shuffle { .. }
                | SimdOp::I8x16ExtractLaneS { .. } | SimdOp::I8x16ExtractLaneU { .. }
                | SimdOp::I8x16ReplaceLane { .. } | SimdOp::I16x8ExtractLaneS { .. }
                | SimdOp::I16x8ExtractLaneU { .. } | SimdOp::I16x8ReplaceLane { .. }
                | SimdOp::I32x4ExtractLane { .. } | SimdOp::I32x4ReplaceLane { .. }
                | SimdOp::I64x2ExtractLane { .. } | SimdOp::I64x2ReplaceLane { .. }
                | SimdOp::F32x4ExtractLane { .. } | SimdOp::F32x4ReplaceLane { .. }
                | SimdOp::F64x2ExtractLane { .. } | SimdOp::F64x2ReplaceLane { .. }
                // All unit variants
                | SimdOp::I8x16Swizzle | SimdOp::I8x16Splat | SimdOp::I16x8Splat
                | SimdOp::I32x4Splat | SimdOp::I64x2Splat | SimdOp::F32x4Splat
                | SimdOp::F64x2Splat
                | SimdOp::I8x16Eq | SimdOp::I8x16Ne | SimdOp::I8x16LtS | SimdOp::I8x16LtU
                | SimdOp::I8x16GtS | SimdOp::I8x16GtU | SimdOp::I8x16LeS | SimdOp::I8x16LeU
                | SimdOp::I8x16GeS | SimdOp::I8x16GeU
                | SimdOp::I16x8Eq | SimdOp::I16x8Ne | SimdOp::I16x8LtS | SimdOp::I16x8LtU
                | SimdOp::I16x8GtS | SimdOp::I16x8GtU | SimdOp::I16x8LeS | SimdOp::I16x8LeU
                | SimdOp::I16x8GeS | SimdOp::I16x8GeU
                | SimdOp::I32x4Eq | SimdOp::I32x4Ne | SimdOp::I32x4LtS | SimdOp::I32x4LtU
                | SimdOp::I32x4GtS | SimdOp::I32x4GtU | SimdOp::I32x4LeS | SimdOp::I32x4LeU
                | SimdOp::I32x4GeS | SimdOp::I32x4GeU
                | SimdOp::I64x2Eq | SimdOp::I64x2Ne | SimdOp::I64x2LtS | SimdOp::I64x2GtS
                | SimdOp::I64x2LeS | SimdOp::I64x2GeS
                | SimdOp::F32x4Eq | SimdOp::F32x4Ne | SimdOp::F32x4Lt | SimdOp::F32x4Gt
                | SimdOp::F32x4Le | SimdOp::F32x4Ge
                | SimdOp::F64x2Eq | SimdOp::F64x2Ne | SimdOp::F64x2Lt | SimdOp::F64x2Gt
                | SimdOp::F64x2Le | SimdOp::F64x2Ge
                | SimdOp::V128Not | SimdOp::V128And | SimdOp::V128AndNot | SimdOp::V128Or
                | SimdOp::V128Xor | SimdOp::V128Bitselect | SimdOp::V128AnyTrue
                | SimdOp::I8x16Abs | SimdOp::I8x16Neg | SimdOp::I8x16Popcnt
                | SimdOp::I8x16AllTrue | SimdOp::I8x16Bitmask
                | SimdOp::I8x16NarrowI16x8S | SimdOp::I8x16NarrowI16x8U
                | SimdOp::F32x4Ceil | SimdOp::F32x4Floor | SimdOp::F32x4Trunc
                | SimdOp::F32x4Nearest
                | SimdOp::I8x16Shl | SimdOp::I8x16ShrS | SimdOp::I8x16ShrU
                | SimdOp::I8x16Add | SimdOp::I8x16AddSatS | SimdOp::I8x16AddSatU
                | SimdOp::I8x16Sub | SimdOp::I8x16SubSatS | SimdOp::I8x16SubSatU
                | SimdOp::F64x2Ceil | SimdOp::F64x2Floor
                | SimdOp::I8x16MinS | SimdOp::I8x16MinU | SimdOp::I8x16MaxS
                | SimdOp::I8x16MaxU | SimdOp::F64x2Trunc | SimdOp::I8x16AvgrU
                | SimdOp::I16x8ExtAddPairwiseI8x16S | SimdOp::I16x8ExtAddPairwiseI8x16U
                | SimdOp::I32x4ExtAddPairwiseI16x8S | SimdOp::I32x4ExtAddPairwiseI16x8U
                | SimdOp::I16x8Abs | SimdOp::I16x8Neg | SimdOp::I16x8Q15MulrSatS
                | SimdOp::I16x8AllTrue | SimdOp::I16x8Bitmask
                | SimdOp::I16x8NarrowI32x4S | SimdOp::I16x8NarrowI32x4U
                | SimdOp::I16x8ExtendLowI8x16S | SimdOp::I16x8ExtendHighI8x16S
                | SimdOp::I16x8ExtendLowI8x16U | SimdOp::I16x8ExtendHighI8x16U
                | SimdOp::I16x8Shl | SimdOp::I16x8ShrS | SimdOp::I16x8ShrU
                | SimdOp::I16x8Add | SimdOp::I16x8AddSatS | SimdOp::I16x8AddSatU
                | SimdOp::I16x8Sub | SimdOp::I16x8SubSatS | SimdOp::I16x8SubSatU
                | SimdOp::F64x2Nearest | SimdOp::I16x8Mul
                | SimdOp::I16x8MinS | SimdOp::I16x8MinU | SimdOp::I16x8MaxS
                | SimdOp::I16x8MaxU | SimdOp::I16x8AvgrU
                | SimdOp::I16x8ExtMulLowI8x16S | SimdOp::I16x8ExtMulHighI8x16S
                | SimdOp::I16x8ExtMulLowI8x16U | SimdOp::I16x8ExtMulHighI8x16U
                | SimdOp::I32x4Abs | SimdOp::I32x4Neg | SimdOp::I32x4AllTrue
                | SimdOp::I32x4Bitmask
                | SimdOp::I32x4ExtendLowI16x8S | SimdOp::I32x4ExtendHighI16x8S
                | SimdOp::I32x4ExtendLowI16x8U | SimdOp::I32x4ExtendHighI16x8U
                | SimdOp::I32x4Shl | SimdOp::I32x4ShrS | SimdOp::I32x4ShrU
                | SimdOp::I32x4Add | SimdOp::I32x4Sub | SimdOp::I32x4Mul
                | SimdOp::I32x4MinS | SimdOp::I32x4MinU | SimdOp::I32x4MaxS
                | SimdOp::I32x4MaxU | SimdOp::I32x4DotI16x8S
                | SimdOp::I32x4ExtMulLowI16x8S | SimdOp::I32x4ExtMulHighI16x8S
                | SimdOp::I32x4ExtMulLowI16x8U | SimdOp::I32x4ExtMulHighI16x8U
                | SimdOp::I64x2Abs | SimdOp::I64x2Neg | SimdOp::I64x2AllTrue
                | SimdOp::I64x2Bitmask
                | SimdOp::I64x2ExtendLowI32x4S | SimdOp::I64x2ExtendHighI32x4S
                | SimdOp::I64x2ExtendLowI32x4U | SimdOp::I64x2ExtendHighI32x4U
                | SimdOp::I64x2Shl | SimdOp::I64x2ShrS | SimdOp::I64x2ShrU
                | SimdOp::I64x2Add | SimdOp::I64x2Sub | SimdOp::I64x2Mul
                | SimdOp::I64x2ExtMulLowI32x4S | SimdOp::I64x2ExtMulHighI32x4S
                | SimdOp::I64x2ExtMulLowI32x4U | SimdOp::I64x2ExtMulHighI32x4U
                | SimdOp::F32x4Abs | SimdOp::F32x4Neg | SimdOp::F32x4Sqrt
                | SimdOp::F32x4Add | SimdOp::F32x4Sub | SimdOp::F32x4Mul
                | SimdOp::F32x4Div | SimdOp::F32x4Min | SimdOp::F32x4Max
                | SimdOp::F32x4PMin | SimdOp::F32x4PMax
                | SimdOp::F64x2Abs | SimdOp::F64x2Neg | SimdOp::F64x2Sqrt
                | SimdOp::F64x2Add | SimdOp::F64x2Sub | SimdOp::F64x2Mul
                | SimdOp::F64x2Div | SimdOp::F64x2Min | SimdOp::F64x2Max
                | SimdOp::F64x2PMin | SimdOp::F64x2PMax
                | SimdOp::I32x4TruncSatF32x4S | SimdOp::I32x4TruncSatF32x4U
                | SimdOp::F32x4ConvertI32x4S | SimdOp::F32x4ConvertI32x4U
                | SimdOp::I32x4TruncSatF64x2SZero | SimdOp::I32x4TruncSatF64x2UZero
                | SimdOp::F64x2ConvertLowI32x4S | SimdOp::F64x2ConvertLowI32x4U
                | SimdOp::F32x4DemoteF64x2Zero | SimdOp::F64x2PromoteLowF32x4
                => {}
            }
        }

        let m = MemArg { align: 0, offset: 0 };

        vec![
            // Memory (memarg only)
            SimdOp::V128Load { memarg: m },
            SimdOp::V128Load8x8S { memarg: m },
            SimdOp::V128Load8x8U { memarg: m },
            SimdOp::V128Load16x4S { memarg: m },
            SimdOp::V128Load16x4U { memarg: m },
            SimdOp::V128Load32x2S { memarg: m },
            SimdOp::V128Load32x2U { memarg: m },
            SimdOp::V128Load8Splat { memarg: m },
            SimdOp::V128Load16Splat { memarg: m },
            SimdOp::V128Load32Splat { memarg: m },
            SimdOp::V128Load64Splat { memarg: m },
            SimdOp::V128Store { memarg: m },
            SimdOp::V128Load32Zero { memarg: m },
            SimdOp::V128Load64Zero { memarg: m },
            // Memory + lane
            SimdOp::V128Load8Lane { memarg: m, lane: 0 },
            SimdOp::V128Load16Lane { memarg: m, lane: 0 },
            SimdOp::V128Load32Lane { memarg: m, lane: 0 },
            SimdOp::V128Load64Lane { memarg: m, lane: 0 },
            SimdOp::V128Store8Lane { memarg: m, lane: 0 },
            SimdOp::V128Store16Lane { memarg: m, lane: 0 },
            SimdOp::V128Store32Lane { memarg: m, lane: 0 },
            SimdOp::V128Store64Lane { memarg: m, lane: 0 },
            // Const and shuffle (16-byte payloads)
            SimdOp::V128Const { value: [0; 16] },
            SimdOp::I8x16Shuffle { lanes: [0; 16] },
            // Splat / swizzle (no operands)
            SimdOp::I8x16Swizzle,
            SimdOp::I8x16Splat,
            SimdOp::I16x8Splat,
            SimdOp::I32x4Splat,
            SimdOp::I64x2Splat,
            SimdOp::F32x4Splat,
            SimdOp::F64x2Splat,
            // Lane extract/replace (1-byte lane)
            SimdOp::I8x16ExtractLaneS { lane: 0 },
            SimdOp::I8x16ExtractLaneU { lane: 0 },
            SimdOp::I8x16ReplaceLane { lane: 0 },
            SimdOp::I16x8ExtractLaneS { lane: 0 },
            SimdOp::I16x8ExtractLaneU { lane: 0 },
            SimdOp::I16x8ReplaceLane { lane: 0 },
            SimdOp::I32x4ExtractLane { lane: 0 },
            SimdOp::I32x4ReplaceLane { lane: 0 },
            SimdOp::I64x2ExtractLane { lane: 0 },
            SimdOp::I64x2ReplaceLane { lane: 0 },
            SimdOp::F32x4ExtractLane { lane: 0 },
            SimdOp::F32x4ReplaceLane { lane: 0 },
            SimdOp::F64x2ExtractLane { lane: 0 },
            SimdOp::F64x2ReplaceLane { lane: 0 },
            // i8x16 comparisons
            SimdOp::I8x16Eq,
            SimdOp::I8x16Ne,
            SimdOp::I8x16LtS,
            SimdOp::I8x16LtU,
            SimdOp::I8x16GtS,
            SimdOp::I8x16GtU,
            SimdOp::I8x16LeS,
            SimdOp::I8x16LeU,
            SimdOp::I8x16GeS,
            SimdOp::I8x16GeU,
            // i16x8 comparisons
            SimdOp::I16x8Eq,
            SimdOp::I16x8Ne,
            SimdOp::I16x8LtS,
            SimdOp::I16x8LtU,
            SimdOp::I16x8GtS,
            SimdOp::I16x8GtU,
            SimdOp::I16x8LeS,
            SimdOp::I16x8LeU,
            SimdOp::I16x8GeS,
            SimdOp::I16x8GeU,
            // i32x4 comparisons
            SimdOp::I32x4Eq,
            SimdOp::I32x4Ne,
            SimdOp::I32x4LtS,
            SimdOp::I32x4LtU,
            SimdOp::I32x4GtS,
            SimdOp::I32x4GtU,
            SimdOp::I32x4LeS,
            SimdOp::I32x4LeU,
            SimdOp::I32x4GeS,
            SimdOp::I32x4GeU,
            // f32x4 comparisons
            SimdOp::F32x4Eq,
            SimdOp::F32x4Ne,
            SimdOp::F32x4Lt,
            SimdOp::F32x4Gt,
            SimdOp::F32x4Le,
            SimdOp::F32x4Ge,
            // f64x2 comparisons
            SimdOp::F64x2Eq,
            SimdOp::F64x2Ne,
            SimdOp::F64x2Lt,
            SimdOp::F64x2Gt,
            SimdOp::F64x2Le,
            SimdOp::F64x2Ge,
            // v128 bitwise
            SimdOp::V128Not,
            SimdOp::V128And,
            SimdOp::V128AndNot,
            SimdOp::V128Or,
            SimdOp::V128Xor,
            SimdOp::V128Bitselect,
            SimdOp::V128AnyTrue,
            // Conversions (no operands)
            SimdOp::F32x4DemoteF64x2Zero,
            SimdOp::F64x2PromoteLowF32x4,
            // i8x16 operations
            SimdOp::I8x16Abs,
            SimdOp::I8x16Neg,
            SimdOp::I8x16Popcnt,
            SimdOp::I8x16AllTrue,
            SimdOp::I8x16Bitmask,
            SimdOp::I8x16NarrowI16x8S,
            SimdOp::I8x16NarrowI16x8U,
            SimdOp::F32x4Ceil,
            SimdOp::F32x4Floor,
            SimdOp::F32x4Trunc,
            SimdOp::F32x4Nearest,
            SimdOp::I8x16Shl,
            SimdOp::I8x16ShrS,
            SimdOp::I8x16ShrU,
            SimdOp::I8x16Add,
            SimdOp::I8x16AddSatS,
            SimdOp::I8x16AddSatU,
            SimdOp::I8x16Sub,
            SimdOp::I8x16SubSatS,
            SimdOp::I8x16SubSatU,
            SimdOp::F64x2Ceil,
            SimdOp::F64x2Floor,
            SimdOp::I8x16MinS,
            SimdOp::I8x16MinU,
            SimdOp::I8x16MaxS,
            SimdOp::I8x16MaxU,
            SimdOp::F64x2Trunc,
            SimdOp::I8x16AvgrU,
            // Pairwise extension
            SimdOp::I16x8ExtAddPairwiseI8x16S,
            SimdOp::I16x8ExtAddPairwiseI8x16U,
            SimdOp::I32x4ExtAddPairwiseI16x8S,
            SimdOp::I32x4ExtAddPairwiseI16x8U,
            // i16x8 operations
            SimdOp::I16x8Abs,
            SimdOp::I16x8Neg,
            SimdOp::I16x8Q15MulrSatS,
            SimdOp::I16x8AllTrue,
            SimdOp::I16x8Bitmask,
            SimdOp::I16x8NarrowI32x4S,
            SimdOp::I16x8NarrowI32x4U,
            SimdOp::I16x8ExtendLowI8x16S,
            SimdOp::I16x8ExtendHighI8x16S,
            SimdOp::I16x8ExtendLowI8x16U,
            SimdOp::I16x8ExtendHighI8x16U,
            SimdOp::I16x8Shl,
            SimdOp::I16x8ShrS,
            SimdOp::I16x8ShrU,
            SimdOp::I16x8Add,
            SimdOp::I16x8AddSatS,
            SimdOp::I16x8AddSatU,
            SimdOp::I16x8Sub,
            SimdOp::I16x8SubSatS,
            SimdOp::I16x8SubSatU,
            SimdOp::F64x2Nearest,
            SimdOp::I16x8Mul,
            SimdOp::I16x8MinS,
            SimdOp::I16x8MinU,
            SimdOp::I16x8MaxS,
            SimdOp::I16x8MaxU,
            SimdOp::I16x8AvgrU,
            SimdOp::I16x8ExtMulLowI8x16S,
            SimdOp::I16x8ExtMulHighI8x16S,
            SimdOp::I16x8ExtMulLowI8x16U,
            SimdOp::I16x8ExtMulHighI8x16U,
            // i32x4 operations
            SimdOp::I32x4Abs,
            SimdOp::I32x4Neg,
            SimdOp::I32x4AllTrue,
            SimdOp::I32x4Bitmask,
            SimdOp::I32x4ExtendLowI16x8S,
            SimdOp::I32x4ExtendHighI16x8S,
            SimdOp::I32x4ExtendLowI16x8U,
            SimdOp::I32x4ExtendHighI16x8U,
            SimdOp::I32x4Shl,
            SimdOp::I32x4ShrS,
            SimdOp::I32x4ShrU,
            SimdOp::I32x4Add,
            SimdOp::I32x4Sub,
            SimdOp::I32x4Mul,
            SimdOp::I32x4MinS,
            SimdOp::I32x4MinU,
            SimdOp::I32x4MaxS,
            SimdOp::I32x4MaxU,
            SimdOp::I32x4DotI16x8S,
            SimdOp::I32x4ExtMulLowI16x8S,
            SimdOp::I32x4ExtMulHighI16x8S,
            SimdOp::I32x4ExtMulLowI16x8U,
            SimdOp::I32x4ExtMulHighI16x8U,
            // i64x2 operations
            SimdOp::I64x2Abs,
            SimdOp::I64x2Neg,
            SimdOp::I64x2AllTrue,
            SimdOp::I64x2Bitmask,
            SimdOp::I64x2ExtendLowI32x4S,
            SimdOp::I64x2ExtendHighI32x4S,
            SimdOp::I64x2ExtendLowI32x4U,
            SimdOp::I64x2ExtendHighI32x4U,
            SimdOp::I64x2Shl,
            SimdOp::I64x2ShrS,
            SimdOp::I64x2ShrU,
            SimdOp::I64x2Add,
            SimdOp::I64x2Sub,
            SimdOp::I64x2Mul,
            SimdOp::I64x2Eq,
            SimdOp::I64x2Ne,
            SimdOp::I64x2LtS,
            SimdOp::I64x2GtS,
            SimdOp::I64x2LeS,
            SimdOp::I64x2GeS,
            SimdOp::I64x2ExtMulLowI32x4S,
            SimdOp::I64x2ExtMulHighI32x4S,
            SimdOp::I64x2ExtMulLowI32x4U,
            SimdOp::I64x2ExtMulHighI32x4U,
            // f32x4 operations
            SimdOp::F32x4Abs,
            SimdOp::F32x4Neg,
            SimdOp::F32x4Sqrt,
            SimdOp::F32x4Add,
            SimdOp::F32x4Sub,
            SimdOp::F32x4Mul,
            SimdOp::F32x4Div,
            SimdOp::F32x4Min,
            SimdOp::F32x4Max,
            SimdOp::F32x4PMin,
            SimdOp::F32x4PMax,
            // f64x2 operations
            SimdOp::F64x2Abs,
            SimdOp::F64x2Neg,
            SimdOp::F64x2Sqrt,
            SimdOp::F64x2Add,
            SimdOp::F64x2Sub,
            SimdOp::F64x2Mul,
            SimdOp::F64x2Div,
            SimdOp::F64x2Min,
            SimdOp::F64x2Max,
            SimdOp::F64x2PMin,
            SimdOp::F64x2PMax,
            // Truncation/conversion
            SimdOp::I32x4TruncSatF32x4S,
            SimdOp::I32x4TruncSatF32x4U,
            SimdOp::F32x4ConvertI32x4S,
            SimdOp::F32x4ConvertI32x4U,
            SimdOp::I32x4TruncSatF64x2SZero,
            SimdOp::I32x4TruncSatF64x2UZero,
            SimdOp::F64x2ConvertLowI32x4S,
            SimdOp::F64x2ConvertLowI32x4U,
        ]
    }

    #[test]
    fn simd_subopcode_roundtrip() {
        let variants = all_simd_variants();
        assert_eq!(
            variants.len(),
            236,
            "Variant count changed. Update all_simd_variants() and _assert_exhaustive().",
        );

        // Verify no duplicate subopcodes (catches copy-paste errors in subopcode())
        let mut seen_opcodes = std::collections::HashMap::new();
        for op in &variants {
            let code = op.subopcode();
            if let Some(prev_mnemonic) = seen_opcodes.insert(code, op.mnemonic()) {
                panic!(
                    "Duplicate subopcode 0x{:02X}: {} and {}",
                    code,
                    prev_mnemonic,
                    op.mnemonic()
                );
            }
        }

        // Round-trip: encode operands → decode via decode_0xfd → verify same subopcode
        for op in &variants {
            let original_subopcode = op.subopcode();
            let original_mnemonic = op.mnemonic();

            // Encode operands only (subopcode is passed separately to decode_0xfd)
            let mut operand_bytes = Vec::new();
            op.encode_operands(&mut operand_bytes);

            // Decode through the real decoder
            let mut reader = Reader::new(operand_bytes);
            let decoded = decode_0xfd(original_subopcode, &mut reader).unwrap_or_else(|e| {
                panic!(
                    "decode_0xfd failed for {} (0x{:02X}): {:?}",
                    original_mnemonic, original_subopcode, e
                )
            });

            // Extract the SimdOp and verify subopcode matches
            let decoded_op = match decoded {
                InstructionKind::Simd(op) => op,
                other => panic!(
                    "Expected Simd variant for {} (0x{:02X}), got {:?}",
                    original_mnemonic, original_subopcode, other
                ),
            };

            assert_eq!(
                decoded_op.subopcode(),
                original_subopcode,
                "Subopcode drift for {}: encode says 0x{:02X}, decode produces {} (0x{:02X})",
                original_mnemonic,
                original_subopcode,
                decoded_op.mnemonic(),
                decoded_op.subopcode()
            );

            // Also verify mnemonic matches (catches variant name mismatches)
            assert_eq!(
                decoded_op.mnemonic(),
                original_mnemonic,
                "Mnemonic drift at subopcode 0x{:02X}: encode says {}, decode produces {}",
                original_subopcode,
                original_mnemonic,
                decoded_op.mnemonic()
            );

            // Verify all operand bytes were consumed
            assert!(
                !reader.has_at_least(1),
                "{} (0x{:02X}): {} unconsumed byte(s) after decode",
                original_mnemonic,
                original_subopcode,
                reader.remaining()
            );
        }
    }
}
