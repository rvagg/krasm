//! The operand and result types of each instruction.
//!
//! Most instructions have one type signature, whatever the module or stack:
//! `i32.add` is `[i32 i32] -> [i32]`, `v128.load` is `[i32] -> [v128]`.
//! [`fixed_signature`] gives these.
//!
//! Others depend on what their immediate names. `local.get 2` pushes whatever
//! type local 2 was declared with, `call 5` takes function 5's parameters and
//! returns its results, and `table.get 0` returns table 0's element type.
//! [`context_signature`] looks these up.
//!
//! That leaves `drop`, `select` and `ref.is_null`, whose types come from the
//! operands on the stack, for each consumer to handle. Validation and
//! compilation both read these signatures, so they cannot disagree about what
//! an instruction consumes and produces.

use super::{InstructionKind, SimdOp};
use crate::parser::index_spaces::IndexSpaceTypes;
use crate::parser::module::ValueType::{self, *};
use crate::parser::module::{FunctionType, RefType, TypeSection};
use crate::parser::validate::ValidationError;
use InstructionKind::*;
use SimdOp::*;

/// Parameter and result types, parameters in stack order: the last is on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signature<'a> {
    pub params: &'a [ValueType],
    pub results: &'a [ValueType],
}

impl FunctionType {
    pub fn signature(&self) -> Signature<'_> {
        Signature {
            params: &self.parameters,
            results: &self.return_types,
        }
    }
}

const fn sig(params: &'static [ValueType], results: &'static [ValueType]) -> Option<Signature<'static>> {
    Some(Signature { params, results })
}

/// The signature of an instruction whose types depend only on its opcode,
/// or `None` when they depend on the module or the operand stack.
pub fn fixed_signature(kind: &InstructionKind) -> Option<Signature<'static>> {
    match kind {
        // Types come from labels, blocks, callees, locals, globals, tables or the stack.
        Unreachable
        | Block { .. }
        | Loop { .. }
        | If { .. }
        | Else
        | End
        | Br { .. }
        | BrIf { .. }
        | BrTable { .. }
        | Return
        | Call { .. }
        | CallIndirect { .. }
        | RefNull { .. }
        | RefIsNull
        | Drop
        | Select
        | SelectTyped { .. }
        | LocalGet { .. }
        | LocalSet { .. }
        | LocalTee { .. }
        | GlobalGet { .. }
        | GlobalSet { .. }
        | TableGet { .. }
        | TableSet { .. }
        | TableGrow { .. }
        | TableFill { .. } => None,

        Nop | ElemDrop { .. } | DataDrop { .. } => sig(&[], &[]),
        RefFunc { .. } => sig(&[], &[FuncRef]),
        TableSize { .. } | MemorySize => sig(&[], &[I32]),
        TableInit { .. } | TableCopy { .. } | MemoryInit { .. } | MemoryCopy | MemoryFill => sig(&[I32, I32, I32], &[]),
        MemoryGrow => sig(&[I32], &[I32]),

        I32Load { .. } | I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } => {
            sig(&[I32], &[I32])
        }
        I64Load { .. }
        | I64Load8S { .. }
        | I64Load8U { .. }
        | I64Load16S { .. }
        | I64Load16U { .. }
        | I64Load32S { .. }
        | I64Load32U { .. } => sig(&[I32], &[I64]),
        F32Load { .. } => sig(&[I32], &[F32]),
        F64Load { .. } => sig(&[I32], &[F64]),
        I32Store { .. } | I32Store8 { .. } | I32Store16 { .. } => sig(&[I32, I32], &[]),
        I64Store { .. } | I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } => sig(&[I32, I64], &[]),
        F32Store { .. } => sig(&[I32, F32], &[]),
        F64Store { .. } => sig(&[I32, F64], &[]),

        I32Const { .. } => sig(&[], &[I32]),
        I64Const { .. } => sig(&[], &[I64]),
        F32Const { .. } => sig(&[], &[F32]),
        F64Const { .. } => sig(&[], &[F64]),

        I32Eqz | I32Clz | I32Ctz | I32Popcnt | I32Extend8S | I32Extend16S => sig(&[I32], &[I32]),
        I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU | I32Add | I32Sub
        | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or | I32Xor | I32Shl | I32ShrS | I32ShrU
        | I32Rotl | I32Rotr => sig(&[I32, I32], &[I32]),

        I64Eqz | I32WrapI64 => sig(&[I64], &[I32]),
        I64Clz | I64Ctz | I64Popcnt | I64Extend8S | I64Extend16S | I64Extend32S => sig(&[I64], &[I64]),
        I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS | I64GeU => {
            sig(&[I64, I64], &[I32])
        }
        I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And | I64Or | I64Xor | I64Shl
        | I64ShrS | I64ShrU | I64Rotl | I64Rotr => sig(&[I64, I64], &[I64]),

        F32Abs | F32Neg | F32Ceil | F32Floor | F32Trunc | F32Nearest | F32Sqrt => sig(&[F32], &[F32]),
        F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge => sig(&[F32, F32], &[I32]),
        F32Add | F32Sub | F32Mul | F32Div | F32Min | F32Max | F32Copysign => sig(&[F32, F32], &[F32]),

        F64Abs | F64Neg | F64Ceil | F64Floor | F64Trunc | F64Nearest | F64Sqrt => sig(&[F64], &[F64]),
        F64Eq | F64Ne | F64Lt | F64Gt | F64Le | F64Ge => sig(&[F64, F64], &[I32]),
        F64Add | F64Sub | F64Mul | F64Div | F64Min | F64Max | F64Copysign => sig(&[F64, F64], &[F64]),

        I32TruncF32S | I32TruncF32U | I32TruncSatF32S | I32TruncSatF32U | I32ReinterpretF32 => sig(&[F32], &[I32]),
        I32TruncF64S | I32TruncF64U | I32TruncSatF64S | I32TruncSatF64U => sig(&[F64], &[I32]),
        I64ExtendI32S | I64ExtendI32U => sig(&[I32], &[I64]),
        I64TruncF32S | I64TruncF32U | I64TruncSatF32S | I64TruncSatF32U => sig(&[F32], &[I64]),
        I64TruncF64S | I64TruncF64U | I64TruncSatF64S | I64TruncSatF64U | I64ReinterpretF64 => sig(&[F64], &[I64]),
        F32ConvertI32S | F32ConvertI32U | F32ReinterpretI32 => sig(&[I32], &[F32]),
        F32ConvertI64S | F32ConvertI64U => sig(&[I64], &[F32]),
        F32DemoteF64 => sig(&[F64], &[F32]),
        F64ConvertI32S | F64ConvertI32U => sig(&[I32], &[F64]),
        F64ConvertI64S | F64ConvertI64U | F64ReinterpretI64 => sig(&[I64], &[F64]),
        F64PromoteF32 => sig(&[F32], &[F64]),

        Simd(op) => Some(simd_signature(op)),
    }
}

/// The signature of a local, global, table or call instruction, found by
/// looking up the local, global, table or function its immediate names.
/// `None` for any other instruction. `locals` holds the function's parameter
/// then declared local types, by local index.
pub fn context_signature<'a>(
    kind: &InstructionKind,
    types: &'a TypeSection,
    spaces: &IndexSpaceTypes,
    locals: &[ValueType],
) -> Result<Option<ContextSignature<'a>>, ValidationError> {
    let local = |index: u32| {
        (locals.get(index as usize).copied())
            .map(one)
            .ok_or(ValidationError::UnknownLocal(index))
    };
    let global = |index: u32| (spaces.global_types.get(index as usize)).ok_or(ValidationError::UnknownGlobal(index));
    let table = |index: u32| {
        (spaces.table_types.get(index as usize).map(|table| table.ref_type)).ok_or(ValidationError::UnknownTable)
    };
    let callee = |type_idx: u32| types.get(type_idx).ok_or(ValidationError::UnknownFunctionType);
    let signature = match kind {
        LocalGet { local_idx } => Signature {
            params: &[],
            results: local(*local_idx)?,
        },
        LocalSet { local_idx } => Signature {
            params: local(*local_idx)?,
            results: &[],
        },
        LocalTee { local_idx } => {
            let local = local(*local_idx)?;
            Signature {
                params: local,
                results: local,
            }
        }
        GlobalGet { global_idx } => Signature {
            params: &[],
            results: one(global(*global_idx)?.value_type),
        },
        GlobalSet { global_idx } => {
            let global = global(*global_idx)?;
            if !global.mutable {
                return Err(ValidationError::GlobalIsImmutable);
            }
            Signature {
                params: one(global.value_type),
                results: &[],
            }
        }
        TableGet { table_idx } => Signature {
            params: &[I32],
            results: one(table(*table_idx)?.into()),
        },
        TableSet { table_idx } => Signature {
            params: match table(*table_idx)? {
                RefType::FuncRef => &[I32, FuncRef],
                RefType::ExternRef => &[I32, ExternRef],
            },
            results: &[],
        },
        TableGrow { table_idx } => Signature {
            params: match table(*table_idx)? {
                RefType::FuncRef => &[FuncRef, I32],
                RefType::ExternRef => &[ExternRef, I32],
            },
            results: &[I32],
        },
        TableFill { table_idx } => Signature {
            params: match table(*table_idx)? {
                RefType::FuncRef => &[I32, FuncRef, I32],
                RefType::ExternRef => &[I32, ExternRef, I32],
            },
            results: &[],
        },
        Call { func_idx } => {
            let type_idx = (spaces.function_type_indices.get(*func_idx as usize))
                .ok_or(ValidationError::UnknownFunction(*func_idx))?;
            callee(*type_idx)?.signature()
        }
        CallIndirect { type_idx, table_idx } => {
            let element = table(*table_idx)?;
            let signature = callee(*type_idx)?.signature();
            if element != RefType::FuncRef {
                return Err(ValidationError::TypeMismatch);
            }
            return Ok(Some(ContextSignature::Indirect(signature)));
        }
        _ => return Ok(None),
    };
    Ok(Some(ContextSignature::Direct(signature)))
}

/// The signature [`context_signature`] finds for an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextSignature<'a> {
    Direct(Signature<'a>),
    /// `call_indirect`: the callee's signature. The instruction also pops an
    /// i32 table index, which sits above the callee's arguments.
    Indirect(Signature<'a>),
}

impl<'a> ContextSignature<'a> {
    /// The signatures to apply in order: `call_indirect` first pops its table
    /// index, then the callee's arguments.
    pub fn steps(self) -> impl Iterator<Item = Signature<'a>> {
        let (table_index, signature) = match self {
            ContextSignature::Direct(signature) => (None, signature),
            ContextSignature::Indirect(signature) => (sig(&[I32], &[]), signature),
        };
        table_index.into_iter().chain([signature])
    }
}

/// A one-type slice with a static lifetime, so a signature built from a
/// local, global or table type borrows nothing.
fn one(value_type: ValueType) -> &'static [ValueType] {
    match value_type {
        I32 => &[I32],
        I64 => &[I64],
        F32 => &[F32],
        F64 => &[F64],
        V128 => &[V128],
        FuncRef => &[FuncRef],
        ExternRef => &[ExternRef],
    }
}

/// Every SIMD instruction has a fixed signature.
pub(crate) fn simd_signature(op: &SimdOp) -> Signature<'static> {
    let (params, results): (&'static [ValueType], &'static [ValueType]) = match op {
        V128Load { .. }
        | V128Load8x8S { .. }
        | V128Load8x8U { .. }
        | V128Load16x4S { .. }
        | V128Load16x4U { .. }
        | V128Load32x2S { .. }
        | V128Load32x2U { .. }
        | V128Load8Splat { .. }
        | V128Load16Splat { .. }
        | V128Load32Splat { .. }
        | V128Load64Splat { .. }
        | V128Load32Zero { .. }
        | V128Load64Zero { .. } => (&[I32], &[V128]),
        V128Store { .. }
        | V128Store8Lane { .. }
        | V128Store16Lane { .. }
        | V128Store32Lane { .. }
        | V128Store64Lane { .. } => (&[I32, V128], &[]),
        V128Load8Lane { .. } | V128Load16Lane { .. } | V128Load32Lane { .. } | V128Load64Lane { .. } => {
            (&[I32, V128], &[V128])
        }
        V128Const { .. } => (&[], &[V128]),

        I8x16Splat | I16x8Splat | I32x4Splat => (&[I32], &[V128]),
        I64x2Splat => (&[I64], &[V128]),
        F32x4Splat => (&[F32], &[V128]),
        F64x2Splat => (&[F64], &[V128]),

        I8x16ExtractLaneS { .. }
        | I8x16ExtractLaneU { .. }
        | I16x8ExtractLaneS { .. }
        | I16x8ExtractLaneU { .. }
        | I32x4ExtractLane { .. } => (&[V128], &[I32]),
        I64x2ExtractLane { .. } => (&[V128], &[I64]),
        F32x4ExtractLane { .. } => (&[V128], &[F32]),
        F64x2ExtractLane { .. } => (&[V128], &[F64]),
        I8x16ReplaceLane { .. } | I16x8ReplaceLane { .. } | I32x4ReplaceLane { .. } => (&[V128, I32], &[V128]),
        I64x2ReplaceLane { .. } => (&[V128, I64], &[V128]),
        F32x4ReplaceLane { .. } => (&[V128, F32], &[V128]),
        F64x2ReplaceLane { .. } => (&[V128, F64], &[V128]),

        V128AnyTrue | I8x16AllTrue | I8x16Bitmask | I16x8AllTrue | I16x8Bitmask | I32x4AllTrue | I32x4Bitmask
        | I64x2AllTrue | I64x2Bitmask => (&[V128], &[I32]),

        I8x16Shl | I8x16ShrS | I8x16ShrU | I16x8Shl | I16x8ShrS | I16x8ShrU | I32x4Shl | I32x4ShrS | I32x4ShrU
        | I64x2Shl | I64x2ShrS | I64x2ShrU => (&[V128, I32], &[V128]),

        V128Bitselect => (&[V128, V128, V128], &[V128]),

        V128Not
        | I8x16Abs
        | I8x16Neg
        | I8x16Popcnt
        | I16x8Abs
        | I16x8Neg
        | I16x8ExtendLowI8x16S
        | I16x8ExtendHighI8x16S
        | I16x8ExtendLowI8x16U
        | I16x8ExtendHighI8x16U
        | I16x8ExtAddPairwiseI8x16S
        | I16x8ExtAddPairwiseI8x16U
        | I32x4Abs
        | I32x4Neg
        | I32x4ExtendLowI16x8S
        | I32x4ExtendHighI16x8S
        | I32x4ExtendLowI16x8U
        | I32x4ExtendHighI16x8U
        | I32x4ExtAddPairwiseI16x8S
        | I32x4ExtAddPairwiseI16x8U
        | I64x2Abs
        | I64x2Neg
        | I64x2ExtendLowI32x4S
        | I64x2ExtendHighI32x4S
        | I64x2ExtendLowI32x4U
        | I64x2ExtendHighI32x4U
        | F32x4Abs
        | F32x4Neg
        | F32x4Sqrt
        | F32x4Ceil
        | F32x4Floor
        | F32x4Trunc
        | F32x4Nearest
        | F64x2Abs
        | F64x2Neg
        | F64x2Sqrt
        | F64x2Ceil
        | F64x2Floor
        | F64x2Trunc
        | F64x2Nearest
        | I32x4TruncSatF32x4S
        | I32x4TruncSatF32x4U
        | F32x4ConvertI32x4S
        | F32x4ConvertI32x4U
        | I32x4TruncSatF64x2SZero
        | I32x4TruncSatF64x2UZero
        | F64x2ConvertLowI32x4S
        | F64x2ConvertLowI32x4U
        | F32x4DemoteF64x2Zero
        | F64x2PromoteLowF32x4 => (&[V128], &[V128]),

        I8x16Shuffle { .. }
        | I8x16Swizzle
        | I8x16Eq
        | I8x16Ne
        | I8x16LtS
        | I8x16LtU
        | I8x16GtS
        | I8x16GtU
        | I8x16LeS
        | I8x16LeU
        | I8x16GeS
        | I8x16GeU
        | I16x8Eq
        | I16x8Ne
        | I16x8LtS
        | I16x8LtU
        | I16x8GtS
        | I16x8GtU
        | I16x8LeS
        | I16x8LeU
        | I16x8GeS
        | I16x8GeU
        | I32x4Eq
        | I32x4Ne
        | I32x4LtS
        | I32x4LtU
        | I32x4GtS
        | I32x4GtU
        | I32x4LeS
        | I32x4LeU
        | I32x4GeS
        | I32x4GeU
        | I64x2Eq
        | I64x2Ne
        | I64x2LtS
        | I64x2GtS
        | I64x2LeS
        | I64x2GeS
        | F32x4Eq
        | F32x4Ne
        | F32x4Lt
        | F32x4Gt
        | F32x4Le
        | F32x4Ge
        | F64x2Eq
        | F64x2Ne
        | F64x2Lt
        | F64x2Gt
        | F64x2Le
        | F64x2Ge
        | V128And
        | V128AndNot
        | V128Or
        | V128Xor
        | I8x16NarrowI16x8S
        | I8x16NarrowI16x8U
        | I16x8NarrowI32x4S
        | I16x8NarrowI32x4U
        | I8x16Add
        | I8x16AddSatS
        | I8x16AddSatU
        | I8x16Sub
        | I8x16SubSatS
        | I8x16SubSatU
        | I8x16MinS
        | I8x16MinU
        | I8x16MaxS
        | I8x16MaxU
        | I8x16AvgrU
        | I16x8Add
        | I16x8AddSatS
        | I16x8AddSatU
        | I16x8Sub
        | I16x8SubSatS
        | I16x8SubSatU
        | I16x8Mul
        | I16x8MinS
        | I16x8MinU
        | I16x8MaxS
        | I16x8MaxU
        | I16x8AvgrU
        | I16x8Q15MulrSatS
        | I16x8ExtMulLowI8x16S
        | I16x8ExtMulHighI8x16S
        | I16x8ExtMulLowI8x16U
        | I16x8ExtMulHighI8x16U
        | I32x4Add
        | I32x4Sub
        | I32x4Mul
        | I32x4MinS
        | I32x4MinU
        | I32x4MaxS
        | I32x4MaxU
        | I32x4DotI16x8S
        | I32x4ExtMulLowI16x8S
        | I32x4ExtMulHighI16x8S
        | I32x4ExtMulLowI16x8U
        | I32x4ExtMulHighI16x8U
        | I64x2Add
        | I64x2Sub
        | I64x2Mul
        | I64x2ExtMulLowI32x4S
        | I64x2ExtMulHighI32x4S
        | I64x2ExtMulLowI32x4U
        | I64x2ExtMulHighI32x4U
        | F32x4Add
        | F32x4Sub
        | F32x4Mul
        | F32x4Div
        | F32x4Min
        | F32x4Max
        | F32x4PMin
        | F32x4PMax
        | F64x2Add
        | F64x2Sub
        | F64x2Mul
        | F64x2Div
        | F64x2Min
        | F64x2Max
        | F64x2PMin
        | F64x2PMax => (&[V128, V128], &[V128]),
    };
    Signature { params, results }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::instruction::MemArg;

    fn fixed(kind: InstructionKind) -> (&'static [ValueType], &'static [ValueType]) {
        let signature = fixed_signature(&kind).expect("fixed signature");
        (signature.params, signature.results)
    }

    #[test]
    fn parameters_are_in_stack_order() {
        let memarg = MemArg { align: 3, offset: 0 };
        // The address is pushed first, so the stored value is on top.
        assert_eq!(fixed(InstructionKind::I64Store { memarg }), (&[I32, I64][..], &[][..]));
        assert_eq!(
            fixed(InstructionKind::Simd(SimdOp::I32x4Shl)),
            (&[V128, I32][..], &[V128][..])
        );
        assert_eq!(
            fixed(InstructionKind::Simd(SimdOp::F64x2ReplaceLane { lane: 1 })),
            (&[V128, F64][..], &[V128][..])
        );
        assert_eq!(fixed(InstructionKind::I64Eqz), (&[I64][..], &[I32][..]));
    }

    /// The params and results of each step of `kind`'s context signature in
    /// the first function of `source`.
    type Steps = Vec<(Vec<ValueType>, Vec<ValueType>)>;
    fn context(source: &str, kind: InstructionKind) -> Result<Steps, ValidationError> {
        let module = crate::wat::parse(source).expect("WAT parse failed");
        let ftype = module.get_function_type(0).expect("a function");
        let locals = module.code.code[0].locals.types(&ftype.parameters);
        let signature = context_signature(&kind, &module.types, module.index_space_types(), &locals)?;
        let steps = signature.expect("a context signature").steps();
        Ok(steps
            .map(|step| (step.params.to_vec(), step.results.to_vec()))
            .collect())
    }

    #[test]
    fn context_signatures_read_locals_globals_tables_and_callees() {
        let source = "(module
            (type $t (func (param i64) (result f32)))
            (table 1 externref)
            (global i32 (i32.const 0))
            (func (param i32) (local f64)))";
        let steps = |kind| context(source, kind).unwrap();
        let one = |params: &[ValueType], results: &[ValueType]| vec![(params.to_vec(), results.to_vec())];

        assert_eq!(steps(InstructionKind::LocalTee { local_idx: 1 }), one(&[F64], &[F64]));
        assert_eq!(steps(InstructionKind::GlobalGet { global_idx: 0 }), one(&[], &[I32]));
        assert_eq!(
            steps(InstructionKind::TableFill { table_idx: 0 }),
            one(&[I32, ExternRef, I32], &[])
        );
        assert_eq!(steps(InstructionKind::Call { func_idx: 0 }), one(&[I32], &[]));
    }

    #[test]
    fn call_indirect_pops_its_table_index_before_the_arguments() {
        let source = "(module (type (func (param i64) (result f32))) (table 1 funcref) (func))";
        let kind = InstructionKind::CallIndirect {
            type_idx: 0,
            table_idx: 0,
        };
        assert_eq!(
            context(source, kind).unwrap(),
            vec![(vec![I32], vec![]), (vec![I64], vec![F32])]
        );
    }

    #[test]
    fn context_signatures_reject_what_the_context_forbids() {
        let source = "(module (table 1 externref) (global i32 (i32.const 0)) (func (local i32)))";
        let error = |kind| context(source, kind).unwrap_err().to_string();
        assert_eq!(error(InstructionKind::LocalGet { local_idx: 1 }), "unknown local 1");
        assert_eq!(
            error(InstructionKind::GlobalSet { global_idx: 0 }),
            "global is immutable"
        );
        assert_eq!(error(InstructionKind::TableGet { table_idx: 1 }), "unknown table");
        let call_indirect = InstructionKind::CallIndirect {
            type_idx: 0,
            table_idx: 0,
        };
        assert_eq!(error(call_indirect), "type mismatch");
    }

    #[test]
    fn context_dependent_instructions_have_no_fixed_signature() {
        for kind in [
            InstructionKind::Drop,
            InstructionKind::Select,
            InstructionKind::LocalGet { local_idx: 0 },
            InstructionKind::GlobalSet { global_idx: 0 },
            InstructionKind::Call { func_idx: 0 },
            InstructionKind::TableGet { table_idx: 0 },
            InstructionKind::RefNull {
                ref_type: ValueType::FuncRef,
            },
            InstructionKind::Unreachable,
        ] {
            assert_eq!(fixed_signature(&kind), None, "{kind:?}");
        }
    }
}
