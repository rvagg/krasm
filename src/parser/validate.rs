//! Module-level type validation for WebAssembly.
//!
//! The binary parser validates structural constraints inline as it decodes
//! (section ordering, index bounds, count consistency). This module performs
//! the deeper type-level validation that the WAT parser needs after building
//! a [`Module`] from text: operand stack typing, control flow label scoping,
//! function signature agreement, constant expression legality, and the full
//! set of `assert_invalid` checks exercised by the spec test suite.
//!
//! Entry point: [`validate_module()`], called automatically by
//! [`crate::wat::parse()`].

use super::module::{
    DataMode, ElementMode, ExportIndex, ExternalKind, FunctionType, GlobalType, Limits, Module, Positional, RefType,
    ValueType, ValueType::*,
};
use super::structured::StructuredInstruction;
use crate::parser::instruction::{BlockType, ByteRange, Instruction, InstructionKind, SimdOp};
use MaybeValue::{Unknown, Val};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("constant expression required")]
    ConstantExpressionRequired,

    #[error("type mismatch")]
    TypeMismatch,

    #[error("unknown type")]
    UnknownFunctionType,

    #[error("unknown function {0}")]
    UnknownFunction(u32),

    #[error("unknown table")]
    UnknownTable,

    #[error("unknown table {0}")]
    UnknownTableWithIndex(u32),

    #[error("unknown memory")]
    UnknownMemory,

    #[error("unknown memory {0}")]
    UnknownMemoryWithIndex(u32),

    #[error("unknown element")]
    UnknownElement,

    #[error("unknown elem segment {0}")]
    UnknownElemSegment(u32),

    #[error("unexpected token")]
    UnexpectedToken,

    #[error("unknown local {0}")]
    UnknownLocal(u32),

    #[error("unknown global {0}")]
    UnknownGlobal(u32),

    #[error("unknown label")]
    UnknownLabel,

    #[error("invalid instruction")]
    InvalidInstruction,

    #[error("unexpected end of section or function")]
    UnexpectedEndOfFunction,

    #[error("data count section required")]
    DataCountSectionRequired,

    #[error("unknown data segment")]
    UnknownDataSegment,

    #[error("unknown data segment {0}")]
    UnknownDataSegmentWithIndex(u32),

    #[error("unknown block type")]
    UnknownBlockType,

    #[error("undeclared function reference")]
    UndeclaredFunctionReference,

    #[error("unimplemented instruction")]
    UnimplementedInstruction,

    #[error("alignment must not be larger than natural")]
    BadAlignment,

    #[error("global is immutable")]
    GlobalIsImmutable,

    #[error("invalid result arity")]
    InvalidResultArity,

    #[error("invalid lane index")]
    InvalidLaneIndex,

    #[error("duplicate export name")]
    DuplicateExportName,

    #[error("unknown function {0}")]
    UnknownFunctionExport(u32),

    #[error("unknown table {0}")]
    UnknownTableExport(u32),

    #[error("unknown memory {0}")]
    UnknownMemoryExport(u32),

    #[error("unknown global {0}")]
    UnknownGlobalExport(u32),

    #[error("unknown type")]
    UnknownType,

    #[error("multiple memories")]
    MultipleMemories,

    #[error("size minimum must not be greater than maximum")]
    MinGreaterThanMax,

    #[error("memory size must be at most 65536 pages (4GiB)")]
    MemorySizeTooLarge,

    #[error("table size must be at most 2^32-1")]
    TableSizeTooLarge,

    #[error("unknown function")]
    UnknownStartFunction,

    #[error("start function")]
    StartFunctionType,
}

pub trait Validator {
    fn validate(&mut self, inst: &Instruction) -> Result<(), ValidationError>;
    fn ended(&mut self) -> bool;
    fn finalise(&mut self) -> Result<(), ValidationError> {
        Ok(())
    }
}

pub struct ConstantExpressionValidator<'a> {
    // globals: &'a Vec<GlobalType>,
    imports: &'a super::module::ImportSection,
    return_type: ValueType,
    instr_count: u32,
    has_end: bool,
    total_functions: Option<u32>, // Total number of functions (imports + declared)
}

impl ConstantExpressionValidator<'_> {
    pub fn new(
        imports: &super::module::ImportSection,
        return_type: super::module::ValueType,
    ) -> ConstantExpressionValidator<'_> {
        ConstantExpressionValidator {
            imports,
            return_type,
            instr_count: 0,
            has_end: false,
            total_functions: None,
        }
    }

    pub fn with_function_count(mut self, count: u32) -> Self {
        self.total_functions = Some(count);
        self
    }

    fn global(&mut self, global_index: u32) -> Result<&GlobalType, ValidationError> {
        // constant expressions can only reference imported globals
        self.imports
            .get_global_import(global_index)
            .map(|import| match &import.external_kind {
                ExternalKind::Global(global_type) => {
                    if global_type.mutable {
                        Err(ValidationError::ConstantExpressionRequired)
                    } else {
                        Ok(global_type)
                    }
                }
                _ => Err(ValidationError::ConstantExpressionRequired),
            })
            .unwrap_or(Err(ValidationError::UnknownGlobal(global_index)))
    }
}

impl Validator for ConstantExpressionValidator<'_> {
    fn validate(&mut self, inst: &Instruction) -> Result<(), ValidationError> {
        let kind = &inst.kind;
        use InstructionKind::*;

        match kind {
            I32Const { .. }
            | I64Const { .. }
            | F32Const { .. }
            | F64Const { .. }
            | RefNull { .. }
            | RefFunc { .. }
            | GlobalGet { .. }
            | Simd(SimdOp::V128Const { .. })
            | End => {}
            _ => return Err(ValidationError::ConstantExpressionRequired),
        }
        self.instr_count += 1;
        if self.instr_count >= 2 && !matches!(kind, End) {
            return Err(ValidationError::TypeMismatch);
        }
        match &kind {
            // const type must match self.return_type
            I32Const { .. } => (self.return_type == I32)
                .then_some(())
                .ok_or(ValidationError::TypeMismatch),
            I64Const { .. } => (self.return_type == I64)
                .then_some(())
                .ok_or(ValidationError::TypeMismatch),
            F32Const { .. } => (self.return_type == F32)
                .then_some(())
                .ok_or(ValidationError::TypeMismatch),
            F64Const { .. } => (self.return_type == F64)
                .then_some(())
                .ok_or(ValidationError::TypeMismatch),
            RefNull { ref_type } => {
                if self.return_type == FuncRef && *ref_type == FuncRef
                    || self.return_type == ExternRef && *ref_type == ExternRef
                {
                    Ok(())
                } else {
                    Err(ValidationError::TypeMismatch)
                }
            }
            RefFunc { func_idx } => {
                // First check the return type
                if self.return_type != FuncRef && self.return_type != ExternRef {
                    return Err(ValidationError::TypeMismatch);
                }
                // Then check if the function index is valid
                if let Some(total) = self.total_functions
                    && *func_idx >= total
                {
                    return Err(ValidationError::UnknownFunction(*func_idx));
                }
                Ok(())
            }
            Simd(SimdOp::V128Const { .. }) => (self.return_type == V128)
                .then_some(())
                .ok_or(ValidationError::TypeMismatch),
            GlobalGet { global_idx } => {
                let value_type = self.global(*global_idx)?.value_type;
                if value_type == self.return_type {
                    Ok(())
                } else {
                    Err(ValidationError::TypeMismatch)
                }
            }
            End => match self.instr_count {
                1 => Err(ValidationError::TypeMismatch),
                2 => {
                    self.has_end = true;
                    Ok(())
                }
                _ => Err(ValidationError::ConstantExpressionRequired),
            },
            _ => Err(ValidationError::ConstantExpressionRequired),
        }
    }

    fn ended(&mut self) -> bool {
        self.has_end
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum MaybeValue {
    Val(ValueType),
    Unknown,
}

#[allow(dead_code)]
impl MaybeValue {
    fn is_num(&self) -> bool {
        match self {
            Val(v) => *v == I32 || *v == I64 || *v == F32 || *v == F64,
            Unknown => true,
        }
    }

    fn is_vec(&self) -> bool {
        match self {
            Val(v) => *v == V128,
            Unknown => true,
        }
    }

    fn is_ref(&self) -> bool {
        match self {
            Val(v) => *v == FuncRef || *v == ExternRef,
            Unknown => true,
        }
    }
}

const I32_VALUE: ValueType = I32;
const I64_VALUE: ValueType = I64;
const F32_VALUE: ValueType = F32;
const F64_VALUE: ValueType = F64;
#[allow(dead_code)]
const V128_VALUE: ValueType = V128;

#[derive(Clone)]
struct CtrlFrame {
    instruction: InstructionKind,
    start_types: Vec<MaybeValue>,
    end_types: Vec<MaybeValue>,
    height: u32,
    unreachable: bool,
}

pub struct CodeValidator<'a> {
    module: &'a super::module::Module,
    ctx: &'a super::module::ValidationContext,
    locals: &'a super::module::Locals,
    function_type: &'a FunctionType,
    vals: Vec<MaybeValue>,
    ctrls: Vec<CtrlFrame>,
    need_end: bool,
    current_function_index: u32,
}

impl<'a> CodeValidator<'a> {
    #[inline]
    fn validate_memory_exists(&self) -> Result<(), ValidationError> {
        if !self.ctx.has_memory {
            return Err(ValidationError::UnknownMemory);
        }
        Ok(())
    }

    pub fn new(
        module: &'a super::module::Module,
        ctx: &'a super::module::ValidationContext,
        locals: &'a super::module::Locals,
        function_type: &'a FunctionType,
        current_function_index: u32,
    ) -> CodeValidator<'a> {
        let mut v = CodeValidator {
            module,
            ctx,
            locals,
            function_type,
            vals: vec![],
            ctrls: vec![],
            need_end: false,
            current_function_index,
        };

        // push the return types but leave the parameters blank as they aren't on the stack
        // until explicitly loaded with local.get
        let end_types = function_type.return_types.iter().map(|v| Val(*v)).collect();
        // Use a dummy block for the function body
        v.push_ctrl(
            InstructionKind::Block {
                block_type: BlockType::Empty,
            },
            vec![],
            end_types,
        );

        v
    }

    fn push_val(&mut self, val_type: MaybeValue) -> Option<()> {
        self.vals.push(val_type);
        Some(())
    }

    fn push_vals(&mut self, val_types: Vec<MaybeValue>) -> Option<()> {
        for val_type in val_types {
            self.push_val(val_type)?;
        }
        Some(())
    }

    fn pop_val(&mut self) -> Option<MaybeValue> {
        if self.vals.len() as u32 == self.ctrls.last()?.height && self.ctrls.last()?.unreachable {
            Some(Unknown)
        } else if self.vals.len() as u32 == self.ctrls.last()?.height || self.vals.is_empty() {
            None
        } else {
            self.vals.pop()
        }
    }

    fn pop_expected(&mut self, val_type: MaybeValue) -> Option<MaybeValue> {
        let popped = self.pop_val()?;
        if popped != val_type && popped != Unknown && val_type != Unknown {
            None
        } else {
            Some(popped)
        }
    }

    fn pop_expecteds(&mut self, mut val_types: Vec<MaybeValue>) -> Option<Vec<MaybeValue>> {
        let mut popped = Vec::new();
        val_types.reverse();
        for val_type in val_types {
            popped.push(self.pop_expected(val_type)?);
        }
        Some(popped)
    }

    fn push_ctrl(&mut self, instruction: InstructionKind, start_types: Vec<MaybeValue>, end_types: Vec<MaybeValue>) {
        self.ctrls.push(CtrlFrame {
            instruction,
            start_types: start_types.clone(),
            end_types,
            height: self.vals.len() as u32,
            unreachable: false,
        });
        self.push_vals(start_types);
        self.need_end = true;
    }

    fn pop_ctrl(&mut self) -> Result<CtrlFrame, ValidationError> {
        if self.ctrls.is_empty() {
            Err(ValidationError::UnexpectedToken)
        } else {
            let ctrl = self.ctrls.last().unwrap();
            let end_types_clone = ctrl.end_types.clone();
            let height = ctrl.height as usize;

            if self.pop_expecteds(end_types_clone).is_none() || self.vals.len() != height {
                Err(ValidationError::TypeMismatch)
            } else {
                Ok(self.ctrls.pop().unwrap())
            }
        }
    }

    fn frame_types(&mut self, frame: CtrlFrame) -> Vec<MaybeValue> {
        if matches!(frame.instruction, InstructionKind::Loop { .. }) {
            frame.start_types
        } else {
            frame.end_types
        }
    }

    fn unreachable(&mut self) -> Option<()> {
        if let Some(ctrl) = self.ctrls.last_mut() {
            let h = ctrl.height as usize;
            if self.vals.len() > h {
                self.vals.truncate(h);
            }
            ctrl.unreachable = true;
            Some(())
        } else {
            None
        }
    }

    fn sig_unary(&mut self, in_type: MaybeValue, out_type: MaybeValue) -> Result<(), ValidationError> {
        self.pop_expected(in_type).ok_or(ValidationError::TypeMismatch)?;
        self.push_val(out_type).ok_or(ValidationError::TypeMismatch)
    }

    fn sig_binary(
        &mut self,
        in1_type: MaybeValue,
        in2_type: MaybeValue,
        out_type: MaybeValue,
    ) -> Result<(), ValidationError> {
        self.pop_expected(in1_type).ok_or(ValidationError::TypeMismatch)?;
        self.pop_expected(in2_type).ok_or(ValidationError::TypeMismatch)?;
        self.push_val(out_type).ok_or(ValidationError::TypeMismatch)
    }

    fn local(&mut self, local_index: u32) -> Result<&ValueType, ValidationError> {
        let li = local_index;
        let param_count = self.function_type.parameters.len();
        let local: Option<&ValueType> = if (li as usize) < param_count {
            self.function_type.parameters.get(li as usize)
        } else {
            let li = li - param_count as u32;
            self.locals.get(li)
        };

        local.ok_or(ValidationError::UnknownLocal(local_index))
    }

    fn global(&mut self, global_index: u32) -> Result<&GlobalType, ValidationError> {
        // TODO: get the imported global type
        let imported_global_count = self.module.imports.global_count();
        let global = if global_index < imported_global_count as u32 {
            self.module
                .imports
                .get_global_import(global_index)
                .and_then(|import| match &import.external_kind {
                    ExternalKind::Global(global_type) => Some(global_type),
                    _ => None,
                })
        } else {
            self.module
                .globals
                .get(global_index - imported_global_count as u32)
                .map(|g| &g.global_type)
        };
        match global {
            Some(global) => Ok(global),
            None => Err(ValidationError::UnknownGlobal(global_index)),
        }
    }

    fn label_types_at(&mut self, li: u32) -> Result<Vec<MaybeValue>, ValidationError> {
        if self.ctrls.len() <= li as usize {
            return Err(ValidationError::UnknownLabel);
        }
        let index = self.ctrls.len() - li as usize - 1;
        let frame = self.ctrls.get(index).ok_or(ValidationError::UnknownLabel)?.clone();
        Ok(self.frame_types(frame))
    }

    fn get_function_type(&self, fi: u32) -> Result<&FunctionType, ValidationError> {
        // Use pre-computed function type indices for O(1) lookup
        self.ctx
            .function_type_indices
            .get(fi as usize)
            .ok_or(ValidationError::UnknownFunction(fi))
            .and_then(|&type_idx| self.get_type(type_idx))
    }

    fn get_type(&self, ti: u32) -> Result<&FunctionType, ValidationError> {
        self.module.types.get(ti).ok_or(ValidationError::UnknownFunctionType)
    }

    /// Validate a SIMD instruction's stack effects, memory alignment, and lane bounds.
    /// Stack pushes/pops occur before alignment and lane checks. This is fine because
    /// any validation error rejects the entire function — partial stack state is discarded.
    fn validate_simd(&mut self, op: &SimdOp) -> Result<(), ValidationError> {
        use SimdOp::*;
        let v = Val(V128);
        let i32v = Val(I32_VALUE);
        let i64v = Val(I64_VALUE);
        let f32v = Val(F32_VALUE);
        let f64v = Val(F64_VALUE);

        match op {
            // [i32] → [v128] : loads (pop address, push result)
            V128Load { memarg } => {
                self.validate_memory_exists()?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 4)
            }
            V128Load8x8S { memarg }
            | V128Load8x8U { memarg }
            | V128Load16x4S { memarg }
            | V128Load16x4U { memarg }
            | V128Load32x2S { memarg }
            | V128Load32x2U { memarg }
            | V128Load64Splat { memarg } => {
                self.validate_memory_exists()?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 3)
            }
            V128Load32Splat { memarg } | V128Load32Zero { memarg } => {
                self.validate_memory_exists()?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 2)
            }
            V128Load16Splat { memarg } => {
                self.validate_memory_exists()?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 1)
            }
            V128Load8Splat { memarg } => {
                self.validate_memory_exists()?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 0)
            }
            V128Load64Zero { memarg } => {
                self.validate_memory_exists()?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 3)
            }

            // [i32 v128] → [] : stores (pop v128 value, pop i32 address)
            V128Store { memarg } => {
                self.validate_memory_exists()?;
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 4)
            }

            // [i32 v128] → [v128] : load lane (pop v128, pop i32, push v128)
            V128Load8Lane { memarg, lane } => {
                validate_lane_index(*lane, 16)?;
                self.validate_memory_exists()?;
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 0)
            }
            V128Load16Lane { memarg, lane } => {
                validate_lane_index(*lane, 8)?;
                self.validate_memory_exists()?;
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 1)
            }
            V128Load32Lane { memarg, lane } => {
                validate_lane_index(*lane, 4)?;
                self.validate_memory_exists()?;
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 2)
            }
            V128Load64Lane { memarg, lane } => {
                validate_lane_index(*lane, 2)?;
                self.validate_memory_exists()?;
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 3)
            }

            // [i32 v128] → [] : store lane (pop v128, pop i32)
            V128Store8Lane { memarg, lane } => {
                validate_lane_index(*lane, 16)?;
                self.validate_memory_exists()?;
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 0)
            }
            V128Store16Lane { memarg, lane } => {
                validate_lane_index(*lane, 8)?;
                self.validate_memory_exists()?;
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 1)
            }
            V128Store32Lane { memarg, lane } => {
                validate_lane_index(*lane, 4)?;
                self.validate_memory_exists()?;
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 2)
            }
            V128Store64Lane { memarg, lane } => {
                validate_lane_index(*lane, 2)?;
                self.validate_memory_exists()?;
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(i32v).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 3)
            }

            // [] → [v128] : constant
            V128Const { .. } => self.push_val(v).ok_or(ValidationError::TypeMismatch),

            // [v128 v128] → [v128] : shuffle (lane indices must be 0-31)
            I8x16Shuffle { lanes } => {
                for &lane in lanes.iter() {
                    if lane >= 32 {
                        return Err(ValidationError::InvalidLaneIndex);
                    }
                }
                self.sig_binary(v.clone(), v.clone(), v)
            }

            // [v128 v128] → [v128] : swizzle, binary arithmetic, comparisons,
            // narrow, extmul, dot, q15mulr, add_sat, sub_sat, avgr
            I8x16Swizzle
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
            | F64x2PMax => self.sig_binary(v.clone(), v.clone(), v),

            // [v128 v128 v128] → [v128] : bitselect
            V128Bitselect => {
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(v.clone()).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(v).ok_or(ValidationError::TypeMismatch)
            }

            // [v128] → [v128] : unary v128→v128 (abs, neg, extend, convert, etc.)
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
            | F64x2PromoteLowF32x4 => self.sig_unary(v.clone(), v),

            // [v128] → [i32] : test/bitmask
            V128AnyTrue | I8x16AllTrue | I8x16Bitmask | I16x8AllTrue | I16x8Bitmask | I32x4AllTrue | I32x4Bitmask
            | I64x2AllTrue | I64x2Bitmask => self.sig_unary(v, i32v),

            // [v128] → [i32] : extract lane to i32 (with lane bounds check)
            I8x16ExtractLaneS { lane } | I8x16ExtractLaneU { lane } => {
                validate_lane_index(*lane, 16)?;
                self.sig_unary(v, i32v)
            }
            I16x8ExtractLaneS { lane } | I16x8ExtractLaneU { lane } => {
                validate_lane_index(*lane, 8)?;
                self.sig_unary(v, i32v)
            }
            I32x4ExtractLane { lane } => {
                validate_lane_index(*lane, 4)?;
                self.sig_unary(v, i32v)
            }

            // [v128] → [i64] : extract i64 lane
            I64x2ExtractLane { lane } => {
                validate_lane_index(*lane, 2)?;
                self.sig_unary(v, i64v)
            }

            // [v128] → [f32] : extract f32 lane
            F32x4ExtractLane { lane } => {
                validate_lane_index(*lane, 4)?;
                self.sig_unary(v, f32v)
            }

            // [v128] → [f64] : extract f64 lane
            F64x2ExtractLane { lane } => {
                validate_lane_index(*lane, 2)?;
                self.sig_unary(v, f64v)
            }

            // [v128 i32] → [v128] : shifts
            I8x16Shl | I8x16ShrS | I8x16ShrU | I16x8Shl | I16x8ShrS | I16x8ShrU | I32x4Shl | I32x4ShrS | I32x4ShrU
            | I64x2Shl | I64x2ShrS | I64x2ShrU => self.sig_binary(i32v, v.clone(), v),

            // [v128 i32] → [v128] : replace lane (i32 replacement value)
            I8x16ReplaceLane { lane } => {
                validate_lane_index(*lane, 16)?;
                self.sig_binary(i32v, v.clone(), v)
            }
            I16x8ReplaceLane { lane } => {
                validate_lane_index(*lane, 8)?;
                self.sig_binary(i32v, v.clone(), v)
            }
            I32x4ReplaceLane { lane } => {
                validate_lane_index(*lane, 4)?;
                self.sig_binary(i32v, v.clone(), v)
            }

            // [v128 i64] → [v128] : i64 replace lane
            I64x2ReplaceLane { lane } => {
                validate_lane_index(*lane, 2)?;
                self.sig_binary(i64v, v.clone(), v)
            }

            // [v128 f32] → [v128] : f32 replace lane
            F32x4ReplaceLane { lane } => {
                validate_lane_index(*lane, 4)?;
                self.sig_binary(f32v, v.clone(), v)
            }

            // [v128 f64] → [v128] : f64 replace lane
            F64x2ReplaceLane { lane } => {
                validate_lane_index(*lane, 2)?;
                self.sig_binary(f64v, v.clone(), v)
            }

            // [i32] → [v128] : splat from i32
            I8x16Splat | I16x8Splat | I32x4Splat => self.sig_unary(i32v, v),

            // [i64] → [v128] : splat from i64
            I64x2Splat => self.sig_unary(i64v, v),

            // [f32] → [v128] : splat from f32
            F32x4Splat => self.sig_unary(f32v, v),

            // [f64] → [v128] : splat from f64
            F64x2Splat => self.sig_unary(f64v, v),
        }
    }
}

impl Validator for CodeValidator<'_> {
    fn validate(&mut self, inst: &Instruction) -> Result<(), ValidationError> {
        let kind = &inst.kind;
        use InstructionKind::*;

        match kind {
            I32Const { .. } => self.push_val(Val(I32_VALUE)).ok_or(ValidationError::TypeMismatch),

            // iunop (i32):i32
            I32Clz | I32Ctz | I32Popcnt => self.sig_unary(Val(I32_VALUE), Val(I32_VALUE)),

            // ibinop (i32,i32):i32
            I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or | I32Xor | I32Shl
            | I32ShrS | I32ShrU | I32Rotr | I32Rotl => self.sig_binary(Val(I32_VALUE), Val(I32_VALUE), Val(I32_VALUE)),

            // itestop (i32):i32
            I32Eqz => self.sig_unary(Val(I32_VALUE), Val(I32_VALUE)),

            // irelop (i32,i32):i32
            I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU => {
                self.sig_binary(Val(I32_VALUE), Val(I32_VALUE), Val(I32_VALUE))
            }

            // cvtop (i32):i32
            I32Extend8S | I32Extend16S => self.sig_unary(Val(I32_VALUE), Val(I32_VALUE)),

            // cvtop (i64):i32
            I32WrapI64 => self.sig_unary(Val(I64_VALUE), Val(I32_VALUE)),

            // cvtop (f32):i32
            I32TruncF32S | I32TruncF32U | I32TruncSatF32S | I32TruncSatF32U | I32ReinterpretF32 => {
                self.sig_unary(Val(F32_VALUE), Val(I32_VALUE))
            }

            // cvtop (f64):i32
            I32TruncF64S | I32TruncF64U | I32TruncSatF64S | I32TruncSatF64U => {
                self.sig_unary(Val(F64_VALUE), Val(I32_VALUE))
            }

            I64Const { .. } => self.push_val(Val(I64_VALUE)).ok_or(ValidationError::TypeMismatch),

            // iunop (i64):i64
            I64Clz | I64Ctz | I64Popcnt => self.sig_unary(Val(I64_VALUE), Val(I64_VALUE)),

            // ibinop (i64,i64):i64
            I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And | I64Or | I64Xor | I64Shl
            | I64ShrS | I64ShrU | I64Rotr | I64Rotl => self.sig_binary(Val(I64_VALUE), Val(I64_VALUE), Val(I64_VALUE)),

            // itestop (i64):i32
            I64Eqz => self.sig_unary(Val(I64_VALUE), Val(I32_VALUE)),

            // irelop (i64,i64):i32
            I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS | I64GeU => {
                self.sig_binary(Val(I64_VALUE), Val(I64_VALUE), Val(I32_VALUE))
            }

            // cvtop (i64):i64
            I64Extend8S | I64Extend16S | I64Extend32S => self.sig_unary(Val(I64_VALUE), Val(I64_VALUE)),

            // cvtop (i32):i64
            I64ExtendI32S | I64ExtendI32U => self.sig_unary(Val(I32_VALUE), Val(I64_VALUE)),

            // cvtop (f32):i64
            I64TruncF32S | I64TruncF32U | I64TruncSatF32S | I64TruncSatF32U => {
                self.sig_unary(Val(F32_VALUE), Val(I64_VALUE))
            }

            // cvtop (f64):i64
            I64TruncF64S | I64TruncF64U | I64TruncSatF64S | I64TruncSatF64U | I64ReinterpretF64 => {
                self.sig_unary(Val(F64_VALUE), Val(I64_VALUE))
            }

            F32Const { .. } => self.push_val(Val(F32_VALUE)).ok_or(ValidationError::TypeMismatch),

            // funop (f32):f32
            F32Abs | F32Neg | F32Sqrt | F32Ceil | F32Floor | F32Trunc | F32Nearest => {
                self.sig_unary(Val(F32_VALUE), Val(F32_VALUE))
            }

            // fbinop (f32,f32):f32
            F32Add | F32Sub | F32Mul | F32Div | F32Min | F32Max | F32Copysign => {
                self.sig_binary(Val(F32_VALUE), Val(F32_VALUE), Val(F32_VALUE))
            }

            // frelop (f32,f32):i32
            F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge => {
                self.sig_binary(Val(F32_VALUE), Val(F32_VALUE), Val(I32_VALUE))
            }

            // cvtop (f64):f32
            F32DemoteF64 => self.sig_unary(Val(F64_VALUE), Val(F32_VALUE)),

            // cvtop (i32):f32
            F32ConvertI32S | F32ConvertI32U | F32ReinterpretI32 => self.sig_unary(Val(I32_VALUE), Val(F32_VALUE)),

            // cvtop (i64):f32
            F32ConvertI64S | F32ConvertI64U => self.sig_unary(Val(I64_VALUE), Val(F32_VALUE)),

            F64Const { .. } => self.push_val(Val(F64_VALUE)).ok_or(ValidationError::TypeMismatch),

            // funop (f64):f64
            F64Abs | F64Neg | F64Sqrt | F64Ceil | F64Floor | F64Trunc | F64Nearest => {
                self.sig_unary(Val(F64_VALUE), Val(F64_VALUE))
            }

            // fbinop (f64,f64):f64
            F64Add | F64Sub | F64Mul | F64Div | F64Min | F64Max | F64Copysign => {
                self.sig_binary(Val(F64_VALUE), Val(F64_VALUE), Val(F64_VALUE))
            }

            // frelop (f64,f64):i32
            F64Eq | F64Ne | F64Lt | F64Gt | F64Le | F64Ge => {
                self.sig_binary(Val(F64_VALUE), Val(F64_VALUE), Val(I32_VALUE))
            }

            // cvtop (f32):f64
            F64PromoteF32 => self.sig_unary(Val(F32_VALUE), Val(F64_VALUE)),

            // cvtop (i32):f64
            F64ConvertI32S | F64ConvertI32U => self.sig_unary(Val(I32_VALUE), Val(F64_VALUE)),

            // cvtop (i64):f64
            F64ConvertI64S | F64ConvertI64U | F64ReinterpretI64 => self.sig_unary(Val(I64_VALUE), Val(F64_VALUE)),

            LocalGet { local_idx } => {
                let local = *self.local(*local_idx)?;
                self.push_val(Val(local)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            LocalSet { local_idx } => {
                let local = *self.local(*local_idx)?;
                self.pop_expected(Val(local)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            LocalTee { local_idx } => {
                let local = *self.local(*local_idx)?;
                self.pop_expected(Val(local)).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(Val(local)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            GlobalGet { global_idx } => {
                let global = *self.global(*global_idx)?;
                self.push_val(Val(global.value_type))
                    .ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            GlobalSet { global_idx } => {
                let global = *self.global(*global_idx)?;
                if !global.mutable {
                    return Err(ValidationError::GlobalIsImmutable);
                }
                self.pop_expected(Val(global.value_type))
                    .ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            I32Load { memarg }
            | I32Load8S { memarg }
            | I32Load8U { memarg }
            | I32Load16S { memarg }
            | I32Load16U { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                let align_exp = match &kind {
                    I32Load { .. } => 2,                        // 4 bytes = 2^2
                    I32Load8S { .. } | I32Load8U { .. } => 0,   // 1 byte = 2^0
                    I32Load16S { .. } | I32Load16U { .. } => 1, // 2 bytes = 2^1
                    _ => unreachable!(),
                };
                check_alignment(memarg, align_exp)
            }

            I64Load { memarg }
            | I64Load8S { memarg }
            | I64Load8U { memarg }
            | I64Load16S { memarg }
            | I64Load16U { memarg }
            | I64Load32S { memarg }
            | I64Load32U { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(Val(I64)).ok_or(ValidationError::TypeMismatch)?;
                let align_exp = match &kind {
                    I64Load { .. } => 3,                        // 8 bytes = 2^3
                    I64Load8S { .. } | I64Load8U { .. } => 0,   // 1 byte = 2^0
                    I64Load16S { .. } | I64Load16U { .. } => 1, // 2 bytes = 2^1
                    I64Load32S { .. } | I64Load32U { .. } => 2, // 4 bytes = 2^2
                    _ => unreachable!(),
                };
                check_alignment(memarg, align_exp)
            }

            F32Load { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(Val(F32)).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 2 /* 4 bytes = 2^2 */)
            }

            F64Load { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(Val(F64)).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 3 /* 8 bytes = 2^3 */)
            }

            I32Store { memarg } | I32Store8 { memarg } | I32Store16 { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                let align_exp = match &kind {
                    I32Store { .. } => 2,   // 4 bytes = 2^2
                    I32Store8 { .. } => 0,  // 1 byte = 2^0
                    I32Store16 { .. } => 1, // 2 bytes = 2^1
                    _ => unreachable!(),
                };
                check_alignment(memarg, align_exp)
            }

            I64Store { memarg } | I64Store8 { memarg } | I64Store16 { memarg } | I64Store32 { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.pop_expected(Val(I64)).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                let align_exp = match &kind {
                    I64Store { .. } => 3,   // 8 bytes = 2^3
                    I64Store8 { .. } => 0,  // 1 byte = 2^0
                    I64Store16 { .. } => 1, // 2 bytes = 2^1
                    I64Store32 { .. } => 2, // 4 bytes = 2^2
                    _ => unreachable!(),
                };
                check_alignment(memarg, align_exp)
            }

            F32Store { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.pop_expected(Val(F32)).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 2 /* 4 bytes = 2^2 */)
            }

            F64Store { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.pop_expected(Val(F64)).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                check_alignment(memarg, 3 /* 8 bytes = 2^3 */)
            }

            MemorySize => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.push_val(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            MemoryGrow => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            MemoryFill | MemoryCopy => {
                // Check that memory exists (imported or declared)
                if self.module.memory.is_empty() && self.module.imports.memory_count() == 0 {
                    return Err(ValidationError::UnknownMemoryWithIndex(0));
                }
                self.pop_expecteds(vec![Val(I32), Val(I32), Val(I32)])
                    .ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            MemoryInit { data_idx } => {
                // Check that memory exists (imported or declared)
                if self.module.memory.is_empty() && self.module.imports.memory_count() == 0 {
                    return Err(ValidationError::UnknownMemoryWithIndex(0));
                }

                // Bulk memory ops require knowledge of data segment count.
                // Binary modules signal this via a DataCount section (non-zero position);
                // WAT-parsed modules always have the count available (non-zero count).
                let has_data_count = self.module.data_count.count > 0
                    || self.module.data_count.position.start != 0
                    || self.module.data_count.position.end != 0;
                if !has_data_count {
                    return Err(ValidationError::DataCountSectionRequired);
                }

                // Check if data_idx >= data_count
                if *data_idx >= self.module.data_count.count {
                    return Err(ValidationError::UnknownDataSegmentWithIndex(*data_idx));
                }
                self.pop_expecteds(vec![Val(I32), Val(I32), Val(I32)])
                    .ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            TableInit { elem_idx, table_idx } => {
                self.pop_expecteds(vec![Val(I32), Val(I32), Val(I32)])
                    .ok_or(ValidationError::TypeMismatch)?;
                let table = self
                    .module
                    .table
                    .tables
                    .get(*table_idx as usize)
                    .ok_or(ValidationError::UnknownTableWithIndex(*table_idx))?;
                let elem = self
                    .module
                    .elements
                    .elements
                    .get(*elem_idx as usize)
                    .ok_or(ValidationError::UnknownElement)?;
                if elem.ref_type != table.ref_type {
                    return Err(ValidationError::TypeMismatch);
                }
                Ok(())
            }

            TableCopy { src_table, dst_table } => {
                self.pop_expecteds(vec![Val(I32), Val(I32), Val(I32)])
                    .ok_or(ValidationError::TypeMismatch)?;
                let src_table_type = self.module.get_table(*src_table).ok_or(ValidationError::UnknownTable)?;
                let dst_table_type = self.module.get_table(*dst_table).ok_or(ValidationError::UnknownTable)?;
                if src_table_type.ref_type != dst_table_type.ref_type {
                    return Err(ValidationError::TypeMismatch);
                }
                Ok(())
            }

            TableGet { table_idx } => {
                let table = self.module.get_table(*table_idx).ok_or(ValidationError::UnknownTable)?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                self.push_val(Val(table.ref_type.into()))
                    .ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            TableSet { table_idx } => {
                let table = self.module.get_table(*table_idx).ok_or(ValidationError::UnknownTable)?;
                self.pop_expected(Val(table.ref_type.into()))
                    .ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            ElemDrop { elem_idx } => {
                self.module
                    .elements
                    .elements
                    .get(*elem_idx as usize)
                    .ok_or(ValidationError::UnknownElemSegment(*elem_idx))?;
                Ok(())
            }

            DataDrop { data_idx } => {
                let has_data_count = self.module.data_count.count > 0
                    || self.module.data_count.position.start != 0
                    || self.module.data_count.position.end != 0;
                let has_memory = !self.module.memory.is_empty() || self.module.imports.memory_count() > 0;

                if !has_data_count {
                    if has_memory {
                        return Err(ValidationError::DataCountSectionRequired);
                    } else {
                        return Err(ValidationError::UnknownDataSegment);
                    }
                }

                // DataCount section exists, check the index
                if *data_idx >= self.module.data_count.count {
                    Err(ValidationError::UnknownDataSegmentWithIndex(*data_idx))
                } else {
                    Ok(())
                }
            }

            Block { block_type } | Loop { block_type } | If { block_type } => {
                // special case for If we need to pop an i32
                if matches!(kind, If { .. }) {
                    self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                }

                let mut start_types = Vec::new();
                let mut end_types = Vec::new();

                match block_type {
                    BlockType::Empty => {
                        // Do nothing, types remain empty
                    }
                    BlockType::Value(t) => {
                        end_types.push(Val(*t));
                    }
                    BlockType::FuncType(ti) => {
                        let ftype = self.module.types.get(*ti).ok_or(ValidationError::UnknownBlockType)?;
                        // parameters are stack ordered, so pick them in reverse
                        ftype.parameters.iter().rev().try_for_each(|v| {
                            match self.pop_expected(Val(*v)) {
                                Some(_) => {
                                    // insert at start because we're working in reverse
                                    start_types.insert(0, Val(*v));
                                    Ok(())
                                }
                                None => Err(ValidationError::TypeMismatch),
                            }
                        })?;
                        ftype.return_types.iter().for_each(|v| {
                            end_types.push(Val(*v));
                        });
                    }
                };

                self.push_ctrl(kind.clone(), start_types, end_types);
                Ok(())
            }

            Else => {
                let ctrl = self.pop_ctrl()?;
                if !matches!(ctrl.instruction, If { .. }) {
                    Err(ValidationError::InvalidInstruction)
                } else {
                    // Mark that we had an else by pushing Else control frame
                    self.push_ctrl(Else, ctrl.start_types, ctrl.end_types);
                    Ok(())
                }
            }

            End => {
                let ctrl = self.pop_ctrl()?;

                // Special check: typed if without else is invalid UNLESS
                // the start types match the end types (parameters flow through)
                if matches!(ctrl.instruction, If { .. }) && !ctrl.end_types.is_empty() && !ctrl.unreachable {
                    // Check if start_types == end_types (parameters flow through)
                    if ctrl.start_types != ctrl.end_types {
                        return Err(ValidationError::TypeMismatch);
                    }
                }

                self.push_vals(ctrl.end_types).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            Return => {
                // Return targets the outermost label
                let li = self.ctrls.len() as u32 - 1;
                let label_types = self.label_types_at(li)?;
                self.pop_expecteds(label_types).ok_or(ValidationError::TypeMismatch)?;
                self.unreachable().ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            Br { label_idx } => {
                let label_types = self.label_types_at(*label_idx)?;
                self.pop_expecteds(label_types).ok_or(ValidationError::TypeMismatch)?;
                self.unreachable().ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            BrIf { label_idx } => {
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                let label_types = self.label_types_at(*label_idx)?;
                self.pop_expecteds(label_types.clone())
                    .ok_or(ValidationError::TypeMismatch)?;
                self.push_vals(label_types).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            BrTable { labels, default } => {
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                if self.ctrls.len() < *default as usize {
                    return Err(ValidationError::UnknownLabel);
                }

                // First, check that all labels exist
                for &li in labels.iter() {
                    if self.ctrls.len() <= li as usize {
                        return Err(ValidationError::UnknownLabel);
                    }
                }

                // Get the result types for the default label
                let default_label_types = self.label_types_at(*default)?;
                let arity = default_label_types.len();

                // All branch targets must have the same arity
                for &li in labels.iter() {
                    let label_types = self.label_types_at(li)?;
                    if label_types.len() != arity {
                        return Err(ValidationError::TypeMismatch);
                    }
                }

                // Check if we're in an unreachable state
                let is_unreachable = self.ctrls.last().map(|c| c.unreachable).unwrap_or(false);

                if is_unreachable {
                    // In unreachable state, we need to be more careful
                    // We still validate, but we allow polymorphic values
                    let saved_stack_height = self.vals.len();

                    // Try to validate each branch
                    let mut validation_failed = false;
                    for &li in labels.iter() {
                        let label_types = self.label_types_at(li)?;

                        // Try to pop expected types
                        let mut can_satisfy = true;
                        for expected_type in label_types.iter().rev() {
                            if let Some(actual) = self.pop_val() {
                                // If we get Unknown (polymorphic), it can satisfy any type
                                // If we get a concrete type, it must match
                                if actual != Unknown && actual != *expected_type {
                                    can_satisfy = false;
                                    break;
                                }
                            } else {
                                can_satisfy = false;
                                break;
                            }
                        }

                        // Restore stack for next iteration
                        self.vals.truncate(saved_stack_height);

                        if !can_satisfy {
                            validation_failed = true;
                            break;
                        }
                    }

                    if validation_failed {
                        return Err(ValidationError::TypeMismatch);
                    }

                    // Also validate the default branch
                    for expected_type in default_label_types.iter().rev() {
                        if let Some(actual) = self.pop_val() {
                            if actual != Unknown && actual != *expected_type {
                                self.vals.truncate(saved_stack_height);
                                return Err(ValidationError::TypeMismatch);
                            }
                        } else {
                            self.vals.truncate(saved_stack_height);
                            return Err(ValidationError::TypeMismatch);
                        }
                    }
                } else {
                    // Normal validation
                    labels.iter().try_for_each(|&li| {
                        let label_types = self.label_types_at(li)?;
                        let popped = self.pop_expecteds(label_types.clone());
                        if popped.is_none() {
                            Err(ValidationError::TypeMismatch)
                        } else {
                            self.push_vals(label_types).ok_or(ValidationError::TypeMismatch)?;
                            Ok(())
                        }
                    })?;

                    // Pop for the default branch
                    self.pop_expecteds(default_label_types)
                        .ok_or(ValidationError::TypeMismatch)?;
                }
                self.unreachable().ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            Unreachable => {
                self.unreachable().ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            Call { func_idx } => {
                // Get type index directly from pre-computed data
                let type_idx = self
                    .ctx
                    .function_type_indices
                    .get(*func_idx as usize)
                    .ok_or(ValidationError::UnknownFunction(*func_idx))?;

                let ftype = self
                    .module
                    .types
                    .get(*type_idx)
                    .ok_or(ValidationError::UnknownFunctionType)?;

                // parameters are stack ordered, so pick them in reverse
                for param in ftype.parameters.iter().rev() {
                    self.pop_expected(Val(*param)).ok_or(ValidationError::TypeMismatch)?;
                }

                for return_type in &ftype.return_types {
                    self.push_val(Val(*return_type)).ok_or(ValidationError::TypeMismatch)?;
                }
                Ok(())
            }

            CallIndirect { type_idx, table_idx } => {
                // Validate table exists using pre-computed data
                if *table_idx as usize >= self.ctx.has_table.len() || !self.ctx.has_table[*table_idx as usize] {
                    return Err(ValidationError::UnknownTable);
                }

                let table = self.module.get_table(*table_idx).ok_or(ValidationError::UnknownTable)?;

                if ValueType::from(table.ref_type) != FuncRef {
                    return Err(ValidationError::TypeMismatch);
                }

                // operand that directs us to the table entry
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;

                // table entry must have this function signature
                let ftype = self
                    .module
                    .types
                    .get(*type_idx)
                    .ok_or(ValidationError::UnknownFunctionType)?;

                // parameters are stack ordered, so pick them in reverse
                for param in ftype.parameters.iter().rev() {
                    self.pop_expected(Val(*param)).ok_or(ValidationError::TypeMismatch)?;
                }

                for return_type in &ftype.return_types {
                    self.push_val(Val(*return_type)).ok_or(ValidationError::TypeMismatch)?;
                }
                Ok(())
            }

            Select => {
                // Select pops: [t t i32] and pushes: [t]
                // Note: We report TypeMismatch for insufficient operands rather than InvalidResultArity.
                // This is technically correct at the binary level, though some validators (like the
                // reference interpreter) may report InvalidResultArity when the original WAT had
                // explicit result arity declarations that are lost during WAT→WASM compilation.

                // Pop condition (i32)
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                // Pop two values of same type
                let t2 = self.pop_val().ok_or(ValidationError::TypeMismatch)?.clone();
                let t1 = self.pop_val().ok_or(ValidationError::TypeMismatch)?.clone();

                // Check if types are compatible
                if (t1.is_num() && t2.is_num()) || (t1.is_vec() && t2.is_vec()) {
                    if t1 != t2 && t1 != Unknown && t2 != Unknown {
                        Err(ValidationError::TypeMismatch)
                    } else {
                        // Push the result type
                        self.push_val(if t1 == Unknown { t2.clone() } else { t1.clone() })
                            .ok_or(ValidationError::TypeMismatch)?;
                        Ok(())
                    }
                } else {
                    Err(ValidationError::TypeMismatch)
                }
            }

            SelectTyped { val_types } => {
                // SelectT has explicit type immediates in the instruction data
                if val_types.is_empty() {
                    // Empty result arity is a specific error
                    return Err(ValidationError::InvalidResultArity);
                }
                if val_types.len() != 1 {
                    // Select should have exactly one result type.
                    return Err(ValidationError::InvalidResultArity);
                }
                let result_type = val_types[0];

                // Pop condition (i32)
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;

                // Pop two values of the result type
                let t2 = self.pop_val().ok_or(ValidationError::TypeMismatch)?;
                let t1 = self.pop_val().ok_or(ValidationError::TypeMismatch)?;

                // Check that popped values match the expected type
                let expected = Val(result_type);
                if !matches!(t1, Unknown) && t1 != expected {
                    return Err(ValidationError::TypeMismatch);
                }
                if !matches!(t2, Unknown) && t2 != expected {
                    return Err(ValidationError::TypeMismatch);
                }

                // Push result type
                self.push_val(Val(result_type)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            RefFunc { func_idx } => {
                self.get_function_type(*func_idx)?;
                // Check if this is a self-reference
                if *func_idx == self.current_function_index {
                    // Allow self-reference if the function is declared in a declarative element
                    // Declarative elements exist specifically to allow functions to be referenced
                    // before they are fully validated, enabling self-references and forward references
                    let is_declared_in_declarative = self.module.elements.elements.iter().any(|elem| {
                        matches!(elem.mode, ElementMode::Declarative)
                            && elem.init.iter().any(|init_exprs| {
                                init_exprs.iter().any(|instr| {
                                    matches!(instr.kind, InstructionKind::RefFunc { func_idx: idx } if idx == *func_idx)
                                })
                            })
                    });

                    if !is_declared_in_declarative {
                        return Err(ValidationError::UndeclaredFunctionReference);
                    }
                }
                self.push_val(Val(FuncRef)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            RefNull { ref_type } => {
                self.push_val(Val(*ref_type)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            RefIsNull => {
                // ref.is_null pops a reference type and pushes i32
                let ref_type = self.pop_val().ok_or(ValidationError::TypeMismatch)?;
                // Check that popped value is a reference type
                if !ref_type.is_ref() {
                    return Err(ValidationError::TypeMismatch);
                }
                self.push_val(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            Drop => {
                self.pop_val().ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            Nop => Ok(()),

            TableSize { table_idx } => {
                // Verify table exists
                self.module.get_table(*table_idx).ok_or(ValidationError::UnknownTable)?;
                // table.size pushes the current size as i32
                self.push_val(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            TableGrow { table_idx } => {
                let table = self.module.get_table(*table_idx).ok_or(ValidationError::UnknownTable)?;
                // table.grow pops: [n:i32, init:reftype] and pushes: [prev_size:i32]
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(Val(table.ref_type.into()))
                    .ok_or(ValidationError::TypeMismatch)?;
                self.push_val(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            TableFill { table_idx } => {
                let table = self.module.get_table(*table_idx).ok_or(ValidationError::UnknownTable)?;
                // table.fill pops: [i:i32, val:reftype, n:i32]
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(Val(table.ref_type.into()))
                    .ok_or(ValidationError::TypeMismatch)?;
                self.pop_expected(Val(I32)).ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            Simd(op) => self.validate_simd(op),
        }
    }

    fn ended(&mut self) -> bool {
        self.need_end && self.ctrls.is_empty()
    }

    fn finalise(&mut self) -> Result<(), ValidationError> {
        // After all instructions are processed, check that the stack state
        // matches the expected function return types
        if !self.ctrls.is_empty() {
            return Err(ValidationError::UnexpectedEndOfFunction);
        }

        // Check that the stack has exactly the expected return values
        let expected_count = self.function_type.return_types.len();
        let actual_count = self.vals.len();

        if actual_count != expected_count {
            return Err(ValidationError::InvalidResultArity);
        }

        // Check that the types match
        for (i, expected_type) in self.function_type.return_types.iter().enumerate() {
            if let Some(Val(actual_type)) = self.vals.get(i)
                && actual_type != expected_type
            {
                return Err(ValidationError::TypeMismatch);
            }
        }

        Ok(())
    }
}

fn check_alignment(memarg: &crate::parser::instruction::MemArg, align_exponent: u32) -> Result<(), ValidationError> {
    if memarg.align > align_exponent {
        return Err(ValidationError::BadAlignment);
    }
    Ok(())
}

fn validate_lane_index(lane: u8, num_lanes: u8) -> Result<(), ValidationError> {
    if lane >= num_lanes {
        return Err(ValidationError::InvalidLaneIndex);
    }
    Ok(())
}

// ============================================================================
// Module-level validation
// ============================================================================

/// Validate an already-parsed Module (constant expressions and function bodies).
///
/// This runs the same type-level validation that the binary parser performs inline
/// during decoding, enabling WAT-parsed modules to be validated without an
/// encode→decode round-trip.
pub(crate) fn validate_module(module: &Module) -> Result<(), ValidationError> {
    let total_functions = (module.imports.function_count() + module.functions.len()) as u32;
    let total_tables = (module.imports.table_count() + module.table.tables.len()) as u32;
    let total_memories = (module.imports.memory_count() + module.memory.memory.len()) as u32;
    let total_globals = (module.imports.global_count() + module.globals.globals.len()) as u32;
    let total_types = module.types.types.len() as u32;

    // Multiple memories are not allowed in WebAssembly 1.0
    if total_memories > 1 {
        return Err(ValidationError::MultipleMemories);
    }

    // Validate memory limits
    for mem in &module.memory.memory {
        validate_limits(&mem.limits, 65536, ValidationError::MemorySizeTooLarge)?;
    }

    // Validate table limits
    for table in &module.table.tables {
        validate_limits(&table.limits, u32::MAX, ValidationError::TableSizeTooLarge)?;
    }

    // Validate imported memory/table limits and function type references
    for import in &module.imports.imports {
        match &import.external_kind {
            ExternalKind::Memory(limits) => {
                validate_limits(limits, 65536, ValidationError::MemorySizeTooLarge)?;
            }
            ExternalKind::Table(table) => {
                validate_limits(&table.limits, u32::MAX, ValidationError::TableSizeTooLarge)?;
            }
            ExternalKind::Function(type_idx) => {
                if *type_idx >= total_types {
                    return Err(ValidationError::UnknownType);
                }
            }
            ExternalKind::Global(_) => {}
        }
    }

    // Validate start function
    if module.start.has_position() {
        let start_idx = module.start.start;
        if start_idx >= total_functions {
            return Err(ValidationError::UnknownStartFunction);
        }
        // Start function must have type [] -> []
        if let Some(ft) = module.get_function_type_by_idx(start_idx)
            && (!ft.parameters.is_empty() || !ft.return_types.is_empty())
        {
            return Err(ValidationError::StartFunctionType);
        }
    }

    // Validate exports: check for duplicate names and valid indices
    {
        let mut seen_names = std::collections::HashSet::new();
        for export in &module.exports.exports {
            if !seen_names.insert(&export.name) {
                return Err(ValidationError::DuplicateExportName);
            }
            match export.index {
                ExportIndex::Function(idx) => {
                    if idx >= total_functions {
                        return Err(ValidationError::UnknownFunctionExport(idx));
                    }
                }
                ExportIndex::Table(idx) => {
                    if idx >= total_tables {
                        return Err(ValidationError::UnknownTableExport(idx));
                    }
                }
                ExportIndex::Memory(idx) => {
                    if idx >= total_memories {
                        return Err(ValidationError::UnknownMemoryExport(idx));
                    }
                }
                ExportIndex::Global(idx) => {
                    if idx >= total_globals {
                        return Err(ValidationError::UnknownGlobalExport(idx));
                    }
                }
            }
        }
    }

    // Validate function type indices
    for func in &module.functions.functions {
        if func.ftype_index >= total_types {
            return Err(ValidationError::UnknownType);
        }
    }

    // Validate element segment table references and ref_type compatibility
    for elem in &module.elements.elements {
        if let ElementMode::Active { table_index, .. } = &elem.mode {
            if *table_index >= total_tables {
                return Err(ValidationError::UnknownTableWithIndex(*table_index));
            }
            // Element ref_type must match target table's ref_type
            let table_ref_type = get_table_ref_type(module, *table_index);
            if let Some(trt) = table_ref_type
                && trt != elem.ref_type
            {
                return Err(ValidationError::TypeMismatch);
            }
        }
    }

    // Validate data segment memory references
    for data in &module.data.data {
        if let DataMode::Active { memory_index, .. } = &data.mode
            && *memory_index >= total_memories
        {
            return Err(ValidationError::UnknownMemoryWithIndex(*memory_index));
        }
    }

    // Validate global initialiser expressions
    for global in &module.globals.globals {
        let mut v = ConstantExpressionValidator::new(&module.imports, global.global_type.value_type)
            .with_function_count(total_functions);
        validate_const_expr(&global.init, &mut v)?;
    }

    // Validate element segments
    for elem in &module.elements.elements {
        // Active element offset must evaluate to i32
        if let ElementMode::Active { offset, .. } = &elem.mode {
            let mut v = ConstantExpressionValidator::new(&module.imports, I32);
            validate_const_expr(offset, &mut v)?;
        }

        // Each init expression must produce the element's ref type
        let init_type = match elem.ref_type {
            RefType::FuncRef => FuncRef,
            RefType::ExternRef => ExternRef,
        };
        for init in &elem.init {
            let mut v =
                ConstantExpressionValidator::new(&module.imports, init_type).with_function_count(total_functions);
            validate_const_expr(init, &mut v)?;
        }
    }

    // Validate data segment offsets
    for data in &module.data.data {
        if let DataMode::Active { offset, .. } = &data.mode {
            let mut v = ConstantExpressionValidator::new(&module.imports, I32);
            validate_const_expr(offset, &mut v)?;
        }
    }

    // Validate function bodies
    let ctx = module.validation_context();
    let import_func_count = module.imports.function_count() as u32;

    for (i, body) in module.code.code.iter().enumerate() {
        let func_index = import_func_count + i as u32;
        let ftype = module
            .get_function_type(i)
            .ok_or(ValidationError::UnknownFunctionType)?;

        let mut v = CodeValidator::new(module, ctx, &body.locals, ftype, func_index);
        validate_structured(&body.body.body, &mut v)?;
        v.validate(&validation_instr(InstructionKind::End))?;
        v.finalise()?;
    }

    Ok(())
}

// Walk a structured instruction tree, feeding each instruction to the validator
// in the flat order it expects (Block → body → End, etc.).
fn validate_structured(body: &[StructuredInstruction], validator: &mut impl Validator) -> Result<(), ValidationError> {
    for si in body {
        match si {
            StructuredInstruction::Plain(instr) => {
                validator.validate(instr)?;
            }
            StructuredInstruction::Block { block_type, body, .. } => {
                validator.validate(&validation_instr(InstructionKind::Block {
                    block_type: *block_type,
                }))?;
                validate_structured(body, validator)?;
                validator.validate(&validation_instr(InstructionKind::End))?;
            }
            StructuredInstruction::Loop { block_type, body, .. } => {
                validator.validate(&validation_instr(InstructionKind::Loop {
                    block_type: *block_type,
                }))?;
                validate_structured(body, validator)?;
                validator.validate(&validation_instr(InstructionKind::End))?;
            }
            StructuredInstruction::If {
                block_type,
                then_branch,
                else_branch,
                ..
            } => {
                validator.validate(&validation_instr(InstructionKind::If {
                    block_type: *block_type,
                }))?;
                validate_structured(then_branch, validator)?;
                if let Some(else_body) = else_branch {
                    validator.validate(&validation_instr(InstructionKind::Else))?;
                    validate_structured(else_body, validator)?;
                }
                validator.validate(&validation_instr(InstructionKind::End))?;
            }
        }
    }
    Ok(())
}

// Look up a table's ref_type by index (imports first, then local tables).
fn get_table_ref_type(module: &Module, table_index: u32) -> Option<RefType> {
    let import_table_count = module.imports.table_count() as u32;
    if table_index < import_table_count {
        let mut table_idx = 0u32;
        for import in &module.imports.imports {
            if let ExternalKind::Table(tt) = &import.external_kind {
                if table_idx == table_index {
                    return Some(tt.ref_type);
                }
                table_idx += 1;
            }
        }
        None
    } else {
        let local_idx = (table_index - import_table_count) as usize;
        module.table.tables.get(local_idx).map(|t| t.ref_type)
    }
}

// Validate limits: min <= max (if present), and both <= absolute_max.
fn validate_limits(limits: &Limits, absolute_max: u32, too_large: ValidationError) -> Result<(), ValidationError> {
    if let Some(max) = limits.max {
        if limits.min > max {
            return Err(ValidationError::MinGreaterThanMax);
        }
        if max > absolute_max {
            return Err(too_large);
        }
    }
    if limits.min > absolute_max {
        return Err(too_large);
    }
    Ok(())
}

// Feed a flat constant expression through the validator and finalise.
fn validate_const_expr(instrs: &[Instruction], validator: &mut impl Validator) -> Result<(), ValidationError> {
    for instr in instrs {
        validator.validate(instr)?;
    }
    validator.finalise()
}

/// Synthesise a minimal Instruction for validation (only `kind` is inspected).
fn validation_instr(kind: InstructionKind) -> Instruction {
    Instruction {
        kind,
        position: ByteRange { offset: 0, length: 0 },
        original_bytes: Vec::new(),
    }
}
