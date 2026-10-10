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
//!
//! Validation is the runtime's trust boundary. [`ValidatedModule`] is the only
//! form a module can be instantiated in, and execution relies on what validation
//! proves (operand types, stack heights, index bounds) instead of rechecking
//! them per instruction.

use super::index_spaces::IndexSpaceTypes;
use super::module::{
    DataMode, ElementMode, ExportIndex, ExternalKind, FunctionType, GlobalType, Limits, Module, Positional, RefType,
    TableType, ValueType, ValueType::*,
};
use super::structured::StructuredInstruction;
use crate::parser::instruction::signature::{Signature, context_signature, fixed_signature, simd_signature};
use crate::parser::instruction::{BlockType, ByteRange, Instruction, InstructionKind, SimdOp};
use InstructionKind::*;
use MaybeValue::{Unknown, Val};
use SimdOp::*;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};
use thiserror::Error;

/// A module that passed validation: the form [`Store::create_instance`] accepts.
///
/// The parsers produce it directly. A module built or edited by hand becomes
/// one through [`ValidatedModule::new`]. It is shared and immutable, so what was
/// validated is what runs; read it through `Deref`, and recover the `Module`
/// with [`ValidatedModule::into_inner`] to change it and validate it again.
///
/// [`Store::create_instance`]: crate::Store::create_instance
#[derive(Debug, Clone)]
pub struct ValidatedModule(Arc<Module>);

impl ValidatedModule {
    /// Validate a module, typically one built or edited by hand.
    pub fn new(module: Module) -> Result<Self, ValidationError> {
        validate_module(&module)?;
        Ok(Self(Arc::new(module)))
    }

    /// Wrap a module whose producer validated it as it was built.
    pub(crate) fn from_validated(module: Module) -> Self {
        Self(Arc::new(module))
    }

    /// Recover the module for editing, or return `self` while it is shared
    /// with instances or other handles.
    pub fn into_inner(self) -> Result<Module, Self> {
        let mut module = Arc::try_unwrap(self.0).map_err(Self)?;
        // Edits can invalidate the cached lookups.
        module.index_space_types = OnceLock::new();
        Ok(module)
    }

    pub(crate) fn shared(&self) -> &Arc<Module> {
        &self.0
    }
}

impl Deref for ValidatedModule {
    type Target = Module;

    fn deref(&self) -> &Module {
        &self.0
    }
}

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("constant expression required")]
    ConstantExpressionRequired,

    #[error("type mismatch")]
    TypeMismatch,

    #[error("function and code section have inconsistent lengths")]
    FunctionCodeCountMismatch,

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

    #[error("too many locals")]
    TooManyLocals,

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
    spaces: &'a IndexSpaceTypes,
    /// Parameter then declared local types, by local index.
    locals: Vec<ValueType>,
    function_type: &'a FunctionType,
    vals: Vec<MaybeValue>,
    ctrls: Vec<CtrlFrame>,
    need_end: bool,
    current_function_index: u32,
}

impl<'a> CodeValidator<'a> {
    fn table(&self, index: u32) -> Option<TableType> {
        self.spaces.table_types.get(index as usize).copied()
    }

    #[inline]
    fn validate_memory_exists(&self) -> Result<(), ValidationError> {
        if !self.spaces.has_memory {
            return Err(ValidationError::UnknownMemory);
        }
        Ok(())
    }

    pub fn new(
        module: &'a super::module::Module,
        locals: &'a super::module::Locals,
        function_type: &'a FunctionType,
        current_function_index: u32,
    ) -> CodeValidator<'a> {
        let mut v = CodeValidator {
            module,
            spaces: module.index_space_types(),
            locals: locals.types(&function_type.parameters),
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

    fn push_val(&mut self, val_type: MaybeValue) {
        self.vals.push(val_type);
    }

    fn push_vals(&mut self, val_types: Vec<MaybeValue>) {
        self.vals.extend(val_types);
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

    /// Pop a signature's parameters, last first, and push its results.
    fn apply_signature(&mut self, signature: Signature) -> Result<(), ValidationError> {
        for param in signature.params.iter().rev() {
            self.pop_expected(Val(*param)).ok_or(ValidationError::TypeMismatch)?;
        }
        for result in signature.results {
            self.push_val(Val(*result));
        }
        Ok(())
    }

    /// Apply the signature an instruction's opcode fixes; see [`fixed_signature`].
    fn apply_fixed(&mut self, kind: &InstructionKind) -> Result<(), ValidationError> {
        let signature = fixed_signature(kind).ok_or(ValidationError::InvalidInstruction)?;
        self.apply_signature(signature)
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
        self.spaces
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
        match op {
            // [i32] → [v128] : loads (pop address, push result)
            V128Load { memarg } => {
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
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
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 3)
            }
            V128Load32Splat { memarg } | V128Load32Zero { memarg } => {
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 2)
            }
            V128Load16Splat { memarg } => {
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 1)
            }
            V128Load8Splat { memarg } => {
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 0)
            }
            V128Load64Zero { memarg } => {
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 3)
            }

            // [i32 v128] → [] : stores (pop v128 value, pop i32 address)
            V128Store { memarg } => {
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 4)
            }

            // [i32 v128] → [v128] : load lane (pop v128, pop i32, push v128)
            V128Load8Lane { memarg, lane } => {
                validate_lane_index(*lane, 16)?;
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 0)
            }
            V128Load16Lane { memarg, lane } => {
                validate_lane_index(*lane, 8)?;
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 1)
            }
            V128Load32Lane { memarg, lane } => {
                validate_lane_index(*lane, 4)?;
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 2)
            }
            V128Load64Lane { memarg, lane } => {
                validate_lane_index(*lane, 2)?;
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 3)
            }

            // [i32 v128] → [] : store lane (pop v128, pop i32)
            V128Store8Lane { memarg, lane } => {
                validate_lane_index(*lane, 16)?;
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 0)
            }
            V128Store16Lane { memarg, lane } => {
                validate_lane_index(*lane, 8)?;
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 1)
            }
            V128Store32Lane { memarg, lane } => {
                validate_lane_index(*lane, 4)?;
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 2)
            }
            V128Store64Lane { memarg, lane } => {
                validate_lane_index(*lane, 2)?;
                self.validate_memory_exists()?;
                self.apply_signature(simd_signature(op))?;
                check_alignment(memarg, 3)
            }

            // [v128 v128] → [v128] : shuffle (lane indices must be 0-31)
            I8x16Shuffle { lanes } => {
                for &lane in lanes.iter() {
                    if lane >= 32 {
                        return Err(ValidationError::InvalidLaneIndex);
                    }
                }
                self.apply_signature(simd_signature(op))
            }

            // [v128] → [i32] : extract lane to i32 (with lane bounds check)
            I8x16ExtractLaneS { lane } | I8x16ExtractLaneU { lane } => {
                validate_lane_index(*lane, 16)?;
                self.apply_signature(simd_signature(op))
            }
            I16x8ExtractLaneS { lane } | I16x8ExtractLaneU { lane } => {
                validate_lane_index(*lane, 8)?;
                self.apply_signature(simd_signature(op))
            }
            I32x4ExtractLane { lane } => {
                validate_lane_index(*lane, 4)?;
                self.apply_signature(simd_signature(op))
            }

            // [v128] → [i64] : extract i64 lane
            I64x2ExtractLane { lane } => {
                validate_lane_index(*lane, 2)?;
                self.apply_signature(simd_signature(op))
            }

            // [v128] → [f32] : extract f32 lane
            F32x4ExtractLane { lane } => {
                validate_lane_index(*lane, 4)?;
                self.apply_signature(simd_signature(op))
            }

            // [v128] → [f64] : extract f64 lane
            F64x2ExtractLane { lane } => {
                validate_lane_index(*lane, 2)?;
                self.apply_signature(simd_signature(op))
            }

            // [v128 i32] → [v128] : replace lane (i32 replacement value)
            I8x16ReplaceLane { lane } => {
                validate_lane_index(*lane, 16)?;
                self.apply_signature(simd_signature(op))
            }
            I16x8ReplaceLane { lane } => {
                validate_lane_index(*lane, 8)?;
                self.apply_signature(simd_signature(op))
            }
            I32x4ReplaceLane { lane } => {
                validate_lane_index(*lane, 4)?;
                self.apply_signature(simd_signature(op))
            }

            // [v128 i64] → [v128] : i64 replace lane
            I64x2ReplaceLane { lane } => {
                validate_lane_index(*lane, 2)?;
                self.apply_signature(simd_signature(op))
            }

            // [v128 f32] → [v128] : f32 replace lane
            F32x4ReplaceLane { lane } => {
                validate_lane_index(*lane, 4)?;
                self.apply_signature(simd_signature(op))
            }

            // [v128 f64] → [v128] : f64 replace lane
            F64x2ReplaceLane { lane } => {
                validate_lane_index(*lane, 2)?;
                self.apply_signature(simd_signature(op))
            }

            // Every other SIMD instruction is checked by its signature alone.
            _ => self.apply_signature(simd_signature(op)),
        }
    }
}

impl Validator for CodeValidator<'_> {
    fn validate(&mut self, inst: &Instruction) -> Result<(), ValidationError> {
        let kind = &inst.kind;
        if let Some(signature) = context_signature(kind, &self.module.types, self.spaces, &self.locals)? {
            return signature.steps().try_for_each(|step| self.apply_signature(step));
        }
        match kind {
            I32Load { memarg }
            | I32Load8S { memarg }
            | I32Load8U { memarg }
            | I32Load16S { memarg }
            | I32Load16U { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.apply_fixed(kind)?;
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
                self.apply_fixed(kind)?;
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
                self.apply_fixed(kind)?;
                check_alignment(memarg, 2 /* 4 bytes = 2^2 */)
            }

            F64Load { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.apply_fixed(kind)?;
                check_alignment(memarg, 3 /* 8 bytes = 2^3 */)
            }

            I32Store { memarg } | I32Store8 { memarg } | I32Store16 { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.apply_fixed(kind)?;
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
                self.apply_fixed(kind)?;
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
                self.apply_fixed(kind)?;
                check_alignment(memarg, 2 /* 4 bytes = 2^2 */)
            }

            F64Store { memarg } => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.apply_fixed(kind)?;
                check_alignment(memarg, 3 /* 8 bytes = 2^3 */)
            }

            MemorySize => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.apply_fixed(kind)
            }

            MemoryGrow => {
                // Check that memory exists (imported or declared)
                self.validate_memory_exists()?;
                self.apply_fixed(kind)
            }

            MemoryFill | MemoryCopy => {
                // Check that memory exists (imported or declared)
                if self.module.memory.is_empty() && self.module.imports.memory_count() == 0 {
                    return Err(ValidationError::UnknownMemoryWithIndex(0));
                }
                self.apply_fixed(kind)
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
                self.apply_fixed(kind)
            }

            TableInit { elem_idx, table_idx } => {
                self.apply_fixed(kind)?;
                let table = self
                    .table(*table_idx)
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
                self.apply_fixed(kind)?;
                let src_table_type = self.table(*src_table).ok_or(ValidationError::UnknownTable)?;
                let dst_table_type = self.table(*dst_table).ok_or(ValidationError::UnknownTable)?;
                if src_table_type.ref_type != dst_table_type.ref_type {
                    return Err(ValidationError::TypeMismatch);
                }
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

                // An if without else has an implicit empty else, which passes
                // its parameters through as its results: their types must be
                // equal, however the then-branch ended.
                if matches!(ctrl.instruction, If { .. }) && ctrl.start_types != ctrl.end_types {
                    return Err(ValidationError::TypeMismatch);
                }

                self.push_vals(ctrl.end_types);
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
                self.push_vals(label_types);
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
                            self.push_vals(label_types);
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
                        self.push_val(if t1 == Unknown { t2.clone() } else { t1.clone() });
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
                self.push_val(Val(result_type));
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
                self.apply_fixed(kind)
            }

            RefNull { ref_type } => {
                if !matches!(ref_type, FuncRef | ExternRef) {
                    return Err(ValidationError::TypeMismatch);
                }
                self.push_val(Val(*ref_type));
                Ok(())
            }

            RefIsNull => {
                // ref.is_null pops a reference type and pushes i32
                let ref_type = self.pop_val().ok_or(ValidationError::TypeMismatch)?;
                // Check that popped value is a reference type
                if !ref_type.is_ref() {
                    return Err(ValidationError::TypeMismatch);
                }
                self.push_val(Val(I32));
                Ok(())
            }

            Drop => {
                self.pop_val().ok_or(ValidationError::TypeMismatch)?;
                Ok(())
            }

            Nop => Ok(()),

            TableSize { table_idx } => {
                // Verify table exists
                self.table(*table_idx).ok_or(ValidationError::UnknownTable)?;
                self.apply_fixed(kind)
            }

            Simd(op) => self.validate_simd(op),

            // Every other instruction is checked by its signature alone.
            _ => self.apply_fixed(kind),
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

    if module.functions.len() != module.code.code.len() {
        return Err(ValidationError::FunctionCodeCountMismatch);
    }

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
            if let Some(table) = module.get_table(*table_index)
                && table.ref_type != elem.ref_type
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
    let import_func_count = module.imports.function_count() as u32;

    for (i, body) in module.code.code.iter().enumerate() {
        let func_index = import_func_count + i as u32;
        let ftype = module
            .get_function_type(i)
            .ok_or(ValidationError::UnknownFunctionType)?;
        // The binary decoder applies the same limit before validating.
        if body.locals.len() > u64::from(super::limits::MAX_FUNCTION_LOCALS) {
            return Err(ValidationError::TooManyLocals);
        }

        let mut v = CodeValidator::new(module, &body.locals, ftype, func_index);
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
