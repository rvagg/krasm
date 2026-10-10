//! Compiler from structured instruction tree to flat bytecode.
//!
//! Walks the `StructuredFunction` tree and emits a linear `Vec<Op>` with
//! pre-resolved branch targets. The key transformation is converting wasm's
//! depth-relative branch indices (`br 2`) into absolute bytecode positions
//! (`br -> 47`).
//!
//! The compiler maintains a label stack during tree traversal. Each block/loop
//! pushes a label; `br N` looks up N levels to find the target:
//! - For blocks: the target is the instruction after the block's End
//!   (forward jump, patched after the body is emitted)
//! - For loops: the target is the loop's first body instruction
//!   (backward jump, known at entry)
//!
//! Branch ops carry stack cleanup metadata (arity + stack_depth) so the flat
//! executor can properly handle multi-value blocks without a runtime label
//! stack. The compiler tracks the operand types during emission, as
//! validation does, to compute these values in words, each function's
//! maximum stack size, and the word width of `drop`, `select` and locals.
//! It checks each instruction's operands against those types, so any
//! disagreement with validation fails compilation instead of producing
//! bytecode that relies on a false assumption.

use super::RuntimeError;
use super::bytecode::{BrTarget, CompiledFunction, Op, value_words};
use crate::parser::instruction::signature::{Signature, context_signature, fixed_signature};
use crate::parser::instruction::{BlockType, Instruction, InstructionKind, SimdOp};
use crate::parser::module::{FunctionType, Locals, Module, ValueType};
use crate::parser::structured::{StructuredFunction, StructuredInstruction};
use std::fmt;

/// A forward branch whose target is not yet known at the time of emission.
///
/// When the compiler emits a branch into a block, the block's end position
/// hasn't been determined yet. The branch is emitted with `target: 0` as a
/// placeholder, and a Fixup is recorded in the block's label. When the block
/// ends and we know the target position, all accumulated fixups are applied
/// to write the real target into the emitted ops.
///
/// Loop labels don't need fixups because their target (the loop start) is
/// already known when the branch is emitted.
enum Fixup {
    /// A Br, BrIf, or Label op at this position needs its target patched.
    Branch(usize),
    /// A BrTable op at `op_pos` needs one of its entries patched.
    /// `Some(i)` patches `targets[i].pc`, `None` patches `default.pc`.
    TableEntry { op_pos: usize, index: Option<usize> },
}

/// A label on the compile-time stack: where its branches go, the operand
/// depth they unwind to, and the types they carry (a block's results, or a
/// loop's parameters).
struct Label<'a> {
    target: LabelTarget,
    stack_depth: u32,
    types: &'a [ValueType],
    /// `stack_depth` and the words `types` occupy, as branch ops count them.
    word_depth: u32,
    word_arity: u16,
}

enum LabelTarget {
    /// Block, if or function body: branches go to the end (forward), unknown
    /// until the body is emitted, so their positions are recorded for patching.
    End(Vec<Fixup>),
    /// Loop: branches go to the start (backward), known at entry.
    Start(u32),
}

type Result<T = ()> = std::result::Result<T, RuntimeError>;

/// A module that validation should have rejected.
fn invalid(what: impl fmt::Display) -> RuntimeError {
    RuntimeError::Compile(what.to_string())
}

/// A parameter, result or branch-arity count, which validation keeps far below
/// `u16::MAX` for decoded modules; others are checked rather than truncated.
fn count(len: usize) -> Result<u16> {
    u16::try_from(len).map_err(|_| invalid(format!("{len} values exceed the supported count")))
}

/// Words a sequence of values occupies.
fn words_of(types: &[ValueType]) -> u32 {
    types.iter().map(|value_type| value_words(*value_type)).sum()
}

/// Compile all local functions in a module to flat bytecode.
///
/// The module must be valid; compilation relies on validation and reports any
/// inconsistency it meets as [`RuntimeError::Compile`] rather than guessing.
pub fn compile_module(module: &Module) -> Result<Vec<CompiledFunction>> {
    if module.code.code.len() != module.functions.functions.len() {
        return Err(invalid("function and code counts differ"));
    }
    module
        .code
        .code
        .iter()
        .zip(&module.functions.functions)
        .map(|(func, decl)| {
            let ftype = module
                .types
                .get(decl.ftype_index)
                .ok_or_else(|| invalid(format!("unknown function type {}", decl.ftype_index)))?;
            compile(&func.body, ftype, &func.locals, module)
        })
        .collect()
}

/// Compile a structured function into flat bytecode.
///
/// `ftype` is the function's signature and `locals` its declared
/// (non-parameter) locals, which also give the typed zero defaults for frame
/// initialisation. `module` is where block types, callees, globals and tables
/// are looked up.
pub fn compile<'a>(
    func: &'a StructuredFunction,
    ftype: &'a FunctionType,
    locals: &Locals,
    module: &'a Module,
) -> Result<CompiledFunction> {
    let locals = locals.types(&ftype.parameters);
    let local_offsets = (locals.iter())
        .scan(0, |next, value_type| {
            let first = *next;
            *next += value_words(*value_type);
            Some(first)
        })
        .collect();
    let mut ctx = CompileContext {
        ops: Vec::new(),
        labels: Vec::new(),
        stack: Vec::new(),
        words: 0,
        max_words: 0,
        reachable: true,
        locals,
        local_offsets,
        module,
    };

    // The function body is the outermost label. A `br` to it is a return; a
    // `br_if` or `br_table` entry jumps to the final End.
    ctx.open_label(LabelTarget::End(Vec::new()), 0, &ftype.return_types)?;
    ctx.emit_body(&func.body)?;
    if ctx.reachable {
        ctx.check_top(0, &ftype.return_types, "function end")?;
    }
    ctx.close_label();
    ctx.ops.push(Op::End);

    Ok(CompiledFunction {
        ops: ctx.ops,
        result_types: ftype.return_types.clone().into_boxed_slice(),
        param_words: words_of(&ftype.parameters),
        local_words: words_of(&ctx.locals),
        result_words: words_of(&ftype.return_types),
        max_stack_words: ctx.max_words,
    })
}

struct CompileContext<'a> {
    ops: Vec<Op>,
    /// Compile-time label stack. Innermost label is last.
    labels: Vec<Label<'a>>,
    /// Types of the operand stack's values, top last. Its length is the depth.
    stack: Vec<ValueType>,
    /// Words the operand stack occupies, and the most it reaches.
    words: u32,
    max_words: u32,
    /// False after an unconditional transfer (br, br_table, return,
    /// unreachable) until the enclosing block ends. Dead code is skipped
    /// entirely: validation allows it to be stack-polymorphic, so tracking
    /// depth through it is meaningless (and would underflow).
    reachable: bool,
    /// Parameter then declared local types, by local index.
    locals: Vec<ValueType>,
    /// Each local's first word in the frame, by local index.
    local_offsets: Vec<u32>,
    /// Block and callee signatures, and function, global and table types.
    module: &'a Module,
}

impl<'a> CompileContext<'a> {
    /// Number of values on the operand stack.
    fn depth(&self) -> u32 {
        self.stack.len() as u32
    }

    fn push(&mut self, value_type: ValueType) {
        self.stack.push(value_type);
        self.words += value_words(value_type);
        self.max_words = self.max_words.max(self.words);
    }

    /// The depth the innermost label unwinds to. Operands beneath it belong
    /// to enclosing blocks and cannot be popped.
    fn floor(&self) -> u32 {
        self.labels.last().map_or(0, |label| label.stack_depth)
    }

    fn pop(&mut self) -> Result<ValueType> {
        let depth = self.depth_below(1)?;
        let popped = self.stack[depth as usize];
        self.truncate(depth);
        Ok(popped)
    }

    fn pop_expect(&mut self, expected: ValueType) -> Result {
        match self.pop()? {
            found if found == expected => Ok(()),
            found => Err(invalid(format!("expected {expected:?} operand, found {found:?}"))),
        }
    }

    /// Pop parameters, last first, and push results.
    fn apply_signature(&mut self, signature: Signature) -> Result {
        for param in signature.params.iter().rev() {
            self.pop_expect(*param)?;
        }
        for result in signature.results {
            self.push(*result);
        }
        Ok(())
    }

    /// Discard operands down to `depth`.
    fn truncate(&mut self, depth: u32) {
        let depth = self.stack.len().min(depth as usize);
        for value_type in self.stack.drain(depth..) {
            self.words -= value_words(value_type);
        }
    }

    /// Check that the operands above `depth` are exactly `expected`.
    fn check_top(&self, depth: u32, expected: &[ValueType], what: &str) -> Result {
        let above = self.stack.get(depth as usize..).unwrap_or_default();
        if above != expected {
            return Err(invalid(format!("{what} has {above:?}, expected {expected:?}")));
        }
        Ok(())
    }

    /// The depth beneath the top `count` operands, which must sit above the floor.
    fn depth_below(&self, count: usize) -> Result<u32> {
        (self.depth().checked_sub(count as u32))
            .filter(|depth| *depth >= self.floor())
            .ok_or_else(|| invalid("operand stack underflow"))
    }

    /// Resolve a BlockType to its parameter and result types.
    fn block_signature(&self, block_type: &'a BlockType) -> Result<Signature<'a>> {
        Ok(match block_type {
            BlockType::Empty => Signature {
                params: &[],
                results: &[],
            },
            BlockType::Value(value_type) => Signature {
                params: &[],
                results: std::slice::from_ref(value_type),
            },
            BlockType::FuncType(idx) => (self.module.types.get(*idx))
                .ok_or_else(|| invalid(format!("unknown block type {idx}")))?
                .signature(),
        })
    }

    /// Apply a plain instruction's effect on the operand types.
    fn apply_types(&mut self, kind: &InstructionKind) -> Result {
        if let Some(signature) = fixed_signature(kind) {
            return self.apply_signature(signature);
        }
        let module = self.module;
        let context = context_signature(kind, &module.types, module.index_space_types(), &self.locals);
        if let Some(signature) = context.map_err(invalid)? {
            return signature.steps().try_for_each(|step| self.apply_signature(step));
        }
        match kind {
            InstructionKind::Drop => self.pop().map(|_| ()),
            InstructionKind::Select => {
                self.pop_expect(ValueType::I32)?;
                // The first operand must match the second, and the result takes their type.
                let second = self.pop()?;
                self.apply_signature(Signature {
                    params: &[second],
                    results: &[second],
                })
            }
            InstructionKind::SelectTyped { val_types } => match val_types.as_slice() {
                [value_type] => self.apply_signature(Signature {
                    params: &[*value_type, *value_type, ValueType::I32],
                    results: std::slice::from_ref(value_type),
                }),
                _ => Err(invalid("select must name one type")),
            },
            InstructionKind::RefNull { ref_type } => {
                self.push(*ref_type);
                Ok(())
            }
            InstructionKind::RefIsNull => {
                let operand = self.pop()?;
                if !matches!(operand, ValueType::FuncRef | ValueType::ExternRef) {
                    return Err(invalid(format!("ref.is_null of {operand:?}")));
                }
                self.push(ValueType::I32);
                Ok(())
            }
            InstructionKind::BrIf { .. } | InstructionKind::BrTable { .. } => self.pop_expect(ValueType::I32),
            // Control leaves; what follows is unreachable and not emitted.
            InstructionKind::Br { .. } | InstructionKind::Return | InstructionKind::Unreachable => Ok(()),
            _ => Err(invalid(format!("no operand types for {kind:?}"))),
        }
    }

    /// A local's first word and its type. Typing has already checked the index.
    fn local(&self, index: u32) -> (u32, ValueType) {
        (self.local_offsets[index as usize], self.locals[index as usize])
    }

    /// Index into `labels` of the label `depth` levels out from the innermost.
    fn label_index(&self, depth: u32) -> Result<usize> {
        (self.labels.len().checked_sub(1 + depth as usize)).ok_or_else(|| invalid(format!("unknown label {depth}")))
    }

    /// Current emit position (index of the next op to be pushed).
    fn pos(&self) -> u32 {
        self.ops.len() as u32
    }

    /// Emit an op; the one-line arm form for mechanical 1:1 instruction mappings.
    fn op(&mut self, op: Op) {
        self.emit(op);
    }

    /// Emit an op and return its position. Operand types follow the source
    /// instruction (`apply_types`), not the ops emitted for it.
    fn emit(&mut self, op: Op) -> usize {
        let pos = self.ops.len();
        self.ops.push(op);
        pos
    }

    /// Patch the target pc of a Br, BrIf or Label op, or of one br_table entry.
    fn apply_fixup(&mut self, fixup: &Fixup, target: u32) {
        match fixup {
            Fixup::Branch(pos) => match &mut self.ops[*pos] {
                Op::Br { target: t, .. } | Op::BrIf { target: t, .. } => *t = target,
                Op::Label { end_target } => *end_target = target,
                _ => {}
            },
            Fixup::TableEntry { op_pos, index } => {
                if let Op::BrTable { targets, default } = &mut self.ops[*op_pos] {
                    match index {
                        Some(i) => targets[*i].pc = target,
                        None => default.pc = target,
                    }
                }
            }
        }
    }

    /// Push a label, checking its arity fits in a branch op.
    fn open_label(&mut self, target: LabelTarget, stack_depth: u32, types: &'a [ValueType]) -> Result {
        let word_arity = count(words_of(types) as usize)?;
        self.labels.push(Label {
            target,
            stack_depth,
            types,
            word_depth: words_of(&self.stack[..stack_depth as usize]),
            word_arity,
        });
        Ok(())
    }

    /// Pop the innermost label, pointing its forward branches here.
    fn close_label(&mut self) {
        let end = self.pos();
        if let Some(Label {
            target: LabelTarget::End(fixups),
            ..
        }) = self.labels.pop()
        {
            for fixup in &fixups {
                self.apply_fixup(fixup, end);
            }
        }
    }

    /// Continue after a block: reachable, with `types` above `depth`.
    fn end_block(&mut self, depth: u32, types: &[ValueType]) {
        self.truncate(depth);
        for value_type in types {
            self.push(*value_type);
        }
        self.reachable = true;
    }

    /// Emit a sequence of structured instructions.
    fn emit_body(&mut self, body: &'a [StructuredInstruction]) -> Result {
        body.iter().try_for_each(|inst| self.emit_instruction(inst))
    }

    /// Emit a single structured instruction.
    fn emit_instruction(&mut self, inst: &'a StructuredInstruction) -> Result {
        // Dead code after an unconditional transfer is never emitted.
        // Branch targets cannot point into it, so nothing is lost.
        if !self.reachable {
            return Ok(());
        }
        match inst {
            StructuredInstruction::Plain(i) => self.emit_plain(i)?,

            StructuredInstruction::Block { block_type, body, .. } => {
                let Signature { params, results } = self.block_signature(block_type)?;
                // Params are inside the block: branches unwind to below them.
                let sd = self.depth_below(params.len())?;
                self.check_top(sd, params, "block entry")?;

                let label_pos = self.emit(Op::Label { end_target: 0 });
                self.open_label(LabelTarget::End(vec![Fixup::Branch(label_pos)]), sd, results)?;
                self.emit_body(body)?;
                if self.reachable {
                    self.check_top(sd, results, "block end")?;
                }
                self.close_label();
                // Reached by fall-through or by a branch to the label: either
                // way the block's results sit above sd.
                self.end_block(sd, results);
            }

            StructuredInstruction::Loop { block_type, body, .. } => {
                let Signature { params, results } = self.block_signature(block_type)?;
                let sd = self.depth_below(params.len())?;
                self.check_top(sd, params, "loop entry")?;

                // Branches restart the loop, carrying its params.
                let loop_start = self.emit(Op::Label { end_target: 0 });
                self.open_label(LabelTarget::Start(loop_start as u32 + 1), sd, params)?;
                self.emit_body(body)?;
                if self.reachable {
                    self.check_top(sd, results, "loop end")?;
                }
                let end = self.pos();
                self.apply_fixup(&Fixup::Branch(loop_start), end);
                self.labels.pop();
                self.end_block(sd, results);
            }

            StructuredInstruction::If {
                block_type,
                then_branch,
                else_branch,
                ..
            } => {
                let Signature { params, results } = self.block_signature(block_type)?;
                // The i32.eqz + br_if below consume the condition; the block's
                // params sit directly beneath it.
                self.pop_expect(ValueType::I32)?;
                let sd = self.depth_below(params.len())?;
                self.check_top(sd, params, "if entry")?;

                // Skip the then-body when the condition is 0, keeping the params.
                self.emit(Op::I32Eqz);
                let skip_then = self.emit(Op::BrIf {
                    target: 0,
                    arity: 0,
                    stack_depth: self.words,
                });
                self.open_label(LabelTarget::End(Vec::new()), sd, results)?;
                self.emit_body(then_branch)?;
                if self.reachable {
                    self.check_top(sd, results, "if end")?;
                }

                if let Some(else_body) = else_branch {
                    // The then-body jumps past the else, carrying the results.
                    let label = &self.labels[self.labels.len() - 1];
                    let skip_else = self.emit(Op::Br {
                        target: 0,
                        arity: label.word_arity,
                        stack_depth: label.word_depth,
                    });
                    if let Some(Label {
                        target: LabelTarget::End(fixups),
                        ..
                    }) = self.labels.last_mut()
                    {
                        fixups.push(Fixup::Branch(skip_else));
                    }
                    // The skip enters the else with the params on the stack,
                    // however the then-body ended.
                    let else_start = self.pos();
                    self.apply_fixup(&Fixup::Branch(skip_then), else_start);
                    self.end_block(sd, params);
                    self.emit_body(else_body)?;
                    if self.reachable {
                        self.check_top(sd, results, "else end")?;
                    }
                } else {
                    // The implicit empty else passes the params through as
                    // the results.
                    if params != results {
                        return Err(invalid(format!("if without else turns {params:?} into {results:?}")));
                    }
                    let end = self.pos();
                    self.apply_fixup(&Fixup::Branch(skip_then), end);
                }
                self.close_label();
                self.end_block(sd, results);
            }
        }
        Ok(())
    }

    /// Emit a plain (non-control-flow-structure) instruction.
    fn emit_plain(&mut self, inst: &'a Instruction) -> Result {
        let dropped = self.stack.last().copied();
        self.apply_types(&inst.kind)?;
        match &inst.kind {
            InstructionKind::I32Const { value } => self.op(Op::I32Const(*value)),
            InstructionKind::I64Const { value } => self.op(Op::I64Const(*value)),
            InstructionKind::F32Const { value } => self.op(Op::F32Const(*value)),
            InstructionKind::F64Const { value } => self.op(Op::F64Const(*value)),
            InstructionKind::Simd(SimdOp::V128Const { value }) => self.op(Op::V128Const(*value)),
            InstructionKind::Simd(SimdOp::I8x16Abs) => self.op(Op::I8x16Abs),
            InstructionKind::Simd(SimdOp::I8x16Neg) => self.op(Op::I8x16Neg),
            InstructionKind::Simd(SimdOp::I8x16Popcnt) => self.op(Op::I8x16Popcnt),
            InstructionKind::Simd(SimdOp::I16x8Abs) => self.op(Op::I16x8Abs),
            InstructionKind::Simd(SimdOp::I16x8Neg) => self.op(Op::I16x8Neg),
            InstructionKind::Simd(SimdOp::I32x4Abs) => self.op(Op::I32x4Abs),
            InstructionKind::Simd(SimdOp::I32x4Neg) => self.op(Op::I32x4Neg),
            InstructionKind::Simd(SimdOp::I64x2Abs) => self.op(Op::I64x2Abs),
            InstructionKind::Simd(SimdOp::I64x2Neg) => self.op(Op::I64x2Neg),
            InstructionKind::Simd(SimdOp::I8x16Splat) => self.op(Op::I8x16Splat),
            InstructionKind::Simd(SimdOp::I16x8Splat) => self.op(Op::I16x8Splat),
            InstructionKind::Simd(SimdOp::I32x4Splat) => self.op(Op::I32x4Splat),
            InstructionKind::Simd(SimdOp::I64x2Splat) => self.op(Op::I64x2Splat),
            InstructionKind::Simd(SimdOp::F32x4Splat) => self.op(Op::F32x4Splat),
            InstructionKind::Simd(SimdOp::F64x2Splat) => self.op(Op::F64x2Splat),
            InstructionKind::Simd(SimdOp::I8x16Shuffle { lanes }) => self.op(Op::I8x16Shuffle(*lanes)),
            InstructionKind::Simd(SimdOp::I8x16Swizzle) => self.op(Op::I8x16Swizzle),
            InstructionKind::Simd(SimdOp::I8x16Eq) => self.op(Op::I8x16Eq),
            InstructionKind::Simd(SimdOp::I8x16Ne) => self.op(Op::I8x16Ne),
            InstructionKind::Simd(SimdOp::I8x16LtS) => self.op(Op::I8x16LtS),
            InstructionKind::Simd(SimdOp::I8x16LtU) => self.op(Op::I8x16LtU),
            InstructionKind::Simd(SimdOp::I8x16GtS) => self.op(Op::I8x16GtS),
            InstructionKind::Simd(SimdOp::I8x16GtU) => self.op(Op::I8x16GtU),
            InstructionKind::Simd(SimdOp::I8x16LeS) => self.op(Op::I8x16LeS),
            InstructionKind::Simd(SimdOp::I8x16LeU) => self.op(Op::I8x16LeU),
            InstructionKind::Simd(SimdOp::I8x16GeS) => self.op(Op::I8x16GeS),
            InstructionKind::Simd(SimdOp::I8x16GeU) => self.op(Op::I8x16GeU),
            InstructionKind::Simd(SimdOp::I16x8Eq) => self.op(Op::I16x8Eq),
            InstructionKind::Simd(SimdOp::I16x8Ne) => self.op(Op::I16x8Ne),
            InstructionKind::Simd(SimdOp::I16x8LtS) => self.op(Op::I16x8LtS),
            InstructionKind::Simd(SimdOp::I16x8LtU) => self.op(Op::I16x8LtU),
            InstructionKind::Simd(SimdOp::I16x8GtS) => self.op(Op::I16x8GtS),
            InstructionKind::Simd(SimdOp::I16x8GtU) => self.op(Op::I16x8GtU),
            InstructionKind::Simd(SimdOp::I16x8LeS) => self.op(Op::I16x8LeS),
            InstructionKind::Simd(SimdOp::I16x8LeU) => self.op(Op::I16x8LeU),
            InstructionKind::Simd(SimdOp::I16x8GeS) => self.op(Op::I16x8GeS),
            InstructionKind::Simd(SimdOp::I16x8GeU) => self.op(Op::I16x8GeU),
            InstructionKind::Simd(SimdOp::I32x4Eq) => self.op(Op::I32x4Eq),
            InstructionKind::Simd(SimdOp::I32x4Ne) => self.op(Op::I32x4Ne),
            InstructionKind::Simd(SimdOp::I32x4LtS) => self.op(Op::I32x4LtS),
            InstructionKind::Simd(SimdOp::I32x4LtU) => self.op(Op::I32x4LtU),
            InstructionKind::Simd(SimdOp::I32x4GtS) => self.op(Op::I32x4GtS),
            InstructionKind::Simd(SimdOp::I32x4GtU) => self.op(Op::I32x4GtU),
            InstructionKind::Simd(SimdOp::I32x4LeS) => self.op(Op::I32x4LeS),
            InstructionKind::Simd(SimdOp::I32x4LeU) => self.op(Op::I32x4LeU),
            InstructionKind::Simd(SimdOp::I32x4GeS) => self.op(Op::I32x4GeS),
            InstructionKind::Simd(SimdOp::I32x4GeU) => self.op(Op::I32x4GeU),
            InstructionKind::Simd(SimdOp::F32x4Eq) => self.op(Op::F32x4Eq),
            InstructionKind::Simd(SimdOp::F32x4Ne) => self.op(Op::F32x4Ne),
            InstructionKind::Simd(SimdOp::F32x4Lt) => self.op(Op::F32x4Lt),
            InstructionKind::Simd(SimdOp::F32x4Gt) => self.op(Op::F32x4Gt),
            InstructionKind::Simd(SimdOp::F32x4Le) => self.op(Op::F32x4Le),
            InstructionKind::Simd(SimdOp::F32x4Ge) => self.op(Op::F32x4Ge),
            InstructionKind::Simd(SimdOp::I64x2Eq) => self.op(Op::I64x2Eq),
            InstructionKind::Simd(SimdOp::I64x2Ne) => self.op(Op::I64x2Ne),
            InstructionKind::Simd(SimdOp::I64x2LtS) => self.op(Op::I64x2LtS),
            InstructionKind::Simd(SimdOp::I64x2GtS) => self.op(Op::I64x2GtS),
            InstructionKind::Simd(SimdOp::I64x2LeS) => self.op(Op::I64x2LeS),
            InstructionKind::Simd(SimdOp::I64x2GeS) => self.op(Op::I64x2GeS),
            InstructionKind::Simd(SimdOp::F64x2Eq) => self.op(Op::F64x2Eq),
            InstructionKind::Simd(SimdOp::F64x2Ne) => self.op(Op::F64x2Ne),
            InstructionKind::Simd(SimdOp::F64x2Lt) => self.op(Op::F64x2Lt),
            InstructionKind::Simd(SimdOp::F64x2Gt) => self.op(Op::F64x2Gt),
            InstructionKind::Simd(SimdOp::F64x2Le) => self.op(Op::F64x2Le),
            InstructionKind::Simd(SimdOp::F64x2Ge) => self.op(Op::F64x2Ge),
            InstructionKind::Simd(SimdOp::I8x16Shl) => self.op(Op::I8x16Shl),
            InstructionKind::Simd(SimdOp::I8x16ShrS) => self.op(Op::I8x16ShrS),
            InstructionKind::Simd(SimdOp::I8x16ShrU) => self.op(Op::I8x16ShrU),
            InstructionKind::Simd(SimdOp::I16x8Shl) => self.op(Op::I16x8Shl),
            InstructionKind::Simd(SimdOp::I16x8ShrS) => self.op(Op::I16x8ShrS),
            InstructionKind::Simd(SimdOp::I16x8ShrU) => self.op(Op::I16x8ShrU),
            InstructionKind::Simd(SimdOp::I32x4Shl) => self.op(Op::I32x4Shl),
            InstructionKind::Simd(SimdOp::I32x4ShrS) => self.op(Op::I32x4ShrS),
            InstructionKind::Simd(SimdOp::I32x4ShrU) => self.op(Op::I32x4ShrU),
            InstructionKind::Simd(SimdOp::I64x2Shl) => self.op(Op::I64x2Shl),
            InstructionKind::Simd(SimdOp::I64x2ShrS) => self.op(Op::I64x2ShrS),
            InstructionKind::Simd(SimdOp::I64x2ShrU) => self.op(Op::I64x2ShrU),
            InstructionKind::Simd(SimdOp::I8x16Add) => self.op(Op::I8x16Add),
            InstructionKind::Simd(SimdOp::I8x16Sub) => self.op(Op::I8x16Sub),
            InstructionKind::Simd(SimdOp::I8x16AddSatS) => self.op(Op::I8x16AddSatS),
            InstructionKind::Simd(SimdOp::I8x16AddSatU) => self.op(Op::I8x16AddSatU),
            InstructionKind::Simd(SimdOp::I8x16SubSatS) => self.op(Op::I8x16SubSatS),
            InstructionKind::Simd(SimdOp::I8x16SubSatU) => self.op(Op::I8x16SubSatU),
            InstructionKind::Simd(SimdOp::I8x16MinS) => self.op(Op::I8x16MinS),
            InstructionKind::Simd(SimdOp::I8x16MinU) => self.op(Op::I8x16MinU),
            InstructionKind::Simd(SimdOp::I8x16MaxS) => self.op(Op::I8x16MaxS),
            InstructionKind::Simd(SimdOp::I8x16MaxU) => self.op(Op::I8x16MaxU),
            InstructionKind::Simd(SimdOp::I8x16AvgrU) => self.op(Op::I8x16AvgrU),
            InstructionKind::Simd(SimdOp::I16x8Add) => self.op(Op::I16x8Add),
            InstructionKind::Simd(SimdOp::I16x8Sub) => self.op(Op::I16x8Sub),
            InstructionKind::Simd(SimdOp::I16x8AddSatS) => self.op(Op::I16x8AddSatS),
            InstructionKind::Simd(SimdOp::I16x8AddSatU) => self.op(Op::I16x8AddSatU),
            InstructionKind::Simd(SimdOp::I16x8SubSatS) => self.op(Op::I16x8SubSatS),
            InstructionKind::Simd(SimdOp::I16x8SubSatU) => self.op(Op::I16x8SubSatU),
            InstructionKind::Simd(SimdOp::I16x8MinS) => self.op(Op::I16x8MinS),
            InstructionKind::Simd(SimdOp::I16x8MinU) => self.op(Op::I16x8MinU),
            InstructionKind::Simd(SimdOp::I16x8MaxS) => self.op(Op::I16x8MaxS),
            InstructionKind::Simd(SimdOp::I16x8MaxU) => self.op(Op::I16x8MaxU),
            InstructionKind::Simd(SimdOp::I16x8AvgrU) => self.op(Op::I16x8AvgrU),
            InstructionKind::Simd(SimdOp::I16x8Mul) => self.op(Op::I16x8Mul),
            InstructionKind::Simd(SimdOp::I32x4Add) => self.op(Op::I32x4Add),
            InstructionKind::Simd(SimdOp::I32x4Sub) => self.op(Op::I32x4Sub),
            InstructionKind::Simd(SimdOp::I32x4MinS) => self.op(Op::I32x4MinS),
            InstructionKind::Simd(SimdOp::I32x4MinU) => self.op(Op::I32x4MinU),
            InstructionKind::Simd(SimdOp::I32x4MaxS) => self.op(Op::I32x4MaxS),
            InstructionKind::Simd(SimdOp::I32x4MaxU) => self.op(Op::I32x4MaxU),
            InstructionKind::Simd(SimdOp::I32x4Mul) => self.op(Op::I32x4Mul),
            InstructionKind::Simd(SimdOp::I64x2Add) => self.op(Op::I64x2Add),
            InstructionKind::Simd(SimdOp::I64x2Sub) => self.op(Op::I64x2Sub),
            InstructionKind::Simd(SimdOp::I64x2Mul) => self.op(Op::I64x2Mul),
            InstructionKind::Simd(SimdOp::F32x4Abs) => self.op(Op::F32x4Abs),
            InstructionKind::Simd(SimdOp::F32x4Neg) => self.op(Op::F32x4Neg),
            InstructionKind::Simd(SimdOp::F32x4Sqrt) => self.op(Op::F32x4Sqrt),
            InstructionKind::Simd(SimdOp::F32x4Ceil) => self.op(Op::F32x4Ceil),
            InstructionKind::Simd(SimdOp::F32x4Floor) => self.op(Op::F32x4Floor),
            InstructionKind::Simd(SimdOp::F32x4Trunc) => self.op(Op::F32x4Trunc),
            InstructionKind::Simd(SimdOp::F32x4Nearest) => self.op(Op::F32x4Nearest),
            InstructionKind::Simd(SimdOp::F32x4Add) => self.op(Op::F32x4Add),
            InstructionKind::Simd(SimdOp::F32x4Sub) => self.op(Op::F32x4Sub),
            InstructionKind::Simd(SimdOp::F32x4Mul) => self.op(Op::F32x4Mul),
            InstructionKind::Simd(SimdOp::F32x4Div) => self.op(Op::F32x4Div),
            InstructionKind::Simd(SimdOp::F64x2Abs) => self.op(Op::F64x2Abs),
            InstructionKind::Simd(SimdOp::F64x2Neg) => self.op(Op::F64x2Neg),
            InstructionKind::Simd(SimdOp::F64x2Sqrt) => self.op(Op::F64x2Sqrt),
            InstructionKind::Simd(SimdOp::F64x2Ceil) => self.op(Op::F64x2Ceil),
            InstructionKind::Simd(SimdOp::F64x2Floor) => self.op(Op::F64x2Floor),
            InstructionKind::Simd(SimdOp::F64x2Trunc) => self.op(Op::F64x2Trunc),
            InstructionKind::Simd(SimdOp::F64x2Nearest) => self.op(Op::F64x2Nearest),
            InstructionKind::Simd(SimdOp::F64x2Add) => self.op(Op::F64x2Add),
            InstructionKind::Simd(SimdOp::F64x2Sub) => self.op(Op::F64x2Sub),
            InstructionKind::Simd(SimdOp::F64x2Mul) => self.op(Op::F64x2Mul),
            InstructionKind::Simd(SimdOp::F64x2Div) => self.op(Op::F64x2Div),
            InstructionKind::Simd(SimdOp::F32x4Min) => self.op(Op::F32x4Min),
            InstructionKind::Simd(SimdOp::F32x4Max) => self.op(Op::F32x4Max),
            InstructionKind::Simd(SimdOp::F32x4PMin) => self.op(Op::F32x4PMin),
            InstructionKind::Simd(SimdOp::F32x4PMax) => self.op(Op::F32x4PMax),
            InstructionKind::Simd(SimdOp::F64x2Min) => self.op(Op::F64x2Min),
            InstructionKind::Simd(SimdOp::F64x2Max) => self.op(Op::F64x2Max),
            InstructionKind::Simd(SimdOp::F64x2PMin) => self.op(Op::F64x2PMin),
            InstructionKind::Simd(SimdOp::F64x2PMax) => self.op(Op::F64x2PMax),
            InstructionKind::Simd(SimdOp::I32x4TruncSatF32x4S) => self.op(Op::I32x4TruncSatF32x4S),
            InstructionKind::Simd(SimdOp::I32x4TruncSatF32x4U) => self.op(Op::I32x4TruncSatF32x4U),
            InstructionKind::Simd(SimdOp::F32x4ConvertI32x4S) => self.op(Op::F32x4ConvertI32x4S),
            InstructionKind::Simd(SimdOp::F32x4ConvertI32x4U) => self.op(Op::F32x4ConvertI32x4U),
            InstructionKind::Simd(SimdOp::I32x4TruncSatF64x2SZero) => self.op(Op::I32x4TruncSatF64x2SZero),
            InstructionKind::Simd(SimdOp::I32x4TruncSatF64x2UZero) => self.op(Op::I32x4TruncSatF64x2UZero),
            InstructionKind::Simd(SimdOp::F64x2ConvertLowI32x4S) => self.op(Op::F64x2ConvertLowI32x4S),
            InstructionKind::Simd(SimdOp::F64x2ConvertLowI32x4U) => self.op(Op::F64x2ConvertLowI32x4U),
            InstructionKind::Simd(SimdOp::F32x4DemoteF64x2Zero) => self.op(Op::F32x4DemoteF64x2Zero),
            InstructionKind::Simd(SimdOp::F64x2PromoteLowF32x4) => self.op(Op::F64x2PromoteLowF32x4),
            InstructionKind::Simd(SimdOp::I8x16NarrowI16x8S) => self.op(Op::I8x16NarrowI16x8S),
            InstructionKind::Simd(SimdOp::I8x16NarrowI16x8U) => self.op(Op::I8x16NarrowI16x8U),
            InstructionKind::Simd(SimdOp::I16x8NarrowI32x4S) => self.op(Op::I16x8NarrowI32x4S),
            InstructionKind::Simd(SimdOp::I16x8NarrowI32x4U) => self.op(Op::I16x8NarrowI32x4U),
            InstructionKind::Simd(SimdOp::I16x8ExtendLowI8x16S) => self.op(Op::I16x8ExtendLowI8x16S),
            InstructionKind::Simd(SimdOp::I16x8ExtendHighI8x16S) => self.op(Op::I16x8ExtendHighI8x16S),
            InstructionKind::Simd(SimdOp::I16x8ExtendLowI8x16U) => self.op(Op::I16x8ExtendLowI8x16U),
            InstructionKind::Simd(SimdOp::I16x8ExtendHighI8x16U) => self.op(Op::I16x8ExtendHighI8x16U),
            InstructionKind::Simd(SimdOp::I32x4ExtendLowI16x8S) => self.op(Op::I32x4ExtendLowI16x8S),
            InstructionKind::Simd(SimdOp::I32x4ExtendHighI16x8S) => self.op(Op::I32x4ExtendHighI16x8S),
            InstructionKind::Simd(SimdOp::I32x4ExtendLowI16x8U) => self.op(Op::I32x4ExtendLowI16x8U),
            InstructionKind::Simd(SimdOp::I32x4ExtendHighI16x8U) => self.op(Op::I32x4ExtendHighI16x8U),
            InstructionKind::Simd(SimdOp::I64x2ExtendLowI32x4S) => self.op(Op::I64x2ExtendLowI32x4S),
            InstructionKind::Simd(SimdOp::I64x2ExtendHighI32x4S) => self.op(Op::I64x2ExtendHighI32x4S),
            InstructionKind::Simd(SimdOp::I64x2ExtendLowI32x4U) => self.op(Op::I64x2ExtendLowI32x4U),
            InstructionKind::Simd(SimdOp::I64x2ExtendHighI32x4U) => self.op(Op::I64x2ExtendHighI32x4U),
            InstructionKind::Simd(SimdOp::I16x8ExtMulLowI8x16S) => self.op(Op::I16x8ExtMulLowI8x16S),
            InstructionKind::Simd(SimdOp::I16x8ExtMulHighI8x16S) => self.op(Op::I16x8ExtMulHighI8x16S),
            InstructionKind::Simd(SimdOp::I16x8ExtMulLowI8x16U) => self.op(Op::I16x8ExtMulLowI8x16U),
            InstructionKind::Simd(SimdOp::I16x8ExtMulHighI8x16U) => self.op(Op::I16x8ExtMulHighI8x16U),
            InstructionKind::Simd(SimdOp::I32x4ExtMulLowI16x8S) => self.op(Op::I32x4ExtMulLowI16x8S),
            InstructionKind::Simd(SimdOp::I32x4ExtMulHighI16x8S) => self.op(Op::I32x4ExtMulHighI16x8S),
            InstructionKind::Simd(SimdOp::I32x4ExtMulLowI16x8U) => self.op(Op::I32x4ExtMulLowI16x8U),
            InstructionKind::Simd(SimdOp::I32x4ExtMulHighI16x8U) => self.op(Op::I32x4ExtMulHighI16x8U),
            InstructionKind::Simd(SimdOp::I64x2ExtMulLowI32x4S) => self.op(Op::I64x2ExtMulLowI32x4S),
            InstructionKind::Simd(SimdOp::I64x2ExtMulHighI32x4S) => self.op(Op::I64x2ExtMulHighI32x4S),
            InstructionKind::Simd(SimdOp::I64x2ExtMulLowI32x4U) => self.op(Op::I64x2ExtMulLowI32x4U),
            InstructionKind::Simd(SimdOp::I64x2ExtMulHighI32x4U) => self.op(Op::I64x2ExtMulHighI32x4U),
            InstructionKind::Simd(SimdOp::I16x8ExtAddPairwiseI8x16S) => self.op(Op::I16x8ExtAddPairwiseI8x16S),
            InstructionKind::Simd(SimdOp::I16x8ExtAddPairwiseI8x16U) => self.op(Op::I16x8ExtAddPairwiseI8x16U),
            InstructionKind::Simd(SimdOp::I32x4ExtAddPairwiseI16x8S) => self.op(Op::I32x4ExtAddPairwiseI16x8S),
            InstructionKind::Simd(SimdOp::I32x4ExtAddPairwiseI16x8U) => self.op(Op::I32x4ExtAddPairwiseI16x8U),
            InstructionKind::Simd(SimdOp::I32x4DotI16x8S) => self.op(Op::I32x4DotI16x8S),
            InstructionKind::Simd(SimdOp::I16x8Q15MulrSatS) => self.op(Op::I16x8Q15MulrSatS),
            InstructionKind::Simd(SimdOp::V128Not) => self.op(Op::V128Not),
            InstructionKind::Simd(SimdOp::V128And) => self.op(Op::V128And),
            InstructionKind::Simd(SimdOp::V128AndNot) => self.op(Op::V128AndNot),
            InstructionKind::Simd(SimdOp::V128Or) => self.op(Op::V128Or),
            InstructionKind::Simd(SimdOp::V128Xor) => self.op(Op::V128Xor),
            InstructionKind::Simd(SimdOp::V128Bitselect) => self.op(Op::V128Bitselect),
            InstructionKind::Simd(SimdOp::V128AnyTrue) => self.op(Op::V128AnyTrue),
            InstructionKind::Simd(SimdOp::I8x16AllTrue) => self.op(Op::I8x16AllTrue),
            InstructionKind::Simd(SimdOp::I8x16Bitmask) => self.op(Op::I8x16Bitmask),
            InstructionKind::Simd(SimdOp::I16x8AllTrue) => self.op(Op::I16x8AllTrue),
            InstructionKind::Simd(SimdOp::I16x8Bitmask) => self.op(Op::I16x8Bitmask),
            InstructionKind::Simd(SimdOp::I32x4AllTrue) => self.op(Op::I32x4AllTrue),
            InstructionKind::Simd(SimdOp::I32x4Bitmask) => self.op(Op::I32x4Bitmask),
            InstructionKind::Simd(SimdOp::I64x2AllTrue) => self.op(Op::I64x2AllTrue),
            InstructionKind::Simd(SimdOp::I64x2Bitmask) => self.op(Op::I64x2Bitmask),
            InstructionKind::Simd(SimdOp::I8x16ExtractLaneS { lane }) => self.op(Op::I8x16ExtractLaneS(*lane)),
            InstructionKind::Simd(SimdOp::I8x16ExtractLaneU { lane }) => self.op(Op::I8x16ExtractLaneU(*lane)),
            InstructionKind::Simd(SimdOp::I16x8ExtractLaneS { lane }) => self.op(Op::I16x8ExtractLaneS(*lane)),
            InstructionKind::Simd(SimdOp::I16x8ExtractLaneU { lane }) => self.op(Op::I16x8ExtractLaneU(*lane)),
            InstructionKind::Simd(SimdOp::I32x4ExtractLane { lane }) => self.op(Op::I32x4ExtractLane(*lane)),
            InstructionKind::Simd(SimdOp::I64x2ExtractLane { lane }) => self.op(Op::I64x2ExtractLane(*lane)),
            InstructionKind::Simd(SimdOp::F32x4ExtractLane { lane }) => self.op(Op::F32x4ExtractLane(*lane)),
            InstructionKind::Simd(SimdOp::F64x2ExtractLane { lane }) => self.op(Op::F64x2ExtractLane(*lane)),
            InstructionKind::Simd(SimdOp::I8x16ReplaceLane { lane }) => self.op(Op::I8x16ReplaceLane(*lane)),
            InstructionKind::Simd(SimdOp::I16x8ReplaceLane { lane }) => self.op(Op::I16x8ReplaceLane(*lane)),
            InstructionKind::Simd(SimdOp::I32x4ReplaceLane { lane }) => self.op(Op::I32x4ReplaceLane(*lane)),
            InstructionKind::Simd(SimdOp::I64x2ReplaceLane { lane }) => self.op(Op::I64x2ReplaceLane(*lane)),
            InstructionKind::Simd(SimdOp::F32x4ReplaceLane { lane }) => self.op(Op::F32x4ReplaceLane(*lane)),
            InstructionKind::Simd(SimdOp::F64x2ReplaceLane { lane }) => self.op(Op::F64x2ReplaceLane(*lane)),
            InstructionKind::I32Add => self.op(Op::I32Add),
            InstructionKind::I32Sub => self.op(Op::I32Sub),
            InstructionKind::I32Mul => self.op(Op::I32Mul),
            InstructionKind::I32DivS => self.op(Op::I32DivS),
            InstructionKind::I32DivU => self.op(Op::I32DivU),
            InstructionKind::I32RemS => self.op(Op::I32RemS),
            InstructionKind::I32RemU => self.op(Op::I32RemU),
            InstructionKind::I32Clz => self.op(Op::I32Clz),
            InstructionKind::I32Ctz => self.op(Op::I32Ctz),
            InstructionKind::I32Popcnt => self.op(Op::I32Popcnt),
            InstructionKind::I32And => self.op(Op::I32And),
            InstructionKind::I32Or => self.op(Op::I32Or),
            InstructionKind::I32Xor => self.op(Op::I32Xor),
            InstructionKind::I32Shl => self.op(Op::I32Shl),
            InstructionKind::I32ShrS => self.op(Op::I32ShrS),
            InstructionKind::I32ShrU => self.op(Op::I32ShrU),
            InstructionKind::I32Rotl => self.op(Op::I32Rotl),
            InstructionKind::I32Rotr => self.op(Op::I32Rotr),
            InstructionKind::I32Extend8S => self.op(Op::I32Extend8S),
            InstructionKind::I32Extend16S => self.op(Op::I32Extend16S),
            InstructionKind::I64Add => self.op(Op::I64Add),
            InstructionKind::I64Sub => self.op(Op::I64Sub),
            InstructionKind::I64Mul => self.op(Op::I64Mul),
            InstructionKind::I64DivS => self.op(Op::I64DivS),
            InstructionKind::I64DivU => self.op(Op::I64DivU),
            InstructionKind::I64RemS => self.op(Op::I64RemS),
            InstructionKind::I64RemU => self.op(Op::I64RemU),
            InstructionKind::I64Clz => self.op(Op::I64Clz),
            InstructionKind::I64Ctz => self.op(Op::I64Ctz),
            InstructionKind::I64Popcnt => self.op(Op::I64Popcnt),
            InstructionKind::I64And => self.op(Op::I64And),
            InstructionKind::I64Or => self.op(Op::I64Or),
            InstructionKind::I64Xor => self.op(Op::I64Xor),
            InstructionKind::I64Shl => self.op(Op::I64Shl),
            InstructionKind::I64ShrS => self.op(Op::I64ShrS),
            InstructionKind::I64ShrU => self.op(Op::I64ShrU),
            InstructionKind::I64Rotl => self.op(Op::I64Rotl),
            InstructionKind::I64Rotr => self.op(Op::I64Rotr),
            InstructionKind::I64Eqz => self.op(Op::I64Eqz),
            InstructionKind::I64Eq => self.op(Op::I64Eq),
            InstructionKind::I64Ne => self.op(Op::I64Ne),
            InstructionKind::I64LtS => self.op(Op::I64LtS),
            InstructionKind::I64LtU => self.op(Op::I64LtU),
            InstructionKind::I64GtS => self.op(Op::I64GtS),
            InstructionKind::I64GtU => self.op(Op::I64GtU),
            InstructionKind::I64LeS => self.op(Op::I64LeS),
            InstructionKind::I64LeU => self.op(Op::I64LeU),
            InstructionKind::I64GeS => self.op(Op::I64GeS),
            InstructionKind::I64GeU => self.op(Op::I64GeU),
            InstructionKind::I64Extend8S => self.op(Op::I64Extend8S),
            InstructionKind::I64Extend16S => self.op(Op::I64Extend16S),
            InstructionKind::I64Extend32S => self.op(Op::I64Extend32S),
            InstructionKind::I64ExtendI32S => self.op(Op::I64ExtendI32S),
            InstructionKind::I64ExtendI32U => self.op(Op::I64ExtendI32U),
            InstructionKind::I32WrapI64 => self.op(Op::I32WrapI64),
            InstructionKind::F32Add => self.op(Op::F32Add),
            InstructionKind::F32Sub => self.op(Op::F32Sub),
            InstructionKind::F32Mul => self.op(Op::F32Mul),
            InstructionKind::F32Div => self.op(Op::F32Div),
            InstructionKind::F32Min => self.op(Op::F32Min),
            InstructionKind::F32Max => self.op(Op::F32Max),
            InstructionKind::F32Copysign => self.op(Op::F32Copysign),
            InstructionKind::F64Add => self.op(Op::F64Add),
            InstructionKind::F64Sub => self.op(Op::F64Sub),
            InstructionKind::F64Mul => self.op(Op::F64Mul),
            InstructionKind::F64Div => self.op(Op::F64Div),
            InstructionKind::F64Min => self.op(Op::F64Min),
            InstructionKind::F64Max => self.op(Op::F64Max),
            InstructionKind::F64Copysign => self.op(Op::F64Copysign),
            InstructionKind::F32Abs => self.op(Op::F32Abs),
            InstructionKind::F32Neg => self.op(Op::F32Neg),
            InstructionKind::F32Sqrt => self.op(Op::F32Sqrt),
            InstructionKind::F32Ceil => self.op(Op::F32Ceil),
            InstructionKind::F32Floor => self.op(Op::F32Floor),
            InstructionKind::F32Trunc => self.op(Op::F32Trunc),
            InstructionKind::F32Nearest => self.op(Op::F32Nearest),
            InstructionKind::F64Abs => self.op(Op::F64Abs),
            InstructionKind::F64Neg => self.op(Op::F64Neg),
            InstructionKind::F64Sqrt => self.op(Op::F64Sqrt),
            InstructionKind::F64Ceil => self.op(Op::F64Ceil),
            InstructionKind::F64Floor => self.op(Op::F64Floor),
            InstructionKind::F64Trunc => self.op(Op::F64Trunc),
            InstructionKind::F64Nearest => self.op(Op::F64Nearest),
            InstructionKind::F32Eq => self.op(Op::F32Eq),
            InstructionKind::F32Ne => self.op(Op::F32Ne),
            InstructionKind::F32Lt => self.op(Op::F32Lt),
            InstructionKind::F32Gt => self.op(Op::F32Gt),
            InstructionKind::F32Le => self.op(Op::F32Le),
            InstructionKind::F32Ge => self.op(Op::F32Ge),
            InstructionKind::F64Eq => self.op(Op::F64Eq),
            InstructionKind::F64Ne => self.op(Op::F64Ne),
            InstructionKind::F64Lt => self.op(Op::F64Lt),
            InstructionKind::F64Gt => self.op(Op::F64Gt),
            InstructionKind::F64Le => self.op(Op::F64Le),
            InstructionKind::F64Ge => self.op(Op::F64Ge),
            InstructionKind::F32ConvertI32S => self.op(Op::F32ConvertI32S),
            InstructionKind::F32ConvertI32U => self.op(Op::F32ConvertI32U),
            InstructionKind::F32ConvertI64S => self.op(Op::F32ConvertI64S),
            InstructionKind::F32ConvertI64U => self.op(Op::F32ConvertI64U),
            InstructionKind::F64ConvertI32S => self.op(Op::F64ConvertI32S),
            InstructionKind::F64ConvertI32U => self.op(Op::F64ConvertI32U),
            InstructionKind::F64ConvertI64S => self.op(Op::F64ConvertI64S),
            InstructionKind::F64ConvertI64U => self.op(Op::F64ConvertI64U),
            InstructionKind::F32DemoteF64 => self.op(Op::F32DemoteF64),
            InstructionKind::F64PromoteF32 => self.op(Op::F64PromoteF32),
            InstructionKind::I32TruncF32S => self.op(Op::I32TruncF32S),
            InstructionKind::I32TruncF32U => self.op(Op::I32TruncF32U),
            InstructionKind::I32TruncF64S => self.op(Op::I32TruncF64S),
            InstructionKind::I32TruncF64U => self.op(Op::I32TruncF64U),
            InstructionKind::I64TruncF32S => self.op(Op::I64TruncF32S),
            InstructionKind::I64TruncF32U => self.op(Op::I64TruncF32U),
            InstructionKind::I64TruncF64S => self.op(Op::I64TruncF64S),
            InstructionKind::I64TruncF64U => self.op(Op::I64TruncF64U),
            InstructionKind::I32TruncSatF32S => self.op(Op::I32TruncSatF32S),
            InstructionKind::I32TruncSatF32U => self.op(Op::I32TruncSatF32U),
            InstructionKind::I32TruncSatF64S => self.op(Op::I32TruncSatF64S),
            InstructionKind::I32TruncSatF64U => self.op(Op::I32TruncSatF64U),
            InstructionKind::I64TruncSatF32S => self.op(Op::I64TruncSatF32S),
            InstructionKind::I64TruncSatF32U => self.op(Op::I64TruncSatF32U),
            InstructionKind::I64TruncSatF64S => self.op(Op::I64TruncSatF64S),
            InstructionKind::I64TruncSatF64U => self.op(Op::I64TruncSatF64U),
            InstructionKind::I32ReinterpretF32 => self.op(Op::I32ReinterpretF32),
            InstructionKind::I64ReinterpretF64 => self.op(Op::I64ReinterpretF64),
            InstructionKind::F32ReinterpretI32 => self.op(Op::F32ReinterpretI32),
            InstructionKind::F64ReinterpretI64 => self.op(Op::F64ReinterpretI64),
            InstructionKind::I32Eqz => self.op(Op::I32Eqz),
            InstructionKind::I32Eq => self.op(Op::I32Eq),
            InstructionKind::I32Ne => self.op(Op::I32Ne),
            InstructionKind::I32LtS => self.op(Op::I32LtS),
            InstructionKind::I32LtU => self.op(Op::I32LtU),
            InstructionKind::I32GtS => self.op(Op::I32GtS),
            InstructionKind::I32GtU => self.op(Op::I32GtU),
            InstructionKind::I32LeS => self.op(Op::I32LeS),
            InstructionKind::I32LeU => self.op(Op::I32LeU),
            InstructionKind::I32GeS => self.op(Op::I32GeS),
            InstructionKind::I32GeU => self.op(Op::I32GeU),

            InstructionKind::LocalGet { local_idx } => self.op(match self.local(*local_idx) {
                (index, ValueType::V128) => Op::LocalGetV128 { index },
                (index, _) => Op::LocalGet { index },
            }),
            InstructionKind::LocalSet { local_idx } => self.op(match self.local(*local_idx) {
                (index, ValueType::V128) => Op::LocalSetV128 { index },
                (index, _) => Op::LocalSet { index },
            }),
            InstructionKind::LocalTee { local_idx } => self.op(match self.local(*local_idx) {
                (index, ValueType::V128) => Op::LocalTeeV128 { index },
                (index, _) => Op::LocalTee { index },
            }),

            InstructionKind::GlobalGet { global_idx } => {
                self.emit(Op::GlobalGet { index: *global_idx });
            }
            InstructionKind::GlobalSet { global_idx } => {
                self.emit(Op::GlobalSet { index: *global_idx });
            }

            // Memory
            InstructionKind::I32Load { memarg } => {
                self.emit(Op::I32Load(*memarg));
            }
            InstructionKind::I32Load8S { memarg } => {
                self.emit(Op::I32Load8S(*memarg));
            }
            InstructionKind::I32Load8U { memarg } => {
                self.emit(Op::I32Load8U(*memarg));
            }
            InstructionKind::I32Load16S { memarg } => {
                self.emit(Op::I32Load16S(*memarg));
            }
            InstructionKind::I32Load16U { memarg } => {
                self.emit(Op::I32Load16U(*memarg));
            }
            InstructionKind::I64Load { memarg } => {
                self.emit(Op::I64Load(*memarg));
            }
            InstructionKind::I64Load8S { memarg } => {
                self.emit(Op::I64Load8S(*memarg));
            }
            InstructionKind::I64Load8U { memarg } => {
                self.emit(Op::I64Load8U(*memarg));
            }
            InstructionKind::I64Load16S { memarg } => {
                self.emit(Op::I64Load16S(*memarg));
            }
            InstructionKind::I64Load16U { memarg } => {
                self.emit(Op::I64Load16U(*memarg));
            }
            InstructionKind::I64Load32S { memarg } => {
                self.emit(Op::I64Load32S(*memarg));
            }
            InstructionKind::I64Load32U { memarg } => {
                self.emit(Op::I64Load32U(*memarg));
            }
            InstructionKind::F32Load { memarg } => {
                self.emit(Op::F32Load(*memarg));
            }
            InstructionKind::F64Load { memarg } => {
                self.emit(Op::F64Load(*memarg));
            }
            InstructionKind::Simd(SimdOp::V128Load { memarg }) => self.op(Op::V128Load(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load8x8S { memarg }) => self.op(Op::V128Load8x8S(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load8x8U { memarg }) => self.op(Op::V128Load8x8U(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load16x4S { memarg }) => self.op(Op::V128Load16x4S(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load16x4U { memarg }) => self.op(Op::V128Load16x4U(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load32x2S { memarg }) => self.op(Op::V128Load32x2S(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load32x2U { memarg }) => self.op(Op::V128Load32x2U(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load8Splat { memarg }) => self.op(Op::V128Load8Splat(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load16Splat { memarg }) => self.op(Op::V128Load16Splat(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load32Splat { memarg }) => self.op(Op::V128Load32Splat(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load64Splat { memarg }) => self.op(Op::V128Load64Splat(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load32Zero { memarg }) => self.op(Op::V128Load32Zero(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load64Zero { memarg }) => self.op(Op::V128Load64Zero(*memarg)),
            InstructionKind::Simd(SimdOp::V128Load8Lane { memarg, lane }) => self.op(Op::V128Load8Lane {
                memarg: *memarg,
                lane: *lane,
            }),
            InstructionKind::Simd(SimdOp::V128Load16Lane { memarg, lane }) => self.op(Op::V128Load16Lane {
                memarg: *memarg,
                lane: *lane,
            }),
            InstructionKind::Simd(SimdOp::V128Load32Lane { memarg, lane }) => self.op(Op::V128Load32Lane {
                memarg: *memarg,
                lane: *lane,
            }),
            InstructionKind::Simd(SimdOp::V128Load64Lane { memarg, lane }) => self.op(Op::V128Load64Lane {
                memarg: *memarg,
                lane: *lane,
            }),
            InstructionKind::I32Store { memarg } => {
                self.emit(Op::I32Store(*memarg));
            }
            InstructionKind::I32Store8 { memarg } => {
                self.emit(Op::I32Store8(*memarg));
            }
            InstructionKind::I32Store16 { memarg } => {
                self.emit(Op::I32Store16(*memarg));
            }
            InstructionKind::I64Store { memarg } => {
                self.emit(Op::I64Store(*memarg));
            }
            InstructionKind::I64Store8 { memarg } => {
                self.emit(Op::I64Store8(*memarg));
            }
            InstructionKind::I64Store16 { memarg } => {
                self.emit(Op::I64Store16(*memarg));
            }
            InstructionKind::I64Store32 { memarg } => {
                self.emit(Op::I64Store32(*memarg));
            }
            InstructionKind::F32Store { memarg } => {
                self.emit(Op::F32Store(*memarg));
            }
            InstructionKind::F64Store { memarg } => {
                self.emit(Op::F64Store(*memarg));
            }
            InstructionKind::Simd(SimdOp::V128Store { memarg }) => self.op(Op::V128Store(*memarg)),
            InstructionKind::Simd(SimdOp::V128Store8Lane { memarg, lane }) => self.op(Op::V128Store8Lane {
                memarg: *memarg,
                lane: *lane,
            }),
            InstructionKind::Simd(SimdOp::V128Store16Lane { memarg, lane }) => self.op(Op::V128Store16Lane {
                memarg: *memarg,
                lane: *lane,
            }),
            InstructionKind::Simd(SimdOp::V128Store32Lane { memarg, lane }) => self.op(Op::V128Store32Lane {
                memarg: *memarg,
                lane: *lane,
            }),
            InstructionKind::Simd(SimdOp::V128Store64Lane { memarg, lane }) => self.op(Op::V128Store64Lane {
                memarg: *memarg,
                lane: *lane,
            }),
            InstructionKind::MemorySize => self.op(Op::MemorySize),
            InstructionKind::MemoryGrow => self.op(Op::MemoryGrow),
            InstructionKind::MemoryCopy => self.op(Op::MemoryCopy),
            InstructionKind::MemoryFill => self.op(Op::MemoryFill),

            InstructionKind::Call { func_idx } => self.op(Op::Call { func_idx: *func_idx }),
            InstructionKind::CallIndirect { type_idx, table_idx } => self.op(Op::CallIndirect {
                type_idx: *type_idx,
                table_idx: *table_idx,
            }),

            InstructionKind::Br { label_idx } => self.emit_br(*label_idx)?,
            InstructionKind::BrIf { label_idx } => self.emit_br_if(*label_idx)?,
            InstructionKind::BrTable { labels, default } => self.emit_br_table(labels, *default)?,
            InstructionKind::Return => self.emit_return()?,
            InstructionKind::Nop => self.op(Op::Nop),
            InstructionKind::Unreachable => {
                self.op(Op::Unreachable);
                self.reachable = false;
            }
            InstructionKind::Drop => self.op(match dropped {
                Some(ValueType::V128) => Op::DropV128,
                _ => Op::Drop,
            }),
            // The result, now on top, has the type of both operands.
            InstructionKind::Select | InstructionKind::SelectTyped { .. } => self.op(match self.stack.last() {
                Some(ValueType::V128) => Op::SelectV128,
                _ => Op::Select,
            }),
            InstructionKind::RefNull { ref_type } => self.op(Op::RefNull(*ref_type)),
            InstructionKind::RefIsNull => self.op(Op::RefIsNull),
            InstructionKind::RefFunc { func_idx } => self.op(Op::RefFunc { func_idx: *func_idx }),
            InstructionKind::TableGet { table_idx } => self.op(Op::TableGet { table_idx: *table_idx }),
            InstructionKind::TableSet { table_idx } => self.op(Op::TableSet { table_idx: *table_idx }),
            InstructionKind::TableSize { table_idx } => self.op(Op::TableSize { table_idx: *table_idx }),
            InstructionKind::TableGrow { table_idx } => self.op(Op::TableGrow { table_idx: *table_idx }),
            InstructionKind::TableFill { table_idx } => self.op(Op::TableFill { table_idx: *table_idx }),
            InstructionKind::TableCopy { dst_table, src_table } => self.op(Op::TableCopy {
                dst_table: *dst_table,
                src_table: *src_table,
            }),
            InstructionKind::TableInit { elem_idx, table_idx } => self.op(Op::TableInit {
                elem_idx: *elem_idx,
                table_idx: *table_idx,
            }),
            InstructionKind::ElemDrop { elem_idx } => self.op(Op::ElemDrop { elem_idx: *elem_idx }),
            InstructionKind::MemoryInit { data_idx } => self.op(Op::MemoryInit { data_idx: *data_idx }),
            InstructionKind::DataDrop { data_idx } => self.op(Op::DataDrop { data_idx: *data_idx }),

            InstructionKind::Block { .. }
            | InstructionKind::Loop { .. }
            | InstructionKind::If { .. }
            | InstructionKind::Else
            | InstructionKind::End => unreachable!("structured control instruction passed to emit_plain"),
        }
        Ok(())
    }

    /// Resolve a branch `depth` labels out: check the operands it carries,
    /// record `fixup` if its target is still unknown, and return the target.
    fn branch_target(&mut self, depth: u32, fixup: Fixup) -> Result<BrTarget> {
        let index = self.label_index(depth)?;
        let Label {
            stack_depth,
            types,
            word_depth,
            word_arity,
            ..
        } = self.labels[index];
        self.check_branch(stack_depth, types)?;
        let pc = match &mut self.labels[index].target {
            LabelTarget::Start(start) => *start,
            LabelTarget::End(fixups) => {
                fixups.push(fixup);
                0
            }
        };
        Ok(BrTarget {
            pc,
            arity: word_arity,
            stack_depth: word_depth,
        })
    }

    /// Check that a branch's operands are on top, above the depth it unwinds to.
    fn check_branch(&self, stack_depth: u32, types: &[ValueType]) -> Result {
        match self.depth().checked_sub(types.len() as u32) {
            Some(start) if start >= stack_depth => match &self.stack[start as usize..] {
                carried if carried == types => Ok(()),
                carried => Err(invalid(format!("branch carries {carried:?}, label expects {types:?}"))),
            },
            _ => Err(invalid(format!("branch needs {types:?} above depth {stack_depth}"))),
        }
    }

    /// `return`, or a `br` to the function body: leave with the results on top.
    fn emit_return(&mut self) -> Result {
        self.check_branch(0, self.labels[0].types)?;
        self.emit(Op::Return);
        self.reachable = false;
        Ok(())
    }

    fn emit_br(&mut self, depth: u32) -> Result {
        if self.label_index(depth)? == 0 {
            return self.emit_return();
        }
        let BrTarget { pc, arity, stack_depth } = self.branch_target(depth, Fixup::Branch(self.ops.len()))?;
        self.emit(Op::Br {
            target: pc,
            arity,
            stack_depth,
        });
        self.reachable = false;
        Ok(())
    }

    fn emit_br_if(&mut self, depth: u32) -> Result {
        let BrTarget { pc, arity, stack_depth } = self.branch_target(depth, Fixup::Branch(self.ops.len()))?;
        self.emit(Op::BrIf {
            target: pc,
            arity,
            stack_depth,
        });
        Ok(())
    }

    fn emit_br_table(&mut self, labels: &[u32], default: u32) -> Result {
        let op_pos = self.ops.len();
        let targets = (labels.iter().enumerate())
            .map(|(i, depth)| self.branch_target(*depth, Fixup::TableEntry { op_pos, index: Some(i) }))
            .collect::<Result<_>>()?;
        let default = self.branch_target(default, Fixup::TableEntry { op_pos, index: None })?;
        self.emit(Op::BrTable { targets, default });
        self.reachable = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wat;

    /// A valid module, recovered for editing into an invalid one.
    fn editable(source: &str) -> Module {
        wat::parse(source).expect("WAT parse failed").into_inner().unwrap()
    }

    /// The plain instruction at `index` in a body, for editing.
    fn plain_mut(body: &mut [StructuredInstruction], index: usize) -> &mut Instruction {
        match &mut body[index] {
            StructuredInstruction::Plain(instruction) => instruction,
            other => panic!("expected a plain instruction, got {other:?}"),
        }
    }

    /// The body of the block at `index`, for editing.
    fn block_mut(body: &mut [StructuredInstruction], index: usize) -> &mut Vec<StructuredInstruction> {
        match &mut body[index] {
            StructuredInstruction::Block { body, .. } => body,
            other => panic!("expected a block, got {other:?}"),
        }
    }

    fn compile_error(module: &Module) -> String {
        match compile_module(module) {
            Err(RuntimeError::Compile(what)) => what,
            other => panic!("expected a compile error, got {other:?}"),
        }
    }

    fn compile_first_func(source: &str) -> CompiledFunction {
        let module = wat::parse(source).expect("WAT parse failed");
        compile_module(&module).expect("compile failed").remove(0)
    }

    /// The function's ops, one numbered line each.
    fn listing(source: &str) -> String {
        let cf = compile_first_func(source);
        cf.ops
            .iter()
            .enumerate()
            .map(|(i, op)| format!("{i}: {op}\n"))
            .collect()
    }

    #[test]
    fn inconsistent_modules_are_rejected_not_guessed() {
        let mut module = editable("(module (func))");
        module.functions.functions[0].ftype_index = 9;
        assert_eq!(compile_error(&module), "unknown function type 9");

        let mut module = editable("(module (func (call 0)))");
        plain_mut(&mut module.code.code[0].body.body, 0).kind = InstructionKind::Call { func_idx: 9 };
        assert_eq!(compile_error(&module), "unknown function 9");

        // `i32.add` with nothing beneath it.
        let mut module = editable("(module (func (result i32) (i32.add (i32.const 1) (i32.const 2))))");
        module.code.code[0].body.body.drain(..2);
        assert_eq!(compile_error(&module), "operand stack underflow");

        // `drop` reaching beneath its block for the function's operand.
        let mut module = editable("(module (func (i32.const 1) (drop) (block (nop))))");
        module.code.code[0].body.body.remove(1);
        plain_mut(block_mut(&mut module.code.code[0].body.body, 1), 0).kind = InstructionKind::Drop;
        assert_eq!(compile_error(&module), "operand stack underflow");
    }

    #[test]
    fn operand_types_that_disagree_with_validation_are_rejected() {
        let mut module = editable("(module (func (result i32) (i32.add (i32.const 1) (i32.const 2))))");
        plain_mut(&mut module.code.code[0].body.body, 1).kind = InstructionKind::I64Const { value: 2 };
        assert_eq!(compile_error(&module), "expected I32 operand, found I64");

        let mut module = editable("(module (func (result i32) (block (result i32) (i32.const 1))))");
        let block = block_mut(&mut module.code.code[0].body.body, 0);
        plain_mut(block, 0).kind = InstructionKind::F32Const { value: 1.0 };
        assert_eq!(compile_error(&module), "block end has [F32], expected [I32]");
    }

    #[test]
    fn if_without_else_must_pass_its_params_through() {
        // Removing the else leaves an implicit empty one that cannot produce the i32.
        let mut module = editable(
            "(module (func (result i32)
               (if (result i32) (i32.const 0) (then (unreachable)) (else (i32.const 2)))))",
        );
        match &mut module.code.code[0].body.body[1] {
            StructuredInstruction::If { else_branch, .. } => *else_branch = None,
            other => panic!("expected an if, got {other:?}"),
        }
        assert_eq!(compile_error(&module), "if without else turns [] into [I32]");
    }

    #[test]
    fn branch_operands_that_disagree_with_validation_are_rejected() {
        let mut module = editable("(module (func (result i32) (block (result i32) (br 0 (i32.const 1)))))");
        let block = block_mut(&mut module.code.code[0].body.body, 0);
        plain_mut(block, 0).kind = InstructionKind::I64Const { value: 1 };
        assert_eq!(compile_error(&module), "branch carries [I64], label expects [I32]");

        // br_if keeps its operand beneath the condition; remove the operand.
        let mut module = editable(
            "(module (func (result i32)
               (block (result i32) (br_if 0 (i32.const 1) (i32.const 1)))))",
        );
        block_mut(&mut module.code.code[0].body.body, 0).remove(0);
        assert_eq!(compile_error(&module), "branch needs [I32] above depth 0");

        let mut module = editable("(module (func (result i32) (return (i32.const 1))))");
        plain_mut(&mut module.code.code[0].body.body, 0).kind = InstructionKind::F32Const { value: 1.0 };
        assert_eq!(compile_error(&module), "branch carries [F32], label expects [I32]");
    }

    #[test]
    fn max_stack_words_is_the_deepest_point_counting_v128_as_two() {
        let words = |source: &str| compile_first_func(source).max_stack_words;
        assert_eq!(
            words("(module (func (result i32) (i32.add (i32.const 1) (i32.const 2))))"),
            2
        );
        assert_eq!(
            words("(module (func (result v128) (i32x4.add (v128.const i64x2 0 0) (v128.const i64x2 1 1))))"),
            4
        );
        // Across a block and a br_if carrying its result.
        assert_eq!(
            words(
                "(module (func (param i32) (result i64)
                   (i64.const 1)
                   (block (result i64)
                     (drop (f64.add (f64.const 1) (f64.const 2)))
                     (br_if 0 (i64.const 2) (local.get 0))
                     (drop)
                     (i64.const 3))
                   (i64.add)))"
            ),
            3
        );
        // The deeper arm of an if decides.
        assert_eq!(
            words(
                "(module (func (param i32) (result v128)
                   (if (result v128) (local.get 0)
                     (then (v128.const i64x2 0 0))
                     (else (i8x16.add (v128.const i64x2 0 0) (v128.const i64x2 1 1))))))"
            ),
            4
        );
        // A loop's v128 parameter stays on the stack beneath its body.
        assert_eq!(
            words(
                "(module (func (result v128)
                   (v128.const i64x2 1 1)
                   (loop (param v128) (result v128)
                     (i32x4.add (v128.const i64x2 2 2)))))"
            ),
            4
        );
        // Locals and parameters are not on the operand stack.
        assert_eq!(words("(module (func (param v128 v128) (local i64)))"), 0);
    }

    #[test]
    fn fib_iterative_frame_and_lowering() {
        let source = include_str!("../../benches/modules/fib_iterative.wat");
        let cf = compile_first_func(source);
        assert_eq!((cf.param_words, cf.result_words, cf.local_words), (1, 1, 5));
        assert_eq!(
            listing(source),
            "0: local.get 0\n1: i32.eqz\n2: i32.eqz\n3: br_if -> 6\n4: i32.const 0\n5: return\n\
             6: i32.const 0\n7: local.set 1\n8: i32.const 1\n9: local.set 2\n10: i32.const 1\n\
             11: local.set 4\n12: label (end -> 31)\n13: label (end -> 31)\n14: local.get 4\n\
             15: local.get 0\n16: i32.ge_u\n17: br_if -> 31\n18: local.get 1\n19: local.get 2\n\
             20: i32.add\n21: local.set 3\n22: local.get 2\n23: local.set 1\n24: local.get 3\n\
             25: local.set 2\n26: local.get 4\n27: i32.const 1\n28: i32.add\n29: local.set 4\n\
             30: br -> 14\n31: local.get 2\n32: end\n"
        );
    }

    #[test]
    fn noop_loop_branches_back_to_the_loop_body() {
        assert_eq!(
            listing(include_str!("../../benches/modules/noop_loop.wat")),
            "0: label (end -> 11)\n1: label (end -> 11)\n2: local.get 1\n3: local.get 0\n\
             4: i32.ge_u\n5: br_if -> 11\n6: local.get 1\n7: i32.const 1\n8: i32.add\n\
             9: local.set 1\n10: br -> 2\n11: local.get 1\n12: end\n"
        );
    }

    #[test]
    fn block_branches_forward_and_loop_branches_back() {
        assert_eq!(
            listing("(module (func (block (br 0))))"),
            "0: label (end -> 2)\n1: br -> 2\n2: end\n"
        );
        assert_eq!(
            listing("(module (func (loop (br 0))))"),
            "0: label (end -> 2)\n1: br -> 1\n2: end\n"
        );
    }

    #[test]
    fn branch_to_a_block_with_a_result_keeps_it() {
        assert_eq!(
            listing("(module (func (result i32) (block (result i32) (i32.const 42) (br 0))))"),
            "0: label (end -> 3)\n1: i32.const 42\n2: br -> 3 (stack 0, keep 1 values)\n3: end\n"
        );
    }

    #[test]
    fn if_skips_its_then_body_on_zero() {
        assert_eq!(
            listing("(module (func (param i32) (if (local.get 0) (then (nop)))))"),
            "0: local.get 0\n1: i32.eqz\n2: br_if -> 4\n3: nop\n4: end\n"
        );
        // With an else, the then-body jumps past it carrying the result.
        assert_eq!(
            listing(
                "(module (func (param i32) (result i32)
                   (if (result i32) (local.get 0) (then (i32.const 1)) (else (i32.const 2)))))"
            ),
            "0: local.get 0\n1: i32.eqz\n2: br_if -> 5\n3: i32.const 1\n\
             4: br -> 6 (stack 0, keep 1 values)\n5: i32.const 2\n6: end\n"
        );
    }

    #[test]
    fn return_inside_an_if_leaves_the_function() {
        assert_eq!(
            listing(
                "(module (func (param i32) (result i32)
                   (if (i32.eqz (local.get 0)) (then (return (i32.const 0))))
                   (i32.const 1)))"
            ),
            "0: local.get 0\n1: i32.eqz\n2: i32.eqz\n3: br_if -> 6\n4: i32.const 0\n5: return\n\
             6: i32.const 1\n7: end\n"
        );
    }

    #[test]
    fn br_table_entries_resolve_to_their_labels() {
        assert_eq!(
            listing(
                "(module (func (param i32) (result i32)
                   (block $a (result i32)
                     (block $b (result i32)
                       (i32.const 10)
                       (local.get 0)
                       (br_table 0 1 0)))))"
            ),
            "0: label (end -> 5)\n1: label (end -> 5)\n2: i32.const 10\n3: local.get 0\n\
             4: br_table [5, 5] default -> 5\n5: end\n"
        );
    }
}
