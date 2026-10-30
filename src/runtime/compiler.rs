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
use crate::parser::instruction::{BlockType, Instruction, InstructionKind};
use crate::parser::module::{FunctionType, Locals, Module, ValueType};
use crate::parser::structured::{StructuredFunction, StructuredInstruction};
use emit::{Emitter, Fixup};
use lower::lower;
use std::fmt;

mod emit;
mod lower;

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
        emitter: Emitter::default(),
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
    ctx.close_label()?;
    ctx.op(Op::End);
    let (ops, entry_cost) = ctx.emitter.finish()?;

    Ok(CompiledFunction {
        code: super::flat_executor::thread(ops),
        result_types: ftype.return_types.clone().into_boxed_slice(),
        param_words: words_of(&ftype.parameters),
        local_words: words_of(&ctx.locals),
        result_words: words_of(&ftype.return_types),
        max_stack_words: ctx.max_words,
        entry_cost,
    })
}

struct CompileContext<'a> {
    emitter: Emitter,
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

    /// Emit an op; the one-line arm form for mechanical 1:1 instruction
    /// mappings. Operand types follow the source instruction
    /// (`apply_types`), not the ops emitted for it.
    fn op(&mut self, op: Op) {
        self.emitter.emit(op);
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
    fn close_label(&mut self) -> Result {
        let end = self.emitter.pos();
        if let Some(Label {
            target: LabelTarget::End(fixups),
            ..
        }) = self.labels.pop()
        {
            for fixup in &fixups {
                self.emitter.patch(fixup, end)?;
            }
        }
        Ok(())
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

                self.open_label(LabelTarget::End(Vec::new()), sd, results)?;
                self.emit_body(body)?;
                if self.reachable {
                    self.check_top(sd, results, "block end")?;
                }
                self.close_label()?;
                // Reached by fall-through or by a branch to the label: either
                // way the block's results sit above sd.
                self.end_block(sd, results);
            }

            StructuredInstruction::Loop { block_type, body, .. } => {
                let Signature { params, results } = self.block_signature(block_type)?;
                let sd = self.depth_below(params.len())?;
                self.check_top(sd, params, "loop entry")?;

                // Branches restart the loop, carrying its params.
                self.open_label(LabelTarget::Start(self.emitter.pos()), sd, params)?;
                self.emit_body(body)?;
                if self.reachable {
                    self.check_top(sd, results, "loop end")?;
                }
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
                self.op(Op::I32Eqz);
                let skip_then = Fixup {
                    op: self.emitter.emit(Op::BrIf {
                        target: BrTarget {
                            pc: 0,
                            arity: 0,
                            stack_depth: self.words,
                            cost: 0,
                        },
                        fallthrough_cost: 0,
                    }),
                    entry: 0,
                };
                self.open_label(LabelTarget::End(Vec::new()), sd, results)?;
                self.emit_body(then_branch)?;
                if self.reachable {
                    self.check_top(sd, results, "if end")?;
                }

                if let Some(else_body) = else_branch {
                    // The then-body jumps past the else, carrying the results.
                    if self.reachable {
                        self.emit_br(0)?;
                    }
                    // The skip enters the else with the params on the stack,
                    // however the then-body ended.
                    self.emitter.patch(&skip_then, self.emitter.pos())?;
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
                    self.emitter.patch(&skip_then, self.emitter.pos())?;
                }
                self.close_label()?;
                self.end_block(sd, results);
            }
        }
        Ok(())
    }

    /// Type a plain (non-structured) instruction and emit its op.
    fn emit_plain(&mut self, inst: &'a Instruction) -> Result {
        let dropped = self.stack.last().copied();
        self.apply_types(&inst.kind)?;
        match &inst.kind {
            // Branches resolve their targets from the label stack.
            InstructionKind::Br { label_idx } => self.emit_br(*label_idx)?,
            InstructionKind::BrIf { label_idx } => self.emit_br_if(*label_idx)?,
            InstructionKind::BrTable { labels, default } => self.emit_br_table(labels, *default)?,
            InstructionKind::Return => self.emit_return()?,
            InstructionKind::Unreachable => {
                self.op(Op::Unreachable);
                self.reachable = false;
            }

            // A v128 local or operand takes two words.
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

            InstructionKind::Drop => self.op(match dropped {
                Some(ValueType::V128) => Op::DropV128,
                _ => Op::Drop,
            }),
            // The result, now on top, has the type of both operands.
            InstructionKind::Select | InstructionKind::SelectTyped { .. } => self.op(match self.stack.last() {
                Some(ValueType::V128) => Op::SelectV128,
                _ => Op::Select,
            }),

            InstructionKind::Block { .. }
            | InstructionKind::Loop { .. }
            | InstructionKind::If { .. }
            | InstructionKind::Else
            | InstructionKind::End => unreachable!("structured control instruction passed to emit_plain"),

            kind => self.op(lower(kind)),
        }
        Ok(())
    }

    /// The target of a branch `depth` labels out, after checking the operands
    /// it carries. A block end is not yet known; [`Self::defer_forward`]
    /// records where to patch it once the branch is emitted.
    fn branch_target(&self, depth: u32) -> Result<BrTarget> {
        let label = &self.labels[self.label_index(depth)?];
        self.check_branch(label.stack_depth, label.types)?;
        let pc = match label.target {
            LabelTarget::Start(start) => start,
            LabelTarget::End(_) => 0,
        };
        Ok(BrTarget {
            pc,
            arity: label.word_arity,
            stack_depth: label.word_depth,
            cost: 0,
        })
    }

    /// Record a fixup for each target of the op at `op` that lands on a block
    /// end, given the label depth of each target in [`Op::targets`] order.
    fn defer_forward(&mut self, op: usize, depths: impl IntoIterator<Item = u32>) -> Result {
        for (entry, depth) in depths.into_iter().enumerate() {
            let index = self.label_index(depth)?;
            if let LabelTarget::End(fixups) = &mut self.labels[index].target {
                fixups.push(Fixup { op, entry });
            }
        }
        Ok(())
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
        self.op(Op::Return);
        self.reachable = false;
        Ok(())
    }

    fn emit_br(&mut self, depth: u32) -> Result {
        if self.label_index(depth)? == 0 {
            return self.emit_return();
        }
        let target = self.branch_target(depth)?;
        let op = self.emitter.emit(Op::Br { target });
        self.reachable = false;
        self.defer_forward(op, [depth])
    }

    fn emit_br_if(&mut self, depth: u32) -> Result {
        let target = self.branch_target(depth)?;
        let op = self.emitter.emit(Op::BrIf {
            target,
            fallthrough_cost: 0,
        });
        self.defer_forward(op, [depth])
    }

    fn emit_br_table(&mut self, labels: &[u32], default: u32) -> Result {
        let targets = (labels.iter())
            .map(|depth| self.branch_target(*depth))
            .collect::<Result<_>>()?;
        let default_target = self.branch_target(default)?;
        let op = self.emitter.emit(Op::BrTable {
            targets,
            default: default_target,
        });
        self.reachable = false;
        self.defer_forward(op, labels.iter().copied().chain([default]))
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
        cf.ops().enumerate().map(|(i, op)| format!("{i}: {op}\n")).collect()
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
            "0: local.get 0\n1: i32.eqz\n2: i32.eqz\n3: br_if -> 6 [10, else 2]\n4: i32.const 0\n5: return\n\
             6: i32.const 0\n7: local.set 1\n8: i32.const 1\n9: local.set 2\n10: i32.const 1\n\
             11: local.set 4\n12: local.get 4\n13: local.get 0\n14: i32.ge_u\n15: br_if -> 29 [2, else 13]\n\
             16: local.get 1\n17: local.get 2\n18: i32.add\n19: local.set 3\n20: local.get 2\n21: local.set 1\n\
             22: local.get 3\n23: local.set 2\n24: local.get 4\n25: i32.const 1\n26: i32.add\n27: local.set 4\n\
             28: br -> 12 [4]\n29: local.get 2\n30: end\n"
        );
    }

    #[test]
    fn noop_loop_branches_back_to_the_loop_body() {
        assert_eq!(
            listing(include_str!("../../benches/modules/noop_loop.wat")),
            "0: local.get 1\n1: local.get 0\n2: i32.ge_u\n3: br_if -> 9 [2, else 5]\n4: local.get 1\n\
             5: i32.const 1\n6: i32.add\n7: local.set 1\n8: br -> 0 [4]\n9: local.get 1\n10: end\n"
        );
    }

    #[test]
    fn block_branches_forward_and_loop_branches_back() {
        assert_eq!(listing("(module (func (block (br 0))))"), "0: br -> 1 [1]\n1: end\n");
        assert_eq!(listing("(module (func (loop (br 0))))"), "0: br -> 0 [1]\n1: end\n");
    }

    #[test]
    fn branch_to_a_block_with_a_result_keeps_it() {
        assert_eq!(
            listing("(module (func (result i32) (block (result i32) (i32.const 42) (br 0))))"),
            "0: i32.const 42\n1: br -> 2 [1] (stack 0, keep 1 values)\n2: end\n"
        );
    }

    #[test]
    fn if_skips_its_then_body_on_zero() {
        assert_eq!(
            listing("(module (func (param i32) (if (local.get 0) (then (nop)))))"),
            "0: local.get 0\n1: i32.eqz\n2: br_if -> 4 [1, else 2]\n3: nop\n4: end\n"
        );
        // With an else, the then-body jumps past it carrying the result.
        assert_eq!(
            listing(
                "(module (func (param i32) (result i32)
                   (if (result i32) (local.get 0) (then (i32.const 1)) (else (i32.const 2)))))"
            ),
            "0: local.get 0\n1: i32.eqz\n2: br_if -> 5 [2, else 2]\n3: i32.const 1\n\
             4: br -> 6 [1] (stack 0, keep 1 values)\n5: i32.const 2\n6: end\n"
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
            "0: local.get 0\n1: i32.eqz\n2: i32.eqz\n3: br_if -> 6 [2, else 2]\n4: i32.const 0\n5: return\n\
             6: i32.const 1\n7: end\n"
        );
    }

    /// The cost of the function's first run, and its listing.
    fn metered(source: &str) -> (u32, String) {
        (compile_first_func(source).entry_cost, listing(source))
    }

    #[test]
    fn runs_cost_their_ops_up_to_the_next_exit() {
        assert_eq!(metered("(module (func))"), (1, "0: end\n".into()));
        assert_eq!(
            metered("(module (func (result i32) (i32.const 7)))"),
            (2, "0: i32.const 7\n1: end\n".into())
        );
        // Blocks emit no ops, and the inner block's end, reached by falling
        // through, lies inside the fall-through run.
        assert_eq!(
            metered("(module (func (param i32) (block (block (br_if 1 (local.get 0))) (nop)) (nop)))"),
            (
                2,
                "0: local.get 0\n1: br_if -> 3 [2, else 3]\n2: nop\n3: nop\n4: end\n".into()
            )
        );
        // A br_if to the function label lands on the final end.
        assert_eq!(
            metered("(module (func (param i32) (br_if 0 (local.get 0)) (nop)))"),
            (2, "0: local.get 0\n1: br_if -> 3 [1, else 2]\n2: nop\n3: end\n".into())
        );
    }

    #[test]
    fn loop_back_edges_charge_from_the_loop_body() {
        // The back edge lands on the loop's first op: here, the branch itself.
        assert_eq!(
            metered("(module (func (loop (block (br 1)))))"),
            (1, "0: br -> 0 [1]\n1: end\n".into())
        );
        assert_eq!(
            metered("(module (func (loop (nop) (br 0))))"),
            (2, "0: nop\n1: br -> 0 [2]\n2: end\n".into())
        );
    }

    #[test]
    fn unreachable_ends_a_run() {
        assert_eq!(
            metered("(module (func (block (unreachable)) (i32.const 1) (drop)))"),
            (1, "0: unreachable\n1: i32.const 1\n2: drop\n3: end\n".into())
        );
    }

    #[test]
    fn br_table_entries_cost_the_run_each_lands_in() {
        assert_eq!(
            metered("(module (func (param i32) (block (block (br_table 0 1 (local.get 0))) (nop) (nop))))"),
            (
                2,
                "0: local.get 0\n1: br_table [2 [3]] default -> 4 [1]\n2: nop\n3: nop\n4: end\n".into()
            )
        );
    }

    #[test]
    fn then_body_jumps_over_the_else_only_if_it_falls_through() {
        assert_eq!(
            listing(
                "(module (func (param i32) (result i32)
                   (if (result i32) (local.get 0) (then (return (i32.const 1))) (else (i32.const 2)))))"
            ),
            "0: local.get 0\n1: i32.eqz\n2: br_if -> 5 [2, else 2]\n3: i32.const 1\n4: return\n\
             5: i32.const 2\n6: end\n"
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
            "0: i32.const 10\n1: local.get 0\n\
             2: br_table [3 [1], 3 [1]] default -> 3 [1] (stack 0, keep 1 values)\n3: end\n"
        );
    }
}
