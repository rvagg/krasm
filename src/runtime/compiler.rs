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
//! stack. The compiler tracks stack depth during emission to compute these
//! values.

use super::bytecode::{BrTarget, CompiledFunction, Op};
use crate::parser::instruction::{BlockType, Instruction, InstructionKind};
use crate::parser::module::{ExternalKind, FunctionType, Module};
use crate::parser::structured::{StructuredFunction, StructuredInstruction};

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

/// A label on the compile-time stack, tracking where branches should go
/// and how to clean up the stack on branch.
enum Label {
    /// Block/if: branch goes to end (forward). Target unknown until body is
    /// emitted, so we record the positions that need patching.
    Block {
        fixups: Vec<Fixup>,
        stack_depth: u32,
        arity: u16,
    },
    /// Loop: branch goes to start (backward). Target is known at entry.
    Loop { start: u32, stack_depth: u32, arity: u16 },
}

impl Label {
    fn stack_depth(&self) -> u32 {
        match self {
            Label::Block { stack_depth, .. } | Label::Loop { stack_depth, .. } => *stack_depth,
        }
    }

    fn arity(&self) -> u16 {
        match self {
            Label::Block { arity, .. } | Label::Loop { arity, .. } => *arity,
        }
    }
}

/// Resolve a BlockType to (param_count, result_count).
fn block_signature(block_type: &BlockType, types: &[FunctionType]) -> (u16, u16) {
    match block_type {
        BlockType::Empty => (0, 0),
        BlockType::Value(_) => (0, 1),
        BlockType::FuncType(idx) => {
            if let Some(ft) = types.get(*idx as usize) {
                (ft.parameters.len() as u16, ft.return_types.len() as u16)
            } else {
                (0, 0)
            }
        }
    }
}

/// Build a `(param_count, result_count)` table for all functions in a module,
/// indexed by module-level function index (imports first, then locals).
pub fn build_func_sigs(module: &Module) -> Vec<(u16, u16)> {
    let mut sigs = Vec::new();
    // Imported functions
    for imp in &module.imports.imports {
        if let ExternalKind::Function(type_idx) = imp.external_kind
            && let Some(ft) = module.types.get(type_idx)
        {
            sigs.push((ft.parameters.len() as u16, ft.return_types.len() as u16));
        }
    }
    // Local functions
    for func in &module.functions.functions {
        if let Some(ft) = module.types.get(func.ftype_index) {
            sigs.push((ft.parameters.len() as u16, ft.return_types.len() as u16));
        }
    }
    sigs
}

/// Compile all local functions in a module to flat bytecode.
pub fn compile_module(module: &Module) -> Vec<CompiledFunction> {
    let func_sigs = build_func_sigs(module);
    module
        .code
        .code
        .iter()
        .enumerate()
        .map(|(i, func)| {
            let ftype_idx = module.functions.functions[i].ftype_index;
            let ftype = module.types.get(ftype_idx).expect("function type not found");
            compile(
                &func.body,
                ftype.parameters.len() as u32,
                &module.types.types,
                &func_sigs,
            )
        })
        .collect()
}

/// Compile a structured function into flat bytecode.
///
/// `types` is the module's type section, needed to resolve `BlockType::FuncType`.
/// `func_sigs` maps module-level function index to `(param_count, result_count)`,
/// needed to compute the stack effect of `call` instructions.
pub fn compile(
    func: &StructuredFunction,
    param_count: u32,
    types: &[FunctionType],
    func_sigs: &[(u16, u16)],
) -> CompiledFunction {
    let mut ctx = CompileContext {
        ops: Vec::new(),
        labels: Vec::new(),
        depth: 0,
        types,
        func_sigs,
    };

    // The function body itself acts as an implicit block. Branches at full
    // depth become Return ops (handled in emit_br). The function label's
    // arity is the result count, and stack_depth is 0 (nothing below it).
    ctx.labels.push(Label::Block {
        fixups: Vec::new(),
        stack_depth: 0,
        arity: func.return_types.len() as u16,
    });

    ctx.emit_body(&func.body);

    // Patch any fixups on the implicit function block.
    let end_pos = ctx.ops.len() as u32;
    if let Some(Label::Block { fixups, .. }) = ctx.labels.pop() {
        for fixup in &fixups {
            ctx.apply_fixup(fixup, end_pos);
        }
    }

    ctx.ops.push(Op::End);

    CompiledFunction {
        ops: ctx.ops,
        local_count: func.local_count as u32,
        param_count,
        result_count: func.return_types.len() as u32,
    }
}

struct CompileContext<'a> {
    ops: Vec<Op>,
    /// Compile-time label stack. Innermost label is last.
    labels: Vec<Label>,
    /// Tracked stack depth (number of values on the operand stack).
    depth: u32,
    /// Module type section for resolving BlockType::FuncType.
    types: &'a [FunctionType],
    /// (param_count, result_count) per module-level function index.
    func_sigs: &'a [(u16, u16)],
}

impl<'a> CompileContext<'a> {
    /// Current emit position (index of the next op to be pushed).
    fn pos(&self) -> u32 {
        self.ops.len() as u32
    }

    /// Emit an op whose stack effect is fully described by `Op::stack_delta`.
    /// The one-line arm form for mechanical 1:1 instruction mappings.
    fn op(&mut self, op: Op) {
        self.emit(op);
    }

    /// Emit an op, adjust stack depth, and return its position.
    fn emit(&mut self, op: Op) -> usize {
        let pos = self.ops.len();
        self.depth = (self.depth as i32 + op.stack_delta()) as u32;
        self.ops.push(op);
        pos
    }

    /// Patch the target pc of a Br, BrIf, or Label at a known position.
    fn patch_target(&mut self, pos: usize, target: u32) {
        match &mut self.ops[pos] {
            Op::Br { target: t, .. } | Op::BrIf { target: t, .. } => *t = target,
            Op::Label { end_target } => *end_target = target,
            _ => {}
        }
    }

    /// Apply a fixup, setting the target pc to `target`.
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

    /// Emit a sequence of structured instructions.
    fn emit_body(&mut self, body: &[StructuredInstruction]) {
        for inst in body {
            self.emit_instruction(inst);
        }
    }

    /// Emit a single structured instruction.
    fn emit_instruction(&mut self, inst: &StructuredInstruction) {
        match inst {
            StructuredInstruction::Plain(i) => self.emit_plain(i),

            StructuredInstruction::Block { block_type, body, .. } => {
                let (params, results) = block_signature(block_type, self.types);

                // stack_depth is below the parameters (params are "inside" the block)
                let sd = self.depth - params as u32;

                let label_pos = self.emit(Op::Label { end_target: 0 });
                self.labels.push(Label::Block {
                    fixups: vec![Fixup::Branch(label_pos)],
                    stack_depth: sd,
                    arity: results,
                });

                self.emit_body(body);

                let end_pos = self.pos();
                if let Some(Label::Block { fixups, .. }) = self.labels.pop() {
                    for fixup in &fixups {
                        self.apply_fixup(fixup, end_pos);
                    }
                }
            }

            StructuredInstruction::Loop { block_type, body, .. } => {
                let (params, _results) = block_signature(block_type, self.types);

                // For loops, branch arity is the param count (restart with params)
                let sd = self.depth - params as u32;

                let loop_start = self.pos();
                self.emit(Op::Label { end_target: 0 });
                self.labels.push(Label::Loop {
                    start: loop_start + 1,
                    stack_depth: sd,
                    arity: params,
                });

                self.emit_body(body);

                let end_pos = self.pos();
                self.patch_target(loop_start as usize, end_pos);

                self.labels.pop();
            }

            StructuredInstruction::If {
                block_type,
                then_branch,
                else_branch,
                ..
            } => {
                let (params, results) = block_signature(block_type, self.types);

                // The condition has been consumed by the i32.eqz + br_if below.
                // stack_depth is below the block params (after condition is popped).
                // At this point, depth includes the condition. After eqz (no net
                // change) and br_if (pops 1), depth = depth - 1. The block params
                // are the next `params` values below the condition.
                let sd = self.depth - 1 - params as u32;

                // Invert condition: skip then-body when condition is 0
                self.emit(Op::I32Eqz);
                let skip_then = self.emit(Op::BrIf {
                    target: 0,
                    arity: 0,
                    stack_depth: self.depth,
                });

                self.labels.push(Label::Block {
                    fixups: Vec::new(),
                    stack_depth: sd,
                    arity: results,
                });

                self.emit_body(then_branch);

                if let Some(else_body) = else_branch {
                    // Jump past else at end of then-body. If the block has
                    // results, this branch carries them.
                    let skip_else = self.emit(Op::Br {
                        target: 0,
                        arity: results,
                        stack_depth: sd,
                    });

                    let else_start = self.pos();
                    self.patch_target(skip_then, else_start);

                    self.emit_body(else_body);

                    let end_pos = self.pos();
                    self.patch_target(skip_else, end_pos);
                    if let Some(Label::Block { fixups, .. }) = self.labels.pop() {
                        for fixup in &fixups {
                            self.apply_fixup(fixup, end_pos);
                        }
                    }
                } else {
                    let end_pos = self.pos();
                    self.patch_target(skip_then, end_pos);
                    if let Some(Label::Block { fixups, .. }) = self.labels.pop() {
                        for fixup in &fixups {
                            self.apply_fixup(fixup, end_pos);
                        }
                    }
                }
            }
        }
    }

    /// Emit a plain (non-control-flow-structure) instruction.
    fn emit_plain(&mut self, inst: &Instruction) {
        match &inst.kind {
            InstructionKind::I32Const { value } => self.op(Op::I32Const(*value)),
            InstructionKind::I64Const { value } => self.op(Op::I64Const(*value)),
            InstructionKind::F32Const { value } => self.op(Op::F32Const(*value)),
            InstructionKind::F64Const { value } => self.op(Op::F64Const(*value)),
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

            InstructionKind::LocalGet { local_idx } => {
                self.emit(Op::LocalGet { index: *local_idx });
            }
            InstructionKind::LocalSet { local_idx } => {
                self.emit(Op::LocalSet { index: *local_idx });
            }
            InstructionKind::LocalTee { local_idx } => {
                self.emit(Op::LocalTee { index: *local_idx });
            }

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
            InstructionKind::MemorySize => self.op(Op::MemorySize),
            InstructionKind::MemoryGrow => self.op(Op::MemoryGrow),
            InstructionKind::MemoryCopy => self.op(Op::MemoryCopy),
            InstructionKind::MemoryFill => self.op(Op::MemoryFill),

            InstructionKind::Call { func_idx } => {
                // Stack effect depends on callee's signature: pops params, pushes results.
                // We bypass self.emit() and adjust depth manually.
                let (params, results) = self.func_sigs.get(*func_idx as usize).copied().unwrap_or((0, 0));
                self.ops.push(Op::Call { func_idx: *func_idx });
                self.depth = (self.depth as i32 - params as i32 + results as i32) as u32;
            }
            InstructionKind::CallIndirect { type_idx, table_idx } => {
                // Pops the i32 element index, then the params; pushes results.
                // The signature comes from the type section, not func_sigs.
                let (params, results) = self
                    .types
                    .get(*type_idx as usize)
                    .map(|ft| (ft.parameters.len() as i32, ft.return_types.len() as i32))
                    .unwrap_or((0, 0));
                self.ops.push(Op::CallIndirect {
                    type_idx: *type_idx,
                    table_idx: *table_idx,
                });
                self.depth = (self.depth as i32 - 1 - params + results) as u32;
            }

            InstructionKind::Br { label_idx } => self.emit_br(*label_idx),
            InstructionKind::BrIf { label_idx } => self.emit_br_if(*label_idx),
            InstructionKind::BrTable { labels, default } => self.emit_br_table(labels, *default),
            InstructionKind::Return => self.op(Op::Return),
            InstructionKind::Nop => self.op(Op::Nop),
            InstructionKind::Unreachable => self.op(Op::Unreachable),
            InstructionKind::Drop => self.op(Op::Drop),
            InstructionKind::Select => self.op(Op::Select),
            InstructionKind::SelectTyped { .. } => self.op(Op::Select),

            // Not yet compiled: emit a named placeholder so execution
            // traps with exactly which instruction is missing, rather than
            // computing anything silently wrong.
            other => self.op(Op::Unsupported(other.mnemonic())),
        }
    }

    /// Emit an unconditional branch to the label at the given depth.
    fn emit_br(&mut self, depth: u32) {
        let label_index = self.labels.len() - 1 - depth as usize;

        // label_index 0 is the implicit function block. A branch targeting
        // it is a return from the function.
        if label_index == 0 {
            self.emit(Op::Return);
            return;
        }

        let arity = self.labels[label_index].arity();
        let stack_depth = self.labels[label_index].stack_depth();

        match &mut self.labels[label_index] {
            Label::Loop { start, .. } => {
                let target = *start;
                self.emit(Op::Br {
                    target,
                    arity,
                    stack_depth,
                });
            }
            Label::Block { fixups, .. } => {
                let pos = self.ops.len();
                fixups.push(Fixup::Branch(pos));
                self.emit(Op::Br {
                    target: 0,
                    arity,
                    stack_depth,
                });
            }
        }
    }

    /// Emit a conditional branch to the label at the given depth.
    fn emit_br_if(&mut self, depth: u32) {
        let label_index = self.labels.len() - 1 - depth as usize;

        // br_if to function-level block: conditional branch to End.
        if label_index == 0 {
            let arity = self.labels[0].arity();
            let stack_depth = self.labels[0].stack_depth();
            if let Label::Block { fixups, .. } = &mut self.labels[label_index] {
                let pos = self.ops.len();
                fixups.push(Fixup::Branch(pos));
                self.emit(Op::BrIf {
                    target: 0,
                    arity,
                    stack_depth,
                });
            }
            return;
        }

        let arity = self.labels[label_index].arity();
        let stack_depth = self.labels[label_index].stack_depth();

        match &mut self.labels[label_index] {
            Label::Loop { start, .. } => {
                let target = *start;
                self.emit(Op::BrIf {
                    target,
                    arity,
                    stack_depth,
                });
            }
            Label::Block { fixups, .. } => {
                let pos = self.ops.len();
                fixups.push(Fixup::Branch(pos));
                self.emit(Op::BrIf {
                    target: 0,
                    arity,
                    stack_depth,
                });
            }
        }
    }

    /// Emit a table branch (br_table).
    fn emit_br_table(&mut self, labels: &[u32], default: u32) {
        let resolve = |depth: u32, ctx: &Self| -> BrTarget {
            let label_index = ctx.labels.len() - 1 - depth as usize;
            let arity = ctx.labels[label_index].arity();
            let stack_depth = ctx.labels[label_index].stack_depth();
            let pc = match &ctx.labels[label_index] {
                Label::Loop { start, .. } => *start,
                Label::Block { .. } => 0, // patched later
            };
            BrTarget { pc, arity, stack_depth }
        };

        let targets: Vec<BrTarget> = labels.iter().map(|d| resolve(*d, self)).collect();
        let default_target = resolve(default, self);

        let pos = self.ops.len();

        // Record fixups for forward (Block) targets
        for (i, depth) in labels.iter().enumerate() {
            let label_index = self.labels.len() - 1 - *depth as usize;
            if let Label::Block { fixups, .. } = &mut self.labels[label_index] {
                fixups.push(Fixup::TableEntry {
                    op_pos: pos,
                    index: Some(i),
                });
            }
        }
        {
            let label_index = self.labels.len() - 1 - default as usize;
            if let Label::Block { fixups, .. } = &mut self.labels[label_index] {
                fixups.push(Fixup::TableEntry {
                    op_pos: pos,
                    index: None,
                });
            }
        }

        self.ops.push(Op::BrTable {
            targets,
            default: default_target,
        });
        // br_table pops an i32 index
        self.depth -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wat;

    /// Helper: parse WAT, extract first function, compile it.
    fn compile_first_func(source: &str) -> CompiledFunction {
        let module = wat::parse(source).expect("WAT parse failed");
        let func_sigs = build_func_sigs(&module);
        let func = &module.code.code[0];
        let ftype_idx = module.functions.functions[0].ftype_index;
        let ftype = module.types.get(ftype_idx).expect("function type not found");
        compile(
            &func.body,
            ftype.parameters.len() as u32,
            &module.types.types,
            &func_sigs,
        )
    }

    #[test]
    fn compile_noop_loop() {
        let cf = compile_first_func(include_str!("../../benches/modules/noop_loop.wat"));
        println!("{cf}");

        assert!(matches!(cf.ops.last().unwrap(), Op::End));

        let has_backward_br = cf
            .ops
            .iter()
            .enumerate()
            .any(|(i, op)| matches!(op, Op::Br { target, .. } if (*target as usize) < i));
        assert!(has_backward_br, "loop should have a backward branch");
    }

    #[test]
    fn compile_fib_iterative() {
        let cf = compile_first_func(include_str!("../../benches/modules/fib_iterative.wat"));
        println!("{cf}");

        assert_eq!(cf.param_count, 1);
        assert_eq!(cf.result_count, 1);
        assert_eq!(cf.local_count, 5);
        assert!(matches!(cf.ops.last().unwrap(), Op::End));
    }

    #[test]
    fn compile_simple_add() {
        let cf = compile_first_func("(module (func (param i32 i32) (result i32) local.get 0 local.get 1 i32.add))");
        println!("{cf}");

        assert_eq!(cf.param_count, 2);
        assert_eq!(cf.ops.len(), 4);
        assert!(matches!(cf.ops[0], Op::LocalGet { index: 0 }));
        assert!(matches!(cf.ops[1], Op::LocalGet { index: 1 }));
        assert!(matches!(cf.ops[2], Op::I32Add));
        assert!(matches!(cf.ops[3], Op::End));
    }

    #[test]
    fn compile_block_br() {
        let cf = compile_first_func("(module (func (block (br 0))))");
        println!("{cf}");

        let br_pos = cf.ops.iter().position(|op| matches!(op, Op::Br { .. })).unwrap();
        if let Op::Br { target, .. } = &cf.ops[br_pos] {
            assert!(*target as usize > br_pos, "block br should jump forward");
        }
    }

    #[test]
    fn compile_loop_br() {
        let cf = compile_first_func("(module (func (loop (br 0))))");
        println!("{cf}");

        let br_pos = cf.ops.iter().position(|op| matches!(op, Op::Br { .. })).unwrap();
        if let Op::Br { target, .. } = &cf.ops[br_pos] {
            assert!(
                (*target as usize) <= br_pos,
                "loop br should jump backward or self-loop"
            );
        }
    }

    #[test]
    fn compile_if_then() {
        let cf = compile_first_func("(module (func (param i32) (if (local.get 0) (then (nop)))))");
        println!("{cf}");

        assert!(cf.ops.iter().any(|op| matches!(op, Op::I32Eqz)));
        assert!(cf.ops.iter().any(|op| matches!(op, Op::BrIf { .. })));
    }

    #[test]
    fn compile_if_then_else() {
        let cf = compile_first_func(
            "(module (func (param i32) (result i32)
                (if (result i32) (local.get 0)
                    (then (i32.const 1))
                    (else (i32.const 2)))))",
        );
        println!("{cf}");

        let consts: Vec<_> = cf
            .ops
            .iter()
            .filter_map(|op| if let Op::I32Const(v) = op { Some(*v) } else { None })
            .collect();
        assert!(consts.contains(&1));
        assert!(consts.contains(&2));
    }

    #[test]
    fn compile_return_from_if() {
        let cf = compile_first_func(
            "(module (func (param i32) (result i32)
                (if (i32.eqz (local.get 0))
                    (then (return (i32.const 0))))
                (i32.const 1)))",
        );
        println!("{cf}");

        assert!(cf.ops.iter().any(|op| matches!(op, Op::Return)));
    }

    #[test]
    fn compile_block_result_arity() {
        // Block with result -- br should carry arity=1
        let cf = compile_first_func(
            "(module (func (result i32)
                (block (result i32)
                    (i32.const 42)
                    (br 0))))",
        );
        println!("{cf}");

        let br = cf.ops.iter().find(|op| matches!(op, Op::Br { .. })).unwrap();
        if let Op::Br { arity, .. } = br {
            assert_eq!(*arity, 1, "block result branch should have arity 1");
        }
    }

    #[test]
    fn compile_br_table() {
        let cf = compile_first_func(
            "(module (func (param i32) (result i32)
                (block $a (result i32)
                    (block $b (result i32)
                        (i32.const 10)
                        (local.get 0)
                        (br_table 0 1 0)))))",
        );
        println!("{cf}");

        assert!(cf.ops.iter().any(|op| matches!(op, Op::BrTable { .. })));
    }
}
