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

use super::bytecode::{CompiledFunction, Op};
use crate::parser::instruction::InstructionKind;
use crate::parser::structured::{StructuredFunction, StructuredInstruction};

/// A label on the compile-time stack, tracking where branches should go.
enum Label {
    /// Block/if: branch goes to end (forward). Target unknown until body is
    /// emitted, so we record the positions that need patching.
    Block { fixups: Vec<usize> },
    /// Loop: branch goes to start (backward). Target is known at entry.
    Loop { start: u32 },
}

/// Compile a structured function into flat bytecode.
pub fn compile(func: &StructuredFunction, param_count: u32) -> CompiledFunction {
    let mut ctx = CompileContext {
        ops: Vec::new(),
        labels: Vec::new(),
    };

    // The function body itself acts as an implicit block: `br` at depth equal
    // to the full label stack depth means return. We model this as a Block
    // label whose fixups would target the End instruction, but since it's the
    // outermost scope, branches to it become Return ops instead. We handle
    // this specially in emit_br.
    ctx.labels.push(Label::Block { fixups: Vec::new() });

    ctx.emit_body(&func.body);

    // Any fixups on the implicit function block point past the end.
    // These should have been emitted as Return, so there shouldn't be
    // pending fixups, but patch them to the End just in case.
    let end_pos = ctx.ops.len() as u32;
    if let Some(Label::Block { fixups }) = ctx.labels.pop() {
        for pos in fixups {
            ctx.patch_target(pos, end_pos);
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

struct CompileContext {
    ops: Vec<Op>,
    /// Compile-time label stack. Innermost label is last.
    labels: Vec<Label>,
}

impl CompileContext {
    /// Current emit position (index of the next op to be pushed).
    fn pos(&self) -> u32 {
        self.ops.len() as u32
    }

    /// Emit an op and return its position.
    fn emit(&mut self, op: Op) -> usize {
        let pos = self.ops.len();
        self.ops.push(op);
        pos
    }

    /// Patch the target of a branch op at the given position.
    fn patch_target(&mut self, pos: usize, target: u32) {
        match &mut self.ops[pos] {
            Op::Br { target: t } | Op::BrIf { target: t } => *t = target,
            Op::Label { end_target } => *end_target = target,
            _ => {}
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

            StructuredInstruction::Block { body, .. } => {
                // Label marks block start; end_target will be patched.
                let label_pos = self.emit(Op::Label { end_target: 0 });
                self.labels.push(Label::Block {
                    fixups: vec![label_pos],
                });

                self.emit_body(body);

                // Patch: all fixups (including the Label) point here.
                let end_pos = self.pos();
                if let Some(Label::Block { fixups }) = self.labels.pop() {
                    for pos in fixups {
                        self.patch_target(pos, end_pos);
                    }
                }
            }

            StructuredInstruction::Loop { body, .. } => {
                // Loop label: branches jump back to the first body instruction.
                let loop_start = self.pos();
                self.emit(Op::Label { end_target: 0 }); // patched below
                self.labels.push(Label::Loop { start: loop_start + 1 });

                self.emit_body(body);

                // Patch the Label's end_target to point past the loop
                // (used by Label display, not by branch resolution).
                let end_pos = self.pos();
                self.patch_target(loop_start as usize, end_pos);

                if let Some(Label::Loop { .. }) = self.labels.pop() {
                    // Loop labels have no fixups; branches resolve immediately.
                }
            }

            StructuredInstruction::If {
                then_branch,
                else_branch,
                ..
            } => {
                // Compile as: BrIf(else_or_end) + then_body [+ Br(end) + else_body]
                //
                // Note: wasm `if` pops an i32 condition. The condition is
                // already on the stack from preceding instructions. We emit a
                // conditional branch that skips the then-body when the
                // condition is zero (i.e. branch on eqz).

                // Emit i32.eqz + br_if to skip then-body when condition is 0
                self.emit(Op::I32Eqz);
                let skip_then = self.emit(Op::BrIf { target: 0 });

                // Push block label for `br` inside the if body
                self.labels.push(Label::Block { fixups: Vec::new() });

                self.emit_body(then_branch);

                if let Some(else_body) = else_branch {
                    // Jump past else at end of then-body
                    let skip_else = self.emit(Op::Br { target: 0 });

                    // Patch: skip_then jumps to start of else
                    let else_start = self.pos();
                    self.patch_target(skip_then, else_start);

                    self.emit_body(else_body);

                    // Patch: skip_else and any br fixups jump here
                    let end_pos = self.pos();
                    self.patch_target(skip_else, end_pos);
                    if let Some(Label::Block { fixups }) = self.labels.pop() {
                        for pos in fixups {
                            self.patch_target(pos, end_pos);
                        }
                    }
                } else {
                    // No else: skip_then jumps past the then-body
                    let end_pos = self.pos();
                    self.patch_target(skip_then, end_pos);
                    if let Some(Label::Block { fixups }) = self.labels.pop() {
                        for pos in fixups {
                            self.patch_target(pos, end_pos);
                        }
                    }
                }
            }
        }
    }

    /// Emit a plain (non-control-flow-structure) instruction.
    fn emit_plain(&mut self, inst: &crate::parser::instruction::Instruction) {
        match &inst.kind {
            InstructionKind::I32Const { value } => {
                self.emit(Op::I32Const(*value));
            }
            InstructionKind::I32Add => {
                self.emit(Op::I32Add);
            }
            InstructionKind::I32Sub => {
                self.emit(Op::I32Sub);
            }
            InstructionKind::I32Mul => {
                self.emit(Op::I32Mul);
            }
            InstructionKind::I32Eqz => {
                self.emit(Op::I32Eqz);
            }
            InstructionKind::I32Eq => {
                self.emit(Op::I32Eq);
            }
            InstructionKind::I32Ne => {
                self.emit(Op::I32Ne);
            }
            InstructionKind::I32LtS => {
                self.emit(Op::I32LtS);
            }
            InstructionKind::I32LtU => {
                self.emit(Op::I32LtU);
            }
            InstructionKind::I32GtS => {
                self.emit(Op::I32GtS);
            }
            InstructionKind::I32GtU => {
                self.emit(Op::I32GtU);
            }
            InstructionKind::I32LeS => {
                self.emit(Op::I32LeS);
            }
            InstructionKind::I32LeU => {
                self.emit(Op::I32LeU);
            }
            InstructionKind::I32GeS => {
                self.emit(Op::I32GeS);
            }
            InstructionKind::I32GeU => {
                self.emit(Op::I32GeU);
            }

            InstructionKind::LocalGet { local_idx } => {
                self.emit(Op::LocalGet { index: *local_idx });
            }
            InstructionKind::LocalSet { local_idx } => {
                self.emit(Op::LocalSet { index: *local_idx });
            }
            InstructionKind::LocalTee { local_idx } => {
                self.emit(Op::LocalTee { index: *local_idx });
            }

            InstructionKind::Br { label_idx } => self.emit_br(*label_idx),
            InstructionKind::BrIf { label_idx } => self.emit_br_if(*label_idx),
            InstructionKind::Return => {
                self.emit(Op::Return);
            }
            InstructionKind::Nop => {
                self.emit(Op::Nop);
            }
            InstructionKind::Unreachable => {
                self.emit(Op::Unreachable);
            }
            InstructionKind::Drop => {
                self.emit(Op::Drop);
            }

            // Not yet compiled -- emit Unreachable as a placeholder so we
            // get a clear trap rather than silent wrong behaviour.
            _ => {
                self.emit(Op::Unreachable);
            }
        }
    }

    /// Emit an unconditional branch to the label at the given depth.
    fn emit_br(&mut self, depth: u32) {
        let label_index = self.labels.len() - 1 - depth as usize;

        // Depth 0 = innermost label, depth N = outermost.
        // If this targets the implicit function block (label_index == 0),
        // emit Return instead of a branch.
        if label_index == 0 {
            self.emit(Op::Return);
            return;
        }

        match &mut self.labels[label_index] {
            Label::Loop { start } => {
                let target = *start;
                self.emit(Op::Br { target });
            }
            Label::Block { fixups } => {
                // Forward branch: target unknown, record for patching.
                let pos = self.ops.len();
                fixups.push(pos);
                self.emit(Op::Br { target: 0 });
            }
        }
    }

    /// Emit a conditional branch to the label at the given depth.
    fn emit_br_if(&mut self, depth: u32) {
        let label_index = self.labels.len() - 1 - depth as usize;

        // br_if to function-level block: conditional return.
        // We handle this by branching to the End instruction.
        // (A real conditional return would need special handling, but
        // for the spike this works because End is at the end of ops.)
        if label_index == 0 {
            if let Label::Block { fixups } = &mut self.labels[label_index] {
                let pos = self.ops.len();
                fixups.push(pos);
                self.emit(Op::BrIf { target: 0 });
            }
            return;
        }

        match &mut self.labels[label_index] {
            Label::Loop { start } => {
                let target = *start;
                self.emit(Op::BrIf { target });
            }
            Label::Block { fixups } => {
                let pos = self.ops.len();
                fixups.push(pos);
                self.emit(Op::BrIf { target: 0 });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wat;

    /// Helper: parse WAT, extract first function, compile it.
    fn compile_first_func(source: &str) -> CompiledFunction {
        let module = wat::parse(source).expect("WAT parse failed");
        let func = &module.code.code[0];
        let ftype_idx = module.functions.functions[0].ftype_index;
        let ftype = module.types.get(ftype_idx).expect("function type not found");
        compile(&func.body, ftype.parameters.len() as u32)
    }

    #[test]
    fn compile_noop_loop() {
        let cf = compile_first_func(include_str!("../../benches/modules/noop_loop.wat"));
        println!("{cf}");

        // Verify structure: should have Label, Label (loop), branch ops, End
        assert!(matches!(cf.ops.last().unwrap(), Op::End));

        // Loop should have a backward branch
        let has_backward_br = cf
            .ops
            .iter()
            .enumerate()
            .any(|(i, op)| matches!(op, Op::Br { target } if (*target as usize) < i));
        assert!(has_backward_br, "loop should have a backward branch");
    }

    #[test]
    fn compile_fib_iterative() {
        let cf = compile_first_func(include_str!("../../benches/modules/fib_iterative.wat"));
        println!("{cf}");

        assert_eq!(cf.param_count, 1);
        assert_eq!(cf.result_count, 1);
        // 5 locals: $n (param), $a, $b, $tmp, $i
        assert_eq!(cf.local_count, 5);
        assert!(matches!(cf.ops.last().unwrap(), Op::End));
    }

    #[test]
    fn compile_simple_add() {
        let cf = compile_first_func("(module (func (param i32 i32) (result i32) local.get 0 local.get 1 i32.add))");
        println!("{cf}");

        assert_eq!(cf.param_count, 2);
        assert_eq!(cf.ops.len(), 4); // local.get, local.get, i32.add, end
        assert!(matches!(cf.ops[0], Op::LocalGet { index: 0 }));
        assert!(matches!(cf.ops[1], Op::LocalGet { index: 1 }));
        assert!(matches!(cf.ops[2], Op::I32Add));
        assert!(matches!(cf.ops[3], Op::End));
    }

    #[test]
    fn compile_block_br() {
        // (block (br 0)) -- branch should jump past the block
        let cf = compile_first_func("(module (func (block (br 0))))");
        println!("{cf}");

        // Find the Br and check it targets past the block
        let br_pos = cf.ops.iter().position(|op| matches!(op, Op::Br { .. })).unwrap();
        if let Op::Br { target } = &cf.ops[br_pos] {
            assert!(*target as usize > br_pos, "block br should jump forward");
        }
    }

    #[test]
    fn compile_loop_br() {
        // (loop (br 0)) -- branch should jump back to loop start
        let cf = compile_first_func("(module (func (loop (br 0))))");
        println!("{cf}");

        let br_pos = cf.ops.iter().position(|op| matches!(op, Op::Br { .. })).unwrap();
        if let Op::Br { target } = &cf.ops[br_pos] {
            // br 0 in a bare loop is a self-loop (target == position)
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

        // Should have: local.get, i32.eqz, br_if(skip), nop, end
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

        // Should have both i32.const 1 and i32.const 2
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
        // The fib pattern: (if (i32.eqz (local.get 0)) (then (return (i32.const 0))))
        let cf = compile_first_func(
            "(module (func (param i32) (result i32)
                (if (i32.eqz (local.get 0))
                    (then (return (i32.const 0))))
                (i32.const 1)))",
        );
        println!("{cf}");

        assert!(cf.ops.iter().any(|op| matches!(op, Op::Return)));
    }
}
