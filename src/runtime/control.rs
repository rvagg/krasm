//! WebAssembly label stack for control flow
//!
//! Implements the label stack as described in the WebAssembly specification.
//! See: <https://webassembly.github.io/spec/core/exec/runtime.html#labels>
//!
//! From the spec:
//! > "Labels carry an arity n and their associated branch target, which is expressed
//! > syntactically as an instruction sequence"
//!
//! The label stack tracks active control constructs (blocks, loops) and enables
//! structured control flow through branch instructions.

use crate::parser::module::ValueType;

/// Type of label construct
///
/// See: <https://webassembly.github.io/spec/core/syntax/instructions.html#control-instructions>
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LabelType {
    /// Block label — branches target the end of the block.
    /// Spec: `block blocktype instr* end`
    Block,
    /// Loop label — branches target the beginning of the loop.
    /// Spec: `loop blocktype instr* end`
    /// Note: "the label of a loop does not target the end, but the beginning of the loop"
    Loop,
}

/// A label on the label stack
///
/// From the WebAssembly spec (4.2.8 Labels):
/// > "Labels carry an arity n and their associated branch target"
///
/// Each label represents an active control construct and contains the information
/// needed for branch instructions to properly exit or continue execution.
#[derive(Debug, Clone)]
pub struct Label {
    /// Type of this label construct
    pub label_type: LabelType,
    /// Height of value stack when entering this label.
    /// Used to restore stack state on branch.
    pub stack_height: usize,
    /// Resolved parameter types (consumed on block entry)
    pub param_types: Vec<ValueType>,
    /// Resolved return types (produced on block exit)
    pub return_types: Vec<ValueType>,
}

/// The label stack for managing nested control structures
///
/// From the WebAssembly spec (4.2.8 Labels):
/// > "The label stack is empty when execution starts and gets extended with new labels
/// > when entering structured control instructions"
#[derive(Debug)]
pub struct LabelStack {
    labels: Vec<Label>,
}

impl Default for LabelStack {
    fn default() -> Self {
        Self::new()
    }
}

impl LabelStack {
    pub fn new() -> Self {
        LabelStack { labels: Vec::new() }
    }

    pub fn from_vec(labels: Vec<Label>) -> Self {
        LabelStack { labels }
    }

    /// Get the nth label from the top (0 = top)
    ///
    /// From the spec (4.4.8 br l):
    /// > "Let L be the l-th label appearing on the stack, starting from the top
    /// > and counting from zero"
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn get(&self, depth: u32) -> Option<&Label> {
        let len = self.labels.len();
        if depth as usize >= len {
            return None;
        }
        self.labels.get(len - 1 - depth as usize)
    }
}

impl Label {
    /// Get the arity (number of values kept) when branching to this label.
    ///
    /// From the spec (4.4.8 Control Instructions):
    /// > "Let n be the arity of L"
    ///
    /// For loops, arity is the parameter count (branch restarts the loop).
    /// For blocks, arity is the return count (branch exits the block).
    pub fn arity(&self) -> usize {
        if self.label_type == LabelType::Loop {
            self.param_types.len()
        } else {
            self.return_types.len()
        }
    }
}
