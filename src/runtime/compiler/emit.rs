//! The op sequence a function compiles to.
//!
//! Lowering hands the [`Emitter`] one op at a time and learns where each
//! landed. Branches to a block end are emitted before the end is known, so
//! their targets are patched once it is. Finishing meters the runs (see
//! [`crate::runtime::bytecode`]).

use super::{Result, invalid};
use crate::runtime::bytecode::Op;

/// A branch target written before its position was known: entry `entry` of
/// the op at `op`, in [`Op::targets`] order.
pub(super) struct Fixup {
    pub(super) op: usize,
    pub(super) entry: usize,
}

#[derive(Default)]
pub(super) struct Emitter {
    ops: Vec<Op>,
}

impl Emitter {
    /// Position of the next op emitted.
    pub(super) fn pos(&self) -> u32 {
        self.ops.len() as u32
    }

    /// Append an op and return its position.
    pub(super) fn emit(&mut self, op: Op) -> usize {
        self.ops.push(op);
        self.ops.len() - 1
    }

    /// Point the branch target `fixup` names at `target`.
    pub(super) fn patch(&mut self, fixup: &Fixup, target: u32) -> Result {
        let entry = (self.ops.get_mut(fixup.op)).and_then(|op| op.targets_mut().nth(fixup.entry));
        let entry = entry.ok_or_else(|| invalid(format!("op {} has no branch target {}", fixup.op, fixup.entry)))?;
        entry.pc = target;
        Ok(())
    }

    /// The ops, with every branch's run costs filled in, and the cost of the
    /// run at op 0.
    pub(super) fn finish(mut self) -> Result<(Vec<Op>, u32)> {
        let entry_cost = meter(&mut self.ops)?;
        Ok((self.ops, entry_cost))
    }
}

/// Fill in each branch's run costs and return the cost of the run at op 0.
///
/// Every op that follows an exit is either a branch target or never runs, and
/// the ops after a call belong to the caller's run, so charging runs where
/// control lands pays for each op executed exactly once.
fn meter(ops: &mut [Op]) -> Result<u32> {
    if u32::try_from(ops.len()).is_err() {
        return Err(invalid(format!("{} ops exceed the supported function size", ops.len())));
    }
    // Ops from each position to the next exit, inclusive, and zero until
    // known. The final `End` is an exit. Filling backwards knows every run
    // ahead of a branch, so only loop back edges wait for a second look.
    let mut run_costs = vec![0u32; ops.len()];
    let mut back_edges = Vec::new();
    let mut next = 0u32;
    for (pc, op) in ops.iter_mut().enumerate().rev() {
        // Only an op that exits can branch.
        let exits = op.exits();
        let rest = if exits { 0 } else { next };
        next = (rest.checked_add(op.weight())).ok_or_else(|| invalid(format!("run at {pc} costs too much")))?;
        run_costs[pc] = next;
        if exits && !fill_landing_costs(op, pc, &run_costs)? {
            back_edges.push(pc);
        }
    }
    for pc in back_edges {
        if !fill_landing_costs(&mut ops[pc], pc, &run_costs)? {
            return Err(invalid(format!("branch at {pc} lands in a run of no ops")));
        }
    }
    (run_costs.first().copied()).ok_or_else(|| invalid("function has no ops"))
}

/// Copy into the branch op at `pc` the cost of each run it can land in, and
/// report whether `run_costs` knew them all.
fn fill_landing_costs(op: &mut Op, pc: usize, run_costs: &[u32]) -> Result<bool> {
    let cost_at = |target: u32| {
        (run_costs.get(target as usize).copied())
            .ok_or_else(|| invalid(format!("branch target {target} is outside the function")))
    };
    let mut known = true;
    for target in op.targets_mut() {
        target.cost = cost_at(target.pc)?;
        known &= target.cost != 0;
    }
    if let Op::BrIf { fallthrough_cost, .. } = op {
        *fallthrough_cost = cost_at(pc as u32 + 1)?;
        known &= *fallthrough_cost != 0;
    }
    Ok(known)
}
