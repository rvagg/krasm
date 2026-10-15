//! Flat bytecode executor.
//!
//! Executes a `CompiledFunction` by walking its `Vec<Op>` with a program
//! counter. Branch targets are pre-resolved absolute indices, so there is no
//! context stack, no label stack, and no multi-level dispatch. Operands and
//! locals live on an untagged stack of 64-bit words (see [`words`]); values
//! take their tagged `Value` form only where they leave the executor.
//!
//! # Frames and registers
//!
//! All calls share one stack. Each function's frame holds its locals,
//! parameters first, then its operands:
//!
//! ```text
//!   ... caller's frame | params | declared locals | operands | free
//!                      ^ fp                       ^ base     ^ sp
//! ```
//!
//! Four registers, held in locals of the dispatch loop, track execution:
//!
//! - `pc`: the next op to run in the current function.
//! - `fp`: the frame's first word. Each local sits at a fixed offset from it,
//!   which the compiler emits as the local's index.
//! - `base`: the first operand word. Branch depths count from here.
//! - `sp`: the next free word. A push writes at `sp` and advances it; a pop
//!   steps back.
//!
//! A call copies nothing: the caller's arguments, on top of its operands,
//! become the callee's parameters, so the callee's `fp` is `sp` less its
//! parameter words. A return copies the results down to `fp` and leaves `sp`
//! just above them. `call_stack` holds each waiting caller's `pc`, `fp` and
//! `base`, as word indices, so the stack can grow and move.
//!
//! # Instruction budget
//!
//! The budget is charged per run, not per op (see [`super::bytecode`]): the
//! entry function and every local call charge their first run, and each
//! branch op charges the run it lands in. Returns and resumes charge nothing,
//! since the caller's run already covered the ops after its call.
//!
//! External calls — imported functions (directly or via call_indirect) and
//! foreign table funcrefs — suspend execution rather than dispatching
//! internally. The caller (ultimately the Store) drives this loop:
//!
//! ```text
//! invoke(funcs, idx, args)
//!    |
//!    +--> Complete(results)                        done
//!    |
//!    +--> NeedsExternalCall { func_addr, args }
//!            |
//!            v
//!         caller performs the call
//!            |
//!            v
//!         resume_with_results(results)
//!            |
//!            +--> Complete(results)                done
//!            |
//!            +--> NeedsExternalCall                repeat: call, resume
//! ```
//!
//! While suspended, the executor retains the stack, the internal call
//! stack, and the frame that issued the call (as `SuspendedState`).
//! Stack discipline at the suspension point: the call's arguments have
//! already been popped; resume pushes the results, exactly as if the call
//! had executed inline. Same-module calls are handled internally on a
//! shared operand stack and never suspend.
//!
//! A suspended instance can be invoked again before it resumes, when the
//! external callee calls back into one of its exports. Each invocation is an
//! activation whose frames and operands sit above those of the activations it
//! interrupted; suspensions form a stack, and resumption is last in, first out.
//!
//! Instruction implementations are delegated to the `ops` module where
//! possible, keeping the dispatch loop thin.

mod words;

use std::ptr::NonNull;

use super::bytecode::{CompiledFunction, Op, value_words};
use super::instance::SegmentState;
use super::memory::Memory;
use super::ops;
use super::store::{FuncAddr, GlobalAddr, MemoryAddr, Resources, TableAddr};
use super::value::Value;
use super::{ExecutionOutcome, ExternalCallRequest, RuntimeError};
use crate::parser::module::{Data, ExternalKind, FunctionType, Module, ValueType};
use words::{Word, pop, pop_value, push, push_value};

/// Execution context providing access to store resources and address maps.
/// Passed to `execute_flat` when the function uses globals or memory.
pub struct ExecContext<'a> {
    pub resources: &'a mut Resources,
    pub global_addrs: &'a [GlobalAddr],
    pub memory_addrs: &'a [MemoryAddr],
    pub table_addrs: &'a [TableAddr],
    /// Module function types, indexed by type index. Used for the argument
    /// and result shapes of imported calls, and for call_indirect signature
    /// checks.
    pub types: &'a [FunctionType],
    /// All module-level functions, imports first (matching the module index
    /// space). `Op::Call { func_idx }` values below `num_imported` suspend
    /// execution with `ExecutionOutcome::NeedsExternalCall`; values at or
    /// above index into the compiled `funcs` slice after subtracting
    /// `num_imported`. call_indirect maps a table funcref back to a callable
    /// through it.
    pub functions: &'a [FuncEntry],
    /// Number of imported functions (a prefix of `functions`).
    pub num_imported: usize,
    /// Per-instance segment state owned by the instance and shared with both
    /// engines.
    pub(crate) segments: &'a mut SegmentState,
    /// The module's data segments, for memory.init.
    pub(crate) data_segments: &'a [Data],
}

/// A module-level function as seen by the flat executor: where it lives in
/// the Store, and its type for signature checks and stack discipline.
#[derive(Debug, Clone, Copy)]
pub struct FuncEntry {
    /// Store address to dispatch external calls to, and the identity that
    /// table funcrefs are matched against.
    pub addr: FuncAddr,
    /// Index into `ExecContext::types`.
    pub type_idx: u32,
}

/// Build a `FuncEntry` per module-level function (imports first), pairing
/// each function's Store address with its type index.
pub(crate) fn build_func_entries(module: &Module, function_addresses: &[FuncAddr]) -> Vec<FuncEntry> {
    let mut type_indices = Vec::with_capacity(function_addresses.len());
    for imp in &module.imports.imports {
        if let ExternalKind::Function(type_idx) = imp.external_kind {
            type_indices.push(type_idx);
        }
    }
    for func in &module.functions.functions {
        type_indices.push(func.ftype_index);
    }

    type_indices
        .into_iter()
        .zip(function_addresses)
        .map(|(type_idx, &addr)| FuncEntry { addr, type_idx })
        .collect()
}

/// Most calls deep an activation may go. A saved frame is 32 bytes, so the
/// stack limit, not this backstop, normally decides how deep recursion runs.
const MAX_CALL_DEPTH: usize = 100_000;

/// Saved frames an idle executor keeps capacity for.
const RETAINED_CALL_FRAMES: usize = 1024;

/// Most words the stack may grow to unless set otherwise (128 MiB).
pub(crate) const DEFAULT_MAX_STACK_WORDS: usize = 1 << 24;

/// Words an idle stack keeps (1 MiB). A deeper spike is released when it
/// ends, so one deep recursion does not pin its memory for the instance's life.
const RETAINED_STACK_WORDS: usize = 1 << 17;

/// A function's place on the stack, saved while it waits on a callee.
#[derive(Debug, Clone, Copy)]
struct CallFrame {
    /// Index into the `funcs` slice.
    func_idx: usize,
    /// Program counter to resume at.
    pc: usize,
    /// Word index of the frame's first local.
    fp: usize,
    /// Word index of its operand base, above the locals. Branch depths count
    /// from here.
    base: usize,
}

impl ExecContext<'_> {
    /// Module-local memory 0, if the module has one.
    fn memory(&mut self) -> Option<&mut Memory> {
        let addr = self.memory_addrs.first()?;
        self.resources.memories.get_mut(addr.0)
    }

    /// The global at module-local `index`.
    fn global(&mut self, index: u32) -> Result<&mut Value, RuntimeError> {
        let addr = self
            .global_addrs
            .get(index as usize)
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(index))?;
        self.resources
            .globals
            .get_mut(addr.0)
            .map(|global| &mut global.value)
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(index))
    }

    /// Resolve module-local table index to the table instance.
    fn table(&self, index: u32) -> Result<&super::table::Table, RuntimeError> {
        let addr = self
            .table_addrs
            .get(index as usize)
            .ok_or(RuntimeError::TableIndexOutOfBounds(index))?;
        self.resources
            .tables
            .get(addr.0)
            .ok_or(RuntimeError::TableIndexOutOfBounds(index))
    }

    /// Resolve module-local table index to its store-wide index.
    fn table_addr(&self, index: u32) -> Result<usize, RuntimeError> {
        self.table_addrs
            .get(index as usize)
            .map(|addr| addr.0)
            .ok_or(RuntimeError::TableIndexOutOfBounds(index))
    }

    /// Resolve module-local table index to a mutable table instance.
    fn table_mut(&mut self, index: u32) -> Result<&mut super::table::Table, RuntimeError> {
        let addr = self
            .table_addrs
            .get(index as usize)
            .ok_or(RuntimeError::TableIndexOutOfBounds(index))?;
        self.resources
            .tables
            .get_mut(addr.0)
            .ok_or(RuntimeError::TableIndexOutOfBounds(index))
    }

    /// Look up a function type by type index.
    fn func_type(&self, type_idx: u32) -> Result<&FunctionType, RuntimeError> {
        self.types
            .get(type_idx as usize)
            .ok_or(RuntimeError::InvalidFunctionType)
    }

    /// Find a funcref in this module's function index space. The Store
    /// allocates a module's defined functions consecutively, so the offset
    /// from the first gives the index; imports are scanned.
    fn module_func_idx(&self, addr: FuncAddr) -> Option<usize> {
        let (imported, defined) = self.functions.split_at(self.num_imported);
        if let Some(first) = defined.first() {
            let offset = addr.0.wrapping_sub(first.addr.0);
            if defined.get(offset).is_some_and(|entry| entry.addr == addr) {
                return Some(self.num_imported + offset);
            }
        }
        imported.iter().position(|entry| entry.addr == addr)
    }
}

/// Require a mutable reference to the execution context, or trap.
macro_rules! require_ctx {
    ($ctx:expr) => {
        $ctx.as_mut()
            .ok_or_else(|| RuntimeError::Trap("operation requires execution context".to_string()))?
    };
}

/// One invocation of the executor, possibly nested inside a suspended one.
#[derive(Debug, Clone, Copy)]
struct Activation {
    /// Entry function, whose result count is collected on completion.
    entry_func_idx: usize,
    /// Call-stack length when the activation began: frames below it belong to
    /// interrupted activations, so the entry function returns at this depth.
    call_base: usize,
}

/// Execution state suspended across an external call.
///
/// Holds the frame that issued the call (with `pc` already past the Call op)
/// and the activation it belongs to.
struct SuspendedState {
    frame: CallFrame,
    activation: Activation,
    /// Result types declared by the suspended import; resume traps on a
    /// mismatch rather than corrupting the stack.
    expected_results: Box<[ValueType]>,
}

/// Grow `stack` to at least `len` words, or fail if that passes `max`.
fn reserve(stack: &mut Vec<Word>, len: usize, max: usize) -> Result<(), RuntimeError> {
    if len > stack.len() {
        if len > max {
            return Err(RuntimeError::CallStackOverflow);
        }
        stack.resize(len.max(stack.len() * 2).min(max), 0);
    }
    Ok(())
}

/// Suspend execution for an external call: pop the callee's arguments per
/// its type, record the suspension, and build the request for the Store.
/// `frame` is the issuing frame with `pc` already past the call instruction.
///
/// # Safety
/// The top words must hold arguments of `ftype`.
unsafe fn suspend_external_call(
    sp: &mut *mut Word,
    suspended: &mut Vec<SuspendedState>,
    frame: CallFrame,
    activation: Activation,
    func_addr: FuncAddr,
    ftype: &FunctionType,
    expected_type: Option<FunctionType>,
) -> ExecutionOutcome {
    let mut args: Vec<Value> = (ftype.parameters.iter().rev())
        .map(|value_type| unsafe { pop_value(sp, *value_type) })
        .collect();
    args.reverse();

    suspended.push(SuspendedState {
        frame,
        activation,
        expected_results: ftype.return_types.clone().into_boxed_slice(),
    });
    ExecutionOutcome::NeedsExternalCall(ExternalCallRequest {
        func_addr,
        args,
        expected_type,
    })
}

/// Flat bytecode executor with resumable external calls.
///
/// Owns the operand stack and call stack so execution can suspend when an
/// imported function is called: `invoke` returns
/// `ExecutionOutcome::NeedsExternalCall`, the Store performs the call, and
/// `resume_with_results` continues from the saved state. This mirrors the
/// structured `Executor`'s outcome-based dispatch, so the Store can drive
/// either engine with the same loop.
pub struct FlatExecutor {
    /// Raw 64-bit words, shared by all activations; the compiled code knows
    /// what each one holds.
    stack: Vec<Word>,
    /// Words in use while no activation is running.
    sp: usize,
    call_stack: Vec<CallFrame>,
    /// Suspended activations, innermost last.
    suspended: Vec<SuspendedState>,
    /// Most words `stack` may grow to.
    max_stack_words: usize,
    /// Budget remaining, charged per run (see [`Self::set_instruction_budget`]);
    /// `u64::MAX` when unlimited.
    instruction_budget: u64,
    #[cfg(feature = "instruction-profile")]
    instruction_counts: Option<Vec<Vec<u64>>>,
}

impl Default for FlatExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl FlatExecutor {
    pub fn new() -> Self {
        FlatExecutor {
            stack: Vec::new(),
            sp: 0,
            call_stack: Vec::new(),
            suspended: Vec::new(),
            max_stack_words: DEFAULT_MAX_STACK_WORDS,
            instruction_budget: u64::MAX,
            #[cfg(feature = "instruction-profile")]
            instruction_counts: None,
        }
    }

    #[cfg(feature = "instruction-profile")]
    pub(crate) fn enable_instruction_profile(&mut self, funcs: &[CompiledFunction]) {
        self.instruction_counts = Some(funcs.iter().map(|func| vec![0; func.ops.len()]).collect());
    }

    #[cfg(feature = "instruction-profile")]
    pub(crate) fn instruction_counts(&self) -> &[Vec<u64>] {
        self.instruction_counts.as_deref().unwrap_or_default()
    }

    /// Limit unfused bytecode operations, including function ends.
    ///
    /// Each straight-line run of ops is charged in full where control enters
    /// it, and traps as exhausted instead of starting when the budget cannot
    /// cover it. Calls do not end a run, so every waiting caller has paid for
    /// the rest of its run too. A call therefore completes exactly when the
    /// ops it executes fit the budget, but exhaustion comes early by the
    /// unexecuted ops those runs paid for, before any trap or side effect
    /// among them; they are also lost when a call traps. The remaining budget
    /// survives calls and suspension. `None` disables the limit.
    pub(crate) fn set_instruction_budget(&mut self, budget: Option<u64>) {
        self.instruction_budget = budget.unwrap_or(u64::MAX);
    }

    /// Limit the stack to `words`; a call that needs more traps as call
    /// stack exhaustion.
    pub(crate) fn set_max_stack_words(&mut self, words: usize) {
        self.max_stack_words = words;
    }

    /// Discard every activation after an error, or when the Store abandons
    /// suspended work, without changing the instruction budget.
    pub(crate) fn reset(&mut self) {
        self.sp = 0;
        self.call_stack.clear();
        self.suspended.clear();
        self.release_idle_stack();
    }

    /// With nothing running or suspended, give back stack and call-stack
    /// memory beyond what an idle executor keeps.
    fn release_idle_stack(&mut self) {
        if !self.suspended.is_empty() {
            return;
        }
        if self.stack.len() > RETAINED_STACK_WORDS {
            self.stack.truncate(RETAINED_STACK_WORDS);
            self.stack.shrink_to_fit();
        }
        self.call_stack.shrink_to(RETAINED_CALL_FRAMES);
    }

    /// Run until the activation completes or suspends, leaving the executor
    /// reusable after an error.
    fn run_activation(
        &mut self,
        funcs: &[CompiledFunction],
        frame: CallFrame,
        activation: Activation,
        ctx: Option<&mut ExecContext<'_>>,
    ) -> Result<ExecutionOutcome, RuntimeError> {
        let result = self.run(funcs, frame, activation, ctx);
        match result {
            Ok(ExecutionOutcome::Complete(_)) => self.release_idle_stack(),
            Ok(ExecutionOutcome::NeedsExternalCall(_)) => {}
            Err(_) => self.reset(),
        }
        result
    }

    /// Execute a compiled function with the given arguments.
    ///
    /// `funcs` is the slice of all compiled local functions for the module.
    /// `func_idx` is the index into `funcs` of the entry function.
    /// `ctx` provides access to globals, memory, and imported functions.
    /// Pass `None` for pure computation with no imports, globals, or memory.
    ///
    /// Returns `Complete` with the function's results, or `NeedsExternalCall`
    /// if an imported function must be dispatched; the caller performs that
    /// call and passes its results to [`FlatExecutor::resume_with_results`].
    pub(crate) fn invoke(
        &mut self,
        funcs: &[CompiledFunction],
        func_idx: usize,
        args: &[Value],
        ctx: Option<&mut ExecContext<'_>>,
    ) -> Result<ExecutionOutcome, RuntimeError> {
        let func = funcs
            .get(func_idx)
            .ok_or(RuntimeError::FunctionIndexOutOfBounds(func_idx as u32))?;

        // Without a suspended activation to return to, nothing below is live.
        if self.suspended.is_empty() {
            self.sp = 0;
            self.call_stack.clear();
        }
        let activation = Activation {
            entry_func_idx: func_idx,
            call_base: self.call_stack.len(),
        };
        let arg_words: u32 = args.iter().map(|arg| value_words(arg.typ())).sum();
        if arg_words != func.param_words {
            return Err(RuntimeError::Trap(format!(
                "entry function takes {} parameter words, given {arg_words}",
                func.param_words
            )));
        }
        let fp = self.sp;
        let base = fp + func.local_words as usize;
        reserve(
            &mut self.stack,
            base + func.max_stack_words as usize,
            self.max_stack_words,
        )?;
        // SAFETY: the frame was reserved above, and the arguments fill
        // exactly the parameter words.
        unsafe {
            let mut sp = self.stack.as_mut_ptr().add(fp);
            for arg in args {
                push_value(&mut sp, *arg);
            }
            sp.write_bytes(0, (func.local_words - func.param_words) as usize);
        }
        self.sp = base;
        let frame = CallFrame {
            func_idx,
            pc: 0,
            fp,
            base,
        };
        self.run_activation(funcs, frame, activation, ctx)
    }

    /// Resume execution after an external call completes, pushing its
    /// results onto the operand stack and continuing from the saved frame.
    pub(crate) fn resume_with_results(
        &mut self,
        funcs: &[CompiledFunction],
        results: Vec<Value>,
        ctx: Option<&mut ExecContext<'_>>,
    ) -> Result<ExecutionOutcome, RuntimeError> {
        let SuspendedState {
            frame,
            activation,
            expected_results,
        } = self
            .suspended
            .pop()
            .ok_or_else(|| RuntimeError::Trap("resume called without saved execution state".to_string()))?;

        // The stack trusts every value's type: a wrong one would corrupt
        // it, so check here even though the Store type-checks host results and
        // Wasm callees match a signature checked when linking or at the
        // indirect call.
        let types = results.iter().map(Value::typ);
        if !types.eq(expected_results.iter().copied()) {
            self.reset();
            return Err(RuntimeError::Trap(format!(
                "external call returned {results:?}, expected {expected_results:?}"
            )));
        }

        // SAFETY: the caller's frame reserved room for the call's results,
        // whose types match the callee's.
        unsafe {
            let mut sp = self.stack.as_mut_ptr().add(self.sp);
            for value in results {
                push_value(&mut sp, value);
            }
            self.sp = sp.offset_from(self.stack.as_ptr()) as usize;
        }

        self.run_activation(funcs, frame, activation, ctx)
    }

    /// The dispatch loop: walk bytecode from `frame` until the entry
    /// function completes or an imported call suspends execution.
    fn run(
        &mut self,
        funcs: &[CompiledFunction],
        frame: CallFrame,
        activation: Activation,
        mut ctx: Option<&mut ExecContext<'_>>,
    ) -> Result<ExecutionOutcome, RuntimeError> {
        // Destructure into disjoint borrows: the loop mutates the stack
        // and call stack independently.
        let FlatExecutor {
            stack,
            sp: saved_sp,
            call_stack,
            suspended,
            max_stack_words,
            instruction_budget,
            #[cfg(feature = "instruction-profile")]
            instruction_counts,
        } = self;

        // SAFETY: the memory stays in place until `run` returns: `memory.grow`
        // resizes its contents, not the instance, and the loop reaches tables
        // and globals without touching the Store's memories, so this pointer
        // is the only path to it.
        let memory: Option<NonNull<Memory>> = ctx.as_mut().and_then(|c| c.memory()).map(NonNull::from);

        // The registers (see the module docs): `pc` indexes the current
        // function's `ops`; `fp`, `base` and `sp`, below, point into `stack`.
        // Few values live across the loop, so each can stay in a register.
        //
        // SAFETY: `pc` stays inside `ops`: every function ends in an `End`,
        // the compiler checks each branch target, and a call resumes at the
        // op after it, which is never the last.
        let mut current_func_idx = frame.func_idx;
        let mut pc = frame.pc;
        let mut ops = funcs[current_func_idx].ops.as_ptr();

        // SAFETY: every frame on the stack was reserved when its function
        // was entered: its locals, and the most operand words the typed
        // compiler found it can use, lie inside `stack`. Validation and the
        // compiler guarantee each op finds the operands it pops, so `sp`
        // stays within the current frame. The pointers are re-derived from
        // `stack` after it grows.
        let (mut fp, mut base, mut sp) = unsafe {
            let start = stack.as_mut_ptr();
            (start.add(frame.fp), start.add(frame.base), start.add(*saved_sp))
        };

        // Pay for the run about to start, or trap without starting it.
        macro_rules! charge {
            ($cost:expr) => {{
                let cost = u64::from($cost);
                debug_assert!(cost > 0, "a run holds at least its exit");
                if *instruction_budget < cost {
                    return Err(RuntimeError::InstructionBudgetExhausted);
                }
                *instruction_budget -= cost;
            }};
        }

        // A fresh activation starts its entry function's first run; a resumed
        // one continues inside a run already paid for, past the call.
        if frame.pc == 0 {
            charge!(funcs[current_func_idx].entry_cost);
        }

        // Word index of a stack pointer.
        macro_rules! index {
            ($p:expr) => {
                $p.offset_from(stack.as_ptr()) as usize
            };
        }

        // Pure operations: pop the operands, push `$f(operands.., args..)`.
        // A `try_` form propagates the operation's trap.
        macro_rules! pure_op {
            (unary, $f:path) => { pure_op!(unary, |a| $f) };
            (binary, $f:path) => { pure_op!(binary, |a, b| $f) };
            (ternary, $f:path) => { pure_op!(ternary, |a, b, c| $f) };
            (try_unary, $f:path) => { pure_op!(try_unary, |a| $f) };
            (try_binary, $f:path) => { pure_op!(try_binary, |a, b| $f) };
            (unary, |$a:ident| $f:path $(, $arg:expr)*) => {{
                let $a = pop(&mut sp);
                push(&mut sp, $f($a $(, $arg)*));
                pc += 1;
            }};
            (binary, |$a:ident, $b:ident| $f:path $(, $arg:expr)*) => {{
                let $b = pop(&mut sp);
                let $a = pop(&mut sp);
                push(&mut sp, $f($a, $b $(, $arg)*));
                pc += 1;
            }};
            (ternary, |$a:ident, $b:ident, $c:ident| $f:path $(, $arg:expr)*) => {{
                let $c = pop(&mut sp);
                let $b = pop(&mut sp);
                let $a = pop(&mut sp);
                push(&mut sp, $f($a, $b, $c $(, $arg)*));
                pc += 1;
            }};
            (try_unary, |$a:ident| $f:path $(, $arg:expr)*) => {{
                let $a = pop(&mut sp);
                push(&mut sp, $f($a $(, $arg)*)?);
                pc += 1;
            }};
            (try_binary, |$a:ident, $b:ident| $f:path $(, $arg:expr)*) => {{
                let $b = pop(&mut sp);
                let $a = pop(&mut sp);
                push(&mut sp, $f($a, $b $(, $arg)*)?);
                pc += 1;
            }};
        }
        // The module's memory, or a trap if it has none.
        macro_rules! memory {
            () => {
                match memory {
                    Some(mut memory) => memory.as_mut(),
                    None => return Err(RuntimeError::MemoryError("no memory".to_string())),
                }
            };
        }

        // Pure operations on linear memory: as `pure_op`, with the memory as
        // first argument.
        macro_rules! mem_op {
            ($adapter:ident, |$($p:ident),*| $f:path $(, $arg:expr)*) => {{
                let memory = memory!();
                mem_op!(@ $adapter, memory, |$($p),*| $f $(, $arg)*)
            }};
            (@ unary, $m:ident, |$a:ident| $f:path $(, $arg:expr)*) => {{
                let $a = pop(&mut sp);
                push(&mut sp, $f($m, $a $(, $arg)*));
                pc += 1;
            }};
            (@ try_unary, $m:ident, |$a:ident| $f:path $(, $arg:expr)*) => {{
                let $a = pop(&mut sp);
                push(&mut sp, $f($m, $a $(, $arg)*)?);
                pc += 1;
            }};
            (@ try_binary, $m:ident, |$a:ident, $b:ident| $f:path $(, $arg:expr)*) => {{
                let $b = pop(&mut sp);
                let $a = pop(&mut sp);
                push(&mut sp, $f($m, $a, $b $(, $arg)*)?);
                pc += 1;
            }};
            (@ try_ternary, $m:ident, |$a:ident, $b:ident, $c:ident| $f:path $(, $arg:expr)*) => {{
                let $c = pop(&mut sp);
                let $b = pop(&mut sp);
                let $a = pop(&mut sp);
                push(&mut sp, $f($m, $a, $b, $c $(, $arg)*)?);
                pc += 1;
            }};
        }
        // Keep the top `arity` words, moved down to `depth` above the
        // operand base, and jump to `target`.
        macro_rules! branch {
            ($target:expr, $arity:expr, $depth:expr) => {{
                let dest = base.add($depth as usize);
                match $arity as usize {
                    0 => {}
                    1 => *dest = *sp.sub(1),
                    arity => dest.copy_from(sp.sub(arity), arity),
                }
                sp = dest.add($arity as usize);
                pc = $target as usize;
            }};
        }

        // Enter local function `$local`: save the caller, reserve the
        // callee's frame above its arguments, and zero its declared locals.
        macro_rules! enter_call {
            ($local:expr, $module_idx:expr) => {{
                let local_idx: usize = $local;
                let Some(callee) = funcs.get(local_idx) else {
                    return Err(RuntimeError::FunctionIndexOutOfBounds($module_idx));
                };
                if call_stack.len() >= MAX_CALL_DEPTH {
                    return Err(RuntimeError::CallStackOverflow);
                }
                call_stack.push(CallFrame {
                    func_idx: current_func_idx,
                    pc: pc + 1,
                    fp: index!(fp),
                    base: index!(base),
                });
                let callee_fp = index!(sp) - callee.param_words as usize;
                let callee_base = callee_fp + callee.local_words as usize;
                if callee_base + callee.max_stack_words as usize > stack.len() {
                    reserve(
                        stack,
                        callee_base + callee.max_stack_words as usize,
                        *max_stack_words,
                    )?;
                }
                charge!(callee.entry_cost);
                let start = stack.as_mut_ptr();
                fp = start.add(callee_fp);
                base = start.add(callee_base);
                fp.add(callee.param_words as usize)
                    .write_bytes(0, (callee.local_words - callee.param_words) as usize);
                sp = base;
                current_func_idx = local_idx;
                ops = callee.ops.as_ptr();
                pc = 0;
            }};
        }

        // Suspend for an external call, keeping the stack for the resume.
        macro_rules! suspend {
            ($addr:expr, $ftype:expr, $expected:expr) => {{
                let frame = CallFrame {
                    func_idx: current_func_idx,
                    pc: pc + 1,
                    fp: index!(fp),
                    base: index!(base),
                };
                // A copy keeps `sp` from escaping into the call, so it can
                // stay in a register throughout the loop.
                let mut top = sp;
                let outcome = suspend_external_call(&mut top, suspended, frame, activation, $addr, $ftype, $expected);
                *saved_sp = index!(top);
                return Ok(outcome);
            }};
        }

        // Move the results down to the frame's first word, then resume the
        // caller, or finish when the entry function returns.
        macro_rules! return_to_caller {
            () => {{
                let results = funcs[current_func_idx].result_words as usize;
                fp.copy_from(sp.sub(results), results);
                sp = fp.add(results);
                if call_stack.len() == activation.call_base {
                    break;
                }
                let Some(frame) = call_stack.pop() else {
                    break;
                };
                current_func_idx = frame.func_idx;
                ops = funcs[current_func_idx].ops.as_ptr();
                pc = frame.pc;
                let start = stack.as_mut_ptr();
                fp = start.add(frame.fp);
                base = start.add(frame.base);
            }};
        }

        loop {
            debug_assert!(
                base <= sp && sp <= base.wrapping_add(funcs[current_func_idx].max_stack_words as usize),
                "operand stack outside its frame"
            );

            debug_assert!(pc < funcs[current_func_idx].ops.len(), "pc outside its function");

            #[cfg(feature = "instruction-profile")]
            if let Some(counts) = instruction_counts {
                counts[current_func_idx][pc] += 1;
            }

            // SAFETY: see the `pc` and frame invariants above.
            unsafe {
                match &*ops.add(pc) {
                    // -- SIMD bitwise --
                    Op::V128Not => pure_op!(unary, ops::simd::v128_not),
                    Op::V128And => pure_op!(binary, ops::simd::v128_and),
                    Op::V128AndNot => pure_op!(binary, ops::simd::v128_andnot),
                    Op::V128Or => pure_op!(binary, ops::simd::v128_or),
                    Op::V128Xor => pure_op!(binary, ops::simd::v128_xor),
                    Op::V128Bitselect => pure_op!(ternary, ops::simd::v128_bitselect),
                    Op::V128AnyTrue => pure_op!(unary, ops::simd::v128_any_true),
                    Op::I8x16AllTrue => pure_op!(unary, ops::simd::i8x16_all_true),
                    Op::I8x16Bitmask => pure_op!(unary, ops::simd::i8x16_bitmask),
                    Op::I16x8AllTrue => pure_op!(unary, ops::simd::i16x8_all_true),
                    Op::I16x8Bitmask => pure_op!(unary, ops::simd::i16x8_bitmask),
                    Op::I32x4AllTrue => pure_op!(unary, ops::simd::i32x4_all_true),
                    Op::I32x4Bitmask => pure_op!(unary, ops::simd::i32x4_bitmask),
                    Op::I64x2AllTrue => pure_op!(unary, ops::simd::i64x2_all_true),
                    Op::I64x2Bitmask => pure_op!(unary, ops::simd::i64x2_bitmask),
                    // -- SIMD integer unary --
                    Op::I8x16Abs => pure_op!(unary, ops::simd::i8x16_abs),
                    Op::I8x16Neg => pure_op!(unary, ops::simd::i8x16_neg),
                    Op::I8x16Popcnt => pure_op!(unary, ops::simd::i8x16_popcnt),
                    Op::I16x8Abs => pure_op!(unary, ops::simd::i16x8_abs),
                    Op::I16x8Neg => pure_op!(unary, ops::simd::i16x8_neg),
                    Op::I32x4Abs => pure_op!(unary, ops::simd::i32x4_abs),
                    Op::I32x4Neg => pure_op!(unary, ops::simd::i32x4_neg),
                    Op::I64x2Abs => pure_op!(unary, ops::simd::i64x2_abs),
                    Op::I64x2Neg => pure_op!(unary, ops::simd::i64x2_neg),
                    // -- SIMD floating-point arithmetic --
                    Op::F32x4Abs => pure_op!(unary, ops::simd::f32x4_abs),
                    Op::F32x4Neg => pure_op!(unary, ops::simd::f32x4_neg),
                    Op::F32x4Sqrt => pure_op!(unary, ops::simd::f32x4_sqrt),
                    Op::F32x4Ceil => pure_op!(unary, ops::simd::f32x4_ceil),
                    Op::F32x4Floor => pure_op!(unary, ops::simd::f32x4_floor),
                    Op::F32x4Trunc => pure_op!(unary, ops::simd::f32x4_trunc),
                    Op::F32x4Nearest => pure_op!(unary, ops::simd::f32x4_nearest),
                    Op::F32x4Add => pure_op!(binary, ops::simd::f32x4_add),
                    Op::F32x4Sub => pure_op!(binary, ops::simd::f32x4_sub),
                    Op::F32x4Mul => pure_op!(binary, ops::simd::f32x4_mul),
                    Op::F32x4Div => pure_op!(binary, ops::simd::f32x4_div),
                    Op::F32x4Min => pure_op!(binary, ops::simd::f32x4_min),
                    Op::F32x4Max => pure_op!(binary, ops::simd::f32x4_max),
                    Op::F32x4PMin => pure_op!(binary, ops::simd::f32x4_pmin),
                    Op::F32x4PMax => pure_op!(binary, ops::simd::f32x4_pmax),
                    Op::F64x2Abs => pure_op!(unary, ops::simd::f64x2_abs),
                    Op::F64x2Neg => pure_op!(unary, ops::simd::f64x2_neg),
                    Op::F64x2Sqrt => pure_op!(unary, ops::simd::f64x2_sqrt),
                    Op::F64x2Ceil => pure_op!(unary, ops::simd::f64x2_ceil),
                    Op::F64x2Floor => pure_op!(unary, ops::simd::f64x2_floor),
                    Op::F64x2Trunc => pure_op!(unary, ops::simd::f64x2_trunc),
                    Op::F64x2Nearest => pure_op!(unary, ops::simd::f64x2_nearest),
                    Op::F64x2Add => pure_op!(binary, ops::simd::f64x2_add),
                    Op::F64x2Sub => pure_op!(binary, ops::simd::f64x2_sub),
                    Op::F64x2Mul => pure_op!(binary, ops::simd::f64x2_mul),
                    Op::F64x2Div => pure_op!(binary, ops::simd::f64x2_div),
                    Op::F64x2Min => pure_op!(binary, ops::simd::f64x2_min),
                    Op::F64x2Max => pure_op!(binary, ops::simd::f64x2_max),
                    Op::F64x2PMin => pure_op!(binary, ops::simd::f64x2_pmin),
                    Op::F64x2PMax => pure_op!(binary, ops::simd::f64x2_pmax),
                    // -- SIMD comparisons --
                    Op::I8x16Eq => pure_op!(binary, ops::simd::i8x16_eq),
                    Op::I8x16Ne => pure_op!(binary, ops::simd::i8x16_ne),
                    Op::I8x16LtS => pure_op!(binary, ops::simd::i8x16_lt_s),
                    Op::I8x16LtU => pure_op!(binary, ops::simd::i8x16_lt_u),
                    Op::I8x16GtS => pure_op!(binary, ops::simd::i8x16_gt_s),
                    Op::I8x16GtU => pure_op!(binary, ops::simd::i8x16_gt_u),
                    Op::I8x16LeS => pure_op!(binary, ops::simd::i8x16_le_s),
                    Op::I8x16LeU => pure_op!(binary, ops::simd::i8x16_le_u),
                    Op::I8x16GeS => pure_op!(binary, ops::simd::i8x16_ge_s),
                    Op::I8x16GeU => pure_op!(binary, ops::simd::i8x16_ge_u),
                    Op::I16x8Eq => pure_op!(binary, ops::simd::i16x8_eq),
                    Op::I16x8Ne => pure_op!(binary, ops::simd::i16x8_ne),
                    Op::I16x8LtS => pure_op!(binary, ops::simd::i16x8_lt_s),
                    Op::I16x8LtU => pure_op!(binary, ops::simd::i16x8_lt_u),
                    Op::I16x8GtS => pure_op!(binary, ops::simd::i16x8_gt_s),
                    Op::I16x8GtU => pure_op!(binary, ops::simd::i16x8_gt_u),
                    Op::I16x8LeS => pure_op!(binary, ops::simd::i16x8_le_s),
                    Op::I16x8LeU => pure_op!(binary, ops::simd::i16x8_le_u),
                    Op::I16x8GeS => pure_op!(binary, ops::simd::i16x8_ge_s),
                    Op::I16x8GeU => pure_op!(binary, ops::simd::i16x8_ge_u),
                    Op::I32x4Eq => pure_op!(binary, ops::simd::i32x4_eq),
                    Op::I32x4Ne => pure_op!(binary, ops::simd::i32x4_ne),
                    Op::I32x4LtS => pure_op!(binary, ops::simd::i32x4_lt_s),
                    Op::I32x4LtU => pure_op!(binary, ops::simd::i32x4_lt_u),
                    Op::I32x4GtS => pure_op!(binary, ops::simd::i32x4_gt_s),
                    Op::I32x4GtU => pure_op!(binary, ops::simd::i32x4_gt_u),
                    Op::I32x4LeS => pure_op!(binary, ops::simd::i32x4_le_s),
                    Op::I32x4LeU => pure_op!(binary, ops::simd::i32x4_le_u),
                    Op::I32x4GeS => pure_op!(binary, ops::simd::i32x4_ge_s),
                    Op::I32x4GeU => pure_op!(binary, ops::simd::i32x4_ge_u),
                    Op::I64x2Eq => pure_op!(binary, ops::simd::i64x2_eq),
                    Op::I64x2Ne => pure_op!(binary, ops::simd::i64x2_ne),
                    Op::I64x2LtS => pure_op!(binary, ops::simd::i64x2_lt_s),
                    Op::I64x2GtS => pure_op!(binary, ops::simd::i64x2_gt_s),
                    Op::I64x2LeS => pure_op!(binary, ops::simd::i64x2_le_s),
                    Op::I64x2GeS => pure_op!(binary, ops::simd::i64x2_ge_s),
                    Op::F32x4Eq => pure_op!(binary, ops::simd::f32x4_eq),
                    Op::F32x4Ne => pure_op!(binary, ops::simd::f32x4_ne),
                    Op::F32x4Lt => pure_op!(binary, ops::simd::f32x4_lt),
                    Op::F32x4Gt => pure_op!(binary, ops::simd::f32x4_gt),
                    Op::F32x4Le => pure_op!(binary, ops::simd::f32x4_le),
                    Op::F32x4Ge => pure_op!(binary, ops::simd::f32x4_ge),
                    Op::F64x2Eq => pure_op!(binary, ops::simd::f64x2_eq),
                    Op::F64x2Ne => pure_op!(binary, ops::simd::f64x2_ne),
                    Op::F64x2Lt => pure_op!(binary, ops::simd::f64x2_lt),
                    Op::F64x2Gt => pure_op!(binary, ops::simd::f64x2_gt),
                    Op::F64x2Le => pure_op!(binary, ops::simd::f64x2_le),
                    Op::F64x2Ge => pure_op!(binary, ops::simd::f64x2_ge),
                    // -- SIMD splat and permutation --
                    Op::I8x16Splat => pure_op!(unary, ops::simd::i8x16_splat),
                    Op::I16x8Splat => pure_op!(unary, ops::simd::i16x8_splat),
                    Op::I32x4Splat => pure_op!(unary, ops::simd::i32x4_splat),
                    Op::I64x2Splat => pure_op!(unary, ops::simd::i64x2_splat),
                    Op::F32x4Splat => pure_op!(unary, ops::simd::f32x4_splat),
                    Op::F64x2Splat => pure_op!(unary, ops::simd::f64x2_splat),
                    Op::I8x16Shuffle(mask) => pure_op!(binary, |a, b| ops::simd::i8x16_shuffle, mask),
                    Op::I8x16Swizzle => pure_op!(binary, ops::simd::i8x16_swizzle),
                    // -- SIMD shifts --
                    Op::I8x16Shl => pure_op!(binary, ops::simd::i8x16_shl),
                    Op::I8x16ShrS => pure_op!(binary, ops::simd::i8x16_shr_s),
                    Op::I8x16ShrU => pure_op!(binary, ops::simd::i8x16_shr_u),
                    Op::I16x8Shl => pure_op!(binary, ops::simd::i16x8_shl),
                    Op::I16x8ShrS => pure_op!(binary, ops::simd::i16x8_shr_s),
                    Op::I16x8ShrU => pure_op!(binary, ops::simd::i16x8_shr_u),
                    Op::I32x4Shl => pure_op!(binary, ops::simd::i32x4_shl),
                    Op::I32x4ShrS => pure_op!(binary, ops::simd::i32x4_shr_s),
                    Op::I32x4ShrU => pure_op!(binary, ops::simd::i32x4_shr_u),
                    Op::I64x2Shl => pure_op!(binary, ops::simd::i64x2_shl),
                    Op::I64x2ShrS => pure_op!(binary, ops::simd::i64x2_shr_s),
                    Op::I64x2ShrU => pure_op!(binary, ops::simd::i64x2_shr_u),
                    // -- SIMD wrapping arithmetic --
                    Op::I8x16Add => pure_op!(binary, ops::simd::i8x16_add),
                    Op::I8x16Sub => pure_op!(binary, ops::simd::i8x16_sub),
                    Op::I16x8Add => pure_op!(binary, ops::simd::i16x8_add),
                    Op::I16x8Sub => pure_op!(binary, ops::simd::i16x8_sub),
                    Op::I16x8Mul => pure_op!(binary, ops::simd::i16x8_mul),
                    Op::I32x4Add => pure_op!(binary, ops::simd::i32x4_add),
                    Op::I32x4Sub => pure_op!(binary, ops::simd::i32x4_sub),
                    Op::I32x4Mul => pure_op!(binary, ops::simd::i32x4_mul),
                    Op::I64x2Add => pure_op!(binary, ops::simd::i64x2_add),
                    Op::I64x2Sub => pure_op!(binary, ops::simd::i64x2_sub),
                    Op::I64x2Mul => pure_op!(binary, ops::simd::i64x2_mul),
                    // -- SIMD min/max --
                    Op::I8x16MinS => pure_op!(binary, ops::simd::i8x16_min_s),
                    Op::I8x16MinU => pure_op!(binary, ops::simd::i8x16_min_u),
                    Op::I8x16MaxS => pure_op!(binary, ops::simd::i8x16_max_s),
                    Op::I8x16MaxU => pure_op!(binary, ops::simd::i8x16_max_u),
                    Op::I16x8MinS => pure_op!(binary, ops::simd::i16x8_min_s),
                    Op::I16x8MinU => pure_op!(binary, ops::simd::i16x8_min_u),
                    Op::I16x8MaxS => pure_op!(binary, ops::simd::i16x8_max_s),
                    Op::I16x8MaxU => pure_op!(binary, ops::simd::i16x8_max_u),
                    Op::I32x4MinS => pure_op!(binary, ops::simd::i32x4_min_s),
                    Op::I32x4MinU => pure_op!(binary, ops::simd::i32x4_min_u),
                    Op::I32x4MaxS => pure_op!(binary, ops::simd::i32x4_max_s),
                    Op::I32x4MaxU => pure_op!(binary, ops::simd::i32x4_max_u),
                    // -- SIMD saturating arithmetic --
                    Op::I8x16AddSatS => pure_op!(binary, ops::simd::i8x16_add_sat_s),
                    Op::I8x16AddSatU => pure_op!(binary, ops::simd::i8x16_add_sat_u),
                    Op::I8x16SubSatS => pure_op!(binary, ops::simd::i8x16_sub_sat_s),
                    Op::I8x16SubSatU => pure_op!(binary, ops::simd::i8x16_sub_sat_u),
                    Op::I16x8AddSatS => pure_op!(binary, ops::simd::i16x8_add_sat_s),
                    Op::I16x8AddSatU => pure_op!(binary, ops::simd::i16x8_add_sat_u),
                    Op::I16x8SubSatS => pure_op!(binary, ops::simd::i16x8_sub_sat_s),
                    Op::I16x8SubSatU => pure_op!(binary, ops::simd::i16x8_sub_sat_u),
                    // -- SIMD rounded averages --
                    Op::I8x16AvgrU => pure_op!(binary, ops::simd::i8x16_avgr_u),
                    Op::I16x8AvgrU => pure_op!(binary, ops::simd::i16x8_avgr_u),
                    // -- SIMD saturating narrowing --
                    Op::I8x16NarrowI16x8S => pure_op!(binary, ops::simd::i8x16_narrow_i16x8_s),
                    Op::I8x16NarrowI16x8U => pure_op!(binary, ops::simd::i8x16_narrow_i16x8_u),
                    Op::I16x8NarrowI32x4S => pure_op!(binary, ops::simd::i16x8_narrow_i32x4_s),
                    Op::I16x8NarrowI32x4U => pure_op!(binary, ops::simd::i16x8_narrow_i32x4_u),
                    // -- SIMD widening extension and multiply --
                    Op::I16x8ExtendLowI8x16S => pure_op!(unary, ops::simd::i16x8_extend_low_i8x16_s),
                    Op::I16x8ExtendHighI8x16S => pure_op!(unary, ops::simd::i16x8_extend_high_i8x16_s),
                    Op::I16x8ExtendLowI8x16U => pure_op!(unary, ops::simd::i16x8_extend_low_i8x16_u),
                    Op::I16x8ExtendHighI8x16U => pure_op!(unary, ops::simd::i16x8_extend_high_i8x16_u),
                    Op::I16x8ExtMulLowI8x16S => pure_op!(binary, ops::simd::i16x8_extmul_low_i8x16_s),
                    Op::I16x8ExtMulHighI8x16S => pure_op!(binary, ops::simd::i16x8_extmul_high_i8x16_s),
                    Op::I16x8ExtMulLowI8x16U => pure_op!(binary, ops::simd::i16x8_extmul_low_i8x16_u),
                    Op::I16x8ExtMulHighI8x16U => pure_op!(binary, ops::simd::i16x8_extmul_high_i8x16_u),
                    Op::I32x4ExtendLowI16x8S => pure_op!(unary, ops::simd::i32x4_extend_low_i16x8_s),
                    Op::I32x4ExtendHighI16x8S => pure_op!(unary, ops::simd::i32x4_extend_high_i16x8_s),
                    Op::I32x4ExtendLowI16x8U => pure_op!(unary, ops::simd::i32x4_extend_low_i16x8_u),
                    Op::I32x4ExtendHighI16x8U => pure_op!(unary, ops::simd::i32x4_extend_high_i16x8_u),
                    Op::I32x4ExtMulLowI16x8S => pure_op!(binary, ops::simd::i32x4_extmul_low_i16x8_s),
                    Op::I32x4ExtMulHighI16x8S => pure_op!(binary, ops::simd::i32x4_extmul_high_i16x8_s),
                    Op::I32x4ExtMulLowI16x8U => pure_op!(binary, ops::simd::i32x4_extmul_low_i16x8_u),
                    Op::I32x4ExtMulHighI16x8U => pure_op!(binary, ops::simd::i32x4_extmul_high_i16x8_u),
                    Op::I64x2ExtendLowI32x4S => pure_op!(unary, ops::simd::i64x2_extend_low_i32x4_s),
                    Op::I64x2ExtendHighI32x4S => pure_op!(unary, ops::simd::i64x2_extend_high_i32x4_s),
                    Op::I64x2ExtendLowI32x4U => pure_op!(unary, ops::simd::i64x2_extend_low_i32x4_u),
                    Op::I64x2ExtendHighI32x4U => pure_op!(unary, ops::simd::i64x2_extend_high_i32x4_u),
                    Op::I64x2ExtMulLowI32x4S => pure_op!(binary, ops::simd::i64x2_extmul_low_i32x4_s),
                    Op::I64x2ExtMulHighI32x4S => pure_op!(binary, ops::simd::i64x2_extmul_high_i32x4_s),
                    Op::I64x2ExtMulLowI32x4U => pure_op!(binary, ops::simd::i64x2_extmul_low_i32x4_u),
                    Op::I64x2ExtMulHighI32x4U => pure_op!(binary, ops::simd::i64x2_extmul_high_i32x4_u),
                    // -- SIMD pairwise widening add --
                    Op::I16x8ExtAddPairwiseI8x16S => pure_op!(unary, ops::simd::i16x8_extadd_pairwise_i8x16_s),
                    Op::I16x8ExtAddPairwiseI8x16U => pure_op!(unary, ops::simd::i16x8_extadd_pairwise_i8x16_u),
                    Op::I32x4ExtAddPairwiseI16x8S => pure_op!(unary, ops::simd::i32x4_extadd_pairwise_i16x8_s),
                    Op::I32x4ExtAddPairwiseI16x8U => pure_op!(unary, ops::simd::i32x4_extadd_pairwise_i16x8_u),
                    // -- SIMD dot product and fixed-point multiply --
                    Op::I32x4DotI16x8S => pure_op!(binary, ops::simd::i32x4_dot_i16x8_s),
                    Op::I16x8Q15MulrSatS => pure_op!(binary, ops::simd::i16x8_q15mulr_sat_s),
                    // -- SIMD truncation and conversion --
                    Op::I32x4TruncSatF32x4S => pure_op!(unary, ops::simd::i32x4_trunc_sat_f32x4_s),
                    Op::I32x4TruncSatF32x4U => pure_op!(unary, ops::simd::i32x4_trunc_sat_f32x4_u),
                    Op::F32x4ConvertI32x4S => pure_op!(unary, ops::simd::f32x4_convert_i32x4_s),
                    Op::F32x4ConvertI32x4U => pure_op!(unary, ops::simd::f32x4_convert_i32x4_u),
                    Op::I32x4TruncSatF64x2SZero => pure_op!(unary, ops::simd::i32x4_trunc_sat_f64x2_s_zero),
                    Op::I32x4TruncSatF64x2UZero => pure_op!(unary, ops::simd::i32x4_trunc_sat_f64x2_u_zero),
                    Op::F64x2ConvertLowI32x4S => pure_op!(unary, ops::simd::f64x2_convert_low_i32x4_s),
                    Op::F64x2ConvertLowI32x4U => pure_op!(unary, ops::simd::f64x2_convert_low_i32x4_u),
                    Op::F32x4DemoteF64x2Zero => pure_op!(unary, ops::simd::f32x4_demote_f64x2_zero),
                    Op::F64x2PromoteLowF32x4 => pure_op!(unary, ops::simd::f64x2_promote_low_f32x4),
                    // -- SIMD lane extraction and replacement --
                    Op::I8x16ExtractLaneS(lane) => pure_op!(unary, |a| ops::simd::i8x16_extract_lane_s, *lane),
                    Op::I8x16ExtractLaneU(lane) => pure_op!(unary, |a| ops::simd::i8x16_extract_lane_u, *lane),
                    Op::I16x8ExtractLaneS(lane) => pure_op!(unary, |a| ops::simd::i16x8_extract_lane_s, *lane),
                    Op::I16x8ExtractLaneU(lane) => pure_op!(unary, |a| ops::simd::i16x8_extract_lane_u, *lane),
                    Op::I32x4ExtractLane(lane) => pure_op!(unary, |a| ops::simd::i32x4_extract_lane, *lane),
                    Op::I64x2ExtractLane(lane) => pure_op!(unary, |a| ops::simd::i64x2_extract_lane, *lane),
                    Op::F32x4ExtractLane(lane) => pure_op!(unary, |a| ops::simd::f32x4_extract_lane, *lane),
                    Op::F64x2ExtractLane(lane) => pure_op!(unary, |a| ops::simd::f64x2_extract_lane, *lane),
                    Op::I8x16ReplaceLane(lane) => pure_op!(binary, |a, b| ops::simd::i8x16_replace_lane, *lane),
                    Op::I16x8ReplaceLane(lane) => pure_op!(binary, |a, b| ops::simd::i16x8_replace_lane, *lane),
                    Op::I32x4ReplaceLane(lane) => pure_op!(binary, |a, b| ops::simd::i32x4_replace_lane, *lane),
                    Op::I64x2ReplaceLane(lane) => pure_op!(binary, |a, b| ops::simd::i64x2_replace_lane, *lane),
                    Op::F32x4ReplaceLane(lane) => pure_op!(binary, |a, b| ops::simd::f32x4_replace_lane, *lane),
                    Op::F64x2ReplaceLane(lane) => pure_op!(binary, |a, b| ops::simd::f64x2_replace_lane, *lane),
                    // -- Arithmetic --
                    Op::I32Add => pure_op!(binary, ops::numeric::i32_add),
                    Op::I32Sub => pure_op!(binary, ops::numeric::i32_sub),
                    Op::I32Mul => pure_op!(binary, ops::numeric::i32_mul),
                    Op::I32DivS => pure_op!(try_binary, ops::numeric::i32_div_s),
                    Op::I32DivU => pure_op!(try_binary, ops::numeric::i32_div_u),
                    Op::I32RemS => pure_op!(try_binary, ops::numeric::i32_rem_s),
                    Op::I32RemU => pure_op!(try_binary, ops::numeric::i32_rem_u),
                    Op::I32Clz => pure_op!(unary, ops::numeric::i32_clz),
                    Op::I32Ctz => pure_op!(unary, ops::numeric::i32_ctz),
                    Op::I32Popcnt => pure_op!(unary, ops::numeric::i32_popcnt),
                    Op::I32And => pure_op!(binary, ops::bitwise::i32_and),
                    Op::I32Or => pure_op!(binary, ops::bitwise::i32_or),
                    Op::I32Xor => pure_op!(binary, ops::bitwise::i32_xor),
                    Op::I32Shl => pure_op!(binary, ops::bitwise::i32_shl),
                    Op::I32ShrS => pure_op!(binary, ops::bitwise::i32_shr_s),
                    Op::I32ShrU => pure_op!(binary, ops::bitwise::i32_shr_u),
                    Op::I32Rotl => pure_op!(binary, ops::bitwise::i32_rotl),
                    Op::I32Rotr => pure_op!(binary, ops::bitwise::i32_rotr),
                    Op::I32Extend8S => pure_op!(unary, ops::conversion::i32_extend8_s),
                    Op::I32Extend16S => pure_op!(unary, ops::conversion::i32_extend16_s),
                    Op::I64Add => pure_op!(binary, ops::numeric::i64_add),
                    Op::I64Sub => pure_op!(binary, ops::numeric::i64_sub),
                    Op::I64Mul => pure_op!(binary, ops::numeric::i64_mul),
                    Op::I64DivS => pure_op!(try_binary, ops::numeric::i64_div_s),
                    Op::I64DivU => pure_op!(try_binary, ops::numeric::i64_div_u),
                    Op::I64RemS => pure_op!(try_binary, ops::numeric::i64_rem_s),
                    Op::I64RemU => pure_op!(try_binary, ops::numeric::i64_rem_u),
                    Op::I64Clz => pure_op!(unary, ops::numeric::i64_clz),
                    Op::I64Ctz => pure_op!(unary, ops::numeric::i64_ctz),
                    Op::I64Popcnt => pure_op!(unary, ops::numeric::i64_popcnt),
                    Op::I64And => pure_op!(binary, ops::bitwise::i64_and),
                    Op::I64Or => pure_op!(binary, ops::bitwise::i64_or),
                    Op::I64Xor => pure_op!(binary, ops::bitwise::i64_xor),
                    Op::I64Shl => pure_op!(binary, ops::bitwise::i64_shl),
                    Op::I64ShrS => pure_op!(binary, ops::bitwise::i64_shr_s),
                    Op::I64ShrU => pure_op!(binary, ops::bitwise::i64_shr_u),
                    Op::I64Rotl => pure_op!(binary, ops::bitwise::i64_rotl),
                    Op::I64Rotr => pure_op!(binary, ops::bitwise::i64_rotr),
                    Op::I64Eqz => pure_op!(unary, ops::comparison::i64_eqz),
                    Op::I64Eq => pure_op!(binary, ops::comparison::i64_eq),
                    Op::I64Ne => pure_op!(binary, ops::comparison::i64_ne),
                    Op::I64LtS => pure_op!(binary, ops::comparison::i64_lt_s),
                    Op::I64LtU => pure_op!(binary, ops::comparison::i64_lt_u),
                    Op::I64GtS => pure_op!(binary, ops::comparison::i64_gt_s),
                    Op::I64GtU => pure_op!(binary, ops::comparison::i64_gt_u),
                    Op::I64LeS => pure_op!(binary, ops::comparison::i64_le_s),
                    Op::I64LeU => pure_op!(binary, ops::comparison::i64_le_u),
                    Op::I64GeS => pure_op!(binary, ops::comparison::i64_ge_s),
                    Op::I64GeU => pure_op!(binary, ops::comparison::i64_ge_u),
                    Op::I64Extend8S => pure_op!(unary, ops::conversion::i64_extend8_s),
                    Op::I64Extend16S => pure_op!(unary, ops::conversion::i64_extend16_s),
                    Op::I64Extend32S => pure_op!(unary, ops::conversion::i64_extend32_s),
                    Op::I64ExtendI32S => pure_op!(unary, ops::conversion::i64_extend_i32_s),
                    Op::I64ExtendI32U => pure_op!(unary, ops::conversion::i64_extend_i32_u),
                    Op::I32WrapI64 => pure_op!(unary, ops::conversion::i32_wrap_i64),
                    Op::F32Add => pure_op!(binary, ops::numeric::f32_add),
                    Op::F32Sub => pure_op!(binary, ops::numeric::f32_sub),
                    Op::F32Mul => pure_op!(binary, ops::numeric::f32_mul),
                    Op::F32Div => pure_op!(binary, ops::numeric::f32_div),
                    Op::F32Min => pure_op!(binary, ops::numeric::f32_min),
                    Op::F32Max => pure_op!(binary, ops::numeric::f32_max),
                    Op::F32Copysign => pure_op!(binary, ops::numeric::f32_copysign),
                    Op::F64Add => pure_op!(binary, ops::numeric::f64_add),
                    Op::F64Sub => pure_op!(binary, ops::numeric::f64_sub),
                    Op::F64Mul => pure_op!(binary, ops::numeric::f64_mul),
                    Op::F64Div => pure_op!(binary, ops::numeric::f64_div),
                    Op::F64Min => pure_op!(binary, ops::numeric::f64_min),
                    Op::F64Max => pure_op!(binary, ops::numeric::f64_max),
                    Op::F64Copysign => pure_op!(binary, ops::numeric::f64_copysign),
                    Op::F32Abs => pure_op!(unary, ops::numeric::f32_abs),
                    Op::F32Neg => pure_op!(unary, ops::numeric::f32_neg),
                    Op::F32Sqrt => pure_op!(unary, ops::numeric::f32_sqrt),
                    Op::F32Ceil => pure_op!(unary, ops::numeric::f32_ceil),
                    Op::F32Floor => pure_op!(unary, ops::numeric::f32_floor),
                    Op::F32Trunc => pure_op!(unary, ops::numeric::f32_trunc),
                    Op::F32Nearest => pure_op!(unary, ops::numeric::f32_nearest),
                    Op::F64Abs => pure_op!(unary, ops::numeric::f64_abs),
                    Op::F64Neg => pure_op!(unary, ops::numeric::f64_neg),
                    Op::F64Sqrt => pure_op!(unary, ops::numeric::f64_sqrt),
                    Op::F64Ceil => pure_op!(unary, ops::numeric::f64_ceil),
                    Op::F64Floor => pure_op!(unary, ops::numeric::f64_floor),
                    Op::F64Trunc => pure_op!(unary, ops::numeric::f64_trunc),
                    Op::F64Nearest => pure_op!(unary, ops::numeric::f64_nearest),
                    Op::F32Eq => pure_op!(binary, ops::comparison::f32_eq),
                    Op::F32Ne => pure_op!(binary, ops::comparison::f32_ne),
                    Op::F32Lt => pure_op!(binary, ops::comparison::f32_lt),
                    Op::F32Gt => pure_op!(binary, ops::comparison::f32_gt),
                    Op::F32Le => pure_op!(binary, ops::comparison::f32_le),
                    Op::F32Ge => pure_op!(binary, ops::comparison::f32_ge),
                    Op::F64Eq => pure_op!(binary, ops::comparison::f64_eq),
                    Op::F64Ne => pure_op!(binary, ops::comparison::f64_ne),
                    Op::F64Lt => pure_op!(binary, ops::comparison::f64_lt),
                    Op::F64Gt => pure_op!(binary, ops::comparison::f64_gt),
                    Op::F64Le => pure_op!(binary, ops::comparison::f64_le),
                    Op::F64Ge => pure_op!(binary, ops::comparison::f64_ge),
                    Op::F32ConvertI32S => pure_op!(unary, ops::conversion::f32_convert_i32_s),
                    Op::F32ConvertI32U => pure_op!(unary, ops::conversion::f32_convert_i32_u),
                    Op::F32ConvertI64S => pure_op!(unary, ops::conversion::f32_convert_i64_s),
                    Op::F32ConvertI64U => pure_op!(unary, ops::conversion::f32_convert_i64_u),
                    Op::F64ConvertI32S => pure_op!(unary, ops::conversion::f64_convert_i32_s),
                    Op::F64ConvertI32U => pure_op!(unary, ops::conversion::f64_convert_i32_u),
                    Op::F64ConvertI64S => pure_op!(unary, ops::conversion::f64_convert_i64_s),
                    Op::F64ConvertI64U => pure_op!(unary, ops::conversion::f64_convert_i64_u),
                    Op::F32DemoteF64 => pure_op!(unary, ops::conversion::f32_demote_f64),
                    Op::F64PromoteF32 => pure_op!(unary, ops::conversion::f64_promote_f32),
                    Op::I32TruncF32S => pure_op!(try_unary, ops::conversion::i32_trunc_f32_s),
                    Op::I32TruncF32U => pure_op!(try_unary, ops::conversion::i32_trunc_f32_u),
                    Op::I32TruncF64S => pure_op!(try_unary, ops::conversion::i32_trunc_f64_s),
                    Op::I32TruncF64U => pure_op!(try_unary, ops::conversion::i32_trunc_f64_u),
                    Op::I64TruncF32S => pure_op!(try_unary, ops::conversion::i64_trunc_f32_s),
                    Op::I64TruncF32U => pure_op!(try_unary, ops::conversion::i64_trunc_f32_u),
                    Op::I64TruncF64S => pure_op!(try_unary, ops::conversion::i64_trunc_f64_s),
                    Op::I64TruncF64U => pure_op!(try_unary, ops::conversion::i64_trunc_f64_u),
                    Op::I32TruncSatF32S => pure_op!(unary, ops::conversion::i32_trunc_sat_f32_s),
                    Op::I32TruncSatF32U => pure_op!(unary, ops::conversion::i32_trunc_sat_f32_u),
                    Op::I32TruncSatF64S => pure_op!(unary, ops::conversion::i32_trunc_sat_f64_s),
                    Op::I32TruncSatF64U => pure_op!(unary, ops::conversion::i32_trunc_sat_f64_u),
                    Op::I64TruncSatF32S => pure_op!(unary, ops::conversion::i64_trunc_sat_f32_s),
                    Op::I64TruncSatF32U => pure_op!(unary, ops::conversion::i64_trunc_sat_f32_u),
                    Op::I64TruncSatF64S => pure_op!(unary, ops::conversion::i64_trunc_sat_f64_s),
                    Op::I64TruncSatF64U => pure_op!(unary, ops::conversion::i64_trunc_sat_f64_u),
                    Op::I32ReinterpretF32 => pure_op!(unary, ops::conversion::i32_reinterpret_f32),
                    Op::I64ReinterpretF64 => pure_op!(unary, ops::conversion::i64_reinterpret_f64),
                    Op::F32ReinterpretI32 => pure_op!(unary, ops::conversion::f32_reinterpret_i32),
                    Op::F64ReinterpretI64 => pure_op!(unary, ops::conversion::f64_reinterpret_i64),
                    // -- Comparison --
                    Op::I32Eqz => pure_op!(unary, ops::comparison::i32_eqz),
                    Op::I32Eq => pure_op!(binary, ops::comparison::i32_eq),
                    Op::I32Ne => pure_op!(binary, ops::comparison::i32_ne),
                    Op::I32LtS => pure_op!(binary, ops::comparison::i32_lt_s),
                    Op::I32LtU => pure_op!(binary, ops::comparison::i32_lt_u),
                    Op::I32GtS => pure_op!(binary, ops::comparison::i32_gt_s),
                    Op::I32GtU => pure_op!(binary, ops::comparison::i32_gt_u),
                    Op::I32LeS => pure_op!(binary, ops::comparison::i32_le_s),
                    Op::I32LeU => pure_op!(binary, ops::comparison::i32_le_u),
                    Op::I32GeS => pure_op!(binary, ops::comparison::i32_ge_s),
                    Op::I32GeU => pure_op!(binary, ops::comparison::i32_ge_u),
                    // -- Memory --
                    Op::I32Load(m) => mem_op!(try_unary, |addr| ops::memory::i32_load, m),
                    Op::I32Load8S(m) => mem_op!(try_unary, |addr| ops::memory::i32_load8_s, m),
                    Op::I32Load8U(m) => mem_op!(try_unary, |addr| ops::memory::i32_load8_u, m),
                    Op::I32Load16S(m) => mem_op!(try_unary, |addr| ops::memory::i32_load16_s, m),
                    Op::I32Load16U(m) => mem_op!(try_unary, |addr| ops::memory::i32_load16_u, m),
                    Op::I64Load(m) => mem_op!(try_unary, |addr| ops::memory::i64_load, m),
                    Op::I64Load8S(m) => mem_op!(try_unary, |addr| ops::memory::i64_load8_s, m),
                    Op::I64Load8U(m) => mem_op!(try_unary, |addr| ops::memory::i64_load8_u, m),
                    Op::I64Load16S(m) => mem_op!(try_unary, |addr| ops::memory::i64_load16_s, m),
                    Op::I64Load16U(m) => mem_op!(try_unary, |addr| ops::memory::i64_load16_u, m),
                    Op::I64Load32S(m) => mem_op!(try_unary, |addr| ops::memory::i64_load32_s, m),
                    Op::I64Load32U(m) => mem_op!(try_unary, |addr| ops::memory::i64_load32_u, m),
                    Op::F32Load(m) => mem_op!(try_unary, |addr| ops::memory::f32_load, m),
                    Op::F64Load(m) => mem_op!(try_unary, |addr| ops::memory::f64_load, m),
                    Op::V128Load(m) => mem_op!(try_unary, |addr| ops::simd::v128_load, m),
                    Op::V128Load8x8S(m) => mem_op!(try_unary, |addr| ops::simd::v128_load8x8_s, m),
                    Op::V128Load8x8U(m) => mem_op!(try_unary, |addr| ops::simd::v128_load8x8_u, m),
                    Op::V128Load16x4S(m) => mem_op!(try_unary, |addr| ops::simd::v128_load16x4_s, m),
                    Op::V128Load16x4U(m) => mem_op!(try_unary, |addr| ops::simd::v128_load16x4_u, m),
                    Op::V128Load32x2S(m) => mem_op!(try_unary, |addr| ops::simd::v128_load32x2_s, m),
                    Op::V128Load32x2U(m) => mem_op!(try_unary, |addr| ops::simd::v128_load32x2_u, m),
                    Op::V128Load8Splat(m) => mem_op!(try_unary, |addr| ops::simd::v128_load8_splat, m),
                    Op::V128Load16Splat(m) => mem_op!(try_unary, |addr| ops::simd::v128_load16_splat, m),
                    Op::V128Load32Splat(m) => mem_op!(try_unary, |addr| ops::simd::v128_load32_splat, m),
                    Op::V128Load64Splat(m) => mem_op!(try_unary, |addr| ops::simd::v128_load64_splat, m),
                    Op::V128Load32Zero(m) => mem_op!(try_unary, |addr| ops::simd::v128_load32_zero, m),
                    Op::V128Load64Zero(m) => mem_op!(try_unary, |addr| ops::simd::v128_load64_zero, m),
                    Op::V128Load8Lane { memarg, lane } => {
                        mem_op!(try_binary, |addr, v| ops::simd::v128_load8_lane, memarg, *lane)
                    }
                    Op::V128Load16Lane { memarg, lane } => {
                        mem_op!(try_binary, |addr, v| ops::simd::v128_load16_lane, memarg, *lane)
                    }
                    Op::V128Load32Lane { memarg, lane } => {
                        mem_op!(try_binary, |addr, v| ops::simd::v128_load32_lane, memarg, *lane)
                    }
                    Op::V128Load64Lane { memarg, lane } => {
                        mem_op!(try_binary, |addr, v| ops::simd::v128_load64_lane, memarg, *lane)
                    }
                    Op::I32Store(m) => mem_op!(try_binary, |addr, value| ops::memory::i32_store, m),
                    Op::I32Store8(m) => mem_op!(try_binary, |addr, value| ops::memory::i32_store8, m),
                    Op::I32Store16(m) => mem_op!(try_binary, |addr, value| ops::memory::i32_store16, m),
                    Op::I64Store(m) => mem_op!(try_binary, |addr, value| ops::memory::i64_store, m),
                    Op::I64Store8(m) => mem_op!(try_binary, |addr, value| ops::memory::i64_store8, m),
                    Op::I64Store16(m) => mem_op!(try_binary, |addr, value| ops::memory::i64_store16, m),
                    Op::I64Store32(m) => mem_op!(try_binary, |addr, value| ops::memory::i64_store32, m),
                    Op::F32Store(m) => mem_op!(try_binary, |addr, value| ops::memory::f32_store, m),
                    Op::F64Store(m) => mem_op!(try_binary, |addr, value| ops::memory::f64_store, m),
                    Op::V128Store(m) => mem_op!(try_binary, |addr, value| ops::simd::v128_store, m),
                    Op::V128Store8Lane { memarg, lane } => {
                        mem_op!(try_binary, |addr, v| ops::simd::v128_store8_lane, memarg, *lane)
                    }
                    Op::V128Store16Lane { memarg, lane } => {
                        mem_op!(try_binary, |addr, v| ops::simd::v128_store16_lane, memarg, *lane)
                    }
                    Op::V128Store32Lane { memarg, lane } => {
                        mem_op!(try_binary, |addr, v| ops::simd::v128_store32_lane, memarg, *lane)
                    }
                    Op::V128Store64Lane { memarg, lane } => {
                        mem_op!(try_binary, |addr, v| ops::simd::v128_store64_lane, memarg, *lane)
                    }
                    Op::MemoryGrow => mem_op!(unary, |delta| ops::memory::memory_grow),
                    Op::MemoryCopy => mem_op!(try_ternary, |dest, src, len| ops::memory::memory_copy),
                    Op::MemoryFill => mem_op!(try_ternary, |dest, value, len| ops::memory::memory_fill),
                    // -- Constants --
                    Op::I32Const(v) => {
                        push(&mut sp, *v);
                        pc += 1;
                    }
                    Op::I64Const(v) => {
                        push(&mut sp, *v);
                        pc += 1;
                    }
                    Op::F32Const(v) => {
                        push(&mut sp, *v);
                        pc += 1;
                    }
                    Op::F64Const(v) => {
                        push(&mut sp, *v);
                        pc += 1;
                    }
                    Op::V128Const(v) => {
                        push(&mut sp, *v);
                        pc += 1;
                    }

                    // -- Locals, read and written in place in the frame --
                    Op::LocalGet { index } => {
                        *sp = *fp.add(*index as usize);
                        sp = sp.add(1);
                        pc += 1;
                    }
                    Op::LocalSet { index } => {
                        sp = sp.sub(1);
                        *fp.add(*index as usize) = *sp;
                        pc += 1;
                    }
                    Op::LocalTee { index } => {
                        *fp.add(*index as usize) = *sp.sub(1);
                        pc += 1;
                    }
                    Op::LocalGetV128 { index } => {
                        sp.copy_from_nonoverlapping(fp.add(*index as usize), 2);
                        sp = sp.add(2);
                        pc += 1;
                    }
                    Op::LocalSetV128 { index } => {
                        sp = sp.sub(2);
                        fp.add(*index as usize).copy_from_nonoverlapping(sp, 2);
                        pc += 1;
                    }
                    Op::LocalTeeV128 { index } => {
                        fp.add(*index as usize).copy_from_nonoverlapping(sp.sub(2), 2);
                        pc += 1;
                    }

                    // -- Globals, stored tagged --
                    Op::GlobalGet { index } => {
                        let value = *require_ctx!(ctx).global(*index)?;
                        push_value(&mut sp, value);
                        pc += 1;
                    }
                    Op::GlobalSet { index } => {
                        let global = require_ctx!(ctx).global(*index)?;
                        *global = pop_value(&mut sp, global.typ());
                        pc += 1;
                    }

                    // -- Parametric --
                    Op::Drop => {
                        sp = sp.sub(1);
                        pc += 1;
                    }
                    Op::DropV128 => {
                        sp = sp.sub(2);
                        pc += 1;
                    }
                    Op::Select => {
                        let condition: i32 = pop(&mut sp);
                        sp = sp.sub(1);
                        if condition == 0 {
                            *sp.sub(1) = *sp;
                        }
                        pc += 1;
                    }
                    Op::SelectV128 => {
                        let condition: i32 = pop(&mut sp);
                        sp = sp.sub(2);
                        if condition == 0 {
                            sp.sub(2).copy_from_nonoverlapping(sp, 2);
                        }
                        pc += 1;
                    }

                    Op::MemorySize => {
                        push(&mut sp, ops::memory::memory_size(memory!()));
                        pc += 1;
                    }

                    // -- Control flow --
                    Op::Br {
                        target,
                        arity,
                        stack_depth,
                        cost,
                    } => {
                        charge!(*cost);
                        branch!(*target, *arity, *stack_depth);
                    }
                    Op::BrIf {
                        target,
                        arity,
                        stack_depth,
                        cost,
                        fallthrough_cost,
                    } => {
                        let condition: i32 = pop(&mut sp);
                        if condition != 0 {
                            charge!(*cost);
                            branch!(*target, *arity, *stack_depth);
                        } else {
                            charge!(*fallthrough_cost);
                            pc += 1;
                        }
                    }
                    Op::BrTable { targets, default } => {
                        let index: i32 = pop(&mut sp);
                        let target = targets.get(index as u32 as usize).unwrap_or(default);
                        charge!(target.cost);
                        branch!(target.pc, target.arity, target.stack_depth);
                    }
                    Op::Call { func_idx } => {
                        let num_imported = ctx.as_ref().map(|c| c.num_imported).unwrap_or(0);

                        // Imported function: suspend and hand the call to the
                        // Store. Execution resumes in `resume_with_results` at
                        // the instruction after the Call.
                        if (*func_idx as usize) < num_imported {
                            let c = require_ctx!(ctx);
                            let entry = c
                                .functions
                                .get(*func_idx as usize)
                                .copied()
                                .ok_or(RuntimeError::FunctionIndexOutOfBounds(*func_idx))?;
                            suspend!(entry.addr, c.func_type(entry.type_idx)?, None);
                        }
                        enter_call!(*func_idx as usize - num_imported, *func_idx);
                    }
                    Op::CallIndirect { type_idx, table_idx } => {
                        let c = require_ctx!(ctx);
                        let elem_idx = pop::<i32>(&mut sp) as u32;

                        // Spec: an out-of-bounds element index is "undefined
                        // element", not "out of bounds table access".
                        let func_ref = c.table(*table_idx)?.get(elem_idx).map_err(|e| match e {
                            RuntimeError::TableIndexOutOfBounds(_) => RuntimeError::UndefinedElement(elem_idx),
                            other => other,
                        })?;
                        let func_addr = match func_ref {
                            Value::FuncRef(Some(addr)) => addr,
                            Value::FuncRef(None) => return Err(RuntimeError::UndefinedElement(elem_idx)),
                            other => {
                                return Err(RuntimeError::TypeMismatch {
                                    expected: "funcref".to_string(),
                                    actual: format!("{:?}", other.typ()),
                                });
                            }
                        };
                        let expected = c.func_type(*type_idx)?;

                        match c.module_func_idx(func_addr) {
                            Some(module_idx) => {
                                // The funcref belongs to this module: the
                                // signature check happens here, before any
                                // arguments are consumed. Equal indices are
                                // equal types; different ones may still match.
                                let actual_idx = c.functions[module_idx].type_idx;
                                let actual = c.func_type(actual_idx)?;
                                if actual_idx != *type_idx && expected != actual {
                                    return Err(RuntimeError::IndirectCallTypeMismatch {
                                        expected: format!("{expected:?}"),
                                        actual: format!("{actual:?}"),
                                    });
                                }
                                if module_idx < c.num_imported {
                                    suspend!(func_addr, expected, None);
                                }
                                // Local function: internal call, as Op::Call.
                                enter_call!(module_idx - c.num_imported, module_idx as u32);
                            }
                            None => {
                                // Foreign funcref: a function from another instance
                                // or the host, placed in a shared table. Its type is
                                // unknown here; the Store checks it before dispatch.
                                let required = Some(expected.clone());
                                suspend!(func_addr, expected, required);
                            }
                        }
                    }
                    // `return` can leave operands beneath its results; validation
                    // guarantees the function's final `end` leaves exactly them.
                    Op::Return | Op::End => return_to_caller!(),
                    Op::Nop => {
                        pc += 1;
                    }

                    // -- References: 0 is null --
                    Op::RefNull(_) => {
                        push(&mut sp, 0i64);
                        pc += 1;
                    }
                    Op::RefIsNull => {
                        let reference: i64 = pop(&mut sp);
                        push(&mut sp, i32::from(reference == 0));
                        pc += 1;
                    }
                    Op::RefFunc { func_idx } => {
                        let c = require_ctx!(ctx);
                        let entry = c
                            .functions
                            .get(*func_idx as usize)
                            .ok_or(RuntimeError::FunctionIndexOutOfBounds(*func_idx))?;
                        push_value(&mut sp, Value::FuncRef(Some(entry.addr)));
                        pc += 1;
                    }

                    // -- Tables and bulk memory --
                    Op::TableGet { table_idx } => {
                        let c = require_ctx!(ctx);
                        let idx: i32 = pop(&mut sp);
                        let value = c.table(*table_idx)?.get(idx as u32)?;
                        push_value(&mut sp, value);
                        pc += 1;
                    }
                    Op::TableSet { table_idx } => {
                        let table = require_ctx!(ctx).table_mut(*table_idx)?;
                        let value = pop_value(&mut sp, table.ref_type().into());
                        let idx: i32 = pop(&mut sp);
                        table.set(idx as u32, Some(value))?;
                        pc += 1;
                    }
                    Op::TableSize { table_idx } => {
                        let size = require_ctx!(ctx).table(*table_idx)?.size();
                        push(&mut sp, size as i32);
                        pc += 1;
                    }
                    Op::TableGrow { table_idx } => {
                        let table = require_ctx!(ctx).table_mut(*table_idx)?;
                        let delta: i32 = pop(&mut sp);
                        let init_value = pop_value(&mut sp, table.ref_type().into());
                        let result = table.grow(delta as u32, Some(init_value))?;
                        push(&mut sp, result as i32);
                        pc += 1;
                    }
                    Op::TableFill { table_idx } => {
                        let table = require_ctx!(ctx).table_mut(*table_idx)?;
                        let count = pop::<i32>(&mut sp) as u32;
                        let value = pop_value(&mut sp, table.ref_type().into());
                        let start = pop::<i32>(&mut sp) as u32;
                        table.fill(start, count, Some(value))?;
                        pc += 1;
                    }
                    Op::TableCopy { dst_table, src_table } => {
                        let c = require_ctx!(ctx);
                        let count = pop::<i32>(&mut sp) as u32;
                        let src_idx = pop::<i32>(&mut sp) as u32;
                        let dst_idx = pop::<i32>(&mut sp) as u32;

                        let dst_addr = c.table_addr(*dst_table)?;
                        let src_addr = c.table_addr(*src_table)?;
                        if dst_addr == src_addr {
                            let table = c.table_mut(*dst_table)?;
                            table.copy_within(dst_idx, src_idx, count)?;
                        } else if src_addr < dst_addr {
                            let (left, right) = c.resources.tables.split_at_mut(dst_addr);
                            right[0].copy_from(dst_idx, &left[src_addr], src_idx, count)?;
                        } else {
                            let (left, right) = c.resources.tables.split_at_mut(src_addr);
                            left[dst_addr].copy_from(dst_idx, &right[0], src_idx, count)?;
                        }
                        pc += 1;
                    }
                    Op::TableInit { elem_idx, table_idx } => {
                        let c = require_ctx!(ctx);
                        let count = pop::<i32>(&mut sp) as u32;
                        let src_idx = pop::<i32>(&mut sp) as u32;
                        let dst_idx = pop::<i32>(&mut sp) as u32;

                        // Field-level access keeps the segment borrow and the
                        // table borrow disjoint.
                        let src_segment = c
                            .segments
                            .element_segments
                            .get(*elem_idx as usize)
                            .ok_or(RuntimeError::ElementIndexOutOfBounds(*elem_idx))?;
                        let addr = c
                            .table_addrs
                            .get(*table_idx as usize)
                            .ok_or(RuntimeError::TableIndexOutOfBounds(*table_idx))?;
                        let table = c
                            .resources
                            .tables
                            .get_mut(addr.0)
                            .ok_or(RuntimeError::TableIndexOutOfBounds(*table_idx))?;
                        table.init(dst_idx, src_segment, src_idx, count)?;
                        pc += 1;
                    }
                    Op::ElemDrop { elem_idx } => {
                        let c = require_ctx!(ctx);
                        let segment = c
                            .segments
                            .element_segments
                            .get_mut(*elem_idx as usize)
                            .ok_or(RuntimeError::ElementIndexOutOfBounds(*elem_idx))?;
                        segment.clear();
                        pc += 1;
                    }
                    Op::MemoryInit { data_idx } => {
                        let memory = memory!();
                        let c = require_ctx!(ctx);
                        let length: i32 = pop(&mut sp);
                        let src_offset: i32 = pop(&mut sp);
                        let dest_addr: i32 = pop(&mut sp);
                        ops::memory::memory_init(
                            memory,
                            dest_addr,
                            src_offset,
                            length,
                            *data_idx,
                            c.data_segments,
                            &c.segments.dropped_data,
                        )?;
                        pc += 1;
                    }
                    Op::DataDrop { data_idx } => {
                        let c = require_ctx!(ctx);
                        c.segments.dropped_data.insert(*data_idx);
                        pc += 1;
                    }
                    Op::Unreachable => {
                        return Err(RuntimeError::Trap("unreachable".to_string()));
                    }
                }
            }
        }

        // The entry function's results sit at its frame's first word.
        let entry = &funcs[activation.entry_func_idx];
        // SAFETY: `return_to_caller` left them there, `result_words` wide.
        let results = unsafe {
            let mut top = fp.add(entry.result_words as usize);
            let mut results: Vec<Value> = (entry.result_types.iter().rev())
                .map(|value_type| pop_value(&mut top, *value_type))
                .collect();
            results.reverse();
            results
        };
        *saved_sp = unsafe { index!(fp) };
        Ok(ExecutionOutcome::Complete(results))
    }
}

/// Execute a compiled function to completion with the given arguments.
///
/// Test wrapper over [`FlatExecutor`] with no Store to dispatch external
/// calls: any external call (an imported function, directly or via
/// call_indirect, or a foreign table funcref) is an error here.
#[cfg(test)]
pub(crate) fn execute_flat(
    funcs: &[CompiledFunction],
    func_idx: usize,
    args: &[Value],
    ctx: Option<&mut ExecContext<'_>>,
) -> Result<Vec<Value>, RuntimeError> {
    let mut executor = FlatExecutor::new();
    match executor.invoke(funcs, func_idx, args, ctx)? {
        ExecutionOutcome::Complete(results) => Ok(results),
        ExecutionOutcome::NeedsExternalCall(_) => {
            Err(RuntimeError::Trap("external call requires Store dispatch".to_string()))
        }
    }
}

#[cfg(test)]
mod tests;
