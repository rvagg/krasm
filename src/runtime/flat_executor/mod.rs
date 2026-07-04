//! Flat bytecode executor.
//!
//! Executes a `CompiledFunction` by walking its `Vec<Op>` with a program
//! counter. Branch targets are pre-resolved absolute indices, so there is no
//! context stack, no label stack, and no multi-level dispatch. The executor
//! reuses the existing `Stack` and `Value` types.
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
//! While suspended, the executor retains the operand stack, the internal
//! call stack, and the frame that issued the call (as `SuspendedState`).
//! Stack discipline at the suspension point: the call's arguments have
//! already been popped; resume pushes the results, exactly as if the call
//! had executed inline. Same-module calls are handled internally on a
//! shared operand stack and never suspend.
//!
//! Instruction implementations are delegated to the `ops` module where
//! possible, keeping the dispatch loop thin.

use super::bytecode::{CompiledFunction, Op};
use super::ops;
use super::stack::Stack;
use super::store::{FuncAddr, GlobalAddr, MemoryAddr, Resources, TableAddr};
use super::value::Value;
use super::{ExecutionOutcome, ExternalCallRequest, RuntimeError};
use crate::parser::module::FunctionType;

/// Perform stack cleanup for a branch: keep `arity` values from the top,
/// discard everything down to `stack_depth`, push the kept values back.
/// When arity is 0, this is a simple truncate.
fn branch_cleanup(stack: &mut Stack, arity: u16, abs_depth: usize) -> Result<(), RuntimeError> {
    if arity == 0 {
        stack.truncate(abs_depth);
        return Ok(());
    }
    let mut kept = Vec::with_capacity(arity as usize);
    for _ in 0..arity {
        kept.push(stack.pop()?);
    }
    stack.truncate(abs_depth);
    for v in kept.into_iter().rev() {
        stack.push(v);
    }
    Ok(())
}

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
    /// `num_imported`. call_indirect scans this slice to map a table
    /// funcref back to a callable.
    pub functions: &'a [FuncEntry],
    /// Number of imported functions (a prefix of `functions`).
    pub num_imported: usize,
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
pub(crate) fn build_func_entries(
    module: &crate::parser::module::Module,
    function_addresses: &[FuncAddr],
) -> Vec<FuncEntry> {
    use crate::parser::module::ExternalKind;

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

const MAX_CALL_DEPTH: usize = 1000;

/// Saved caller state for the call stack.
struct CallFrame {
    /// Index into the `funcs` slice (the function to resume).
    func_idx: usize,
    /// Program counter to resume at in the caller.
    pc: usize,
    /// Caller's stack base (for resolving branch stack_depth).
    stack_base: usize,
    /// Caller's local variables.
    locals: Vec<Value>,
}

impl ExecContext<'_> {
    /// Resolve module-local memory index 0 to the memory instance.
    fn memory(&self) -> Result<&super::memory::Memory, RuntimeError> {
        let addr = self
            .memory_addrs
            .first()
            .ok_or_else(|| RuntimeError::MemoryError("no memory".to_string()))?;
        self.resources
            .memories
            .get(addr.0)
            .ok_or_else(|| RuntimeError::MemoryError("invalid memory address".to_string()))
    }

    /// Resolve module-local memory index 0 to a mutable memory instance.
    fn memory_mut(&mut self) -> Result<&mut super::memory::Memory, RuntimeError> {
        let addr = self
            .memory_addrs
            .first()
            .ok_or_else(|| RuntimeError::MemoryError("no memory".to_string()))?;
        self.resources
            .memories
            .get_mut(addr.0)
            .ok_or_else(|| RuntimeError::MemoryError("invalid memory address".to_string()))
    }

    fn global_get(&self, stack: &mut Stack, index: u32) -> Result<(), RuntimeError> {
        let addr = self
            .global_addrs
            .get(index as usize)
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(index))?;
        let value = self
            .resources
            .globals
            .get(addr.0)
            .copied()
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(index))?;
        stack.push(value);
        Ok(())
    }

    fn global_set(&mut self, stack: &mut Stack, index: u32) -> Result<(), RuntimeError> {
        let val = stack.pop()?;
        let addr = self
            .global_addrs
            .get(index as usize)
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(index))?;
        let slot = self
            .resources
            .globals
            .get_mut(addr.0)
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(index))?;
        *slot = val;
        Ok(())
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

    /// Look up a function type by type index.
    fn func_type(&self, type_idx: u32) -> Result<&FunctionType, RuntimeError> {
        self.types
            .get(type_idx as usize)
            .ok_or(RuntimeError::InvalidFunctionType)
    }
}

/// Require a mutable reference to the execution context, or trap.
macro_rules! require_ctx {
    ($ctx:expr) => {
        $ctx.as_mut()
            .ok_or_else(|| RuntimeError::Trap("operation requires execution context".to_string()))?
    };
}

/// Build locals for the entry function from explicit arguments.
fn init_locals(func: &CompiledFunction, args: &[Value]) -> Vec<Value> {
    let mut locals = Vec::with_capacity(func.local_count as usize);
    locals.extend_from_slice(args);
    locals.resize(func.local_count as usize, Value::I32(0));
    locals
}

/// Build locals for a callee by popping arguments from the stack.
fn init_locals_from_stack(callee: &CompiledFunction, stack: &mut Stack) -> Result<Vec<Value>, RuntimeError> {
    let mut args = Vec::with_capacity(callee.param_count as usize);
    for _ in 0..callee.param_count {
        args.push(stack.pop()?);
    }
    args.reverse();
    let mut locals = Vec::with_capacity(callee.local_count as usize);
    locals.extend(args);
    locals.resize(callee.local_count as usize, Value::I32(0));
    Ok(locals)
}

/// Execution state suspended across an external call.
///
/// Holds the frame that issued the call (with `pc` already past the Call op)
/// and the entry function index, needed to collect the right number of
/// results when execution eventually completes.
struct SuspendedState {
    frame: CallFrame,
    entry_func_idx: usize,
    /// Result count declared by the suspended import; resume traps on
    /// mismatch rather than corrupting the stack.
    expected_results: u16,
}

/// Enter a local function call: bounds- and depth-check, pop the callee's
/// arguments into fresh locals, save the caller's frame, and return the
/// callee's cursor. Shared by `call` and `call_indirect`.
/// `caller` is the issuing frame with `pc` already past the call instruction.
fn enter_local_call(
    stack: &mut Stack,
    call_stack: &mut Vec<CallFrame>,
    funcs: &[CompiledFunction],
    local_idx: usize,
    module_func_idx: u32,
    caller: CallFrame,
) -> Result<CallFrame, RuntimeError> {
    if local_idx >= funcs.len() {
        return Err(RuntimeError::FunctionIndexOutOfBounds(module_func_idx));
    }
    if call_stack.len() >= MAX_CALL_DEPTH {
        return Err(RuntimeError::CallStackOverflow);
    }

    let callee = &funcs[local_idx];
    let locals = init_locals_from_stack(callee, stack)?;
    let stack_base = stack.len();
    call_stack.push(caller);
    Ok(CallFrame {
        func_idx: local_idx,
        pc: 0,
        stack_base,
        locals,
    })
}

/// Suspend execution for an external call: pop the callee's arguments per
/// its type, record the suspension, and build the request for the Store.
/// `frame` is the issuing frame with `pc` already past the call instruction.
fn suspend_external_call(
    stack: &mut Stack,
    suspended: &mut Option<SuspendedState>,
    frame: CallFrame,
    entry_func_idx: usize,
    func_addr: FuncAddr,
    ftype: &FunctionType,
) -> Result<ExecutionOutcome, RuntimeError> {
    let mut args = Vec::with_capacity(ftype.parameters.len());
    for _ in 0..ftype.parameters.len() {
        args.push(stack.pop()?);
    }
    args.reverse();

    *suspended = Some(SuspendedState {
        frame,
        entry_func_idx,
        expected_results: ftype.return_types.len() as u16,
    });
    Ok(ExecutionOutcome::NeedsExternalCall(ExternalCallRequest {
        func_addr,
        args,
    }))
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
    stack: Stack,
    call_stack: Vec<CallFrame>,
    suspended: Option<SuspendedState>,
}

impl Default for FlatExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl FlatExecutor {
    pub fn new() -> Self {
        FlatExecutor {
            stack: Stack::new(),
            call_stack: Vec::new(),
            suspended: None,
        }
    }

    /// Discard all execution state, returning the executor to a reusable
    /// idle state after an error.
    fn reset(&mut self) {
        self.stack.clear();
        self.call_stack.clear();
        self.suspended = None;
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
        self.reset();
        let func = funcs
            .get(func_idx)
            .ok_or(RuntimeError::FunctionIndexOutOfBounds(func_idx as u32))?;

        let frame = CallFrame {
            func_idx,
            pc: 0,
            stack_base: 0,
            locals: init_locals(func, args),
        };
        let result = self.run(funcs, frame, func_idx, ctx);
        if result.is_err() {
            self.reset();
        }
        result
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
            entry_func_idx,
            expected_results,
        } = self
            .suspended
            .take()
            .ok_or_else(|| RuntimeError::Trap("resume called without saved execution state".to_string()))?;

        // A wrong count would silently skew branch stack offsets for the
        // rest of execution; trap here instead. Types are still checked
        // lazily by the typed pops that consume the values.
        if results.len() != expected_results as usize {
            self.reset();
            return Err(RuntimeError::Trap(format!(
                "external call returned {} values, expected {}",
                results.len(),
                expected_results
            )));
        }

        for value in results {
            self.stack.push(value);
        }

        let result = self.run(funcs, frame, entry_func_idx, ctx);
        if result.is_err() {
            self.reset();
        }
        result
    }

    /// The dispatch loop: walk bytecode from `frame` until the entry
    /// function completes or an imported call suspends execution.
    fn run(
        &mut self,
        funcs: &[CompiledFunction],
        frame: CallFrame,
        entry_func_idx: usize,
        mut ctx: Option<&mut ExecContext<'_>>,
    ) -> Result<ExecutionOutcome, RuntimeError> {
        // Destructure into disjoint borrows: the loop mutates the operand
        // stack and call stack independently.
        let FlatExecutor {
            stack,
            call_stack,
            suspended,
        } = self;

        let CallFrame {
            func_idx: mut current_func_idx,
            mut pc,
            // Stack base for the current function. Branch stack_depth values
            // are offsets from this base, since the physical stack is shared
            // across all active call frames.
            mut stack_base,
            mut locals,
        } = frame;

        // One-line dispatch for ops that only touch the operand stack.
        macro_rules! stack_op {
            ($f:path $(, $arg:expr)*) => {{
                $f(stack $(, $arg)*)?;
                pc += 1;
            }};
        }
        // Ops that read (mem_op) or write (mem_op_mut) linear memory.
        macro_rules! mem_op {
            ($f:path $(, $arg:expr)*) => {{
                $f(stack, require_ctx!(ctx).memory()? $(, $arg)*)?;
                pc += 1;
            }};
        }
        macro_rules! mem_op_mut {
            ($f:path $(, $arg:expr)*) => {{
                $f(stack, require_ctx!(ctx).memory_mut()? $(, $arg)*)?;
                pc += 1;
            }};
        }

        loop {
            let ops_slice = &funcs[current_func_idx].ops;
            if pc >= ops_slice.len() {
                break;
            }

            match &ops_slice[pc] {
                // -- Constants --
                Op::I32Const(v) => stack_op!(ops::numeric::i32_const, *v),
                Op::I64Const(v) => stack_op!(ops::numeric::i64_const, *v),
                Op::F32Const(v) => stack_op!(ops::numeric::f32_const, *v),
                Op::F64Const(v) => stack_op!(ops::numeric::f64_const, *v),

                // -- Arithmetic --
                Op::I32Add => stack_op!(ops::numeric::i32_add),
                Op::I32Sub => stack_op!(ops::numeric::i32_sub),
                Op::I32Mul => stack_op!(ops::numeric::i32_mul),
                Op::I32DivS => stack_op!(ops::numeric::i32_div_s),
                Op::I32DivU => stack_op!(ops::numeric::i32_div_u),
                Op::I32RemS => stack_op!(ops::numeric::i32_rem_s),
                Op::I32RemU => stack_op!(ops::numeric::i32_rem_u),
                Op::I32Clz => stack_op!(ops::numeric::i32_clz),
                Op::I32Ctz => stack_op!(ops::numeric::i32_ctz),
                Op::I32Popcnt => stack_op!(ops::numeric::i32_popcnt),
                Op::I32And => stack_op!(ops::bitwise::i32_and),
                Op::I32Or => stack_op!(ops::bitwise::i32_or),
                Op::I32Xor => stack_op!(ops::bitwise::i32_xor),
                Op::I32Shl => stack_op!(ops::bitwise::i32_shl),
                Op::I32ShrS => stack_op!(ops::bitwise::i32_shr_s),
                Op::I32ShrU => stack_op!(ops::bitwise::i32_shr_u),
                Op::I32Rotl => stack_op!(ops::bitwise::i32_rotl),
                Op::I32Rotr => stack_op!(ops::bitwise::i32_rotr),
                Op::I32Extend8S => stack_op!(ops::conversion::i32_extend8_s),
                Op::I32Extend16S => stack_op!(ops::conversion::i32_extend16_s),

                // -- Comparison --
                Op::I32Eqz => stack_op!(ops::comparison::i32_eqz),
                Op::I32Eq => stack_op!(ops::comparison::i32_eq),
                Op::I32Ne => stack_op!(ops::comparison::i32_ne),
                Op::I32LtS => stack_op!(ops::comparison::i32_lt_s),
                Op::I32LtU => stack_op!(ops::comparison::i32_lt_u),
                Op::I32GtS => stack_op!(ops::comparison::i32_gt_s),
                Op::I32GtU => stack_op!(ops::comparison::i32_gt_u),
                Op::I32LeS => stack_op!(ops::comparison::i32_le_s),
                Op::I32LeU => stack_op!(ops::comparison::i32_le_u),
                Op::I32GeS => stack_op!(ops::comparison::i32_ge_s),
                Op::I32GeU => stack_op!(ops::comparison::i32_ge_u),

                // -- Local variables --
                // These interact with the locals array directly; no ops function.
                Op::LocalGet { index } => {
                    let val = locals
                        .get(*index as usize)
                        .copied()
                        .ok_or(RuntimeError::LocalIndexOutOfBounds(*index))?;
                    stack.push(val);
                    pc += 1;
                }
                Op::LocalSet { index } => {
                    let val = stack.pop()?;
                    let slot = locals
                        .get_mut(*index as usize)
                        .ok_or(RuntimeError::LocalIndexOutOfBounds(*index))?;
                    *slot = val;
                    pc += 1;
                }
                Op::LocalTee { index } => {
                    let val = stack.pop()?;
                    let slot = locals
                        .get_mut(*index as usize)
                        .ok_or(RuntimeError::LocalIndexOutOfBounds(*index))?;
                    *slot = val;
                    stack.push(val);
                    pc += 1;
                }

                // -- Global variables --
                Op::GlobalGet { index } => {
                    require_ctx!(ctx).global_get(stack, *index)?;
                    pc += 1;
                }
                Op::GlobalSet { index } => {
                    require_ctx!(ctx).global_set(stack, *index)?;
                    pc += 1;
                }

                // -- Memory --
                Op::I32Load(m) => mem_op!(ops::memory::i32_load, m),
                Op::I32Load8S(m) => mem_op!(ops::memory::i32_load8_s, m),
                Op::I32Load8U(m) => mem_op!(ops::memory::i32_load8_u, m),
                Op::I32Load16S(m) => mem_op!(ops::memory::i32_load16_s, m),
                Op::I32Load16U(m) => mem_op!(ops::memory::i32_load16_u, m),
                Op::I64Load(m) => mem_op!(ops::memory::i64_load, m),
                Op::I64Load8S(m) => mem_op!(ops::memory::i64_load8_s, m),
                Op::I64Load8U(m) => mem_op!(ops::memory::i64_load8_u, m),
                Op::I64Load16S(m) => mem_op!(ops::memory::i64_load16_s, m),
                Op::I64Load16U(m) => mem_op!(ops::memory::i64_load16_u, m),
                Op::I64Load32S(m) => mem_op!(ops::memory::i64_load32_s, m),
                Op::I64Load32U(m) => mem_op!(ops::memory::i64_load32_u, m),
                Op::F32Load(m) => mem_op!(ops::memory::f32_load, m),
                Op::F64Load(m) => mem_op!(ops::memory::f64_load, m),
                Op::I32Store(m) => mem_op_mut!(ops::memory::i32_store, m),
                Op::I32Store8(m) => mem_op_mut!(ops::memory::i32_store8, m),
                Op::I32Store16(m) => mem_op_mut!(ops::memory::i32_store16, m),
                Op::I64Store(m) => mem_op_mut!(ops::memory::i64_store, m),
                Op::I64Store8(m) => mem_op_mut!(ops::memory::i64_store8, m),
                Op::I64Store16(m) => mem_op_mut!(ops::memory::i64_store16, m),
                Op::I64Store32(m) => mem_op_mut!(ops::memory::i64_store32, m),
                Op::F32Store(m) => mem_op_mut!(ops::memory::f32_store, m),
                Op::F64Store(m) => mem_op_mut!(ops::memory::f64_store, m),
                Op::MemorySize => mem_op!(ops::memory::memory_size),
                Op::MemoryGrow => mem_op_mut!(ops::memory::memory_grow),
                Op::MemoryCopy => mem_op_mut!(ops::memory::memory_copy),
                Op::MemoryFill => mem_op_mut!(ops::memory::memory_fill),

                // -- Control flow --
                // These mutate pc directly; no ops function.
                Op::Br {
                    target,
                    arity,
                    stack_depth,
                } => {
                    branch_cleanup(stack, *arity, stack_base + *stack_depth as usize)?;
                    pc = *target as usize;
                }
                Op::BrIf {
                    target,
                    arity,
                    stack_depth,
                } => {
                    let cond = stack.pop_i32()?;
                    if cond != 0 {
                        branch_cleanup(stack, *arity, stack_base + *stack_depth as usize)?;
                        pc = *target as usize;
                    } else {
                        pc += 1;
                    }
                }
                Op::BrTable { targets, default } => {
                    let index = stack.pop_i32()? as u32;
                    let target = if (index as usize) < targets.len() {
                        &targets[index as usize]
                    } else {
                        default
                    };
                    branch_cleanup(stack, target.arity, stack_base + target.stack_depth as usize)?;
                    pc = target.pc as usize;
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
                        let ftype = c.func_type(entry.type_idx)?;
                        let frame = CallFrame {
                            func_idx: current_func_idx,
                            pc: pc + 1,
                            stack_base,
                            locals,
                        };
                        return suspend_external_call(stack, suspended, frame, entry_func_idx, entry.addr, ftype);
                    }

                    let local_idx = *func_idx as usize - num_imported;
                    let caller = CallFrame {
                        func_idx: current_func_idx,
                        pc: pc + 1,
                        stack_base,
                        locals: std::mem::take(&mut locals),
                    };
                    let callee = enter_local_call(stack, call_stack, funcs, local_idx, *func_idx, caller)?;
                    current_func_idx = callee.func_idx;
                    pc = callee.pc;
                    stack_base = callee.stack_base;
                    locals = callee.locals;
                }
                Op::CallIndirect { type_idx, table_idx } => {
                    let c = require_ctx!(ctx);
                    let elem_idx = stack.pop_i32()? as u32;

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

                    match c.functions.iter().position(|e| e.addr == func_addr) {
                        Some(module_idx) => {
                            // The funcref belongs to this module: the
                            // signature check happens here, before any
                            // arguments are consumed.
                            let actual = c.func_type(c.functions[module_idx].type_idx)?;
                            if expected != actual {
                                return Err(RuntimeError::IndirectCallTypeMismatch {
                                    expected: format!("{expected:?}"),
                                    actual: format!("{actual:?}"),
                                });
                            }

                            if module_idx < c.num_imported {
                                let frame = CallFrame {
                                    func_idx: current_func_idx,
                                    pc: pc + 1,
                                    stack_base,
                                    locals,
                                };
                                return suspend_external_call(
                                    stack,
                                    suspended,
                                    frame,
                                    entry_func_idx,
                                    func_addr,
                                    expected,
                                );
                            }

                            // Local function: internal call, same frame
                            // discipline as Op::Call.
                            let local_idx = module_idx - c.num_imported;
                            let caller = CallFrame {
                                func_idx: current_func_idx,
                                pc: pc + 1,
                                stack_base,
                                locals: std::mem::take(&mut locals),
                            };
                            let callee =
                                enter_local_call(stack, call_stack, funcs, local_idx, module_idx as u32, caller)?;
                            current_func_idx = callee.func_idx;
                            pc = callee.pc;
                            stack_base = callee.stack_base;
                            locals = callee.locals;
                        }
                        None => {
                            // Foreign funcref: a function from another module
                            // placed in a shared table. Its type is unknown
                            // here, so the signature check is deferred to the
                            // Store at dispatch (structured executor parity).
                            let frame = CallFrame {
                                func_idx: current_func_idx,
                                pc: pc + 1,
                                stack_base,
                                locals,
                            };
                            return suspend_external_call(stack, suspended, frame, entry_func_idx, func_addr, expected);
                        }
                    }
                }
                Op::Return | Op::End => {
                    if let Some(frame) = call_stack.pop() {
                        current_func_idx = frame.func_idx;
                        pc = frame.pc;
                        stack_base = frame.stack_base;
                        locals = frame.locals;
                    } else {
                        break;
                    }
                }
                Op::Nop | Op::Label { .. } => {
                    pc += 1;
                }
                Op::Drop => stack_op!(ops::parametric::drop),
                Op::Select => stack_op!(ops::parametric::select),
                Op::Unreachable => {
                    return Err(RuntimeError::Trap("unreachable".to_string()));
                }
            }
        }

        // Collect the entry function's return values from the stack
        let result_count = funcs[entry_func_idx].result_count;
        let mut results = Vec::with_capacity(result_count as usize);
        for _ in 0..result_count {
            results.push(stack.pop()?);
        }
        results.reverse();
        Ok(ExecutionOutcome::Complete(results))
    }
}

/// Execute a compiled function to completion with the given arguments.
///
/// Convenience wrapper over [`FlatExecutor`] for callers with no Store to
/// dispatch external calls (benchmarks, tests): any external call — an
/// imported function, directly or via call_indirect, or a foreign table
/// funcref — is an error here.
pub fn execute_flat(
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
