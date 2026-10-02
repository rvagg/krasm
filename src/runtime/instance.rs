//! WebAssembly module instance

use super::bytecode::CompiledFunction;
use super::compiler::compile_module;
use super::flat_executor::{ExecContext, FlatExecutor, FuncEntry, build_func_entries};
use super::{
    EngineKind, ExecutionOutcome, FuncAddr, GlobalAddr, MemoryAddr, RuntimeError, TableAddr, Value, executor::Executor,
    ops, store::Resources,
};
use crate::parser::instruction::{Instruction, InstructionKind, SimdOp};
use crate::parser::module::{DataMode, ElementMode, ExportIndex, ExternalKind, Module, Positional, ValueType};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A WebAssembly module instance.
///
/// An instance is the runtime footprint of a [`Module`] within a [`Store`](super::Store).
/// It holds address maps that translate module-local indices to global addresses
/// in the Store's resource pools, plus an executor that drives interpretation.
///
/// Lifecycle (all called by Store):
/// 1. `new_unlinked` — allocate with resource addresses
/// 2. `link_functions` — populate function addresses, then initialise globals,
///    element segments, and data sections
/// 3. `get_start_function_addr` — resolve the start function for Store to execute
pub struct Instance {
    module: Arc<Module>,
    exports: HashMap<String, u32>, // Maps export name to function index
    /// Maps local function index to global FuncAddr
    function_addresses: Vec<FuncAddr>,
    /// Maps local memory index to global MemoryAddr
    pub(super) memory_addresses: Vec<MemoryAddr>,
    /// Maps local table index to global TableAddr
    table_addresses: Vec<TableAddr>,
    /// Maps local global index to global GlobalAddr
    global_addresses: Vec<GlobalAddr>,
    engine: Engine,
    /// Per-instance segment state, produced at instantiation and mutated by
    /// bulk-memory instructions.
    segments: SegmentState,
}

/// Per-instance segment state, produced at instantiation and mutated by the
/// bulk-memory instructions. Owned by the instance and borrowed by both
/// executors.
#[derive(Default)]
pub(crate) struct SegmentState {
    /// Runtime element segments for table.init; elem.drop empties them.
    pub(crate) element_segments: Vec<Vec<Option<Value>>>,
    /// Data segments dropped via data.drop (active segments are logically
    /// dropped after initialisation, per spec).
    pub(crate) dropped_data: HashSet<u32>,
}

/// Only the selected interpreter's execution state is constructed.
enum Engine {
    Structured(Executor),
    Flat(FlatEngine),
}

/// Flat-bytecode execution state for an instance.
struct FlatEngine {
    /// Compiled local functions, indexed by local function index.
    funcs: Vec<CompiledFunction>,
    /// FuncEntry per module-level function; built at link time when
    /// function addresses become known.
    entries: Vec<FuncEntry>,
    executor: FlatExecutor,
}

impl Instance {
    /// Create a new unlinked instance with resource address maps
    ///
    /// Resources (memories, tables, globals) live in the Store. The instance
    /// holds address maps that translate module-local indices to global addresses.
    /// Function addresses are linked separately via link_functions().
    pub(super) fn new_unlinked(
        module: Arc<Module>,
        memory_addresses: Vec<MemoryAddr>,
        table_addresses: Vec<TableAddr>,
        global_addresses: Vec<GlobalAddr>,
        engine: EngineKind,
    ) -> Result<Self, RuntimeError> {
        let mut exports = HashMap::new();

        for export in &module.exports.exports {
            if let ExportIndex::Function(idx) = export.index {
                exports.insert(export.name.clone(), idx);
            }
        }

        let engine = match engine {
            EngineKind::Structured => Engine::Structured(Executor::new_unlinked(
                Arc::clone(&module),
                memory_addresses.clone(),
                table_addresses.clone(),
                global_addresses.clone(),
            )?),
            EngineKind::Flat => {
                let funcs = compile_module(&module);
                let executor = FlatExecutor::new();
                #[cfg(feature = "instruction-profile")]
                let executor = {
                    let mut executor = executor;
                    executor.enable_instruction_profile(&funcs);
                    executor
                };
                Engine::Flat(FlatEngine {
                    funcs,
                    entries: Vec::new(),
                    executor,
                })
            }
        };

        Ok(Instance {
            module,
            exports,
            function_addresses: Vec::new(),
            memory_addresses,
            table_addresses,
            global_addresses,
            engine,
            segments: SegmentState::default(),
        })
    }

    /// Link function addresses and initialise the instance
    ///
    /// Populates function addresses, then initialises globals, element segments,
    /// and data sections. This must happen after linking because init expressions
    /// can contain ref.func instructions that require the function address mapping.
    pub(super) fn link_functions(
        &mut self,
        function_addresses: Vec<FuncAddr>,
        resources: &mut Resources,
    ) -> Result<(), RuntimeError> {
        match &mut self.engine {
            Engine::Structured(executor) => executor.link_function_addresses(function_addresses.clone()),
            Engine::Flat(flat) => flat.entries = build_func_entries(&self.module, &function_addresses),
        }
        self.function_addresses = function_addresses;

        // Initialise in dependency order: globals first (element segments may
        // reference them), then element segments, then data sections.
        self.initialise_globals(resources)?;
        self.initialise_element_segments(resources)?;
        self.initialise_data_sections(resources)?;

        Ok(())
    }

    /// Resolve the start function's address for Store to execute, if present.
    pub(super) fn get_start_function_addr(&self) -> Result<Option<FuncAddr>, RuntimeError> {
        if !self.module.start.has_position() {
            return Ok(None);
        }
        let func_idx = self.module.start.start;
        self.function_addresses
            .get(func_idx as usize)
            .copied()
            .map(Some)
            .ok_or(RuntimeError::FunctionIndexOutOfBounds(func_idx))
    }

    /// Invoke a function by its local function index
    ///
    /// Used by Store to execute functions. The func_idx includes both imported
    /// and local functions in the module's function index space.
    ///
    /// # Errors
    /// - `Trap` if func_idx refers to an imported function (Store handles these)
    /// - `FunctionIndexOutOfBounds` if the index is invalid
    /// - `TypeMismatch` if argument count or types don't match
    pub(super) fn invoke_by_index(
        &mut self,
        func_idx: u32,
        args: Vec<Value>,
        resources: &mut Resources,
    ) -> Result<ExecutionOutcome, RuntimeError> {
        let num_imported_functions = self.module.imports.function_count();

        if (func_idx as usize) < num_imported_functions {
            return Err(RuntimeError::Trap(format!(
                "imported function {func_idx} must be dispatched by Store"
            )));
        }

        let code_idx = func_idx as usize - num_imported_functions;

        let func = self
            .module
            .functions
            .get((code_idx) as u32)
            .ok_or(RuntimeError::FunctionIndexOutOfBounds(func_idx))?;

        let func_type = self
            .module
            .types
            .get(func.ftype_index)
            .ok_or(RuntimeError::FunctionIndexOutOfBounds(func_idx))?;

        if args.len() != func_type.parameters.len() {
            return Err(RuntimeError::TypeMismatch {
                expected: format!("{} arguments", func_type.parameters.len()),
                actual: format!("{} arguments", args.len()),
            });
        }

        for (i, (arg, expected_type)) in args.iter().zip(&func_type.parameters).enumerate() {
            if arg.typ() != *expected_type {
                return Err(RuntimeError::TypeMismatch {
                    expected: format!("{expected_type:?} for argument {i}"),
                    actual: format!("{:?}", arg.typ()),
                });
            }
        }

        match &mut self.engine {
            Engine::Flat(flat) => {
                let mut ctx = ExecContext {
                    resources,
                    global_addrs: &self.global_addresses,
                    memory_addrs: &self.memory_addresses,
                    table_addrs: &self.table_addresses,
                    types: &self.module.types.types,
                    functions: &flat.entries,
                    num_imported: num_imported_functions,
                    segments: &mut self.segments,
                    data_segments: &self.module.data.data,
                };
                flat.executor.invoke(&flat.funcs, code_idx, &args, Some(&mut ctx))
            }
            Engine::Structured(executor) => {
                let body = self
                    .module
                    .code
                    .get(code_idx as u32)
                    .ok_or(RuntimeError::FunctionIndexOutOfBounds(func_idx))?;

                executor.execute_function_with_locals(
                    &body.body,
                    args,
                    &func_type.return_types,
                    Some(&body.locals),
                    resources,
                    &mut self.segments,
                )
            }
        }
    }

    /// Resume execution after an external call completes
    ///
    /// Called by Store when a cross-module call returns with results.
    pub(super) fn resume_with_results(
        &mut self,
        results: Vec<Value>,
        resources: &mut Resources,
    ) -> Result<ExecutionOutcome, RuntimeError> {
        match &mut self.engine {
            Engine::Flat(flat) => {
                let mut ctx = ExecContext {
                    resources,
                    global_addrs: &self.global_addresses,
                    memory_addrs: &self.memory_addresses,
                    table_addrs: &self.table_addresses,
                    types: &self.module.types.types,
                    functions: &flat.entries,
                    num_imported: self.module.imports.function_count(),
                    segments: &mut self.segments,
                    data_segments: &self.module.data.data,
                };
                flat.executor.resume_with_results(&flat.funcs, results, Some(&mut ctx))
            }
            Engine::Structured(executor) => executor.resume_with_results(results, resources, &mut self.segments),
        }
    }

    /// Get the FuncAddr for an exported function by name
    ///
    /// # Errors
    /// - `UnknownExport` if the export doesn't exist or isn't a function
    pub fn get_function_addr(&self, name: &str) -> Result<FuncAddr, RuntimeError> {
        let func_idx = self
            .exports
            .get(name)
            .ok_or_else(|| RuntimeError::UnknownExport(name.to_string()))?;

        self.function_addresses
            .get(*func_idx as usize)
            .copied()
            .ok_or_else(|| RuntimeError::UnknownExport(format!("function {} not linked", name)))
    }

    /// Get the module reference
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Snapshot cumulative flat-bytecode dispatch counts, including start functions.
    ///
    /// Counts include the operation that traps, but not an operation stopped by
    /// an exhausted instruction budget. Host work is excluded. Structured
    /// instances return `None`; counts survive completed and failed invocations.
    #[cfg(feature = "instruction-profile")]
    pub fn instruction_profile(&self) -> Option<super::profile::InstructionProfile<'_>> {
        let Engine::Flat(flat) = &self.engine else {
            return None;
        };
        Some(super::profile::InstructionProfile::new(
            &self.module,
            &flat.funcs,
            flat.executor.instruction_counts(),
        ))
    }

    /// Get an exported global value by name
    ///
    /// # Errors
    /// - `UnknownExport` if the export doesn't exist or isn't a global
    pub fn get_global_export(&self, name: &str, resources: &Resources) -> Result<Value, RuntimeError> {
        if let ExportIndex::Global(global_idx) = self.find_export(name)? {
            self.get_global(global_idx, resources)
        } else {
            Err(RuntimeError::UnknownExport(format!("{} is not a global export", name)))
        }
    }

    fn get_global(&self, global_idx: u32, resources: &Resources) -> Result<Value, RuntimeError> {
        let addr = self
            .global_addresses
            .get(global_idx as usize)
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(global_idx))?;
        resources
            .globals
            .get(addr.0)
            .copied()
            .ok_or(RuntimeError::GlobalIndexOutOfBounds(global_idx))
    }

    /// Look up an export by name, returning its ExportIndex.
    fn find_export(&self, name: &str) -> Result<ExportIndex, RuntimeError> {
        self.module
            .exports
            .get_by_name(name)
            .map(|e| e.index)
            .ok_or_else(|| RuntimeError::UnknownExport(name.to_string()))
    }

    /// Get the GlobalAddr for an exported global by name
    ///
    /// # Errors
    /// - `UnknownExport` if the export doesn't exist or isn't a global
    pub fn get_global_addr(&self, name: &str) -> Result<GlobalAddr, RuntimeError> {
        if let ExportIndex::Global(idx) = self.find_export(name)? {
            self.global_addresses
                .get(idx as usize)
                .copied()
                .ok_or_else(|| RuntimeError::UnknownExport(format!("global {} not found", name)))
        } else {
            Err(RuntimeError::UnknownExport(format!("{} is not a global export", name)))
        }
    }

    /// Get the MemoryAddr for an exported memory by name
    ///
    /// # Errors
    /// - `UnknownExport` if the export doesn't exist or isn't a memory
    pub fn get_memory_addr(&self, name: &str) -> Result<MemoryAddr, RuntimeError> {
        if let ExportIndex::Memory(idx) = self.find_export(name)? {
            self.memory_addresses
                .get(idx as usize)
                .copied()
                .ok_or_else(|| RuntimeError::UnknownExport(format!("memory {} not found", name)))
        } else {
            Err(RuntimeError::UnknownExport(format!("{} is not a memory export", name)))
        }
    }

    /// Get the TableAddr for an exported table by name
    ///
    /// # Errors
    /// - `UnknownExport` if the export doesn't exist or isn't a table
    pub fn get_table_addr(&self, name: &str) -> Result<TableAddr, RuntimeError> {
        if let ExportIndex::Table(idx) = self.find_export(name)? {
            self.table_addresses
                .get(idx as usize)
                .copied()
                .ok_or_else(|| RuntimeError::UnknownExport(format!("table {} not found", name)))
        } else {
            Err(RuntimeError::UnknownExport(format!("{} is not a table export", name)))
        }
    }

    /// Set an instruction budget limit for execution
    ///
    /// Each interpreter operation consumes one unit; exhaustion traps before
    /// the next operation. Counts depend on the engine's representation.
    /// The limit covers this instance only, persists across calls and
    /// suspension, and excludes host work. Pass `None` to remove it.
    pub fn set_instruction_budget(&mut self, budget: Option<u64>) {
        match &mut self.engine {
            Engine::Flat(flat) => flat.executor.set_instruction_budget(budget),
            Engine::Structured(executor) => executor.set_instruction_budget(budget),
        }
    }

    /// Initialise module globals with their init expressions
    ///
    /// This must be called after function_addresses have been linked, as global init
    /// expressions can contain ref.func instructions that need the address mapping.
    pub(super) fn initialise_globals(&mut self, resources: &mut Resources) -> Result<(), RuntimeError> {
        // Calculate how many imported globals there are
        let num_imported_globals = self
            .module
            .imports
            .imports
            .iter()
            .filter(|import| matches!(import.external_kind, ExternalKind::Global(_)))
            .count();

        // Initialise module's own globals with their init expressions.
        // Later globals can reference earlier ones, so order matters.
        for (idx, global) in self.module.globals.globals.iter().enumerate() {
            let global_idx = num_imported_globals + idx;

            let initial_value = if global.init.is_empty() {
                continue;
            } else {
                self.evaluate_const_expr(&global.init, resources)?
            };

            let addr = self.global_addresses[global_idx];
            resources.globals[addr.0] = initial_value;
        }

        Ok(())
    }

    /// Initialise tables with element segments
    ///
    /// This must be called after function_addresses have been linked, as element
    /// segments can contain ref.func instructions that need the address mapping.
    pub(super) fn initialise_element_segments(&mut self, resources: &mut Resources) -> Result<(), RuntimeError> {
        for element in &self.module.elements.elements {
            let mut values = Vec::new();
            for init_expr in &element.init {
                let val = self.evaluate_const_expr(init_expr, resources)?;
                values.push(Some(val));
            }

            match &element.mode {
                ElementMode::Active { table_index, offset } => {
                    let offset_val = self.evaluate_const_expr(offset, resources)?;
                    let start_idx = match offset_val {
                        Value::I32(v) => v as u32,
                        _ => return Err(RuntimeError::InvalidConstExpr("element offset must be i32".to_string())),
                    };

                    let table_idx = self
                        .table_addresses
                        .get(*table_index as usize)
                        .map(|addr| addr.0)
                        .ok_or(RuntimeError::TableIndexOutOfBounds(*table_index))?;
                    let table = &mut resources.tables[table_idx];
                    table.init(start_idx, &values, 0, values.len() as u32)?;

                    // Active segments are dropped after instantiation per spec
                    self.segments.element_segments.push(Vec::new());
                }
                ElementMode::Declarative => {
                    // Declarative segments are dropped immediately per spec
                    self.segments.element_segments.push(Vec::new());
                }
                ElementMode::Passive => {
                    // Passive segments remain available for table.init
                    self.segments.element_segments.push(values);
                }
            }
        }

        Ok(())
    }

    /// Initialise memory with data from data sections
    pub(super) fn initialise_data_sections(&mut self, resources: &mut Resources) -> Result<(), RuntimeError> {
        for (seg_idx, data_segment) in self.module.data.data.iter().enumerate() {
            match &data_segment.mode {
                DataMode::Active { memory_index, offset } => {
                    if *memory_index != 0 {
                        return Err(RuntimeError::MemoryError(format!(
                            "invalid memory index {} in data segment",
                            memory_index
                        )));
                    }

                    let mem_idx = self
                        .memory_addresses
                        .first()
                        .map(|addr| addr.0)
                        .ok_or_else(|| RuntimeError::MemoryError("no memory instance available".to_string()))?;

                    let offset_value = self.evaluate_const_expr(offset, resources)?;
                    let offset_addr = match offset_value {
                        Value::I32(v) => v as u32,
                        _ => {
                            return Err(RuntimeError::MemoryError(
                                "data segment offset must be an i32".to_string(),
                            ));
                        }
                    };

                    let memory = &mut resources.memories[mem_idx];
                    let data = &data_segment.init;

                    // Bounds check: data must fit within allocated memory pages
                    let end_addr = offset_addr as usize + data.len();
                    let memory_size_bytes = (memory.size() as usize) * 65536; // pages to bytes
                    if end_addr > memory_size_bytes {
                        return Err(RuntimeError::MemoryError("out of bounds memory access".to_string()));
                    }

                    ops::memory::copy_to_memory(memory, offset_addr, data)?;
                    // Active segments are logically dropped after initialisation
                    self.segments.dropped_data.insert(seg_idx as u32);
                }
                DataMode::Passive => {
                    // Passive data segments are used with memory.init instruction
                }
            }
        }

        Ok(())
    }

    /// Evaluate a constant expression (used for data/element segment offsets and global init)
    fn evaluate_const_expr(&self, instructions: &[Instruction], resources: &Resources) -> Result<Value, RuntimeError> {
        // Constant expressions are limited to a small set of instructions
        // They must end with an End instruction
        if instructions.is_empty() {
            return Err(RuntimeError::InvalidConstExpr("empty constant expression".to_string()));
        }

        // Check that the last instruction is End
        match instructions.last() {
            Some(inst) if matches!(inst.kind, InstructionKind::End) => {}
            _ => {
                return Err(RuntimeError::InvalidConstExpr(
                    "constant expression must end with end instruction".to_string(),
                ));
            }
        }

        // For now, handle the common cases (single instruction + End)
        if instructions.len() == 2 {
            match &instructions[0].kind {
                InstructionKind::I32Const { value } => Ok(Value::I32(*value)),
                InstructionKind::I64Const { value } => Ok(Value::I64(*value)),
                InstructionKind::F32Const { value } => Ok(Value::F32(*value)),
                InstructionKind::F64Const { value } => Ok(Value::F64(*value)),
                InstructionKind::GlobalGet { global_idx } => self.get_global(*global_idx, resources),
                InstructionKind::RefNull { ref_type } => match ref_type {
                    ValueType::FuncRef => Ok(Value::FuncRef(None)),
                    ValueType::ExternRef => Ok(Value::ExternRef(None)),
                    _ => Err(RuntimeError::InvalidConstExpr(format!(
                        "invalid reference type for ref.null: {:?}",
                        ref_type
                    ))),
                },
                InstructionKind::Simd(SimdOp::V128Const { value }) => Ok(Value::V128(*value)),
                InstructionKind::RefFunc { func_idx } => {
                    // Validate function exists
                    let total_functions = self.module.imports.function_count() + self.module.functions.functions.len();
                    if (*func_idx as usize) >= total_functions {
                        return Err(RuntimeError::FunctionIndexOutOfBounds(*func_idx));
                    }
                    // Map local func_idx to global FuncAddr
                    let func_addr = self
                        .function_addresses
                        .get(*func_idx as usize)
                        .copied()
                        .ok_or(RuntimeError::FunctionIndexOutOfBounds(*func_idx))?;
                    Ok(Value::FuncRef(Some(func_addr)))
                }
                _ => Err(RuntimeError::InvalidConstExpr(format!(
                    "unsupported instruction in constant expression: {:?}",
                    instructions[0].kind
                ))),
            }
        } else if instructions.len() == 1 && matches!(instructions[0].kind, InstructionKind::End) {
            // Just an End instruction - this shouldn't happen in valid WebAssembly
            Err(RuntimeError::InvalidConstExpr(
                "constant expression cannot be just end".to_string(),
            ))
        } else {
            // TODO: Support more complex constant expressions (e.g., i32.add with two consts)
            Err(RuntimeError::InvalidConstExpr(format!(
                "unsupported constant expression with {} instructions",
                instructions.len()
            )))
        }
    }
}

#[cfg(test)]
impl Instance {
    /// Create a standalone structured executor with its own resources (test-only).
    ///
    /// Allocates memories, tables, and globals from the module definition into
    /// fresh resources. Data sections are initialised; globals remain at their
    /// defaults for tests to configure explicitly.
    pub(crate) fn new_test_executor(module: Arc<Module>) -> Result<(Executor, Resources, SegmentState), RuntimeError> {
        use super::imports::default_value_for_type;
        use super::memory::Memory;
        use super::table::Table;

        let mut resources = Resources::new();

        // Allocate memories from module definition
        let mut memory_addresses = Vec::new();
        if !module.memory.memory.is_empty() {
            if module.memory.memory.len() > 1 {
                return Err(RuntimeError::MemoryError("multiple memories not supported".to_string()));
            }
            let mem_def = &module.memory.memory[0];
            let memory = Memory::new(mem_def.limits.min, mem_def.limits.max)?;
            let addr = MemoryAddr(resources.memories.len());
            resources.memories.push(memory);
            memory_addresses.push(addr);
        }

        // Allocate tables (imported + local)
        let mut table_addresses = Vec::new();
        for import in &module.imports.imports {
            if let ExternalKind::Table(table_type) = &import.external_kind {
                let table = Table::new(table_type.ref_type, table_type.limits)?;
                let addr = TableAddr(resources.tables.len());
                resources.tables.push(table);
                table_addresses.push(addr);
            }
        }
        for table_type in &module.table.tables {
            let table = Table::new(table_type.ref_type, table_type.limits)?;
            let addr = TableAddr(resources.tables.len());
            resources.tables.push(table);
            table_addresses.push(addr);
        }

        // Allocate globals (imported + local)
        let mut global_addresses = Vec::new();
        for import in &module.imports.imports {
            if let ExternalKind::Global(global_type) = &import.external_kind {
                let initial = default_value_for_type(global_type.value_type);
                let addr = GlobalAddr(resources.globals.len());
                resources.globals.push(initial);
                global_addresses.push(addr);
            }
        }
        for global in &module.globals.globals {
            let default = default_value_for_type(global.global_type.value_type);
            let addr = GlobalAddr(resources.globals.len());
            resources.globals.push(default);
            global_addresses.push(addr);
        }

        let mut instance = Self::new_unlinked(
            module,
            memory_addresses,
            table_addresses,
            global_addresses,
            EngineKind::Structured,
        )?;

        // Initialise data sections (writes module data into memory)
        instance.initialise_data_sections(&mut resources)?;

        let Engine::Structured(executor) = instance.engine else {
            unreachable!("test instances use the structured engine");
        };
        Ok((executor, resources, instance.segments))
    }
}
