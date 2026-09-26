//! WebAssembly Store - manages runtime instances and provides global function addressing
//!
//! The Store is the central runtime component that owns all module instances and provides
//! a global address space for functions. This enables proper cross-module function references
//! as required by the WebAssembly specification.
//!
//! # Architecture
//!
//! ```text
//! +-------------------------------------------------------------+
//! |                        Store<T>                             |
//! |  +--------------------------------------------------------+ |
//! |  | data: T  (embedder user data, accessible via Caller)   | |
//! |  +--------------------------------------------------------+ |
//! |  +--------------------------------------------------------+ |
//! |  | Function Space (FuncAddr -> FunctionInstance<T>)        | |
//! |  |  [0]: Host { print }                                   | |
//! |  |  [1]: Host { print_i32 }                               | |
//! |  |  [2]: Wasm { instance: 0, func: 0 }                    | |
//! |  |  [3]: Wasm { instance: 1, func: 0 }                    | |
//! |  +--------------------------------------------------------+ |
//! |  +--------------------------------------------------------+ |
//! |  | Resources (owned directly)                             | |
//! |  |  memories: [Memory]                                    | |
//! |  |  tables:   [Table]                                     | |
//! |  |  globals:  [Value]                                     | |
//! |  +--------------------------------------------------------+ |
//! |  +--------------------------------------------------------+ |
//! |  | Instance Registry                                      | |
//! |  |  [0]: module_a  (holds address maps + element/data)    | |
//! |  |  [1]: module_b  (imports func from module_a)           | |
//! |  +--------------------------------------------------------+ |
//! +-------------------------------------------------------------+
//! ```
//!
//! # Key Design Decisions
//!
//! - **FuncAddr is globally unique**: Allocated by Store, works across module boundaries
//! - **Store owns all resources**: Memories, tables, globals live in Store directly
//! - **Borrow splitting**: During execution, Store's fields (functions, instances, resources,
//!   data) are borrowed independently, allowing the compiler to prove safety
//! - **Caller context**: Host functions receive a `Caller<T>` providing access to the calling
//!   instance's memory and the embedder's user data via `data()`/`data_mut()`
//! - **Resumable execution**: Cross-module calls return `NeedsExternalCall`, Store handles delegation
//!
//! # Cross-Module Execution Flow
//!
//! When execution encounters a function from another module (via import or call_indirect):
//!
//! 1. Instance returns `NeedsExternalCall` with the target FuncAddr
//! 2. Store.execute() pushes the calling instance onto a call stack
//! 3. Store dispatches to the target (wasm instance or host function)
//! 4. When target completes, Store pops the call stack and resumes the caller
//! 5. This continues until the call stack is empty
//!
//! The call stack enables arbitrarily deep cross-module chains (A -> B -> C -> ...)
//! where each module can perform computation before/after its external calls.

use super::imports::{global_value_type, is_global_mutable};
use super::{EngineKind, ExecutionOutcome, Instance, Memory, RuntimeError, Table, Value};
use crate::parser::module::{ExportIndex, ExternalKind, FunctionType, Limits};
use std::sync::Arc;

/// Type alias for host function implementations
///
/// Host functions receive a `Caller` providing access to the calling instance's
/// linear memory and the embedder's user data, plus the function arguments as
/// `Vec<Value>`.
pub type HostFunc<T = ()> = Box<dyn Fn(&mut Caller<'_, T>, Vec<Value>) -> Result<Vec<Value>, RuntimeError>>;

/// Context passed to host functions during execution
///
/// Provides access to the calling instance's linear memory and the embedder's
/// user data stored in `Store<T>`. Constructed by Store for each host function
/// call and destroyed when the call returns.
pub struct Caller<'a, T = ()> {
    memory: Option<&'a mut Memory>,
    data: &'a mut T,
}

impl<'a, T> Caller<'a, T> {
    /// Access the calling instance's linear memory (read-only)
    pub fn memory(&self) -> Option<&Memory> {
        self.memory.as_deref()
    }

    /// Access the calling instance's linear memory (mutable)
    pub fn memory_mut(&mut self) -> Option<&mut Memory> {
        self.memory.as_deref_mut()
    }

    /// Access the embedder's user data (read-only)
    pub fn data(&self) -> &T {
        self.data
    }

    /// Access the embedder's user data (mutable)
    pub fn data_mut(&mut self) -> &mut T {
        self.data
    }

    /// Create a Caller for unit tests
    #[cfg(test)]
    pub fn for_test(memory: Option<&'a mut Memory>, data: &'a mut T) -> Self {
        Caller { memory, data }
    }
}

/// Runtime resources owned directly by the Store
///
/// Grouped as a struct to enable field-level borrow splitting: the compiler can
/// prove that `&mut resources`, `&self.functions`, and `&mut self.data` are
/// disjoint borrows.
#[derive(Default)]
pub struct Resources {
    /// All memory instances, indexed by MemoryAddr
    pub(super) memories: Vec<Memory>,
    /// All table instances, indexed by TableAddr
    pub(super) tables: Vec<Table>,
    /// All global values, indexed by GlobalAddr
    pub(super) globals: Vec<Value>,
}

impl Resources {
    pub fn new() -> Self {
        Self::default()
    }
}

/// The next action in the cross-module execution loop.
///
/// Alternates between calling a new function and resuming a suspended instance
/// with the results of a completed call.
enum PendingAction {
    Call(FuncAddr, Vec<Value>),
    Resume(usize, Vec<Value>),
}

/// Global function address - index into the Store's function space
///
/// FuncAddr provides stable, globally-unique identifiers for functions that work
/// across module boundaries, enabling proper funcref semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncAddr(pub(crate) usize);

/// Global memory address - index into the Store's memory registry
///
/// MemoryAddr provides stable, globally-unique identifiers for memory instances
/// that can be shared across module boundaries for memory imports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryAddr(pub(crate) usize);

/// Global table address - index into the Store's table registry
///
/// TableAddr provides stable, globally-unique identifiers for table instances
/// that can be shared across module boundaries for table imports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableAddr(pub(crate) usize);

/// Global address - index into the Store's global registry
///
/// GlobalAddr provides stable, globally-unique identifiers for global instances
/// that can be shared across module boundaries for mutable global imports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalAddr(pub(crate) usize);

/// A function instance in the Store
///
/// Functions can either be WebAssembly functions (executed by an instance)
/// or host functions (native Rust functions provided for imports).
pub enum FunctionInstance<T = ()> {
    /// WebAssembly function - reference to function within an instance
    Wasm {
        /// Index of the instance in Store.instances
        instance_id: usize,
        /// Function index within the instance's function space (includes imports)
        func_idx: u32,
        /// Cached function type for quick access
        func_type: FunctionType,
    },
    /// Host function - native Rust function
    Host {
        /// The host function implementation
        func: HostFunc<T>,
        /// Function type signature
        func_type: FunctionType,
    },
}

/// Validate that actual limits are compatible with expected limits for an import.
///
/// The imported entity's minimum must be >= the declared minimum, and if a maximum
/// is declared, the imported entity must also have a maximum that is <= the declared one.
fn validate_import_limits(
    actual_min: u32,
    actual_max: Option<u32>,
    expected: &Limits,
    module_name: &str,
    field_name: &str,
) -> Result<(), RuntimeError> {
    if actual_min < expected.min {
        return Err(RuntimeError::IncompatibleImportType(format!(
            "{}.{}",
            module_name, field_name
        )));
    }
    if let Some(expected_max) = expected.max {
        match actual_max {
            Some(actual_max) if actual_max <= expected_max => {}
            _ => {
                return Err(RuntimeError::IncompatibleImportType(format!(
                    "{}.{}",
                    module_name, field_name
                )));
            }
        }
    }
    Ok(())
}

/// The WebAssembly Store - owns all instances and resources
///
/// The Store manages the lifetime of all module instances and provides a global
/// function address space. All function calls are routed through the Store, which
/// delegates to the appropriate instance.
///
/// The generic parameter `T` holds embedder-defined user data accessible from
/// host functions via [`Caller::data()`] / [`Caller::data_mut()`]. Use
/// [`Store::new()`] for `Store<()>` (no user data) or [`Store::with_data()`]
/// to provide a context value.
pub struct Store<T = ()> {
    /// All function instances, indexed by FuncAddr
    pub(super) functions: Vec<FunctionInstance<T>>,

    /// All module instances owned by this store
    instances: Vec<Instance>,

    /// All runtime resources (memories, tables, globals) owned directly
    pub(super) resources: Resources,

    /// Embedder-defined user data, accessible via `Caller<T>` in host functions
    data: T,

    /// Interpreter used by instances created in this store
    engine: EngineKind,
}

impl Default for Store<()> {
    fn default() -> Self {
        Store {
            functions: Vec::new(),
            instances: Vec::new(),
            resources: Resources::new(),
            data: (),
            engine: EngineKind::default(),
        }
    }
}

impl Store<()> {
    /// Create a new empty Store with no user data
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T> Store<T> {
    /// Create a new Store with embedder-defined user data
    ///
    /// The data is accessible from host functions via `caller.data()` and
    /// `caller.data_mut()`, and from the store via `store.data()` and
    /// `store.data_mut()`.
    pub fn with_data(data: T) -> Self {
        Store {
            functions: Vec::new(),
            instances: Vec::new(),
            resources: Resources::new(),
            data,
            engine: EngineKind::default(),
        }
    }

    /// Select the interpreter for instances created after this call.
    ///
    /// Existing instances keep the engine they were created with.
    pub fn set_engine(&mut self, engine: EngineKind) {
        self.engine = engine;
    }

    /// Access the embedder's user data (read-only)
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Access the embedder's user data (mutable)
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Allocate a new function address and register a function instance
    ///
    /// Returns the FuncAddr that can be used to call this function.
    pub fn allocate_function(&mut self, func: FunctionInstance<T>) -> FuncAddr {
        let addr = FuncAddr(self.functions.len());
        self.functions.push(func);
        addr
    }

    /// Register a typed closure as a host function
    ///
    /// The function type is inferred from the closure signature. Supported
    /// parameter types are `i32`, `i64`, `f32`, `f64` (up to 8 parameters).
    /// Return type can be `()`, a single `WasmType`, or `Result<R, RuntimeError>`.
    ///
    /// ```ignore
    /// let add = store.wrap(|a: i32, b: i32| -> i32 { a + b });
    /// ```
    pub fn wrap<Params, Returns, F>(&mut self, func: F) -> FuncAddr
    where
        F: super::host::IntoHostFunc<T, Params, Returns>,
    {
        let (host_func, func_type) = func.into_host_func();
        self.allocate_function(FunctionInstance::Host {
            func: host_func,
            func_type,
        })
    }

    /// Register a typed closure as a host function with caller access
    ///
    /// Like [`wrap`](Self::wrap), but the closure receives `&mut Caller<T>` as
    /// its first argument, providing access to the calling instance's memory
    /// and the embedder's user data.
    ///
    /// ```ignore
    /// let log = store.wrap_with_caller(|caller: &mut Caller<MyState>, val: i32| {
    ///     caller.data_mut().log.push(val);
    /// });
    /// ```
    pub fn wrap_with_caller<Params, Returns, F>(&mut self, func: F) -> FuncAddr
    where
        F: super::host::IntoHostFuncWithCaller<T, Params, Returns>,
    {
        let (host_func, func_type) = func.into_host_func();
        self.allocate_function(FunctionInstance::Host {
            func: host_func,
            func_type,
        })
    }

    /// Allocate a new memory in the Store
    ///
    /// Returns the MemoryAddr that can be used to reference this memory.
    pub fn allocate_memory(&mut self, memory: Memory) -> MemoryAddr {
        let addr = MemoryAddr(self.resources.memories.len());
        self.resources.memories.push(memory);
        addr
    }

    /// Allocate a new table in the Store
    ///
    /// Returns the TableAddr that can be used to reference this table.
    pub fn allocate_table(&mut self, table: Table) -> TableAddr {
        let addr = TableAddr(self.resources.tables.len());
        self.resources.tables.push(table);
        addr
    }

    /// Get a reference to a memory by its address.
    ///
    /// Returns `None` if `addr` does not correspond to an allocated memory.
    pub fn get_memory(&self, addr: MemoryAddr) -> Option<&Memory> {
        self.resources.memories.get(addr.0)
    }

    /// Get a mutable reference to a memory by its address.
    ///
    /// Returns `None` if `addr` does not correspond to an allocated memory.
    pub fn get_memory_mut(&mut self, addr: MemoryAddr) -> Option<&mut Memory> {
        self.resources.memories.get_mut(addr.0)
    }

    /// Get a reference to a table by its address.
    ///
    /// Returns `None` if `addr` does not correspond to an allocated table.
    pub fn get_table(&self, addr: TableAddr) -> Option<&Table> {
        self.resources.tables.get(addr.0)
    }

    /// Allocate a new global in the Store
    ///
    /// Returns the GlobalAddr that can be used to reference this global.
    pub fn allocate_global(&mut self, value: Value) -> GlobalAddr {
        let addr = GlobalAddr(self.resources.globals.len());
        self.resources.globals.push(value);
        addr
    }

    /// Get a global value by its address.
    ///
    /// Returns `None` if `addr` does not correspond to an allocated global.
    pub fn get_global(&self, addr: GlobalAddr) -> Option<Value> {
        self.resources.globals.get(addr.0).copied()
    }

    /// Set a global value by its address.
    ///
    /// Returns `None` if `addr` does not correspond to an allocated global.
    /// Does not check mutability — callers are responsible for that.
    pub fn set_global(&mut self, addr: GlobalAddr, value: Value) -> Option<()> {
        let slot = self.resources.globals.get_mut(addr.0)?;
        *slot = value;
        Some(())
    }

    /// Create and register a new instance in the Store
    ///
    /// Resolves imports, creates local resources, links functions, and initialises
    /// the instance. Returns the instance ID.
    pub fn create_instance(
        &mut self,
        module: Arc<crate::parser::module::Module>,
        imports: Option<&super::ImportObject>,
    ) -> Result<usize, RuntimeError> {
        let instance_id = self.instances.len();

        let (memory_addresses, table_addresses, global_addresses) = self.resolve_resources(&module, imports)?;

        let mut instance = Instance::new_unlinked(
            Arc::clone(&module),
            memory_addresses,
            table_addresses,
            global_addresses,
            self.engine,
        )?;

        let function_addresses = self.resolve_functions(&module, imports, instance_id)?;

        // Link functions, then push instance before propagating errors.
        // The spec requires side effects on shared tables/memories to persist
        // even when instantiation fails (e.g., OOB element segments).
        let link_result = instance.link_functions(function_addresses, &mut self.resources);
        self.instances.push(instance);
        link_result?;

        self.execute_start_function(instance_id)?;

        Ok(instance_id)
    }

    /// Resolve all resource imports (memories, tables, globals) and create local resources.
    ///
    /// Returns the address mappings for (memories, tables, globals).
    #[allow(clippy::type_complexity)]
    fn resolve_resources(
        &mut self,
        module: &crate::parser::module::Module,
        imports: Option<&super::ImportObject>,
    ) -> Result<(Vec<MemoryAddr>, Vec<TableAddr>, Vec<GlobalAddr>), RuntimeError> {
        let memory_addresses = self.resolve_memories(module, imports)?;
        let table_addresses = self.resolve_tables(module, imports)?;
        let global_addresses = self.resolve_globals(module, imports)?;
        Ok((memory_addresses, table_addresses, global_addresses))
    }

    /// Resolve memory imports and create local memories.
    ///
    /// Imported memories come first in the index space, followed by locally defined ones.
    fn resolve_memories(
        &mut self,
        module: &crate::parser::module::Module,
        imports: Option<&super::ImportObject>,
    ) -> Result<Vec<MemoryAddr>, RuntimeError> {
        let mut addresses = Vec::new();

        for import in &module.imports.imports {
            if let ExternalKind::Memory(expected_limits) = &import.external_kind {
                let import_obj = imports.ok_or_else(|| {
                    RuntimeError::MemoryError(format!("memory import {}.{} not found", import.module, import.name))
                })?;

                let mem_addr = import_obj.get_memory(&import.module, &import.name)?;
                let mem = self.resources.memories.get(mem_addr.0).ok_or_else(|| {
                    RuntimeError::MemoryError(format!(
                        "memory import {}.{} not found in store",
                        import.module, import.name
                    ))
                })?;

                validate_import_limits(
                    mem.size(),
                    mem.max_pages(),
                    expected_limits,
                    &import.module,
                    &import.name,
                )?;

                addresses.push(mem_addr);
            }
        }

        for mem_def in &module.memory.memory {
            let memory = Memory::new(mem_def.limits.min, mem_def.limits.max)?;
            let addr = self.allocate_memory(memory);
            addresses.push(addr);
        }

        Ok(addresses)
    }

    /// Resolve table imports and create local tables.
    ///
    /// Unlike memories, tables can have both imports and locals (sequential in the index space).
    fn resolve_tables(
        &mut self,
        module: &crate::parser::module::Module,
        imports: Option<&super::ImportObject>,
    ) -> Result<Vec<TableAddr>, RuntimeError> {
        let mut addresses = Vec::new();

        for import in &module.imports.imports {
            if let ExternalKind::Table(expected_table_type) = &import.external_kind {
                let import_obj = imports
                    .ok_or_else(|| RuntimeError::UnknownFunction(format!("{}.{}", import.module, import.name)))?;
                let table_addr = import_obj.get_table(&import.module, &import.name)?;
                let tbl = self.resources.tables.get(table_addr.0).ok_or_else(|| {
                    RuntimeError::Trap(format!(
                        "table import {}.{} not found in store",
                        import.module, import.name
                    ))
                })?;

                if tbl.ref_type() != expected_table_type.ref_type {
                    return Err(RuntimeError::IncompatibleImportType(format!(
                        "{}.{}",
                        import.module, import.name
                    )));
                }
                validate_import_limits(
                    tbl.size(),
                    tbl.limits().max,
                    &expected_table_type.limits,
                    &import.module,
                    &import.name,
                )?;

                addresses.push(table_addr);
            }
        }

        for table_type in &module.table.tables {
            let table = Table::new(table_type.ref_type, table_type.limits)?;
            let addr = self.allocate_table(table);
            addresses.push(addr);
        }

        Ok(addresses)
    }

    /// Resolve global imports and create local globals.
    ///
    /// Imported globals share the same GlobalAddr so mutations are visible across modules.
    /// Local globals are allocated with default values; init expressions are evaluated later.
    fn resolve_globals(
        &mut self,
        module: &crate::parser::module::Module,
        imports: Option<&super::ImportObject>,
    ) -> Result<Vec<GlobalAddr>, RuntimeError> {
        let mut addresses = Vec::new();

        for import in &module.imports.imports {
            if let ExternalKind::Global(global_type) = &import.external_kind {
                let import_obj = imports
                    .ok_or_else(|| RuntimeError::UnknownFunction(format!("{}.{}", import.module, import.name)))?;
                let global_addr = import_obj.get_global_addr(&import.module, &import.name)?;
                import_obj.validate_global(
                    &import.module,
                    &import.name,
                    global_type.value_type,
                    global_type.mutable,
                )?;

                // Verify the global exists in the store
                if self.resources.globals.get(global_addr.0).is_none() {
                    return Err(RuntimeError::Trap(format!(
                        "global import {}.{} not found in store",
                        import.module, import.name
                    )));
                }

                addresses.push(global_addr);
            }
        }

        for global in &module.globals.globals {
            let default_value = super::imports::default_value_for_type(global.global_type.value_type);
            let addr = self.allocate_global(default_value);
            addresses.push(addr);
        }

        Ok(addresses)
    }

    /// Resolve function imports and allocate local function addresses.
    ///
    /// Imported functions are validated against their expected type signatures.
    /// Local functions are registered as Wasm function instances in the Store.
    fn resolve_functions(
        &mut self,
        module: &crate::parser::module::Module,
        imports: Option<&super::ImportObject>,
        instance_id: usize,
    ) -> Result<Vec<FuncAddr>, RuntimeError> {
        let num_imported_funcs = module.imports.function_count();
        let mut addresses = Vec::new();

        for import in &module.imports.imports {
            if let ExternalKind::Function(type_idx) = &import.external_kind {
                let import_obj = imports.ok_or_else(|| {
                    RuntimeError::UnknownFunction(format!("import {}.{} not found", import.module, import.name))
                })?;

                let addr = import_obj.get_function(&import.module, &import.name)?;
                let expected_type = module.types.get(*type_idx).ok_or(RuntimeError::InvalidFunctionType)?;
                let actual_type = self.get_function_type(addr)?;

                if expected_type != actual_type {
                    return Err(RuntimeError::ImportTypeMismatch {
                        module: import.module.clone(),
                        name: import.name.clone(),
                        expected: format!("{:?}", expected_type),
                        actual: format!("{:?}", actual_type),
                    });
                }

                addresses.push(addr);
            }
        }

        for (local_idx, func) in module.functions.functions.iter().enumerate() {
            let func_idx = (num_imported_funcs + local_idx) as u32;
            let func_type = module
                .types
                .get(func.ftype_index)
                .ok_or(RuntimeError::InvalidFunctionType)?
                .clone();

            let addr = self.allocate_function(FunctionInstance::Wasm {
                instance_id,
                func_idx,
                func_type,
            });
            addresses.push(addr);
        }

        Ok(addresses)
    }

    /// Execute the start function for a newly instantiated module.
    fn execute_start_function(&mut self, instance_id: usize) -> Result<(), RuntimeError> {
        if let Some(func_addr) = self.instances[instance_id].get_start_function_addr()? {
            self.execute_with_caller(func_addr, vec![], Some(instance_id))?;
        }
        Ok(())
    }

    /// Get a reference to an instance by ID
    pub fn get_instance(&self, instance_id: usize) -> Option<&Instance> {
        self.instances.get(instance_id)
    }

    /// Get an exported global value by instance ID and export name
    pub fn get_global_export(&self, instance_id: usize, name: &str) -> Result<Value, RuntimeError> {
        let instance = self
            .instances
            .get(instance_id)
            .ok_or_else(|| RuntimeError::Trap(format!("instance {instance_id} not found")))?;
        instance.get_global_export(name, &self.resources)
    }

    /// Get a mutable reference to an instance by ID
    pub fn get_instance_mut(&mut self, instance_id: usize) -> Option<&mut Instance> {
        self.instances.get_mut(instance_id)
    }

    /// Get the function type for a FuncAddr
    ///
    /// # Errors
    /// - Returns error if FuncAddr is invalid
    pub fn get_function_type(&self, addr: FuncAddr) -> Result<&FunctionType, RuntimeError> {
        match self.functions.get(addr.0) {
            Some(FunctionInstance::Wasm { func_type, .. }) => Ok(func_type),
            Some(FunctionInstance::Host { func_type, .. }) => Ok(func_type),
            None => Err(RuntimeError::FunctionIndexOutOfBounds(addr.0 as u32)),
        }
    }

    /// Execute a single function (host or wasm) and return its outcome
    ///
    /// Uses two-phase dispatch to satisfy the borrow checker: first determine
    /// the dispatch target (releasing the borrow on self.functions), then execute
    /// with the appropriate borrows on self.instances and self.resources.
    fn execute_one(
        &mut self,
        addr: FuncAddr,
        args: Vec<Value>,
        calling_instance: Option<usize>,
    ) -> Result<(ExecutionOutcome, Option<usize>), RuntimeError> {
        // Phase 1: Determine dispatch target
        let is_wasm = match self.functions.get(addr.0) {
            Some(FunctionInstance::Wasm {
                instance_id, func_idx, ..
            }) => Some((*instance_id, *func_idx)),
            Some(FunctionInstance::Host { .. }) => None,
            None => return Err(RuntimeError::FunctionIndexOutOfBounds(addr.0 as u32)),
        };

        // Phase 2: Execute (borrows released from phase 1)
        if let Some((instance_id, func_idx)) = is_wasm {
            let instance = &mut self.instances[instance_id];
            let resources = &mut self.resources;
            let outcome = instance.invoke_by_index(func_idx, args, resources)?;
            Ok((outcome, Some(instance_id)))
        } else {
            // Host function: construct Caller with calling instance's memory and user data.
            // Split the borrow: extract the memory address as a Copy value first (releasing
            // the borrow on self.instances), then borrow self.resources and self.data as
            // disjoint fields.
            let mem_addr = calling_instance
                .and_then(|id| self.instances.get(id))
                .and_then(|inst| inst.memory_addresses.first().copied());

            let memory = mem_addr.and_then(|addr| self.resources.memories.get_mut(addr.0));
            let mut caller = Caller {
                memory,
                data: &mut self.data,
            };

            match &self.functions[addr.0] {
                FunctionInstance::Host { func, .. } => {
                    let results = func(&mut caller, args)?;
                    Ok((ExecutionOutcome::Complete(results), None))
                }
                _ => unreachable!(),
            }
        }
    }

    /// Execute a function by its address, handling cross-module calls.
    ///
    /// Execution alternates between two actions: calling a new function, or resuming
    /// a suspended instance with results from a completed call. The call stack tracks
    /// instances waiting for results.
    ///
    /// ```text
    /// Call(A) -> A completes -> return results   (simple case)
    /// Call(A) -> A needs B   -> push A, Call(B)
    ///                        -> B completes -> pop A, Resume(A, results)
    ///                                       -> A completes -> return results
    /// ```
    pub fn execute(&mut self, addr: FuncAddr, args: Vec<Value>) -> Result<Vec<Value>, RuntimeError> {
        self.execute_with_caller(addr, args, None)
    }

    /// Run the dispatch loop with an optional initial caller for a host function.
    /// Suspended Wasm callers supply their own context for subsequent calls.
    fn execute_with_caller(
        &mut self,
        addr: FuncAddr,
        args: Vec<Value>,
        calling_instance: Option<usize>,
    ) -> Result<Vec<Value>, RuntimeError> {
        let mut call_stack: Vec<usize> = Vec::new();
        let mut action = PendingAction::Call(addr, args);

        loop {
            let (outcome, source_instance) = match action {
                PendingAction::Call(addr, args) => {
                    self.execute_one(addr, args, call_stack.last().copied().or(calling_instance))?
                }
                PendingAction::Resume(instance_id, results) => {
                    let instance = &mut self.instances[instance_id];
                    let resources = &mut self.resources;
                    let outcome = instance.resume_with_results(results, resources)?;
                    (outcome, Some(instance_id))
                }
            };

            match outcome {
                ExecutionOutcome::Complete(results) => {
                    if let Some(caller_id) = call_stack.pop() {
                        action = PendingAction::Resume(caller_id, results);
                    } else {
                        return Ok(results);
                    }
                }
                ExecutionOutcome::NeedsExternalCall(request) => {
                    if let Some(instance_id) = source_instance {
                        call_stack.push(instance_id);
                    }
                    action = PendingAction::Call(request.func_addr, request.args);
                }
            }
        }
    }

    /// Invoke an exported function by name on a specific instance
    ///
    /// This is the recommended way to invoke functions when cross-module calls may occur.
    ///
    /// `Some(n)` allows up to `n` interpreter operations in this instance;
    /// attempting the next returns `RuntimeError::InstructionBudgetExhausted`.
    /// Counts are engine-dependent: flat execution includes bytecode labels
    /// and function ends. Calls and suspension preserve the remaining budget.
    /// Host work and execution in other instances are not charged.
    /// `None` disables the limit; the budget is cleared after execution.
    pub fn invoke_export(
        &mut self,
        instance_id: usize,
        name: &str,
        args: Vec<Value>,
        instruction_budget: Option<u64>,
    ) -> Result<Vec<Value>, RuntimeError> {
        // Set instruction budget on the instance if specified
        if let Some(instance) = self.instances.get_mut(instance_id) {
            instance.set_instruction_budget(instruction_budget);
        }

        let func_addr = {
            let instance = self
                .get_instance(instance_id)
                .ok_or_else(|| RuntimeError::Trap(format!("instance {instance_id} not found")))?;
            instance.get_function_addr(name)?
        };

        let result = self.execute(func_addr, args);

        // Clear the budget after execution
        if let Some(instance) = self.instances.get_mut(instance_id) {
            instance.set_instruction_budget(None);
        }

        result
    }

    /// Register all exports from an instance as imports under a given module name.
    ///
    /// This implements the `.wast` `(register "name")` directive: every function,
    /// global, memory, and table exported by the instance becomes available for
    /// import under `as_name`.
    pub fn register_exports(
        &self,
        instance_id: usize,
        as_name: &str,
        imports: &mut super::ImportObject,
    ) -> Result<(), RuntimeError> {
        let instance = self
            .get_instance(instance_id)
            .ok_or_else(|| RuntimeError::Trap(format!("instance {instance_id} not found")))?;
        let module = instance.module();

        for export in &module.exports.exports {
            match export.index {
                ExportIndex::Function(_) => {
                    if let Ok(addr) = instance.get_function_addr(&export.name) {
                        imports.add_function(as_name, &export.name, addr);
                    }
                }
                ExportIndex::Global(global_idx) => {
                    if let Ok(addr) = instance.get_global_addr(&export.name) {
                        let mutable = is_global_mutable(module, global_idx).unwrap_or(false);
                        let vtype =
                            global_value_type(module, global_idx).unwrap_or(crate::parser::module::ValueType::I32);
                        imports.add_global(as_name, &export.name, addr, vtype, mutable);
                    }
                }
                ExportIndex::Memory(_) => {
                    if let Ok(addr) = instance.get_memory_addr(&export.name) {
                        imports.add_memory(as_name, &export.name, addr);
                    }
                }
                ExportIndex::Table(_) => {
                    if let Ok(addr) = instance.get_table_addr(&export.name) {
                        imports.add_table(as_name, &export.name, addr);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::instruction::{ByteRange, Instruction, InstructionKind};
    use crate::parser::module::{
        CodeSection, Export, ExportIndex, ExportSection, Function, FunctionBody, FunctionSection, Import,
        ImportSection, Locals, Module, SectionPosition, TypeSection, ValueType,
    };
    use crate::parser::structure_builder::StructureBuilder;
    use crate::parser::structured::StructuredFunction;
    use crate::runtime::ImportObject;
    use std::sync::Arc;

    #[test]
    fn store_new_is_empty() {
        let store = Store::new();
        assert!(store.instances.is_empty());
        assert!(store.functions.is_empty());
        assert!(store.resources.memories.is_empty());
        assert!(store.resources.tables.is_empty());
        assert!(store.resources.globals.is_empty());
    }

    #[test]
    fn store_allocate_function() {
        let mut store = Store::new();
        let addr = store.allocate_function(FunctionInstance::Host {
            func: Box::new(|_caller, _args| Ok(vec![Value::I32(42)])),
            func_type: FunctionType {
                parameters: vec![],
                return_types: vec![ValueType::I32],
            },
        });
        assert_eq!(addr.0, 0);

        let addr2 = store.allocate_function(FunctionInstance::Host {
            func: Box::new(|_caller, _args| Ok(vec![])),
            func_type: FunctionType {
                parameters: vec![],
                return_types: vec![],
            },
        });
        assert_eq!(addr2.0, 1);
    }

    #[test]
    fn store_allocate_global() {
        let mut store = Store::new();
        let addr = store.allocate_global(Value::I32(42));
        assert_eq!(store.get_global(addr), Some(Value::I32(42)));

        store.set_global(addr, Value::I32(99));
        assert_eq!(store.get_global(addr), Some(Value::I32(99)));
    }

    #[test]
    fn store_allocate_memory() {
        let mut store = Store::new();
        let memory = Memory::new(1, None).unwrap();
        let addr = store.allocate_memory(memory);
        assert!(store.get_memory(addr).is_some());
    }

    #[test]
    fn caller_memory_access() {
        let mut memory = Memory::new(1, None).unwrap();
        memory.write_u32(0, 42).unwrap();

        let mut data = ();
        let mut caller = Caller {
            memory: Some(&mut memory),
            data: &mut data,
        };

        // Read via immutable access
        let mem = caller.memory().unwrap();
        assert_eq!(mem.read_u32(0).unwrap(), 42);

        // Write via mutable access
        let mem = caller.memory_mut().unwrap();
        mem.write_u32(0, 99).unwrap();
        assert_eq!(mem.read_u32(0).unwrap(), 99);
    }

    #[test]
    fn caller_no_memory() {
        let mut data = ();
        let mut caller = Caller {
            memory: None,
            data: &mut data,
        };
        assert!(caller.memory().is_none());
        assert!(caller.memory_mut().is_none());
    }

    #[test]
    fn caller_data_access() {
        let mut data = 42u32;
        let mut caller: Caller<'_, u32> = Caller {
            memory: None,
            data: &mut data,
        };
        assert_eq!(*caller.data(), 42);
        *caller.data_mut() = 99;
        assert_eq!(*caller.data(), 99);
    }

    #[test]
    fn store_with_data() {
        let store = Store::with_data(42u32);
        assert_eq!(*store.data(), 42);
    }

    #[test]
    fn store_data_mut() {
        let mut store = Store::with_data(String::from("hello"));
        store.data_mut().push_str(" world");
        assert_eq!(store.data(), "hello world");
    }

    #[test]
    fn host_function_accesses_store_data() {
        let mut store = Store::with_data(0u32);

        // Host function that increments the counter and returns the old value
        let addr = store.allocate_function(FunctionInstance::Host {
            func: Box::new(|caller, _args| {
                let count = *caller.data();
                *caller.data_mut() += 1;
                Ok(vec![Value::I32(count as i32)])
            }),
            func_type: FunctionType {
                parameters: vec![],
                return_types: vec![ValueType::I32],
            },
        });

        // Call it three times via Store::execute
        let r1 = store.execute(addr, vec![]).unwrap();
        assert_eq!(r1, vec![Value::I32(0)]);

        let r2 = store.execute(addr, vec![]).unwrap();
        assert_eq!(r2, vec![Value::I32(1)]);

        let r3 = store.execute(addr, vec![]).unwrap();
        assert_eq!(r3, vec![Value::I32(2)]);

        assert_eq!(*store.data(), 3);
    }

    // === Import type checking tests ===

    /// Create a module that imports a function with the given type signature
    fn module_with_function_import(
        module_name: &str,
        func_name: &str,
        type_idx: u32,
        param_types: Vec<ValueType>,
        return_types: Vec<ValueType>,
    ) -> Module {
        let mut module = Module::new("test");

        module.types = TypeSection {
            types: vec![FunctionType {
                parameters: param_types,
                return_types,
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        module.imports = ImportSection {
            imports: vec![Import {
                module: module_name.to_string(),
                name: func_name.to_string(),
                external_kind: ExternalKind::Function(type_idx),
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        module
    }

    #[test]
    fn test_import_type_match_succeeds() {
        let mut store = Store::new();

        // Register a host function: (i32) -> i32
        let host_func_type = FunctionType {
            parameters: vec![ValueType::I32],
            return_types: vec![ValueType::I32],
        };
        let addr = store.allocate_function(FunctionInstance::Host {
            func: Box::new(|_caller, args| Ok(args)),
            func_type: host_func_type,
        });

        let mut imports = ImportObject::new();
        imports.add_function("env", "add_one", addr);

        // Module imports (i32) -> i32 — should match
        let module = module_with_function_import("env", "add_one", 0, vec![ValueType::I32], vec![ValueType::I32]);
        let result = store.create_instance(Arc::new(module), Some(&imports));
        assert!(result.is_ok(), "Expected instantiation to succeed");
    }

    #[test]
    fn test_import_type_mismatch_parameter() {
        let mut store = Store::new();

        // Register a host function: (i32) -> i32
        let host_func_type = FunctionType {
            parameters: vec![ValueType::I32],
            return_types: vec![ValueType::I32],
        };
        let addr = store.allocate_function(FunctionInstance::Host {
            func: Box::new(|_caller, args| Ok(args)),
            func_type: host_func_type,
        });

        let mut imports = ImportObject::new();
        imports.add_function("env", "my_func", addr);

        // Module expects i64, but host provides i32 — parameter type mismatch
        let module = module_with_function_import("env", "my_func", 0, vec![ValueType::I64], vec![ValueType::I32]);
        let result = store.create_instance(Arc::new(module), Some(&imports));
        assert!(result.is_err(), "Expected instantiation to fail");

        let err = result.unwrap_err();
        match err {
            RuntimeError::ImportTypeMismatch {
                module,
                name,
                expected,
                actual,
            } => {
                assert_eq!(module, "env");
                assert_eq!(name, "my_func");
                assert!(expected.contains("I64"), "Expected type should mention I64");
                assert!(actual.contains("I32"), "Actual type should mention I32");
            }
            _ => panic!("Expected ImportTypeMismatch error, got: {:?}", err),
        }
    }

    #[test]
    fn test_import_type_mismatch_return() {
        let mut store = Store::new();

        // Register a host function: () -> i32
        let host_func_type = FunctionType {
            parameters: vec![],
            return_types: vec![ValueType::I32],
        };
        let addr = store.allocate_function(FunctionInstance::Host {
            func: Box::new(|_caller, _args| Ok(vec![Value::I32(42)])),
            func_type: host_func_type,
        });

        let mut imports = ImportObject::new();
        imports.add_function("env", "get_value", addr);

        // Module expects i64 return, but host returns i32
        let module = module_with_function_import("env", "get_value", 0, vec![], vec![ValueType::I64]);
        let result = store.create_instance(Arc::new(module), Some(&imports));
        assert!(result.is_err(), "Expected instantiation to fail");

        let err = result.unwrap_err();
        assert!(
            matches!(err, RuntimeError::ImportTypeMismatch { .. }),
            "Expected ImportTypeMismatch error, got: {:?}",
            err
        );
    }

    #[test]
    fn test_import_type_mismatch_arity() {
        let mut store = Store::new();

        // Host provides (i32, i32) -> i32
        let host_func_type = FunctionType {
            parameters: vec![ValueType::I32, ValueType::I32],
            return_types: vec![ValueType::I32],
        };
        let addr = store.allocate_function(FunctionInstance::Host {
            func: Box::new(|_caller, _args| Ok(vec![Value::I32(0)])),
            func_type: host_func_type,
        });

        let mut imports = ImportObject::new();
        imports.add_function("env", "binary_op", addr);

        // Module expects (i32) -> i32 — different arity
        let module = module_with_function_import("env", "binary_op", 0, vec![ValueType::I32], vec![ValueType::I32]);
        let result = store.create_instance(Arc::new(module), Some(&imports));
        assert!(result.is_err(), "Expected instantiation to fail");

        let err = result.unwrap_err();
        assert!(
            matches!(err, RuntimeError::ImportTypeMismatch { .. }),
            "Expected ImportTypeMismatch error, got: {:?}",
            err
        );
    }

    // === Cross-module call chain tests ===

    fn make_instruction(kind: InstructionKind) -> Instruction {
        Instruction {
            kind,
            position: ByteRange { offset: 0, length: 1 },
            original_bytes: vec![],
        }
    }

    fn build_structured_function(
        instructions: Vec<InstructionKind>,
        local_count: usize,
        return_types: Vec<ValueType>,
    ) -> StructuredFunction {
        let instrs: Vec<Instruction> = instructions.into_iter().map(make_instruction).collect();
        StructureBuilder::build_function(&instrs, local_count, return_types).expect("Structure building should succeed")
    }

    /// Create a module with a single function that returns a constant i32
    fn module_returning_constant(value: i32, export_name: &str) -> Module {
        let mut module = Module::new("const_module");

        // Type: () -> i32
        module.types = TypeSection {
            types: vec![FunctionType {
                parameters: vec![],
                return_types: vec![ValueType::I32],
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        // Function declaration
        module.functions = FunctionSection {
            functions: vec![Function { ftype_index: 0 }],
            position: SectionPosition { start: 0, end: 0 },
        };

        // Function body: i32.const value, end
        let body = build_structured_function(
            vec![InstructionKind::I32Const { value }, InstructionKind::End],
            0,
            vec![ValueType::I32],
        );
        module.code = CodeSection {
            code: vec![FunctionBody {
                locals: Locals::empty(),
                body,
                position: SectionPosition { start: 0, end: 0 },
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        module.exports = ExportSection {
            exports: vec![Export {
                name: export_name.to_string(),
                index: ExportIndex::Function(0),
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        module
    }

    /// Create a module that imports a () -> i32 function and calls it, returning the result
    fn module_calling_import(import_module: &str, import_name: &str, export_name: &str) -> Module {
        let mut module = Module::new("caller_module");

        // Type: () -> i32 (used for both import and local function)
        module.types = TypeSection {
            types: vec![FunctionType {
                parameters: vec![],
                return_types: vec![ValueType::I32],
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        // Import the function (takes index 0 in function space)
        module.imports = ImportSection {
            imports: vec![Import {
                module: import_module.to_string(),
                name: import_name.to_string(),
                external_kind: ExternalKind::Function(0),
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        // Local function declaration (takes index 1, after the import)
        module.functions = FunctionSection {
            functions: vec![Function { ftype_index: 0 }],
            position: SectionPosition { start: 0, end: 0 },
        };

        // Function body: call 0 (the imported function), end
        // Function index 0 is the import, our local function is index 1
        let body = build_structured_function(
            vec![InstructionKind::Call { func_idx: 0 }, InstructionKind::End],
            0,
            vec![ValueType::I32],
        );
        module.code = CodeSection {
            code: vec![FunctionBody {
                locals: Locals::empty(),
                body,
                position: SectionPosition { start: 0, end: 0 },
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        // Export the local function (index 1, since import is index 0)
        module.exports = ExportSection {
            exports: vec![Export {
                name: export_name.to_string(),
                index: ExportIndex::Function(1),
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        module
    }

    /// Create a module that imports () -> i32, calls it, adds a constant, and returns
    fn module_calling_import_and_add(
        import_module: &str,
        import_name: &str,
        add_value: i32,
        export_name: &str,
    ) -> Module {
        let mut module = Module::new("caller_add_module");

        // Type: () -> i32
        module.types = TypeSection {
            types: vec![FunctionType {
                parameters: vec![],
                return_types: vec![ValueType::I32],
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        // Import the function
        module.imports = ImportSection {
            imports: vec![Import {
                module: import_module.to_string(),
                name: import_name.to_string(),
                external_kind: ExternalKind::Function(0),
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        // Local function declaration
        module.functions = FunctionSection {
            functions: vec![Function { ftype_index: 0 }],
            position: SectionPosition { start: 0, end: 0 },
        };

        // call 0; i32.const add_value; i32.add; end
        let body = build_structured_function(
            vec![
                InstructionKind::Call { func_idx: 0 },
                InstructionKind::I32Const { value: add_value },
                InstructionKind::I32Add,
                InstructionKind::End,
            ],
            0,
            vec![ValueType::I32],
        );
        module.code = CodeSection {
            code: vec![FunctionBody {
                locals: Locals::empty(),
                body,
                position: SectionPosition { start: 0, end: 0 },
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        module.exports = ExportSection {
            exports: vec![Export {
                name: export_name.to_string(),
                index: ExportIndex::Function(1),
            }],
            position: SectionPosition { start: 0, end: 0 },
        };

        module
    }

    #[test]
    fn test_wasm_to_host_call() {
        let mut store = Store::new();

        // Host function that returns 42
        let host_addr = store.allocate_function(FunctionInstance::Host {
            func: Box::new(|_caller, _args| Ok(vec![Value::I32(42)])),
            func_type: FunctionType {
                parameters: vec![],
                return_types: vec![ValueType::I32],
            },
        });

        let mut imports = ImportObject::new();
        imports.add_function("host", "get_value", host_addr);

        let module = module_calling_import("host", "get_value", "call_host");
        let instance_id = store
            .create_instance(Arc::new(module), Some(&imports))
            .expect("Instance creation should succeed");

        let result = store
            .invoke_export(instance_id, "call_host", vec![], None)
            .expect("Execution should succeed");

        assert_eq!(result, vec![Value::I32(42)]);
    }

    #[test]
    fn test_wasm_to_wasm_call() {
        let mut store = Store::new();

        // Module B: exports "get_value" returning 100
        let module_b = module_returning_constant(100, "get_value");
        let instance_b = store.create_instance(Arc::new(module_b), None).unwrap();
        let func_addr_b = store
            .get_instance(instance_b)
            .unwrap()
            .get_function_addr("get_value")
            .unwrap();

        // Module A: imports and calls module_b.get_value
        let mut imports_a = ImportObject::new();
        imports_a.add_function("module_b", "get_value", func_addr_b);
        let module_a = module_calling_import("module_b", "get_value", "call_b");
        let instance_a = store.create_instance(Arc::new(module_a), Some(&imports_a)).unwrap();

        let result = store
            .invoke_export(instance_a, "call_b", vec![], None)
            .expect("Execution should succeed");

        assert_eq!(result, vec![Value::I32(100)]);
    }

    #[test]
    fn test_three_module_chain() {
        // A -> B -> C: tests the cross-module call stack
        let mut store = Store::new();

        // Module C: returns 999
        let module_c = module_returning_constant(999, "get_value");
        let instance_c = store.create_instance(Arc::new(module_c), None).unwrap();
        let func_addr_c = store
            .get_instance(instance_c)
            .unwrap()
            .get_function_addr("get_value")
            .unwrap();

        // Module B: calls C, forwards result
        let mut imports_b = ImportObject::new();
        imports_b.add_function("module_c", "get_value", func_addr_c);
        let module_b = module_calling_import("module_c", "get_value", "call_c");
        let instance_b = store.create_instance(Arc::new(module_b), Some(&imports_b)).unwrap();
        let func_addr_b = store
            .get_instance(instance_b)
            .unwrap()
            .get_function_addr("call_c")
            .unwrap();

        // Module A: calls B
        let mut imports_a = ImportObject::new();
        imports_a.add_function("module_b", "call_c", func_addr_b);
        let module_a = module_calling_import("module_b", "call_c", "call_chain");
        let instance_a = store.create_instance(Arc::new(module_a), Some(&imports_a)).unwrap();

        let result = store
            .invoke_export(instance_a, "call_chain", vec![], None)
            .expect("Three-module chain should succeed");

        assert_eq!(result, vec![Value::I32(999)]);
    }

    #[test]
    fn test_three_module_chain_with_computation() {
        // C returns 10, B calls C and adds 100, A calls B and adds 1000.
        // If the call stack is broken, A never resumes and we get 110 instead of 1110.
        let mut store = Store::new();

        // Module C: returns 10
        let module_c = module_returning_constant(10, "get_value");
        let instance_c = store.create_instance(Arc::new(module_c), None).unwrap();
        let func_addr_c = store
            .get_instance(instance_c)
            .unwrap()
            .get_function_addr("get_value")
            .unwrap();

        // Module B: calls C, adds 100
        let mut imports_b = ImportObject::new();
        imports_b.add_function("module_c", "get_value", func_addr_c);
        let module_b = module_calling_import_and_add("module_c", "get_value", 100, "call_c");
        let instance_b = store.create_instance(Arc::new(module_b), Some(&imports_b)).unwrap();
        let func_addr_b = store
            .get_instance(instance_b)
            .unwrap()
            .get_function_addr("call_c")
            .unwrap();

        // Module A: calls B, adds 1000
        let mut imports_a = ImportObject::new();
        imports_a.add_function("module_b", "call_c", func_addr_b);
        let module_a = module_calling_import_and_add("module_b", "call_c", 1000, "call_chain");
        let instance_a = store.create_instance(Arc::new(module_a), Some(&imports_a)).unwrap();

        // Expected: 10 + 100 + 1000 = 1110
        let result = store
            .invoke_export(instance_a, "call_chain", vec![], None)
            .expect("Chain should succeed");

        assert_eq!(
            result,
            vec![Value::I32(1110)],
            "Expected 1110 (10 + 100 + 1000), got {:?}. If 110, the call stack bug exists.",
            result
        );
    }

    #[test]
    fn test_wasm_host_wasm_chain() {
        // A (wasm) -> B (wasm) -> Host -> resume B -> resume A
        let mut store = Store::new();

        // Host function that returns 777
        let host_addr = store.allocate_function(FunctionInstance::Host {
            func: Box::new(|_caller, _args| Ok(vec![Value::I32(777)])),
            func_type: FunctionType {
                parameters: vec![],
                return_types: vec![ValueType::I32],
            },
        });

        // Module B: calls host function
        let mut imports_b = ImportObject::new();
        imports_b.add_function("host", "get_value", host_addr);
        let module_b = module_calling_import("host", "get_value", "call_host");
        let instance_b = store.create_instance(Arc::new(module_b), Some(&imports_b)).unwrap();
        let func_addr_b = store
            .get_instance(instance_b)
            .unwrap()
            .get_function_addr("call_host")
            .unwrap();

        // Module A: calls B
        let mut imports_a = ImportObject::new();
        imports_a.add_function("module_b", "call_host", func_addr_b);
        let module_a = module_calling_import("module_b", "call_host", "start_chain");
        let instance_a = store.create_instance(Arc::new(module_a), Some(&imports_a)).unwrap();

        // A -> B -> Host chain
        let result = store
            .invoke_export(instance_a, "start_chain", vec![], None)
            .expect("Wasm-Host-Wasm chain should succeed");

        assert_eq!(result, vec![Value::I32(777)]);
    }

    // === Resource allocation tests ===

    #[test]
    fn test_multiple_memory_allocations() {
        let mut store = Store::new();

        let mem1 = Memory::new(1, Some(5)).unwrap();
        let mem2 = Memory::new(2, Some(10)).unwrap();
        let mem3 = Memory::new(3, None).unwrap();

        let addr1 = store.allocate_memory(mem1);
        let addr2 = store.allocate_memory(mem2);
        let addr3 = store.allocate_memory(mem3);

        assert_eq!(addr1, MemoryAddr(0));
        assert_eq!(addr2, MemoryAddr(1));
        assert_eq!(addr3, MemoryAddr(2));

        assert_eq!(store.get_memory(addr1).unwrap().size(), 1);
        assert_eq!(store.get_memory(addr2).unwrap().size(), 2);
        assert_eq!(store.get_memory(addr3).unwrap().size(), 3);
    }

    #[test]
    fn test_invalid_memory_address() {
        let store = Store::new();
        assert!(store.get_memory(MemoryAddr(0)).is_none());
        assert!(store.get_memory(MemoryAddr(999)).is_none());
    }

    #[test]
    fn test_invalid_table_address() {
        let store = Store::new();
        assert!(store.get_table(TableAddr(0)).is_none());
        assert!(store.get_table(TableAddr(999)).is_none());
    }

    #[test]
    fn test_memory_modification_via_store() {
        let mut store = Store::new();

        let memory = Memory::new(1, Some(10)).unwrap();
        let addr = store.allocate_memory(memory);

        // Write through mutable access
        store.get_memory_mut(addr).unwrap().write_u32(0, 0xDEAD_BEEF).unwrap();

        // Read back through immutable access
        let value = store.get_memory(addr).unwrap().read_u32(0).unwrap();
        assert_eq!(value, 0xDEAD_BEEF);
    }

    // === Typed host function (wrap) tests ===

    #[test]
    fn wrap_simple_add() {
        let mut store = Store::new();
        let addr = store.wrap(|a: i32, b: i32| -> i32 { a + b });
        let result = store.execute(addr, vec![Value::I32(3), Value::I32(4)]).unwrap();
        assert_eq!(result, vec![Value::I32(7)]);
    }

    #[test]
    fn wrap_no_args_no_return() {
        let mut store = Store::new();
        let addr = store.wrap(|| {});
        let result = store.execute(addr, vec![]).unwrap();
        assert_eq!(result, vec![]);
    }

    #[test]
    fn wrap_fallible() {
        let mut store = Store::new();
        let addr = store.wrap(|x: i32| -> Result<i32, RuntimeError> {
            if x == 0 {
                Err(RuntimeError::Trap("zero".into()))
            } else {
                Ok(x * 2)
            }
        });
        assert_eq!(store.execute(addr, vec![Value::I32(5)]).unwrap(), vec![Value::I32(10)]);
        assert!(store.execute(addr, vec![Value::I32(0)]).is_err());
    }

    #[test]
    fn wrap_with_caller_accesses_data() {
        let mut store = Store::with_data(0u32);
        let addr = store.wrap_with_caller(|caller: &mut Caller<'_, u32>, x: i32| -> i32 {
            let count = *caller.data();
            *caller.data_mut() += x as u32;
            count as i32
        });

        assert_eq!(store.execute(addr, vec![Value::I32(10)]).unwrap(), vec![Value::I32(0)]);
        assert_eq!(store.execute(addr, vec![Value::I32(5)]).unwrap(), vec![Value::I32(10)]);
        assert_eq!(*store.data(), 15);
    }

    // -- Flat engine dispatch --

    /// Parse WAT and instantiate it in a flat-engine store.
    fn flat_instance(wat: &str, imports: Option<&ImportObject>) -> (Store, usize) {
        let module = crate::wat::parse(wat).expect("WAT parse failed");
        let mut store = Store::new();
        store.set_engine(EngineKind::Flat);
        let id = store
            .create_instance(Arc::new(module), imports)
            .expect("instantiation failed");
        (store, id)
    }

    #[test]
    fn flat_engine_invoke_export() {
        let (mut store, id) = flat_instance(
            "(module (func (export \"add\") (param i32 i32) (result i32)
                (i32.add (local.get 0) (local.get 1))))",
            None,
        );
        let result = store.invoke_export(id, "add", vec![Value::I32(3), Value::I32(4)], None);
        assert_eq!(result.unwrap(), vec![Value::I32(7)]);
    }

    #[test]
    fn flat_engine_budget_boundary_and_reuse() {
        let (mut store, id) = flat_instance("(module (func (export \"run\") (result i32) (i32.const 7)))", None);
        assert!(matches!(
            store.invoke_export(id, "run", vec![], Some(0)),
            Err(RuntimeError::InstructionBudgetExhausted)
        ));
        // The constant and function end each consume one bytecode operation.
        assert_eq!(
            store.invoke_export(id, "run", vec![], Some(2)).unwrap(),
            vec![Value::I32(7)]
        );
        assert!(matches!(
            store.invoke_export(id, "run", vec![], Some(1)),
            Err(RuntimeError::InstructionBudgetExhausted)
        ));
        let addr = store.get_instance(id).unwrap().get_function_addr("run").unwrap();
        // Direct execution does not install a fresh budget.
        assert_eq!(store.execute(addr, vec![]).unwrap(), vec![Value::I32(7)]);
    }

    #[test]
    fn flat_engine_budget_spans_local_calls() {
        let (mut store, id) = flat_instance(
            "(module
                (type $value (func (result i32)))
                (table 1 funcref)
                (elem (i32.const 0) $one)
                (func $one (type $value) (i32.const 1))
                (func (export \"direct\") (result i32)
                    (i32.add (i32.add (call $one) (call $one)) (call $one)))
                (func (export \"indirect\") (result i32)
                    (i32.add (call_indirect (type $value) (i32.const 0)) (i32.const 2))))",
            None,
        );
        assert!(matches!(
            store.invoke_export(id, "direct", vec![], Some(4)),
            Err(RuntimeError::InstructionBudgetExhausted)
        ));
        assert_eq!(
            store.invoke_export(id, "direct", vec![], None).unwrap(),
            vec![Value::I32(3)]
        );
        assert!(matches!(
            store.invoke_export(id, "indirect", vec![], Some(3)),
            Err(RuntimeError::InstructionBudgetExhausted)
        ));
        assert_eq!(
            store.invoke_export(id, "indirect", vec![], None).unwrap(),
            vec![Value::I32(3)]
        );
    }

    #[test]
    fn flat_engine_budget_survives_host_calls() {
        let mut store = Store::with_data(0u32);
        store.set_engine(EngineKind::Flat);
        let tick = store.wrap_with_caller(|caller: &mut Caller<'_, u32>| {
            *caller.data_mut() += 1;
        });
        let mut imports = ImportObject::new();
        imports.add_function("env", "tick", tick);
        let module = crate::wat::parse(
            "(module
                (import \"env\" \"tick\" (func $tick))
                (func (export \"run\") (result i32)
                    (call $tick) (call $tick) (i32.const 7)))",
        )
        .unwrap();
        let id = store.create_instance(Arc::new(module), Some(&imports)).unwrap();
        assert!(matches!(
            store.invoke_export(id, "run", vec![], Some(1)),
            Err(RuntimeError::InstructionBudgetExhausted)
        ));
        // The first call is charged before suspension; the second never runs.
        assert_eq!(*store.data(), 1);
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![Value::I32(7)]
        );
        assert_eq!(*store.data(), 3);
    }

    #[test]
    fn flat_engine_host_call_round_trip() {
        // wasm (flat) -> host -> resume wasm, through Store::execute
        let mut store = Store::new();
        store.set_engine(EngineKind::Flat);
        let double = store.wrap(|x: i32| x * 2);

        let mut imports = ImportObject::new();
        imports.add_function("env", "double", double);

        let module = crate::wat::parse(
            "(module
                (import \"env\" \"double\" (func $double (param i32) (result i32)))
                (func (export \"run\") (param i32) (result i32)
                    (i32.add (call $double (local.get 0)) (i32.const 1))))",
        )
        .expect("WAT parse failed");
        let id = store
            .create_instance(Arc::new(module), Some(&imports))
            .expect("instantiation failed");

        let result = store.invoke_export(id, "run", vec![Value::I32(21)], None);
        assert_eq!(result.unwrap(), vec![Value::I32(43)]);
    }

    #[test]
    fn flat_engine_cross_module_call() {
        // Callee instantiated on the structured engine, caller on the flat
        // engine: the Store's dispatch loop bridges the two.
        let mut store = Store::new();
        store.set_engine(EngineKind::Structured);

        let callee = crate::wat::parse("(module (func (export \"ten\") (result i32) (i32.const 10)))")
            .expect("WAT parse failed");
        let callee_id = store.create_instance(Arc::new(callee), None).unwrap();
        let ten_addr = store.get_instance(callee_id).unwrap().get_function_addr("ten").unwrap();

        store.set_engine(EngineKind::Flat);
        let mut imports = ImportObject::new();
        imports.add_function("m", "ten", ten_addr);
        let caller = crate::wat::parse(
            "(module
                (import \"m\" \"ten\" (func $ten (result i32)))
                (func (export \"run\") (result i32)
                    (i32.add (call $ten) (i32.const 100))))",
        )
        .expect("WAT parse failed");
        let caller_id = store.create_instance(Arc::new(caller), Some(&imports)).unwrap();

        assert!(matches!(
            store.invoke_export(caller_id, "run", vec![], Some(1)),
            Err(RuntimeError::InstructionBudgetExhausted)
        ));
        let result = store.invoke_export(caller_id, "run", vec![], None);
        assert_eq!(result.unwrap(), vec![Value::I32(110)]);
    }

    #[test]
    fn function_calls_use_function_indices_with_mixed_imports() {
        for engine in [EngineKind::Structured, EngineKind::Flat] {
            let mut store = Store::new();
            store.set_engine(engine);
            let provider = crate::wat::parse(
                "(module
                    (memory (export \"memory\") 1)
                    (global (export \"global\") i32 (i32.const 5))
                    (table (export \"table\") 1 funcref)
                    (func (export \"wide\") (param i64) (result i64)
                        (i64.add (local.get 0) (i64.const 2))))",
            )
            .unwrap();
            let provider_id = store.create_instance(Arc::new(provider), None).unwrap();
            let mut imports = ImportObject::new();
            store.register_exports(provider_id, "env", &mut imports).unwrap();
            let increment = store.wrap(|x: i32| x + 1);
            imports.add_function("env", "increment", increment);
            let consumer = crate::wat::parse(
                "(module
                    (type $unary (func (param i32) (result i32)))
                    (import \"env\" \"memory\" (memory 1))
                    (import \"env\" \"increment\" (func $increment (param i32) (result i32)))
                    (import \"env\" \"global\" (global i32))
                    (import \"env\" \"wide\" (func $wide (param i64) (result i64)))
                    (import \"env\" \"table\" (table 1 funcref))
                    (elem (i32.const 0) func $increment)
                    (func (export \"direct\") (result i32 i64)
                        (call $increment (i32.const 10)) (call $wide (i64.const 20)))
                    (func (export \"indirect\") (result i32)
                        (call_indirect (type $unary) (i32.const 30) (i32.const 0))))",
            )
            .unwrap();
            let id = store.create_instance(Arc::new(consumer), Some(&imports)).unwrap();
            assert_eq!(
                store.invoke_export(id, "direct", vec![], None).unwrap(),
                vec![Value::I32(11), Value::I64(22)]
            );
            assert_eq!(
                store.invoke_export(id, "indirect", vec![], None).unwrap(),
                vec![Value::I32(31)]
            );
        }
    }

    #[test]
    fn start_resumes_after_host_and_indirect_foreign_calls() {
        for (engine, provider_engine) in [
            (EngineKind::Structured, EngineKind::Flat),
            (EngineKind::Flat, EngineKind::Structured),
        ] {
            let mut store = Store::with_data(Vec::<u32>::new());
            let advance = store.wrap_with_caller(
                |caller: &mut Caller<'_, Vec<u32>>, delta: i32| -> Result<i32, RuntimeError> {
                    let memory = caller.memory_mut().unwrap();
                    let next = memory.read_u32(0)? + delta as u32;
                    memory.write_u32(0, next)?;
                    caller.data_mut().push(next);
                    Ok(next as i32)
                },
            );
            let mut imports = ImportObject::new();
            imports.add_function("host", "advance", advance);
            store.set_engine(provider_engine);
            let provider = crate::wat::parse(
                "(module
                    (import \"host\" \"advance\" (func $advance (param i32) (result i32)))
                    (memory (export \"memory\") 1)
                    (data (i32.const 0) \"\\64\")
                    (func (export \"work\") (result i32)
                        (i32.add (call $advance (i32.const 20)) (i32.const 1))))",
            )
            .unwrap();
            let provider_id = store.create_instance(Arc::new(provider), Some(&imports)).unwrap();
            store.register_exports(provider_id, "provider", &mut imports).unwrap();
            store.set_engine(engine);
            let consumer = crate::wat::parse(
                "(module
                    (type $value (func (result i32)))
                    (import \"host\" \"advance\" (func $advance (param i32) (result i32)))
                    (import \"provider\" \"work\" (func $work (type $value)))
                    (memory 1)
                    (data (i32.const 0) \"\\05\")
                    (table 1 funcref)
                    (elem (i32.const 0) func $work)
                    (func $boot (local $saved i32)
                        (local.set $saved (call $advance (i32.const 2)))
                        (i32.store (i32.const 4)
                            (i32.add (local.get $saved) (call_indirect (type $value) (i32.const 0))))
                        (drop (call $advance (i32.const 3))))
                    (start $boot)
                    (func (export \"read\") (result i32 i32)
                        (i32.load (i32.const 0)) (i32.load (i32.const 4))))",
            )
            .unwrap();
            let id = store.create_instance(Arc::new(consumer), Some(&imports)).unwrap();
            assert_eq!(
                store.invoke_export(id, "read", vec![], None).unwrap(),
                vec![Value::I32(10), Value::I32(128)]
            );
            assert_eq!(store.data(), &[7, 120, 10]);
            let memory = store
                .get_instance(provider_id)
                .unwrap()
                .get_memory_addr("memory")
                .unwrap();
            assert_eq!(store.get_memory(memory).unwrap().read_u32(0).unwrap(), 120);
        }
    }

    #[test]
    fn imported_host_start_has_initialised_caller_memory() {
        for engine in [EngineKind::Structured, EngineKind::Flat] {
            let mut store = Store::with_data(0_u32);
            store.set_engine(engine);
            let decoy = store.allocate_memory(Memory::new(1, None).unwrap());
            store.get_memory_mut(decoy).unwrap().write_u32(0, 17).unwrap();
            let boot = store.wrap_with_caller(|caller: &mut Caller<'_, u32>| -> Result<(), RuntimeError> {
                let memory = caller.memory_mut().unwrap();
                let initial = memory.read_u32(0)?;
                memory.write_u32(0, 42)?;
                *caller.data_mut() = initial;
                Ok(())
            });
            let mut imports = ImportObject::new();
            imports.add_function("host", "boot", boot);
            let module = crate::wat::parse(
                "(module
                    (import \"host\" \"boot\" (func $boot))
                    (memory (export \"memory\") 1)
                    (data (i32.const 0) \"\\09\")
                    (start $boot))",
            )
            .unwrap();
            let id = store.create_instance(Arc::new(module), Some(&imports)).unwrap();
            let memory = store.get_instance(id).unwrap().get_memory_addr("memory").unwrap();
            assert_eq!(*store.data(), 9);
            assert_eq!(store.get_memory(memory).unwrap().read_u32(0).unwrap(), 42);
            assert_eq!(store.get_memory(decoy).unwrap().read_u32(0).unwrap(), 17);
        }
    }

    #[test]
    fn imported_wasm_start_propagates_traps_and_preserves_side_effects() {
        for (engine, provider_engine) in [
            (EngineKind::Structured, EngineKind::Flat),
            (EngineKind::Flat, EngineKind::Structured),
        ] {
            let mut store = Store::with_data(false);
            let touch = store.wrap_with_caller(|caller: &mut Caller<'_, bool>| -> Result<(), RuntimeError> {
                let memory = caller.memory_mut().unwrap();
                memory.write_u32(0, memory.read_u32(0)? + 1)?;
                if *caller.data() {
                    return Err(RuntimeError::Trap("host start failed".into()));
                }
                Ok(())
            });
            let mut imports = ImportObject::new();
            imports.add_function("host", "touch", touch);
            store.set_engine(provider_engine);
            let provider = crate::wat::parse(
                "(module
                    (import \"host\" \"touch\" (func $touch))
                    (memory (export \"memory\") 1)
                    (global $divisor (export \"divisor\") (mut i32) (i32.const 1))
                    (func (export \"boot\")
                        (call $touch)
                        (drop (i32.div_u (i32.const 1) (global.get $divisor)))
                        (i32.store (i32.const 4)
                            (i32.add (i32.load (i32.const 4)) (i32.const 1)))))",
            )
            .unwrap();
            let provider_id = store.create_instance(Arc::new(provider), Some(&imports)).unwrap();
            store.register_exports(provider_id, "provider", &mut imports).unwrap();
            let provider = store.get_instance(provider_id).unwrap();
            let memory = provider.get_memory_addr("memory").unwrap();
            let divisor = provider.get_global_addr("divisor").unwrap();
            store.set_engine(engine);
            let consumer = Arc::new(
                crate::wat::parse(
                    "(module
                        (import \"provider\" \"boot\" (func $boot))
                        (memory (export \"memory\") 1)
                        (data (i32.const 0) \"\\ff\")
                        (start $boot))",
                )
                .unwrap(),
            );
            let id = store.create_instance(Arc::clone(&consumer), Some(&imports)).unwrap();
            let consumer_memory = store.get_instance(id).unwrap().get_memory_addr("memory").unwrap();
            assert_eq!(store.get_memory(consumer_memory).unwrap().read_u32(0).unwrap(), 255);
            assert_eq!(store.get_memory(memory).unwrap().read_u32(0).unwrap(), 1);
            assert_eq!(store.get_memory(memory).unwrap().read_u32(4).unwrap(), 1);

            store.set_global(divisor, Value::I32(0)).unwrap();
            assert!(matches!(
                store.create_instance(Arc::clone(&consumer), Some(&imports)),
                Err(RuntimeError::DivisionByZero)
            ));
            assert_eq!(store.get_memory(memory).unwrap().read_u32(0).unwrap(), 2);
            assert_eq!(store.get_memory(memory).unwrap().read_u32(4).unwrap(), 1);

            store.set_global(divisor, Value::I32(1)).unwrap();
            *store.data_mut() = true;
            assert!(matches!(
                store.create_instance(Arc::clone(&consumer), Some(&imports)),
                Err(RuntimeError::Trap(_))
            ));
            assert_eq!(store.get_memory(memory).unwrap().read_u32(0).unwrap(), 3);
            assert_eq!(store.get_memory(memory).unwrap().read_u32(4).unwrap(), 1);

            *store.data_mut() = false;
            store.create_instance(consumer, Some(&imports)).unwrap();
            assert_eq!(store.get_memory(memory).unwrap().read_u32(0).unwrap(), 4);
            assert_eq!(store.get_memory(memory).unwrap().read_u32(4).unwrap(), 2);
        }
    }

    #[test]
    fn instance_segment_state_survives_start_and_is_isolated() {
        let module = Arc::new(
            crate::wat::parse(
                "(module
                    (type $value (func (result i32)))
                    (memory 1)
                    (table 1 funcref)
                    (data $boot \"A\")
                    (data $later \"B\")
                    (elem $boot_elem func $seven)
                    (elem $later_elem func $seven)
                    (func $seven (type $value) (i32.const 7))
                    (func $start
                        (memory.init $boot (i32.const 0) (i32.const 0) (i32.const 1))
                        (data.drop $boot)
                        (table.init $boot_elem (i32.const 0) (i32.const 0) (i32.const 1))
                        (elem.drop $boot_elem))
                    (start $start)
                    (func (export \"read\") (result i32 i32)
                        (i32.load8_u (i32.const 0)) (call_indirect (type $value) (i32.const 0)))
                    (func (export \"boot_data\")
                        (memory.init $boot (i32.const 0) (i32.const 0) (i32.const 1)))
                    (func (export \"boot_elem\")
                        (table.init $boot_elem (i32.const 0) (i32.const 0) (i32.const 1)))
                    (func (export \"consume\")
                        (memory.init $later (i32.const 0) (i32.const 0) (i32.const 1))
                        (data.drop $later)
                        (table.init $later_elem (i32.const 0) (i32.const 0) (i32.const 1))
                        (elem.drop $later_elem)))",
            )
            .unwrap(),
        );
        for engine in [EngineKind::Structured, EngineKind::Flat] {
            let mut store = Store::new();
            store.set_engine(engine);
            let first = store.create_instance(Arc::clone(&module), None).unwrap();
            let second = store.create_instance(Arc::clone(&module), None).unwrap();
            assert_eq!(
                store.invoke_export(first, "read", vec![], None).unwrap(),
                vec![Value::I32(65), Value::I32(7)]
            );
            assert!(matches!(
                store.invoke_export(first, "boot_data", vec![], None),
                Err(RuntimeError::MemoryError(_))
            ));
            assert!(matches!(
                store.invoke_export(first, "boot_elem", vec![], None),
                Err(RuntimeError::TableIndexOutOfBounds(_))
            ));
            store.invoke_export(first, "consume", vec![], None).unwrap();
            assert!(matches!(
                store.invoke_export(first, "consume", vec![], None),
                Err(RuntimeError::MemoryError(_))
            ));
            assert_eq!(
                store.invoke_export(second, "read", vec![], None).unwrap(),
                vec![Value::I32(65), Value::I32(7)]
            );
            store.invoke_export(second, "consume", vec![], None).unwrap();
            assert_eq!(
                store.invoke_export(second, "read", vec![], None).unwrap(),
                vec![Value::I32(66), Value::I32(7)]
            );
        }
    }

    #[test]
    fn flat_engine_call_indirect_via_element_segment() {
        // Instance initialisation populates the table before flat execution.
        let (mut store, id) = flat_instance(
            "(module
                (type $binop (func (param i32 i32) (result i32)))
                (table 2 funcref)
                (elem (i32.const 0) $add $sub)
                (func $add (type $binop) (i32.add (local.get 0) (local.get 1)))
                (func $sub (type $binop) (i32.sub (local.get 0) (local.get 1)))
                (func (export \"dispatch\") (param i32 i32 i32) (result i32)
                    (call_indirect (type $binop) (local.get 1) (local.get 2) (local.get 0))))",
            None,
        );

        let args = vec![Value::I32(0), Value::I32(10), Value::I32(4)];
        assert_eq!(
            store.invoke_export(id, "dispatch", args, None).unwrap(),
            vec![Value::I32(14)]
        );
        let args = vec![Value::I32(1), Value::I32(10), Value::I32(4)];
        assert_eq!(
            store.invoke_export(id, "dispatch", args, None).unwrap(),
            vec![Value::I32(6)]
        );
    }

    #[test]
    fn flat_engine_splat_load_reads_one_byte() {
        // A splat's output width does not determine its memory access width.
        let (mut store, id) = flat_instance(
            "(module
                (memory 1)
                (data (i32.const 65535) \"\\ab\")
                (func (export \"splat\") (result v128)
                    (v128.load8_splat offset=65535 (i32.const 0)))
                (func (export \"wide\") (result v128)
                    (v128.load offset=65535 (i32.const 0))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "splat", vec![], None).unwrap(),
            vec![Value::V128([0xab; 16])]
        );
        let err = store.invoke_export(id, "wide", vec![], None).unwrap_err();
        assert!(matches!(err, RuntimeError::MemoryError(_)));
    }

    #[test]
    fn flat_engine_lane_replace_and_extract() {
        // Replacing the last i16 lane truncates 0x18001 to 0x8001 (little-endian bytes [1, 128]).
        // Signed extraction yields -32767; unsigned yields 32769. Other lanes stay unchanged.
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"run\") (result v128 i32 i32)
                    (local $vector v128)
                    (local.set $vector
                        (i16x8.replace_lane 7
                            (v128.const i16x8 1 2 3 4 5 6 7 8)
                            (i32.const 0x18001)))
                    (local.get $vector)
                    (i16x8.extract_lane_s 7 (local.get $vector))
                    (i16x8.extract_lane_u 7 (local.get $vector))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 1, 128]),
                Value::I32(-32767),
                Value::I32(32769),
            ]
        );
    }

    #[test]
    fn flat_engine_i8x16_comparisons_preserve_signedness_and_masks() {
        // High-bit lanes have opposite signed and unsigned ordering.
        // True comparisons use all-ones byte masks, not scalar booleans.
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"run\") (result v128 v128)
                    (local $a v128)
                    (local $b v128)
                    (local.set $a (v128.const i8x16 -128 -1 127 0 1 2 3 4 5 6 7 8 9 10 11 12))
                    (local.set $b (v128.const i8x16 127 0 -128 0 1 2 3 4 5 6 7 8 9 10 11 12))
                    (i8x16.lt_s (local.get $a) (local.get $b))
                    (i8x16.lt_u (local.get $a) (local.get $b))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                Value::V128([0, 0, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ]
        );
    }

    #[test]
    fn flat_engine_i16x8_and_i32x4_comparisons_preserve_signedness_and_masks() {
        // High-boundary lanes distinguish signed ordering from unsigned ordering.
        // True lanes fill their complete lane-width masks.
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"i16\") (result v128 v128)
                    (local $a v128)
                    (local $b v128)
                    (local.set $a (v128.const i16x8 -32768 -1 32767 0 1 2 3 4))
                    (local.set $b (v128.const i16x8 32767 0 -32768 0 1 2 3 4))
                    (i16x8.lt_s (local.get $a) (local.get $b))
                    (i16x8.lt_u (local.get $a) (local.get $b)))
                (func (export \"i32\") (result v128 v128)
                    (local $a v128)
                    (local $b v128)
                    (local.set $a (v128.const i32x4 -2147483648 -1 2147483647 0))
                    (local.set $b (v128.const i32x4 2147483647 0 -2147483648 0))
                    (i32x4.lt_s (local.get $a) (local.get $b))
                    (i32x4.lt_u (local.get $a) (local.get $b))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "i16", vec![], None).unwrap(),
            vec![
                Value::V128([0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                Value::V128([0, 0, 0, 0, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ]
        );
        assert_eq!(
            store.invoke_export(id, "i32", vec![], None).unwrap(),
            vec![
                Value::V128([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0]),
                Value::V128([0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]),
            ]
        );
    }

    #[test]
    fn flat_engine_i64x2_comparisons_use_full_width_lanes() {
        // The high 32 bits determine lane zero, while signedness determines lane one.
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"run\") (result v128 v128)
                    (local $a v128)
                    (local $b v128)
                    (local.set $a (v128.const i64x2 4294967296 -9223372036854775808))
                    (local.set $b (v128.const i64x2 0 0))
                    (i64x2.gt_s (local.get $a) (local.get $b))
                    (i64x2.lt_s (local.get $a) (local.get $b))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0]),
                Value::V128([0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]),
            ]
        );
    }

    #[test]
    fn flat_engine_float_comparisons_handle_nan_zero_and_full_width_masks() {
        // NaNs are unordered on either side, including when compared with themselves.
        // Signed zeros compare equal; the finite 1 < 2 lane distinguishes le from eq.
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"f32\") (result v128 v128 v128)
                    (local $a v128) (local $b v128)
                    (local.set $a (v128.const f32x4 nan 1 0 1))
                    (local.set $b (v128.const f32x4 1 nan -0 2))
                    (f32x4.eq (local.get $a) (local.get $b))
                    (f32x4.ne (local.get $a) (local.get $b))
                    (f32x4.le (local.get $a) (local.get $b)))
                (func (export \"f64\") (result v128 v128 v128)
                    (local $a v128) (local $b v128)
                    (local.set $a (v128.const f64x2 nan -0))
                    (local.set $b (v128.const f64x2 nan 0))
                    (f64x2.eq (local.get $a) (local.get $b))
                    (f64x2.ne (local.get $a) (local.get $b))
                    (f64x2.le (local.get $a) (local.get $b))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "f32", vec![], None).unwrap(),
            vec![
                Value::V128([0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]),
                Value::V128([
                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff
                ]),
                Value::V128([0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]),
            ]
        );
        assert_eq!(
            store.invoke_export(id, "f64", vec![], None).unwrap(),
            vec![
                Value::V128([0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]),
                Value::V128([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0]),
                Value::V128([0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]),
            ]
        );
    }

    #[test]
    fn flat_engine_f32x4_abs_and_neg_preserve_zero_and_nan_bits() {
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"run\") (result v128 v128)
                    (local $vector v128)
                    (local.set $vector
                        (v128.const i32x4 0 -2147483648 2139095041 -8388607))
                    (f32x4.abs (local.get $vector))
                    (f32x4.neg (local.get $vector))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 128, 127, 1, 0, 128, 127]),
                Value::V128([0, 0, 0, 128, 0, 0, 0, 0, 1, 0, 128, 255, 1, 0, 128, 127]),
            ]
        );
    }

    #[test]
    fn flat_engine_f64x2_div_and_sqrt_follow_ieee_edge_cases() {
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"div\") (result v128)
                    (f64x2.div
                        (v128.const f64x2 1 -1)
                        (v128.const f64x2 0 -0)))
                (func (export \"sqrt\") (result f64 f64)
                    (local $root v128)
                    (local.set $root (f64x2.sqrt (v128.const f64x2 -0 -1)))
                    (f64x2.extract_lane 0 (local.get $root))
                    (f64x2.extract_lane 1 (local.get $root))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "div", vec![], None).unwrap(),
            vec![Value::V128([0, 0, 0, 0, 0, 0, 240, 127, 0, 0, 0, 0, 0, 0, 240, 127])]
        );

        let roots = store.invoke_export(id, "sqrt", vec![], None).unwrap();
        assert!(matches!(roots[0], Value::F64(value) if value.to_bits() == (-0.0f64).to_bits()));
        assert!(matches!(roots[1], Value::F64(value) if value.is_nan()));
    }

    #[test]
    fn flat_engine_simd_float_rounding_observes_directions_ties_and_signed_zero() {
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"f32\") (result v128 v128 v128 v128)
                    (local $vector v128)
                    (local.set $vector (v128.const f32x4 -0.5 0.5 1.5 2.5))
                    (f32x4.ceil (local.get $vector))
                    (f32x4.floor (local.get $vector))
                    (f32x4.trunc (local.get $vector))
                    (f32x4.nearest (local.get $vector)))
                (func (export \"f64\") (result v128)
                    (f64x2.nearest (v128.const f64x2 -1.5 -2.5))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "f32", vec![], None).unwrap(),
            vec![
                Value::V128([0, 0, 0, 128, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64]),
                Value::V128([0, 0, 128, 191, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64]),
                Value::V128([0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64]),
                Value::V128([0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 64]),
            ]
        );
        assert_eq!(
            store.invoke_export(id, "f64", vec![], None).unwrap(),
            vec![Value::V128([0, 0, 0, 0, 0, 0, 0, 192, 0, 0, 0, 0, 0, 0, 0, 192])]
        );
    }

    #[test]
    fn flat_engine_i16x8_reductions() {
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"run\") (result i32 i32 i32 i32)
                    (local $vector v128)
                    (local.set $vector (v128.const i16x8 1 -1 -32768 2 3 4 5 -1))
                    (i16x8.all_true (local.get $vector))
                    ;; Negative lanes 1, 2 and 7 set bits 1, 2 and 7: 0x86.
                    (i16x8.bitmask (local.get $vector))
                    ;; Zeroing positive lane 0 clears all_true but leaves the sign-bit mask unchanged.
                    (local.set $vector
                        (i16x8.replace_lane 0 (local.get $vector) (i32.const 0)))
                    (i16x8.all_true (local.get $vector))
                    (i16x8.bitmask (local.get $vector))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![Value::I32(1), Value::I32(0x86), Value::I32(0), Value::I32(0x86),]
        );
    }

    #[test]
    fn flat_engine_i16x8_shifts_normalize_count_and_preserve_signedness() {
        // In 16-bit lanes, counts 16 and 17 wrap to 0 and 1.
        // Signed right shift fills with the sign bit; unsigned fills with zeros.
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"run\") (result v128 v128 v128)
                    (local $vector v128)
                    (local.set $vector
                        (v128.const i16x8 -32768 -1 32767 1 2 0 123 -2))
                    (i16x8.shl (local.get $vector) (i32.const 16))
                    (i16x8.shr_s (local.get $vector) (i32.const 17))
                    (i16x8.shr_u (local.get $vector) (i32.const 17))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([0, 128, 255, 255, 255, 127, 1, 0, 2, 0, 0, 0, 123, 0, 254, 255]),
                Value::V128([0, 192, 255, 255, 255, 63, 0, 0, 1, 0, 0, 0, 61, 0, 255, 255]),
                Value::V128([0, 64, 255, 127, 255, 63, 0, 0, 1, 0, 0, 0, 61, 0, 255, 127]),
            ]
        );
    }

    #[test]
    fn flat_engine_lane_memory_preserves_width_and_neighbours() {
        // The two-byte boundary load replaces only lane 7 of the vector.
        // The unaligned store preserves its adjacent sentinel bytes.
        let (mut store, id) = flat_instance(
            "(module
                (memory 1)
                (data (i32.const 65534) \"\\12\\34\")
                (data (i32.const 30) \"\\11\\ff\\ff\\22\")
                (func (export \"run\") (result v128 i32)
                    (local $vector v128)
                    (local.set $vector
                        (v128.load16_lane offset=65534 7
                            (i32.const 0)
                            (v128.const i16x8 1 2 3 4 5 6 7 8)))
                    (v128.store16_lane offset=30 7
                        (i32.const 1)
                        (local.get $vector))
                    (local.get $vector)
                    (i32.load offset=30 (i32.const 0))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 0x12, 0x34]),
                Value::I32(0x22341211),
            ]
        );
    }

    #[test]
    fn flat_engine_shuffle_and_swizzle_boundaries() {
        // Shuffle selects from two vectors using fixed indices 0..31.
        // Swizzle selects from one vector using runtime indices; indices >=16 yield zero.
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"run\") (result v128 v128)
                    (i8x16.shuffle 0 15 16 31 1 14 17 30 2 13 18 29 3 12 19 28
                        (v128.const i8x16 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
                        (v128.const i8x16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32))
                    (i8x16.swizzle
                        (v128.const i8x16 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
                        (v128.const i8x16 0 15 16 31 1 14 128 255 2 13 18 29 3 12 19 28))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([1, 16, 17, 32, 2, 15, 18, 31, 3, 14, 19, 30, 4, 13, 20, 29]),
                Value::V128([1, 16, 0, 0, 2, 15, 0, 0, 3, 14, 0, 0, 4, 13, 0, 0]),
            ]
        );
    }

    #[test]
    fn flat_engine_integer_unary_ops_wrap_and_popcount_per_lane() {
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"run\") (result v128 v128 v128 v128 v128 v128 v128 v128 v128)
                    (local $i8 v128) (local $i16 v128) (local $i32 v128) (local $i64 v128)
                    (local.set $i8
                        (v128.const i8x16 -128 -3 -1 0 1 2 3 127 -127 64 -64 15 -15 85 -86 -2))
                    (local.set $i16 (v128.const i16x8 -32768 -3 -1 0 1 32767 -2 42))
                    (local.set $i32 (v128.const i32x4 -2147483648 -3 0 123))
                    (local.set $i64 (v128.const i64x2 -9223372036854775808 3))
                    (i8x16.abs (local.get $i8))
                    (i8x16.neg (local.get $i8))
                    (i8x16.popcnt (local.get $i8))
                    (i16x8.abs (local.get $i16))
                    (i16x8.neg (local.get $i16))
                    (i32x4.abs (local.get $i32))
                    (i32x4.neg (local.get $i32))
                    (i64x2.abs (local.get $i64))
                    (i64x2.neg (local.get $i64))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([128, 3, 1, 0, 1, 2, 3, 127, 127, 64, 64, 15, 15, 85, 86, 2]),
                Value::V128([128, 3, 1, 0, 255, 254, 253, 129, 127, 192, 64, 241, 15, 171, 86, 2]),
                Value::V128([1, 7, 8, 0, 1, 1, 2, 7, 2, 1, 2, 4, 5, 4, 4, 7]),
                Value::V128([0, 128, 3, 0, 1, 0, 0, 0, 1, 0, 255, 127, 2, 0, 42, 0]),
                Value::V128([0, 128, 3, 0, 1, 0, 0, 0, 255, 255, 1, 128, 2, 0, 214, 255]),
                Value::V128([0, 0, 0, 128, 3, 0, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0]),
                Value::V128([0, 0, 0, 128, 3, 0, 0, 0, 0, 0, 0, 0, 133, 255, 255, 255]),
                Value::V128([0, 0, 0, 0, 0, 0, 0, 128, 3, 0, 0, 0, 0, 0, 0, 0]),
                Value::V128([0, 0, 0, 0, 0, 0, 0, 128, 253, 255, 255, 255, 255, 255, 255, 255]),
            ]
        );
    }

    #[test]
    fn flat_engine_i16x8_wrapping_arithmetic() {
        let (mut store, id) = flat_instance(
            "(module (func (export \"run\") (result v128 v128 v128)
                (local $a v128) (local $b v128)
                (local.set $a (v128.const i16x8 -1 0 32767 -32768 256 -1 3 10))
                (local.set $b (v128.const i16x8 1 1 1 1 256 2 4 0))
                (i16x8.add (local.get $a) (local.get $b))
                (i16x8.sub (local.get $a) (local.get $b))
                (i16x8.mul (local.get $a) (local.get $b))))",
            None,
        );
        // Carry and borrow do not cross lane boundaries.
        // Products truncate to their 16-bit lanes.
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([0, 0, 1, 0, 0, 128, 1, 128, 0, 2, 1, 0, 7, 0, 10, 0]),
                Value::V128([254, 255, 255, 255, 254, 127, 255, 127, 0, 0, 253, 255, 255, 255, 10, 0]),
                Value::V128([255, 255, 0, 0, 255, 127, 0, 128, 0, 0, 254, 255, 12, 0, 0, 0]),
            ]
        );
    }

    #[test]
    fn flat_engine_i16x8_saturating_arithmetic_preserves_signedness_and_operands() {
        let (mut store, id) = flat_instance(
            "(module (func (export \"run\") (result v128 v128 v128 v128)
                (local $a v128) (local $b v128)
                (local.set $a (v128.const i16x8 -32768 32767 -1 0 100 -100 32760 -32760))
                (local.set $b (v128.const i16x8 -1 1 1 1 200 200 10 -10))
                (i16x8.add_sat_s (local.get $a) (local.get $b))
                (i16x8.add_sat_u (local.get $a) (local.get $b))
                (i16x8.sub_sat_s (local.get $a) (local.get $b))
                (i16x8.sub_sat_u (local.get $a) (local.get $b))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([0, 128, 255, 127, 0, 0, 1, 0, 44, 1, 100, 0, 255, 127, 0, 128]),
                Value::V128([255, 255, 0, 128, 255, 255, 1, 0, 44, 1, 255, 255, 2, 128, 255, 255]),
                Value::V128([
                    1, 128, 254, 127, 254, 255, 255, 255, 156, 255, 212, 254, 238, 127, 18, 128
                ]),
                Value::V128([0, 0, 254, 127, 254, 255, 0, 0, 0, 0, 212, 254, 238, 127, 0, 0]),
            ]
        );
    }

    #[test]
    fn flat_engine_i32x4_minmax_preserves_signedness_and_lanes() {
        let (mut store, id) = flat_instance(
            "(module (func (export \"run\") (result v128 v128 v128 v128)
                (local $a v128) (local $b v128)
                (local.set $a (v128.const i32x4 -2147483648 -1 7 42))
                (local.set $b (v128.const i32x4 1 2147483647 7 9))
                (i32x4.min_s (local.get $a) (local.get $b))
                (i32x4.min_u (local.get $a) (local.get $b))
                (i32x4.max_s (local.get $a) (local.get $b))
                (i32x4.max_u (local.get $a) (local.get $b))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([0, 0, 0, 128, 255, 255, 255, 255, 7, 0, 0, 0, 9, 0, 0, 0]),
                Value::V128([1, 0, 0, 0, 255, 255, 255, 127, 7, 0, 0, 0, 9, 0, 0, 0]),
                Value::V128([1, 0, 0, 0, 255, 255, 255, 127, 7, 0, 0, 0, 42, 0, 0, 0]),
                Value::V128([0, 0, 0, 128, 255, 255, 255, 255, 7, 0, 0, 0, 42, 0, 0, 0]),
            ]
        );
    }

    #[test]
    fn flat_engine_unsigned_average_rounds_up_without_overflow() {
        let (mut store, id) = flat_instance(
            "(module (func (export \"run\") (result v128)
                (i16x8.avgr_u
                    (v128.const i16x8 65535 65534 65535 32768 0 1 2 10)
                    (v128.const i16x8 65535 65535 0 32768 1 2 4 20))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![Value::V128([
                255, 255, 255, 255, 0, 128, 0, 128, 1, 0, 2, 0, 3, 0, 15, 0
            ])]
        );
    }

    #[test]
    fn flat_engine_narrowing_clamps_signed_sources_and_packs_in_order() {
        let (mut store, id) = flat_instance(
            "(module (func (export \"run\") (result v128 v128)
                (local $a v128) (local $b v128)
                (local.set $a (v128.const i16x8 -32768 -129 -128 -1 0 127 128 255))
                (local.set $b (v128.const i16x8 256 32767 1 2 3 4 5 -2))
                (i8x16.narrow_i16x8_s (local.get $a) (local.get $b))
                (i8x16.narrow_i16x8_u (local.get $a) (local.get $b))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([128, 128, 128, 255, 0, 127, 127, 127, 127, 127, 1, 2, 3, 4, 5, 254]),
                Value::V128([0, 0, 0, 0, 0, 127, 128, 255, 255, 255, 1, 2, 3, 4, 5, 0]),
            ]
        );
    }

    #[test]
    fn flat_engine_i32x4_extension_selects_half_and_signedness() {
        let (mut store, id) = flat_instance(
            "(module (func (export \"run\") (result v128 v128 v128 v128)
                (local $a v128)
                (local.set $a (v128.const i16x8 -32768 -1 4660 32767 42 -2 -32768 -1))
                (i32x4.extend_low_i16x8_s (local.get $a))
                (i32x4.extend_high_i16x8_s (local.get $a))
                (i32x4.extend_low_i16x8_u (local.get $a))
                (i32x4.extend_high_i16x8_u (local.get $a))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([0, 128, 255, 255, 255, 255, 255, 255, 52, 18, 0, 0, 255, 127, 0, 0]),
                Value::V128([42, 0, 0, 0, 254, 255, 255, 255, 0, 128, 255, 255, 255, 255, 255, 255]),
                Value::V128([0, 128, 0, 0, 255, 255, 0, 0, 52, 18, 0, 0, 255, 127, 0, 0]),
                Value::V128([42, 0, 0, 0, 254, 255, 0, 0, 0, 128, 0, 0, 255, 255, 0, 0]),
            ]
        );
    }

    #[test]
    fn flat_engine_i64x2_extmul_selects_half_and_extends_before_multiplying() {
        let (mut store, id) = flat_instance(
            "(module (func (export \"run\") (result v128 v128 v128 v128)
                (local $a v128) (local $b v128)
                (local.set $a (v128.const i32x4 -2147483648 -1 2 -3))
                (local.set $b (v128.const i32x4 -2147483648 -1 3 -4))
                (i64x2.extmul_low_i32x4_s (local.get $a) (local.get $b))
                (i64x2.extmul_high_i32x4_s (local.get $a) (local.get $b))
                (i64x2.extmul_low_i32x4_u (local.get $a) (local.get $b))
                (i64x2.extmul_high_i32x4_u (local.get $a) (local.get $b))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([0, 0, 0, 0, 0, 0, 0, 64, 1, 0, 0, 0, 0, 0, 0, 0]),
                Value::V128([6, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0]),
                Value::V128([0, 0, 0, 0, 0, 0, 0, 64, 1, 0, 0, 0, 254, 255, 255, 255]),
                Value::V128([6, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 249, 255, 255, 255]),
            ]
        );
    }

    #[test]
    fn flat_engine_pairwise_add_widens_adjacent_signed_or_unsigned_lanes() {
        let (mut store, id) = flat_instance(
            "(module (func (export \"run\") (result v128 v128)
                (local $a v128)
                (local.set $a (v128.const i16x8 -32768 -32768 -1 -1 32767 32767 100 -20))
                (i32x4.extadd_pairwise_i16x8_s (local.get $a))
                (i32x4.extadd_pairwise_i16x8_u (local.get $a))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![
                Value::V128([0, 0, 255, 255, 254, 255, 255, 255, 254, 255, 0, 0, 80, 0, 0, 0]),
                Value::V128([0, 0, 1, 0, 254, 255, 1, 0, 254, 255, 0, 0, 80, 0, 1, 0]),
            ]
        );
    }

    #[test]
    fn flat_engine_dot_product_pairs_signed_products_and_wraps_sum() {
        let (mut store, id) = flat_instance(
            "(module (func (export \"run\") (result v128)
                (i32x4.dot_i16x8_s
                    (v128.const i16x8 -32768 -32768 -1 2 100 -100 32767 -32768)
                    (v128.const i16x8 -32768 -32768 3 4 100 100 32767 32767))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![Value::V128([0, 0, 0, 128, 5, 0, 0, 0, 0, 0, 0, 0, 1, 128, 255, 255])]
        );
    }

    #[test]
    fn flat_engine_q15_multiply_rounds_ties_up_and_saturates() {
        let (mut store, id) = flat_instance(
            "(module (func (export \"run\") (result v128)
                (i16x8.q15mulr_sat_s
                    (v128.const i16x8 -32768 1 -1 1 -1 32767 -32768 16384)
                    (v128.const i16x8 -32768 16384 16384 16383 16385 32767 32767 16384))))",
            None,
        );
        assert_eq!(
            store.invoke_export(id, "run", vec![], None).unwrap(),
            vec![Value::V128([
                255, 127, 1, 0, 0, 0, 0, 0, 255, 255, 254, 127, 1, 128, 0, 32
            ])]
        );
    }

    #[test]
    fn flat_engine_simd_min_max_and_pmin_pmax_preserve_nan_order_and_zero_sign() {
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"f32min\") (result f32)
                    (f32x4.extract_lane 0
                        (f32x4.min (v128.const f32x4 0 0 0 0) (v128.const f32x4 -0 0 0 0))))
                (func (export \"f32max\") (result f32)
                    (f32x4.extract_lane 0
                        (f32x4.max (v128.const f32x4 -0 0 0 0) (v128.const f32x4 0 0 0 0))))
                (func (export \"f32pmin\") (result f32)
                    (f32x4.extract_lane 0
                        (f32x4.pmin (v128.const f32x4 3 0 0 0) (v128.const f32x4 nan 0 0 0))))
                (func (export \"f32pmin_nan_left\") (result f32)
                    (f32x4.extract_lane 0
                        (f32x4.pmin (v128.const f32x4 nan 0 0 0) (v128.const f32x4 3 0 0 0))))
                (func (export \"f32pmax\") (result f32)
                    (f32x4.extract_lane 0
                        (f32x4.pmax (v128.const f32x4 3 0 0 0) (v128.const f32x4 4 0 0 0))))
                (func (export \"f64min\") (result f64)
                    (f64x2.extract_lane 0
                        (f64x2.min (v128.const f64x2 0 0) (v128.const f64x2 -0 0))))
                (func (export \"f64max\") (result f64)
                    (f64x2.extract_lane 0
                        (f64x2.max (v128.const f64x2 -0 0) (v128.const f64x2 0 0))))
                (func (export \"f64pmin\") (result f64)
                    (f64x2.extract_lane 0
                        (f64x2.pmin (v128.const f64x2 3 0) (v128.const f64x2 nan 0))))
                (func (export \"f64pmin_nan_left\") (result f64)
                    (f64x2.extract_lane 0
                        (f64x2.pmin (v128.const f64x2 nan 0) (v128.const f64x2 3 0))))
                (func (export \"f64pmax\") (result f64)
                    (f64x2.extract_lane 0
                        (f64x2.pmax (v128.const f64x2 3 0) (v128.const f64x2 4 0)))))",
            None,
        );

        assert!(matches!(
            store.invoke_export(id, "f32min", vec![], None).unwrap().as_slice(),
            [Value::F32(value)] if value.to_bits() == (-0.0f32).to_bits()
        ));
        assert!(matches!(
            store.invoke_export(id, "f32max", vec![], None).unwrap().as_slice(),
            [Value::F32(value)] if value.to_bits() == 0.0f32.to_bits()
        ));
        assert_eq!(
            store.invoke_export(id, "f32pmin", vec![], None).unwrap(),
            vec![Value::F32(3.0)]
        );
        assert!(matches!(
            store.invoke_export(id, "f32pmin_nan_left", vec![], None).unwrap().as_slice(),
            [Value::F32(value)] if value.is_nan()
        ));
        assert_eq!(
            store.invoke_export(id, "f32pmax", vec![], None).unwrap(),
            vec![Value::F32(4.0)]
        );
        assert!(matches!(
            store.invoke_export(id, "f64min", vec![], None).unwrap().as_slice(),
            [Value::F64(value)] if value.to_bits() == (-0.0f64).to_bits()
        ));
        assert!(matches!(
            store.invoke_export(id, "f64max", vec![], None).unwrap().as_slice(),
            [Value::F64(value)] if value.to_bits() == 0.0f64.to_bits()
        ));
        assert_eq!(
            store.invoke_export(id, "f64pmin", vec![], None).unwrap(),
            vec![Value::F64(3.0)]
        );
        assert!(matches!(
            store.invoke_export(id, "f64pmin_nan_left", vec![], None).unwrap().as_slice(),
            [Value::F64(value)] if value.is_nan()
        ));
        assert_eq!(
            store.invoke_export(id, "f64pmax", vec![], None).unwrap(),
            vec![Value::F64(4.0)]
        );
    }

    #[test]
    fn flat_engine_simd_conversions_saturate_and_select_low_lanes() {
        let (mut store, id) = flat_instance(
            "(module
                (func (export \"f32s\") (result v128)
                    (i32x4.trunc_sat_f32x4_s (v128.const f32x4 nan -1.5 2147483648 -2147483648)))
                (func (export \"f32u\") (result v128)
                    (i32x4.trunc_sat_f32x4_u (v128.const f32x4 nan -1 4294967296 42.9)))
                (func (export \"i32s\") (result v128)
                    (f32x4.convert_i32x4_s (v128.const i32x4 -1 2 -3 4)))
                (func (export \"i32u\") (result v128)
                    (f32x4.convert_i32x4_u (v128.const i32x4 -1 2 -3 4)))
                (func (export \"f64s\") (result v128)
                    (i32x4.trunc_sat_f64x2_s_zero (v128.const f64x2 nan -2147483649)))
                (func (export \"f64u\") (result v128)
                    (i32x4.trunc_sat_f64x2_u_zero (v128.const f64x2 nan 4294967296)))
                (func (export \"low_s\") (result v128)
                    (f64x2.convert_low_i32x4_s (v128.const i32x4 -1 2 99 99)))
                (func (export \"low_u\") (result v128)
                    (f64x2.convert_low_i32x4_u (v128.const i32x4 -1 2 99 99)))
                (func (export \"demote\") (result v128)
                    (f32x4.demote_f64x2_zero (v128.const f64x2 1.5 -2.25)))
                (func (export \"promote\") (result v128)
                    (f64x2.promote_low_f32x4 (v128.const f32x4 1.5 -2.25 99 99))))",
            None,
        );

        assert_eq!(
            store.invoke_export(id, "f32s", vec![], None).unwrap(),
            vec![Value::V128([
                0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 127, 0, 0, 0, 128
            ])]
        );
        assert_eq!(
            store.invoke_export(id, "f32u", vec![], None).unwrap(),
            vec![Value::V128([0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 42, 0, 0, 0])]
        );
        assert_eq!(
            store.invoke_export(id, "i32s", vec![], None).unwrap(),
            vec![Value::V128([0, 0, 128, 191, 0, 0, 0, 64, 0, 0, 64, 192, 0, 0, 128, 64])]
        );
        assert_eq!(
            store.invoke_export(id, "i32u", vec![], None).unwrap(),
            vec![Value::V128([0, 0, 128, 79, 0, 0, 0, 64, 0, 0, 128, 79, 0, 0, 128, 64])]
        );
        assert_eq!(
            store.invoke_export(id, "f64s", vec![], None).unwrap(),
            vec![Value::V128([0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0])]
        );
        assert_eq!(
            store.invoke_export(id, "f64u", vec![], None).unwrap(),
            vec![Value::V128([0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0])]
        );
        assert_eq!(
            store.invoke_export(id, "low_s", vec![], None).unwrap(),
            vec![Value::V128([0, 0, 0, 0, 0, 0, 240, 191, 0, 0, 0, 0, 0, 0, 0, 64])]
        );
        assert_eq!(
            store.invoke_export(id, "low_u", vec![], None).unwrap(),
            vec![Value::V128([
                0, 0, 224, 255, 255, 255, 239, 65, 0, 0, 0, 0, 0, 0, 0, 64
            ])]
        );
        assert_eq!(
            store.invoke_export(id, "demote", vec![], None).unwrap(),
            vec![Value::V128([0, 0, 192, 63, 0, 0, 16, 192, 0, 0, 0, 0, 0, 0, 0, 0])]
        );
        assert_eq!(
            store.invoke_export(id, "promote", vec![], None).unwrap(),
            vec![Value::V128([0, 0, 0, 0, 0, 0, 248, 63, 0, 0, 0, 0, 0, 0, 2, 192])]
        );
    }
}
