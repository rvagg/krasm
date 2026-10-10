//! The types in a module's index spaces.
//!
//! WebAssembly numbers each kind of entity in its own index space (spec 2.5.1):
//! `call 3`, `global.get 3` and `table.get 3` name three unrelated things. For
//! functions, tables, memories and globals, "the indices of imports precede the
//! indices of other definitions in the same index space", although imports and
//! definitions live in different sections. [`IndexSpaceTypes`] flattens each of
//! those spaces into one list in index order, so resolving an index to its type
//! is a single lookup for the validator and the flat compiler alike.

use super::module::{ExternalKind, GlobalType, Module, TableType};

/// The type of every entry in a module's function, global and table index
/// spaces, imports first.
#[derive(Debug)]
pub struct IndexSpaceTypes {
    /// Type-section index of each function's signature.
    pub function_type_indices: Vec<u32>,
    pub global_types: Vec<GlobalType>,
    pub table_types: Vec<TableType>,
    /// Whether the module imports or defines a memory.
    pub has_memory: bool,
}

impl IndexSpaceTypes {
    pub(crate) fn new(module: &Module) -> Self {
        let mut function_type_indices = Vec::new();
        let mut global_types = Vec::new();
        let mut table_types = Vec::new();
        let mut has_memory = !module.memory.is_empty();
        for import in &module.imports.imports {
            match &import.external_kind {
                ExternalKind::Function(type_idx) => function_type_indices.push(*type_idx),
                ExternalKind::Global(global_type) => global_types.push(*global_type),
                ExternalKind::Table(table_type) => table_types.push(*table_type),
                ExternalKind::Memory(_) => has_memory = true,
            }
        }
        function_type_indices.extend(module.functions.functions.iter().map(|func| func.ftype_index));
        global_types.extend(module.globals.globals.iter().map(|global| global.global_type));
        table_types.extend(module.table.tables.iter().copied());
        Self {
            function_type_indices,
            global_types,
            table_types,
            has_memory,
        }
    }
}
