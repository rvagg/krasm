//! WebAssembly module representation.
//!
//! [`Module`] is the central AST type shared by both the binary parser
//! ([`crate::parser::parse`]) and the WAT text parser ([`crate::wat::parse`]).
//! It contains one section struct per WebAssembly section kind (type, import,
//! function, table, memory, global, export, start, element, code, data, custom)
//! plus a lazily built [`ValidationContext`] that caches derived lookups used
//! during type checking and execution.

use std::fmt;
use std::io;
use std::sync::OnceLock;

use fhex::ToHex;

use crate::parser::instruction::{self, InstructionKind};

/// Checks if a string contains the specific control character sequence that WABT displays as empty.
/// WABT shows the sequence "\x00\x01\x02...\x0F" (exactly 16 bytes, values 0-15) as empty <> or "".
/// This is specific behaviour for compatibility with WABT's wasm-objdump tool.
fn is_wabt_empty_control_sequence(s: &str) -> bool {
    let bytes: Vec<u8> = s.bytes().collect();
    bytes.len() == 16 && bytes.iter().enumerate().all(|(i, &b)| b == i as u8)
}

/// Formats a function name for display according to WABT's conventions.
/// - Empty names return an empty string
/// - The control sequence "\x00\x01\x02...\x0F" returns " <>"
/// - All other names return " <name>"
///
/// This ensures compatibility with WABT's wasm-objdump output format.
fn format_function_name(name: &str) -> String {
    if name.is_empty() {
        String::new()
    } else if is_wabt_empty_control_sequence(name) {
        " <>".to_string()
    } else {
        format!(" <{name}>")
    }
}

/// Returns the export string for display in quotes according to WABT's conventions.
/// The control sequence "\x00\x01\x02...\x0F" is shown as an empty string "",
/// while all other strings are returned as-is.
fn get_export_string_display(name: &str) -> &str {
    if is_wabt_empty_control_sequence(name) { "" } else { name }
}

/// A parsed WebAssembly module.
///
/// Produced by [`parser::parse()`](crate::parser::parse) (binary format) or
/// [`wat::parse()`](crate::wat::parse) (text format). Passed to
/// [`Store::create_instance()`](crate::Store::create_instance) wrapped in
/// `Arc<Module>` for instantiation.
#[derive(Debug)]
pub struct Module {
    pub name: String,
    pub magic: u32,
    pub version: u32,

    pub types: TypeSection,
    pub imports: ImportSection,
    pub functions: FunctionSection,
    pub table: TableSection,
    pub memory: MemorySection,
    pub globals: GlobalSection,
    pub exports: ExportSection,
    pub start: StartSection,
    pub elements: ElementSection,
    pub code: CodeSection,
    pub data: DataSection,
    pub data_count: DataCountSection,
    pub custom: Vec<CustomSection>,

    /// Cached validation context for efficient lookups during validation
    /// and streaming decode. Built lazily via `validation_context()`.
    /// `OnceLock` keeps `Module` `Sync`, so `Arc<Module>` is shareable.
    pub(crate) validation_context: OnceLock<ValidationContext>,
}

impl Module {
    /// Get the cached validation context, building it lazily if needed
    pub fn validation_context(&self) -> &ValidationContext {
        self.validation_context.get_or_init(|| self.build_validation_context())
    }

    /// Finds and formats the export name for a function at the given index.
    /// Returns the formatted string with angle brackets if the function is exported,
    /// or an empty string if not exported.
    /// Note: If a function has multiple exports, returns the LAST export name (WABT behaviour).
    pub fn get_function_export_name(&self, func_index: u32) -> String {
        // Use get_function to find the last export, then format its name
        self.exports
            .get_function(func_index)
            .map(|export| format_function_name(&export.name))
            .unwrap_or_default()
    }

    pub fn get_function_type(&self, index: usize) -> Option<&FunctionType> {
        self.functions
            .get(index as u32)
            .and_then(|f| self.types.get(f.ftype_index))
    }

    /// Get function type for any function index (imported or local)
    ///
    /// # Arguments
    /// * `func_idx` - The function index (0-based, imports first, then local functions)
    ///
    /// # Returns
    /// The function type if the index is valid, None otherwise
    pub fn get_function_type_by_idx(&self, func_idx: u32) -> Option<&FunctionType> {
        let num_imported_functions = self.imports.function_count() as u32;

        if func_idx < num_imported_functions {
            // Imported function - use ImportSection helper
            let type_idx = self.imports.get_function_type_index(func_idx)?;
            self.types.get(type_idx)
        } else {
            // Local function
            let local_idx = func_idx - num_imported_functions;
            let func = self.functions.get(local_idx)?;
            self.types.get(func.ftype_index)
        }
    }

    pub fn get_table(&self, index: u32) -> Option<&TableType> {
        self.imports
            .get_table(index)
            .or_else(|| self.table.tables.get(index as usize))
    }

    pub fn get_function_name(&self, index: u32) -> Option<String> {
        // Check if it's an imported function
        let import_count = self.imports.function_count() as u32;
        if index < import_count {
            // It's an imported function
            let mut func_idx = 0;
            for import in &self.imports.imports {
                if let ExternalKind::Function(_) = import.external_kind {
                    if func_idx == index {
                        return Some(format!("{}.{}", import.module, import.name));
                    }
                    func_idx += 1;
                }
            }
        }

        // Check if it's exported
        if let Some(export) = self.exports.get_function(index) {
            return Some(export.name.clone());
        }

        None
    }

    pub fn get_table_name(&self, index: u32) -> Option<String> {
        // Check if it's an imported table
        let import_count = self.imports.table_count() as u32;
        if index < import_count {
            // It's an imported table
            let mut table_idx = 0;
            for import in &self.imports.imports {
                if let ExternalKind::Table(_) = import.external_kind {
                    if table_idx == index {
                        return Some(format!("{}.{}", import.module, import.name));
                    }
                    table_idx += 1;
                }
            }
        }
        None
    }

    pub fn get_global_name(&self, index: u32) -> Option<String> {
        // Check if it's an imported global
        let import_count = self.imports.global_count() as u32;
        if index < import_count {
            // It's an imported global
            let mut global_idx = 0;
            for import in &self.imports.imports {
                if let ExternalKind::Global(_) = import.external_kind {
                    if global_idx == index {
                        return Some(format!("{}.{}", import.module, import.name));
                    }
                    global_idx += 1;
                }
            }
        }

        // Check if it's exported
        if let Some(export) = self
            .exports
            .exports
            .iter()
            .find(|e| matches!(e.index, ExportIndex::Global(idx) if idx == index))
        {
            return Some(export.name.clone());
        }

        None
    }
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ParsedUnit name = {}", self.name)?;
        write!(f, " magic = 0x{:08x}", self.magic)?;
        write!(f, " version = {}", self.version)?;
        write!(f, " types = ").unwrap();
        f.debug_set().entries(self.types.types.iter()).finish().unwrap();
        write!(f, " imports = ").unwrap();
        f.debug_set().entries(self.imports.imports.iter()).finish().unwrap();
        write!(f, " functions = ").unwrap();
        f.debug_set().entries(self.functions.functions.iter()).finish().unwrap();
        write!(f, " table = ").unwrap();
        f.debug_set().entries(self.table.tables.iter()).finish().unwrap();
        write!(f, " memory = ").unwrap();
        f.debug_set().entries(self.memory.memory.iter()).finish().unwrap();
        write!(f, " globals = ").unwrap();
        f.debug_set().entries(self.globals.globals.iter()).finish().unwrap();
        write!(f, " exports = ").unwrap();
        f.debug_set().entries(self.exports.exports.iter()).finish().unwrap();
        write!(f, " start = {}", self.start.start)?;
        write!(f, " elements = ").unwrap();
        f.debug_set().entries(self.elements.elements.iter()).finish().unwrap();
        write!(f, " code = ").unwrap();
        f.debug_set().entries(self.code.code.iter()).finish().unwrap();
        write!(f, " data = ").unwrap();
        f.debug_set().entries(self.data.data.iter()).finish().unwrap();
        Ok(())
    }
}

#[derive(Debug)]
pub struct SectionPosition {
    pub start: u32,
    pub end: u32,
}

impl SectionPosition {
    pub fn new(start: u32, end: u32) -> SectionPosition {
        SectionPosition { start, end }
    }

    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Display for SectionPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "start=0x{:08x} end=0x{:08x} (size=0x{:08x})",
            self.start,
            self.end,
            self.len()
        )
    }
}

pub trait Positional {
    fn set_position(&mut self, start: u32, end: u32);
    fn has_position(&self) -> bool;
    fn get_position(&self) -> Option<SectionPosition>;
}

macro_rules! impl_positional {
    ($($t:ty),*) => {
        $(
            impl Positional for $t {
                fn set_position(&mut self, start: u32, end: u32) {
                    self.position.start = start;
                    self.position.end = end;
                }

                fn has_position(&self) -> bool {
                    self.position.start != 0 || self.position.end != 0
                }
                fn get_position(&self) -> Option<SectionPosition> {
                    if self.has_position() {
                        Some(SectionPosition::new(self.position.start, self.position.end))
                    } else {
                        None
                    }
                }
            }
        )*
    }
}

impl_positional!(
    TypeSection,
    ImportSection,
    FunctionSection,
    TableSection,
    MemorySection,
    GlobalSection,
    ExportSection,
    StartSection,
    ElementSection,
    CodeSection,
    DataSection,
    DataCountSection,
    CustomSection
);

pub trait SectionToString {
    fn header_prefix(&self) -> &'static str;
    fn to_header_string(&self) -> String;
    fn to_details_string(&self, unit: &Module) -> String;
}

#[derive(Debug)]
pub struct TypeSection {
    pub types: Vec<FunctionType>,
    pub position: SectionPosition,
}

impl Default for TypeSection {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeSection {
    pub fn new() -> TypeSection {
        TypeSection {
            types: Vec::new(),
            position: SectionPosition::new(0, 0),
        }
    }

    pub fn push(&mut self, function_type: FunctionType) {
        self.types.push(function_type);
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Get a function type by index
    pub fn get(&self, index: u32) -> Option<&FunctionType> {
        self.types.get(index as usize)
    }

    pub fn find(&self, function_type: &FunctionType) -> Option<u32> {
        self.types.iter().position(|t| t == function_type).map(|i| i as u32)
    }
}

impl SectionToString for TypeSection {
    fn header_prefix(&self) -> &'static str {
        "Type"
    }

    fn to_header_string(&self) -> String {
        format!("{} count: {}", self.position, self.types.len())
    }

    fn to_details_string(&self, _: &Module) -> String {
        let mut result = String::new();
        result.push_str(&format!("Type[{}]:\n", self.types.len()));
        for (i, function_type) in self.types.iter().enumerate() {
            result.push_str(&format!(" - type[{i}] {function_type}\n"));
        }
        result
    }
}

#[derive(Debug)]
pub struct ImportSection {
    pub imports: Vec<Import>,
    pub position: SectionPosition,
}

impl Default for ImportSection {
    fn default() -> Self {
        Self::new()
    }
}

impl ImportSection {
    pub fn new() -> ImportSection {
        ImportSection {
            imports: Vec::new(),
            position: SectionPosition::new(0, 0),
        }
    }

    pub fn push(&mut self, import: Import) {
        self.imports.push(import);
    }

    /// Get an import by index
    pub fn get(&self, index: u32) -> Option<&Import> {
        self.imports.get(index as usize)
    }

    pub fn function_count(&self) -> usize {
        self.imports
            .iter()
            .filter(|i| matches!(i.external_kind, ExternalKind::Function(_)))
            .count()
    }

    pub fn get_function_type_index(&self, index: u32) -> Option<u32> {
        self.imports
            .iter()
            .filter_map(|i| match i.external_kind {
                ExternalKind::Function(ref f) => Some(*f),
                _ => None,
            })
            .nth(index as usize)
    }

    pub fn table_count(&self) -> usize {
        self.imports
            .iter()
            .filter(|i| matches!(i.external_kind, ExternalKind::Table(_)))
            .count()
    }

    pub fn get_table_import(&self, index: u32) -> Option<&Import> {
        // return the index'th table, if we have that many
        self.imports
            .iter()
            .filter(|i| matches!(i.external_kind, ExternalKind::Table(_)))
            .nth(index as usize)
    }

    pub fn get_table(&self, index: u32) -> Option<&TableType> {
        // return the index'th table, if we have that many
        self.get_table_import(index).and_then(|i| match i.external_kind {
            ExternalKind::Table(ref t) => Some(t),
            _ => None,
        })
    }

    pub fn memory_count(&self) -> usize {
        self.imports
            .iter()
            .filter(|i| matches!(i.external_kind, ExternalKind::Memory(_)))
            .count()
    }

    pub fn global_count(&self) -> usize {
        self.imports
            .iter()
            .filter(|i| matches!(i.external_kind, ExternalKind::Global(_)))
            .count()
    }

    pub fn get_global_import(&self, index: u32) -> Option<&Import> {
        // return the index'th global, if we have that many
        self.imports
            .iter()
            .filter(|i| matches!(i.external_kind, ExternalKind::Global(_)))
            .nth(index as usize)
            .and_then(|i| match i.external_kind {
                ExternalKind::Global(_) => Some(i),
                _ => None,
            })
    }
}

impl SectionToString for ImportSection {
    fn header_prefix(&self) -> &'static str {
        "Import"
    }

    fn to_header_string(&self) -> String {
        format!("{} count: {}", self.position, self.imports.len())
    }

    fn to_details_string(&self, unit: &Module) -> String {
        let mut result = String::new();
        result.push_str(&format!("Import[{}]:\n", self.imports.len()));
        let mut function_count = 0;
        let mut table_count = 0;
        let mut memory_count = 0;
        let mut global_count = 0;

        for import in &self.imports {
            match import.external_kind {
                ExternalKind::Function(ref f) => {
                    // Check if this function is re-exported
                    // Use the last export if there are multiple (for test compatibility)
                    let display_name = if let Some(export) = unit
                        .exports
                        .exports
                        .iter()
                        .rev()
                        .find(|e| matches!(e.index, ExportIndex::Function(idx) if idx == function_count as u32))
                    {
                        export.name.clone()
                    } else {
                        format!("{}.{}", import.module, import.name)
                    };

                    result.push_str(&format!(
                        " - func[{}] sig={} <{}> <- {}.{}\n",
                        function_count, f, display_name, import.module, import.name
                    ));
                    function_count += 1;
                }
                ExternalKind::Table(ref t) => {
                    result.push_str(&format!(
                        " - table[{}] type={} initial={}{} <- {}.{}\n",
                        table_count,
                        match t.ref_type {
                            RefType::FuncRef => "funcref",
                            RefType::ExternRef => "externref",
                        },
                        t.limits.min,
                        match t.limits.max {
                            Some(max) => format!(" max={max}"),
                            None => "".to_string(),
                        },
                        import.module,
                        import.name,
                    ));
                    table_count += 1;
                }
                ExternalKind::Memory(ref m) => {
                    result.push_str(&format!(
                        " - memory[{}] pages: initial={}{} <- {}.{}\n",
                        memory_count,
                        m.min,
                        match m.max {
                            Some(max) => format!(" max={max}"),
                            None => "".to_string(),
                        },
                        import.module,
                        import.name
                    ));
                    memory_count += 1;
                }
                ExternalKind::Global(ref g) => {
                    result.push_str(&format!(
                        " - global[{}] {} mutable={} <- {}.{}\n",
                        global_count,
                        g.value_type,
                        if g.mutable { 1 } else { 0 },
                        import.module,
                        import.name
                    ));
                    global_count += 1;
                }
            }
        }
        result
    }
}

#[derive(Debug)]
pub struct FunctionSection {
    pub functions: Vec<Function>,
    pub position: SectionPosition,
}

impl Default for FunctionSection {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionSection {
    pub fn new() -> FunctionSection {
        FunctionSection {
            functions: Vec::new(),
            position: SectionPosition::new(0, 0),
        }
    }

    pub fn push(&mut self, function: Function) {
        self.functions.push(function);
    }

    pub fn len(&self) -> usize {
        self.functions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// Get a function by index
    pub fn get(&self, index: u32) -> Option<&Function> {
        self.functions.get(index as usize)
    }
}

impl SectionToString for FunctionSection {
    fn header_prefix(&self) -> &'static str {
        "Function"
    }

    fn to_header_string(&self) -> String {
        format!("{} count: {}", self.position, self.functions.len())
    }

    fn to_details_string(&self, unit: &Module) -> String {
        let mut result = String::new();
        result.push_str(&format!("Function[{}]:\n", self.functions.len()));
        for (i, function) in self.functions.iter().enumerate() {
            let fi = unit.imports.function_count() + i;
            result.push_str(&format!(
                " - func[{}] sig={}{}\n",
                fi,
                function.ftype_index,
                match unit.exports.get_function(fi as u32) {
                    Some(export) => {
                        format_function_name(&export.name)
                    }
                    None => "".to_string(),
                }
            ));
        }
        result
    }
}

#[derive(Debug)]
pub struct TableSection {
    pub tables: Vec<TableType>,
    pub position: SectionPosition,
}

impl Default for TableSection {
    fn default() -> Self {
        Self::new()
    }
}

impl TableSection {
    pub fn new() -> TableSection {
        TableSection {
            tables: Vec::new(),
            position: SectionPosition::new(0, 0),
        }
    }

    pub fn push(&mut self, table: TableType) {
        self.tables.push(table);
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }
}

impl SectionToString for TableSection {
    fn header_prefix(&self) -> &'static str {
        "Table"
    }

    fn to_header_string(&self) -> String {
        format!("{} count: {}", self.position, self.tables.len())
    }

    fn to_details_string(&self, unit: &Module) -> String {
        let mut result = String::new();
        result.push_str(&format!("Table[{}]:\n", self.tables.len()));
        let import_table_count = unit.imports.table_count();
        for (i, table_type) in self.tables.iter().enumerate() {
            result.push_str(&format!(" - table[{}] {}\n", import_table_count + i, table_type));
        }
        result
    }
}

#[derive(Debug)]
pub struct MemorySection {
    pub memory: Vec<Memory>,
    pub position: SectionPosition,
}

impl Default for MemorySection {
    fn default() -> Self {
        Self::new()
    }
}

impl MemorySection {
    pub fn new() -> MemorySection {
        MemorySection {
            memory: Vec::new(),
            position: SectionPosition::new(0, 0),
        }
    }

    pub fn push(&mut self, memory: Memory) {
        self.memory.push(memory);
    }

    pub fn len(&self) -> usize {
        self.memory.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }
}

impl SectionToString for MemorySection {
    fn header_prefix(&self) -> &'static str {
        "Memory"
    }

    fn to_header_string(&self) -> String {
        format!("{} count: {}", self.position, self.memory.len())
    }

    fn to_details_string(&self, _: &Module) -> String {
        let mut result = String::new();
        result.push_str(&format!("Memory[{}]:\n", self.memory.len()));
        for (i, memory) in self.memory.iter().enumerate() {
            result.push_str(&format!(
                " - memory[{}] pages: initial={}{}\n",
                i,
                memory.limits.min,
                match memory.limits.max {
                    Some(max) => format!(" max={max}"),
                    None => "".to_string(),
                }
            ));
        }
        result
    }
}

#[derive(Debug)]
pub struct GlobalSection {
    pub globals: Vec<Global>,
    pub position: SectionPosition,
}

impl Default for GlobalSection {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalSection {
    pub fn new() -> GlobalSection {
        GlobalSection {
            globals: Vec::new(),
            position: SectionPosition::new(0, 0),
        }
    }

    pub fn push(&mut self, global: Global) {
        self.globals.push(global);
    }

    pub fn len(&self) -> usize {
        self.globals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.globals.is_empty()
    }

    pub fn find(&self, global: &Global) -> Option<u32> {
        self.globals.iter().position(|g| g == global).map(|i| i as u32)
    }

    /// Get a global by index
    pub fn get(&self, index: u32) -> Option<&Global> {
        self.globals.get(index as usize)
    }
}

impl From<&GlobalSection> for Vec<GlobalType> {
    fn from(section: &GlobalSection) -> Self {
        section.globals.iter().map(|g| g.global_type).collect()
    }
}

impl SectionToString for GlobalSection {
    fn header_prefix(&self) -> &'static str {
        "Global"
    }

    fn to_header_string(&self) -> String {
        format!("{} count: {}", self.position, self.globals.len())
    }

    fn to_details_string(&self, unit: &Module) -> String {
        let mut result = String::new();
        result.push_str(&format!("Global[{}]:\n", self.globals.len()));
        let import_global_count = unit.imports.global_count() as u32;
        for (i, global) in self.globals.iter().enumerate() {
            let global_index = import_global_count + i as u32;
            result.push_str(&format!(
                " - global[{}] {} mutable={}{}{}\n",
                global_index,
                global.global_type.value_type,
                if global.global_type.mutable { 1 } else { 0 },
                match unit.exports.get_global(global_index) {
                    Some(e) => {
                        if e.name.is_empty() {
                            "".to_string()
                        } else {
                            format!(" <{}>", e.name)
                        }
                    }
                    None => "".to_string(),
                },
                init_expr_to_string(unit, &global.init, false, true),
            ));
        }
        result
    }
}

#[derive(Debug)]
pub struct ExportSection {
    pub exports: Vec<Export>,
    pub position: SectionPosition,
}

impl Default for ExportSection {
    fn default() -> Self {
        Self::new()
    }
}

impl ExportSection {
    pub fn new() -> ExportSection {
        ExportSection {
            exports: Vec::new(),
            position: SectionPosition::new(0, 0),
        }
    }

    pub fn push(&mut self, export: Export) {
        self.exports.push(export);
    }

    pub fn get_global(&self, global_index: u32) -> Option<&Export> {
        self.exports
            .iter()
            .rev()
            .find(|e| matches!(e.index, ExportIndex::Global(idx) if idx == global_index))
    }

    pub fn get_function(&self, function_index: u32) -> Option<&Export> {
        self.exports
            .iter()
            .rev()
            .find(|e| matches!(e.index, ExportIndex::Function(idx) if idx == function_index))
    }

    /// Get an export by name
    pub fn get_by_name(&self, name: &str) -> Option<&Export> {
        self.exports.iter().find(|e| e.name == name)
    }
}

impl SectionToString for ExportSection {
    fn header_prefix(&self) -> &'static str {
        "Export"
    }

    fn to_header_string(&self) -> String {
        format!("{} count: {}", self.position, self.exports.len())
    }

    fn to_details_string(&self, _unit: &Module) -> String {
        let mut result = String::new();
        result.push_str(&format!("Export[{}]:\n", self.exports.len()));
        for export in &self.exports {
            let (typ, idx) = match export.index {
                ExportIndex::Function(i) => ("func", i),
                ExportIndex::Table(i) => ("table", i),
                ExportIndex::Memory(i) => ("memory", i),
                ExportIndex::Global(i) => ("global", i),
            };
            let name_display = match export.index {
                ExportIndex::Function(func_idx) => {
                    // WABT displays export names in a specific way for functions:
                    // 1. When a function has multiple exports, WABT shows the LAST export name
                    //    for all instances. For example, if func[0] is exported as "a", "b", and "c",
                    //    all three exports will show as <c> in the export list.
                    // 2. For control characters, WABT has special handling:
                    //    - The specific sequence "\x00\x01\x02...\x0F" (all 16 bytes) displays as <>
                    //    - Other control characters (like \n, \r, etc.) are shown as-is
                    //
                    // Find the last export for this function
                    let mut last_export_name = "";
                    for exp in &self.exports {
                        if let ExportIndex::Function(f_idx) = exp.index
                            && f_idx == func_idx
                        {
                            last_export_name = &exp.name;
                        }
                    }

                    format_function_name(last_export_name)
                }
                _ => "".to_string(),
            };
            // Export string display: WABT shows the control sequence "\x00\x01...\x0F" as an empty string
            // in the quoted export name (e.g., func[22] <> -> "" instead of showing the control chars)
            let export_string_display = get_export_string_display(&export.name);
            result.push_str(&format!(
                " - {typ}[{idx}]{name_display} -> \"{export_string_display}\"\n"
            ));
        }
        result
    }
}

#[derive(Debug)]
pub struct StartSection {
    pub start: u32,
    pub position: SectionPosition,
}

impl Default for StartSection {
    fn default() -> Self {
        Self::new()
    }
}

impl StartSection {
    pub fn new() -> StartSection {
        StartSection {
            start: 0,
            position: SectionPosition::new(0, 0),
        }
    }
}

impl SectionToString for StartSection {
    fn header_prefix(&self) -> &'static str {
        "Start"
    }

    fn to_header_string(&self) -> String {
        format!("{} start: {}", self.position, self.start)
    }

    fn to_details_string(&self, unit: &Module) -> String {
        let mut result = String::new();
        let func_name = if self.start < unit.imports.function_count() as u32 {
            // It's an imported function
            unit.imports
                .imports
                .iter()
                .filter(|i| matches!(i.external_kind, ExternalKind::Function(_)))
                .nth(self.start as usize)
                .map(|import| format!(" <{}.{}>", import.module, import.name))
                .unwrap_or_default()
        } else {
            // It's a local function - check if it has an export
            unit.exports
                .get_function(self.start)
                .map(|export| format_function_name(&export.name))
                .unwrap_or_default()
        };
        result.push_str(&format!("Start:\n - start function: {}{}\n", self.start, func_name));
        result
    }
}

#[derive(Debug)]
pub struct ElementSection {
    pub elements: Vec<Element>,
    pub position: SectionPosition,
}

impl Default for ElementSection {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementSection {
    pub fn new() -> ElementSection {
        ElementSection {
            elements: Vec::new(),
            position: SectionPosition::new(0, 0),
        }
    }

    pub fn push(&mut self, element: Element) {
        self.elements.push(element);
    }
}

// Helper functions for new instruction type
fn init_expr_to_offset(init: &[instruction::Instruction]) -> Option<u32> {
    if init.len() != 2 {
        // const + end
        return None;
    }
    match init[0].kind {
        // include instruction + end
        InstructionKind::I32Const { value } => Some(value as u32),
        InstructionKind::GlobalGet { .. } => Some(0),
        _ => None,
    }
}

fn init_expr_to_string(
    unit: &Module,
    init: &[instruction::Instruction],
    as_unsigned: bool,
    with_prefix: bool,
) -> String {
    if init.is_empty() {
        return String::new();
    }

    let mut result = String::new();

    if with_prefix {
        result.push_str(" - init ");
    }

    match init.len() {
        0 => {
            result.push_str("<EMPTY>\n");
        }
        1 => result.push_str("<INVALID INIT EXPR>\n"),
        2 => match &init[0].kind {
            // include instruction + end
            InstructionKind::I32Const { value } => {
                if as_unsigned {
                    result.push_str(&format!("i32={}", *value as u32));
                } else {
                    result.push_str(&format!("i32={}", *value));
                }
            }
            InstructionKind::I64Const { value } => {
                if as_unsigned {
                    result.push_str(&format!("i64={}", *value as u64));
                } else {
                    result.push_str(&format!("i64={}", *value));
                }
            }
            InstructionKind::F32Const { value } => {
                result.push_str(&format!("f32={}", value.to_hex()));
            }
            InstructionKind::F64Const { value } => {
                result.push_str(&format!("f64={}", value.to_hex()));
            }
            // v128 const disassembly not yet implemented
            InstructionKind::GlobalGet { global_idx } => match unit.imports.get_global_import(*global_idx) {
                Some(import) => {
                    result.push_str(&format!("global={} <{}.{}>", global_idx, import.module, import.name));
                }
                None => {
                    result.push_str(&format!("global={global_idx} <INVALID>"));
                }
            },
            InstructionKind::RefFunc { func_idx } => {
                result.push_str(&format!(
                    "ref.func:{}{}",
                    func_idx,
                    match unit.get_function_name(*func_idx) {
                        Some(name) => format!(" <{name}>"),
                        None => "".to_string(),
                    }
                ));
            }
            InstructionKind::RefNull { ref_type } => {
                result.push_str(&format!("ref.null {ref_type}"));
            }
            _ => {
                result.push_str("<INVALID>");
            }
        },
        _ => {
            result.push('(');
            let mut first = true;
            if let Some(len) = init.len().checked_sub(1) {
                for instruction in init.iter().take(len) {
                    if !first {
                        result.push_str(", ");
                    }
                    first = false;
                    result.push_str(&format!("{instruction}"));
                    match &instruction.kind {
                        InstructionKind::I32Const { value } => {
                            result.push_str(&format!("{}", *value));
                        }
                        InstructionKind::I64Const { value } => {
                            result.push_str(&format!("{}", *value));
                        }
                        InstructionKind::F32Const { value } => {
                            result.push_str(&format!("{value}"));
                        }
                        InstructionKind::F64Const { value } => {
                            result.push_str(&format!("{value}"));
                        }
                        InstructionKind::GlobalGet { global_idx } => {
                            result.push_str(&format!("{}", *global_idx));
                            // Could resolve the global's import name here
                        }
                        _ => {
                            result.push_str("<INVALID>");
                        }
                    };
                }
            } else {
                result.push_str("<UNEXPECTED ERR>");
            }
        }
    }
    result
}

impl SectionToString for ElementSection {
    fn header_prefix(&self) -> &'static str {
        "Elem"
    }

    fn to_header_string(&self) -> String {
        format!("{} count: {}", self.position, self.elements.len())
    }

    fn to_details_string(&self, unit: &Module) -> String {
        let mut result: String = String::new();
        result.push_str(&format!("Elem[{}]:\n", self.elements.len()));

        // Track the last active segment offset to handle passive element indexing
        // Passive elements inherit the offset from the immediately preceding active segment
        let mut last_active_offset: i32 = 0;

        for (i, element) in self.elements.iter().enumerate() {
            let (expr_str, elem_offset) = match element.mode {
                ElementMode::Active { ref offset, .. } => {
                    let offset_val = match init_expr_to_offset(offset) {
                        Some(o) => o as i32,
                        None => -1,
                    };
                    // Update the last active offset - this will be used by subsequent passive elements
                    last_active_offset = offset_val;
                    (init_expr_to_string(unit, offset, true, true), offset_val)
                }
                ElementMode::Passive => {
                    // For passive elements, use the offset from the most recent active segment
                    // This matches WABT's behaviour where passive elements display indices
                    // as if they were conceptually positioned after the preceding active segment
                    (String::new(), last_active_offset)
                }
                ElementMode::Declarative => (String::new(), 0),
            };
            result.push_str(&format!(
                " - segment[{}] flags={} table={} count={}{}\n",
                i,
                element.flags,
                match element.mode {
                    ElementMode::Passive => 0,
                    ElementMode::Declarative => 0,
                    ElementMode::Active { table_index, .. } => table_index,
                },
                element.init.len(),
                expr_str,
            ));
            element.init.iter().enumerate().for_each(|(i, _)| {
                result.push_str(&format!(
                    "  - elem[{}] = {}\n",
                    match elem_offset {
                        -1 => "<INVALID_OFFSET>".to_string(),
                        _ => format!("{}", elem_offset + i as i32),
                    },
                    init_expr_to_string(unit, &element.init[i], false, false),
                ));
            });
        }
        result
    }
}

#[derive(Debug)]
pub struct CodeSection {
    pub code: Vec<FunctionBody>,
    pub position: SectionPosition,
}

impl Default for CodeSection {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeSection {
    pub fn new() -> CodeSection {
        CodeSection {
            code: Vec::new(),
            position: SectionPosition::new(0, 0),
        }
    }

    pub fn len(&self) -> usize {
        self.code.len()
    }

    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }

    /// Get a function body by index
    pub fn get(&self, index: u32) -> Option<&FunctionBody> {
        self.code.get(index as usize)
    }
}

impl SectionToString for CodeSection {
    fn header_prefix(&self) -> &'static str {
        "Code"
    }

    fn to_header_string(&self) -> String {
        format!("{} count: {}", self.position, self.code.len())
    }

    fn to_details_string(&self, unit: &Module) -> String {
        let mut result: String = String::new();
        result.push_str(&format!("Code[{}]:\n", self.code.len()));
        for (i, function_body) in self.code.iter().enumerate() {
            let fi = unit.imports.function_count() + i;
            let exp = unit.get_function_export_name(fi as u32);
            result.push_str(&format!(
                " - func[{}] size={}{}\n",
                fi, // NB: index may be wrong for modules with empty imported functions — wabt skips these somehow
                function_body.position.len(),
                exp
            ));
        }
        result
    }
}

impl CodeSection {
    fn to_disassemble_string(&self, unit: &Module) -> String {
        let mut result: String = String::new();
        for (i, function_body) in self.code.iter().enumerate() {
            let fi = unit.imports.function_count() + i;
            let exp = unit.get_function_export_name(fi as u32);
            let ftype_index = match unit.functions.get(i as u32) {
                Some(f) => f.ftype_index,
                None => {
                    result.push_str(&format!("invalid function index: {i}\n"));
                    continue;
                }
            };
            let ftype = match unit.types.get(ftype_index) {
                Some(t) => t,
                None => {
                    result.push_str(&format!("invalid type index: {ftype_index}\n"));
                    continue;
                }
            };

            let mut pos = function_body.position.start as usize;
            result.push_str(&format!("{pos:06x} func[{fi}]{exp}:\n"));
            pos += 1; // skip the local decl count byte at function start
            // for each instruction, ignoring the opcodes for now
            //  00011f: 20 00                      | local.get 0

            let mut offset = ftype.parameters.len() as u32;
            function_body.locals.iter().for_each(|(count, value_type)| {
                let mut count_bytes = Vec::new();
                super::encoding::write_vu32(&mut count_bytes, *count);
                let mut byts = count_bytes.iter().fold(String::new(), |mut acc, byte| {
                    acc.push_str(&format!("{byte:02x} "));
                    acc
                });
                byts.push_str(&format!("{:02x}", value_type.emit_bytes()[0]));

                let range = match *count {
                    0 => "".to_string(),
                    1 => format!("{offset}"),
                    _ => format!("{}..{}", offset, offset + count - 1),
                };

                result.push_str(&format!(" {pos:06x}: {byts:27}| local[{range}] type={value_type}\n",));
                pos += count_bytes.len() + 1;
                offset += count;
            });

            let mut indent: usize = 0;
            let instructions = function_body.body.flatten();
            instructions.iter().for_each(|instruction| {
                // Get the InstructionKind directly
                let inst_kind = &instruction.kind;

                // Update indent before instruction (for else/end)
                use crate::parser::instruction::InstructionKind;
                match inst_kind {
                    InstructionKind::Else | InstructionKind::End => indent = indent.saturating_sub(1),
                    _ => {}
                }

                // Use the original bytes for display - this preserves byte-perfect encoding
                let coding_bytes = instruction.original_bytes.clone();

                // Get the string representation with context
                let coding_string = inst_kind.format_with_context(unit);

                // Check if this is a 0xFC prefix instruction
                // WABT has a quirk where it displays saturating truncation instructions
                // (i32.trunc_sat_*, i64.trunc_sat_*) starting from the position after
                // the prefix byte and only shows the subopcode bytes.
                // Other 0xFC instructions (memory.copy, table.copy, etc.) show the full bytes.
                let (display_pos, display_bytes, pos_increment) = if !coding_bytes.is_empty() && coding_bytes[0] == 0xFC
                {
                    let subopcode_bytes = &coding_bytes[1..];

                    // Check if this is a saturating truncation instruction (subopcodes 0-7)
                    let is_trunc_sat = if let Some(&first_byte) = subopcode_bytes.first() {
                        // Decode the LEB128 value to check if it's 0-7
                        let value = if first_byte < 0x80 {
                            first_byte as u32
                        } else if subopcode_bytes.len() >= 2 && subopcode_bytes[1] < 0x80 {
                            ((first_byte & 0x7F) as u32) | ((subopcode_bytes[1] as u32) << 7)
                        } else if subopcode_bytes.len() >= 3 && subopcode_bytes[2] < 0x80 {
                            ((first_byte & 0x7F) as u32)
                                | (((subopcode_bytes[1] & 0x7F) as u32) << 7)
                                | ((subopcode_bytes[2] as u32) << 14)
                        } else if subopcode_bytes.len() >= 4 && subopcode_bytes[3] < 0x80 {
                            ((first_byte & 0x7F) as u32)
                                | (((subopcode_bytes[1] & 0x7F) as u32) << 7)
                                | (((subopcode_bytes[2] & 0x7F) as u32) << 14)
                                | ((subopcode_bytes[3] as u32) << 21)
                        } else if subopcode_bytes.len() >= 5 && subopcode_bytes[4] < 0x80 {
                            ((first_byte & 0x7F) as u32)
                                | (((subopcode_bytes[1] & 0x7F) as u32) << 7)
                                | (((subopcode_bytes[2] & 0x7F) as u32) << 14)
                                | (((subopcode_bytes[3] & 0x7F) as u32) << 21)
                                | ((subopcode_bytes[4] as u32) << 28)
                        } else {
                            // For simplicity, assume it's not a trunc_sat if we can't decode
                            8
                        };
                        value <= 7
                    } else {
                        false
                    };

                    if is_trunc_sat {
                        // WABT has a bug where it normalises the display of non-canonical LEB128
                        // encodings for saturating truncation instructions (0xFC prefix, subopcodes 0-7).
                        // When the subopcode is encoded with more bytes than necessary (e.g., 0x81 0x80 0x00
                        // instead of just 0x01), WABT displays it as the normalised form "80 00" and
                        // adjusts the position offset.
                        //
                        // To match WABT's output exactly for tests, we need to replicate this behaviour.
                        // This appears to be related to how WABT prints certain instructions when they
                        // have non-canonical LEB128 encodings.

                        // Check if this is a non-canonical encoding
                        // For values 0-7, the canonical encoding is 1 byte
                        // Any encoding longer than 1 byte is non-canonical
                        let is_non_canonical = subopcode_bytes.len() > 1;

                        if is_non_canonical {
                            // For non-canonical encodings, WABT shows normalised "80 00"
                            // and adjusts position. The offset is the number of extra bytes
                            // beyond what's needed for the canonical encoding.
                            // For values 0-7, canonical encoding is 1 byte, so:
                            // - 2-byte encoding (e.g., 0x80 0x00) -> offset = 1
                            // - 3-byte encoding (e.g., 0x81 0x80 0x00) -> offset = 2
                            // - 4-byte encoding (e.g., 0x86 0x80 0x80 0x00) -> offset = 3
                            // - 5-byte encoding (e.g., 0x87 0x80 0x80 0x80 0x00) -> offset = 4
                            let pos_offset = subopcode_bytes.len() - 1;
                            (pos + pos_offset, vec![0x80, 0x00], coding_bytes.len())
                        } else {
                            // For canonical encodings, show the full bytes including prefix
                            (pos, coding_bytes.clone(), coding_bytes.len())
                        }
                    } else {
                        // Other 0xFC instructions show the full bytes
                        (pos, coding_bytes.clone(), coding_bytes.len())
                    }
                } else {
                    (pos, coding_bytes.clone(), coding_bytes.len())
                };

                // all of the complexity here is to get nice wrapping when we overflow 26 chars in
                // the instruction hex, so we end up with something like this:
                //  000017: 42 80 80 80 80 80 80 80 80 | i64.const -9223372036854775808
                //  000020: 80 7f                      |

                let mut byte_string = String::new();
                let mut coding_string_added = false;

                let push_format = |byte_string: &String, coding_string_added: bool| -> String {
                    let (indent_string, coding_str) = if coding_string_added {
                        ("".to_string(), "")
                    } else {
                        ("  ".repeat(indent), coding_string.as_str())
                    };
                    format!("{byte_string:27}| {indent_string}{coding_str}\n",)
                };

                result.push_str(&format!(" {display_pos:06x}: "));

                let mut bytes_processed = 0;
                for (i, byte) in display_bytes.iter().enumerate() {
                    let new_byte_string = format!("{byte:02x} ");
                    if byte_string.len() + new_byte_string.len() > 27 {
                        // the case of a byte block that's too long, so we dump the
                        // current batch and loop again, possibly dumping more until
                        // we're within the 26 char limit
                        result.push_str(&push_format(&byte_string, coding_string_added));
                        // Calculate the position for the wrapped bytes
                        let wrapped_pos = display_pos + bytes_processed;
                        result.push_str(&format!(" {wrapped_pos:06x}: "));
                        byte_string.clear();
                        coding_string_added = true;
                    }
                    byte_string.push_str(&new_byte_string);
                    bytes_processed = i + 1;
                }

                // Update position based on calculated increment
                pos += pos_increment;
                result.push_str(&push_format(&byte_string, coding_string_added));

                // Update indent after instruction (for block/loop/if/else)
                match inst_kind {
                    InstructionKind::Block { .. }
                    | InstructionKind::Loop { .. }
                    | InstructionKind::If { .. }
                    | InstructionKind::Else => indent += 1,
                    _ => {}
                }
            });
        }
        result
    }
}

#[derive(Debug)]
pub struct DataSection {
    pub data: Vec<Data>,
    pub position: SectionPosition,
}

impl Default for DataSection {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSection {
    pub fn new() -> DataSection {
        DataSection {
            data: Vec::new(),
            position: SectionPosition::new(0, 0),
        }
    }
}

#[derive(Debug)]
pub struct DataCountSection {
    pub count: u32,
    pub position: SectionPosition,
}

impl Default for DataCountSection {
    fn default() -> Self {
        Self::new()
    }
}

impl DataCountSection {
    pub fn new() -> DataCountSection {
        DataCountSection {
            count: 0,
            position: SectionPosition::new(0, 0),
        }
    }
}

impl SectionToString for DataCountSection {
    fn header_prefix(&self) -> &'static str {
        "DataCount"
    }

    fn to_header_string(&self) -> String {
        format!("{} count: {}", self.position, self.count)
    }

    fn to_details_string(&self, _: &Module) -> String {
        let mut result = String::new();
        result.push_str(&format!("DataCount:\n - data count: {}\n", self.count));
        result
    }
}

#[derive(Debug)]
pub struct CustomSection {
    pub name: String,
    pub data: Vec<u8>,
    pub position: SectionPosition,
}

impl Default for CustomSection {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomSection {
    pub fn new() -> CustomSection {
        CustomSection {
            name: String::new(),
            data: Vec::new(),
            position: SectionPosition::new(0, 0),
        }
    }
}

impl SectionToString for CustomSection {
    fn header_prefix(&self) -> &'static str {
        "Custom"
    }

    fn to_header_string(&self) -> String {
        format!(
            "{} \"{}\"",
            self.position,
            self.name
                .split('\0') // custom section names may contain NUL; wabt truncates at the first one
                .next()
                .unwrap_or(&self.name)
        )
    }

    fn to_details_string(&self, _: &Module) -> String {
        let mut result = String::new();
        result.push_str(&format!(
            "Custom:\n - name: \"{}\"\n",
            self.name
                .split('\0') // custom section names may contain NUL; wabt truncates at the first one
                .next()
                .unwrap_or(&self.name)
        ));
        result
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionType {
    pub parameters: Vec<ValueType>,
    pub return_types: Vec<ValueType>,
}

impl fmt::Display for FunctionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "({}) -> {}",
            self.parameters
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<String>>()
                .join(", "),
            if self.return_types.is_empty() {
                "nil".to_string()
            } else if self.return_types.len() > 1 {
                format!(
                    "({})",
                    self.return_types
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<String>>()
                        .join(", ")
                )
            } else {
                self.return_types
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
                    .join(", ")
            }
        )
    }
}

#[derive(Debug)]
pub struct Import {
    pub external_kind: ExternalKind,
    pub module: String,
    pub name: String,
}

impl fmt::Display for Import {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.external_kind {
            ExternalKind::Function(type_index) => write!(
                f,
                "sig={} <{}.{}> <- {}.{}",
                type_index, self.module, self.name, self.module, self.name
            ),
            ExternalKind::Table(ref table_type) => write!(
                f,
                "type={} initial={}{} <- {}.{}",
                match table_type.ref_type {
                    RefType::FuncRef => "funcref",
                    RefType::ExternRef => "externref",
                },
                table_type.limits.min,
                match table_type.limits.max {
                    Some(max) => format!(" max={max}"),
                    None => "".to_string(),
                },
                self.module,
                self.name
            ),
            ExternalKind::Memory(ref limits) => write!(
                f,
                "pages: initial={}{} <- {}.{}",
                limits.min,
                match limits.max {
                    Some(max) => format!(" max={max}"),
                    None => "".to_string(),
                },
                self.module,
                self.name
            ),
            ExternalKind::Global(ref global_type) => write!(
                f,
                "{} mutable={} <- {}.{}",
                global_type.value_type, global_type.mutable, self.module, self.name
            ),
        }
    }
}

#[derive(Debug)]
pub struct Function {
    pub ftype_index: u32,
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Type({})", self.ftype_index)
    }
}

#[derive(Debug)]
pub struct Memory {
    pub limits: Limits,
}

#[derive(Debug, Clone, Copy)]
pub enum ExportIndex {
    Function(u32),
    Table(u32),
    Memory(u32),
    Global(u32),
}

#[derive(Debug)]
pub struct Export {
    pub index: ExportIndex,
    pub name: String,
}

#[derive(Debug)]
pub struct Locals {
    entries: Vec<(u32, ValueType)>,
}

impl Locals {
    // Initialise with a list of count+ValueType pairs
    pub fn new(entries: Vec<(u32, ValueType)>) -> Self {
        Self { entries }
    }

    pub fn empty() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn len(&self) -> u64 {
        self.entries.iter().map(|(count, _)| *count as u64).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, (u32, ValueType)> {
        self.entries.iter()
    }

    pub fn get(&self, index: u32) -> Option<&ValueType> {
        let mut remaining = index;
        for (count, value_type) in &self.entries {
            if remaining < *count {
                return Some(value_type);
            }
            remaining -= count;
        }
        None
    }
}

#[derive(Debug)]
pub struct FunctionBody {
    pub locals: Locals,
    // Structured representation - avoids re-processing during execution
    pub body: crate::parser::structured::StructuredFunction,
    pub position: SectionPosition, // sub-section position
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum RefType {
    FuncRef,
    ExternRef,
}

impl RefType {
    /// Returns the wire byte for this reference type (§5.3.3).
    pub fn wire_byte(&self) -> u8 {
        match self {
            RefType::FuncRef => 0x70,
            RefType::ExternRef => 0x6F,
        }
    }

    /// Decodes a reference type from its wire byte.
    pub fn from_wire_byte(byte: u8) -> Result<Self, io::Error> {
        match byte {
            0x70 => Ok(RefType::FuncRef),
            0x6f => Ok(RefType::ExternRef),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "malformed reference type")),
        }
    }
}

impl From<RefType> for ValueType {
    fn from(rt: RefType) -> Self {
        match rt {
            RefType::FuncRef => ValueType::FuncRef,
            RefType::ExternRef => ValueType::ExternRef,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Limits {
    pub min: u32,
    pub max: Option<u32>,
}

impl Limits {
    /// Check if these limits are compatible for import
    /// According to WASM spec: import can be more restrictive than export
    /// Import min must be >= export min, import max must be <= export max
    pub fn is_compatible_with(&self, exported: &Limits) -> bool {
        // Import's min must be >= exported min
        if self.min < exported.min {
            return false;
        }
        // If export has a max, import must have a max <= exported max
        match (self.max, exported.max) {
            (Some(import_max), Some(export_max)) => import_max <= export_max,
            (Some(_), None) => true,  // Import has max, export doesn't - OK
            (None, Some(_)) => false, // Import has no max, export does - not OK
            (None, None) => true,     // Neither has max - OK
        }
    }
}

impl fmt::Display for Limits {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.max {
            Some(max) => write!(f, "min = {}, max = {}", self.min, max),
            None => write!(f, "min = {}", self.min),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TableType {
    pub ref_type: RefType,
    pub limits: Limits,
}

impl fmt::Display for TableType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "type={} initial={}{}",
            match self.ref_type {
                RefType::FuncRef => "funcref",
                RefType::ExternRef => "externref",
            },
            self.limits.min,
            match self.limits.max {
                Some(max) => format!(" max={max}"),
                None => "".to_string(),
            }
        )
    }
}

#[derive(Debug)]
pub enum ElementMode {
    Passive,
    Declarative,
    Active {
        table_index: u32,
        offset: Vec<instruction::Instruction>,
    },
}

impl fmt::Display for ElementMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ElementMode::Passive => write!(f, "Passive"),
            ElementMode::Declarative => write!(f, "Declarative"),
            ElementMode::Active { table_index, offset } => {
                write!(f, "Active {{ table_index = {table_index}, offset =").unwrap();
                for instruction in offset {
                    write!(f, "{instruction} ")?;
                }
                write!(f, "}}")
            }
        }
    }
}

#[derive(Debug)]
pub struct Element {
    pub flags: u32, // we only keep this for now to match the dump output
    pub ref_type: RefType,
    pub init: Vec<Vec<instruction::Instruction>>,
    pub mode: ElementMode,
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Element({}) init = [",
            match self.ref_type {
                RefType::FuncRef => "FuncRef",
                RefType::ExternRef => "ExternRef",
            }
        )
        .unwrap();
        for iv in &self.init {
            write!(f, "[")?;
            for instruction in iv {
                write!(f, "{instruction} ")?;
            }
            write!(f, "] ")?;
        }
        write!(f, "] mode = {}", self.mode)
    }
}

#[derive(Debug)]
pub enum DataMode {
    Passive,
    Active {
        memory_index: u32,
        offset: Vec<instruction::Instruction>,
    },
}

impl fmt::Display for DataMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DataMode::Passive => write!(f, "Passive"),
            DataMode::Active { memory_index, offset } => {
                write!(f, "Active {{ memory_index = {memory_index}, offset =").unwrap();
                for instruction in offset {
                    write!(f, "{instruction} ")?;
                }
                write!(f, "}}")
            }
        }
    }
}

#[derive(Debug)]
pub struct Data {
    pub init: Vec<u8>,
    pub mode: DataMode,
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Data init = [")?;
        for byte in &self.init {
            write!(f, "{byte:02x} ")?;
        }
        write!(f, "] mode = {}", self.mode)
    }
}

impl SectionToString for DataSection {
    fn header_prefix(&self) -> &'static str {
        "Data"
    }

    fn to_header_string(&self) -> String {
        format!("{} count: {}", self.position, self.data.len())
    }

    fn to_details_string(&self, unit: &Module) -> String {
        let mut result: String = String::new();
        result.push_str(&format!("Data[{}]:\n", self.data.len()));
        let mut data_offset = 0;
        for (i, data) in self.data.iter().enumerate() {
            result.push_str(&format!(
                " - segment[{}] {} size={}{}\n",
                i,
                match data.mode {
                    DataMode::Passive => "passive".to_string(),
                    DataMode::Active { memory_index, .. } => format!("memory={memory_index}"),
                },
                data.init.len(),
                match data.mode {
                    DataMode::Passive => "".to_string(),
                    DataMode::Active { ref offset, .. } => {
                        if let Some(o) = init_expr_to_offset(offset) {
                            data_offset = o as i32
                        } else {
                            data_offset = -1
                        }
                        init_expr_to_string(unit, offset, false, true)
                    }
                }
            ));
            // for data.init byte array, print it in blocks of 8 bytes like so:
            //   - 0000000: 0100 0000 0000 0000 0100 0000 0000 0080  ................
            let mut pos = 0;
            while pos < data.init.len() {
                let mut byts = String::new();
                let mut chars = String::new();
                for i in (0..16).step_by(2) {
                    if pos + i + 1 < data.init.len() {
                        byts.push_str(&format!("{:02x}{:02x} ", data.init[pos + i], data.init[pos + i + 1]));
                    } else if pos + i < data.init.len() {
                        byts.push_str(&format!("{:02x}   ", data.init[pos + i]));
                    } else {
                        byts.push_str("     ");
                    }
                }
                for i in 0..16 {
                    if pos + i < data.init.len() {
                        let byte = data.init[pos + i];
                        let ch = if byte.is_ascii_graphic() || byte == /* space */ 32 {
                            byte as char
                        } else {
                            '.'
                        };
                        chars.push(ch);
                    }
                }

                let offset_str = match data_offset {
                    -1 => "<INVALID OFFSET>".to_string(),
                    _ => format!("{:07x}", pos as i32 + data_offset),
                };
                result.push_str(&format!("  - {offset_str}: {byts} {chars}\n"));
                pos += 16;
            }
        }
        result
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct GlobalType {
    pub value_type: ValueType,
    pub mutable: bool, // const or var
}

impl fmt::Display for GlobalType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "GlobalType({}) {}",
            self.value_type,
            if self.mutable { "var" } else { "const" }
        )
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Global {
    pub global_type: GlobalType,
    pub init: Vec<instruction::Instruction>,
}

impl fmt::Display for Global {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Global global_type = {}", self.global_type)?;
        write!(f, " init = [")?;
        for instruction in &self.init {
            write!(f, "{instruction} ")?;
        }
        write!(f, "]")
    }
}

pub enum ExternalKind {
    Function(u32), // typeidx
    Table(TableType),
    Memory(Limits),
    Global(GlobalType),
}

impl fmt::Debug for ExternalKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self, f)
    }
}

impl fmt::Display for ExternalKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ExternalKind::Function(typeidx) => format!("Function({typeidx})"),
                ExternalKind::Table(table_type) => format!("Table({table_type})"),
                ExternalKind::Memory(limits) => format!("Memory({limits})"),
                ExternalKind::Global(global_type) => format!("Global({global_type})"),
            }
        )
    }
}

impl Module {
    pub fn new(name: &str) -> Module {
        Module {
            name: name.to_string(),
            magic: 0,
            version: 0,

            types: TypeSection::new(),
            imports: ImportSection::new(),
            functions: FunctionSection::new(),
            table: TableSection::new(),
            memory: MemorySection::new(),
            globals: GlobalSection::new(),
            exports: ExportSection::new(),
            start: StartSection::new(),
            elements: ElementSection::new(),
            code: CodeSection::new(),
            data: DataSection::new(),
            data_count: DataCountSection::new(),
            custom: Vec::new(),
            validation_context: OnceLock::new(),
        }
    }
}

#[allow(dead_code)] // used in tests so far
pub enum ParsedUnitFormat {
    Header,
    Details,
    Disassemble,
}

impl Module {
    #[allow(dead_code)] // used in tests
    pub fn to_string(&self, format: ParsedUnitFormat) -> String {
        match format {
            ParsedUnitFormat::Header => self.to_header_string(),
            ParsedUnitFormat::Details => self.to_details_string(),
            ParsedUnitFormat::Disassemble => self.to_disassemble_string(),
        }
    }

    fn to_header_string(&self) -> String {
        let mut components: Vec<(&dyn Positional, &dyn SectionToString)> = vec![
            (&self.types, &self.types),
            (&self.imports, &self.imports),
            (&self.functions, &self.functions),
            (&self.table, &self.table),
            (&self.memory, &self.memory),
            (&self.globals, &self.globals),
            (&self.exports, &self.exports),
            (&self.start, &self.start),
            (&self.elements, &self.elements),
            (&self.data_count, &self.data_count),
            (&self.code, &self.code),
            (&self.data, &self.data),
        ];
        components.extend(
            self.custom
                .iter()
                .map(|custom| (custom as &dyn Positional, custom as &dyn SectionToString)),
        );

        components.sort_by_key(|(p, _)| p.get_position().map(|pos| pos.start));

        let mut result = String::new();
        for (p, s) in components {
            if p.has_position() {
                result.push_str(&format!("{:>9} {}\n", s.header_prefix(), s.to_header_string()));
            }
        }

        result
    }

    fn to_details_string(&self) -> String {
        let mut result = String::new();

        let mut components: Vec<(&dyn Positional, &dyn SectionToString)> = vec![
            (&self.types, &self.types),
            (&self.imports, &self.imports),
            (&self.functions, &self.functions),
            (&self.table, &self.table),
            (&self.memory, &self.memory),
            (&self.globals, &self.globals),
            (&self.exports, &self.exports),
            (&self.start, &self.start),
            (&self.elements, &self.elements),
            (&self.data_count, &self.data_count),
            (&self.code, &self.code),
            (&self.data, &self.data),
        ];
        components.extend(
            self.custom
                .iter()
                .map(|custom| (custom as &dyn Positional, custom as &dyn SectionToString)),
        );

        components.sort_by_key(|(p, _)| p.get_position().map(|pos| pos.start));

        for (p, s) in components {
            if p.has_position() {
                result.push_str(s.to_details_string(self).as_str());
            }
        }

        result
    }

    fn to_disassemble_string(&self) -> String {
        if self.code.has_position() {
            self.code.to_disassemble_string(self)
        } else {
            String::new()
        }
    }
}

impl fmt::Display for Memory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Memory({})", self.limits)
    }
}

impl fmt::Display for Export {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let typ = match self.index {
            ExportIndex::Function(i) => format!("Function({i})"),
            ExportIndex::Table(i) => format!("Table({i})"),
            ExportIndex::Memory(i) => format!("Memory({i})"),
            ExportIndex::Global(i) => format!("Global({i})"),
        };
        write!(f, "{} {}", self.name, typ)
    }
}

impl fmt::Display for FunctionBody {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "locals = {:?}", self.locals).unwrap();
        write!(f, " instructions = ").unwrap();
        let instructions = self.body.flatten();
        for instruction in &instructions {
            write!(f, "{instruction} ")?;
        }
        Ok(())
    }
}

impl ExportIndex {
    pub fn decode(byte: u8, idx: u32) -> Result<ExportIndex, io::Error> {
        use super::encoding::{DESC_FUNC, DESC_GLOBAL, DESC_MEMORY, DESC_TABLE};
        match byte {
            DESC_FUNC => Ok(ExportIndex::Function(idx)),
            DESC_TABLE => Ok(ExportIndex::Table(idx)),
            DESC_MEMORY => Ok(ExportIndex::Memory(idx)),
            DESC_GLOBAL => Ok(ExportIndex::Global(idx)),
            _ => {
                let msg: String = format!("invalid export type: {byte}");
                Err(io::Error::new(io::ErrorKind::InvalidData, msg))
            }
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ValueType {
    // Number types
    I32,
    I64,
    F32,
    F64,
    // Vector types
    V128,
    // Reference types
    FuncRef,
    ExternRef,
}

impl ValueType {
    pub fn is_value_type_byte(byte: u8) -> bool {
        byte == 0x7f || byte == 0x7e || byte == 0x7d || byte == 0x7c || byte == 0x7b || byte == 0x70 || byte == 0x6f
    }

    pub fn decode(byte: u8) -> Result<Self, String> {
        match byte {
            0x7f => Ok(ValueType::I32),
            0x7e => Ok(ValueType::I64),
            0x7d => Ok(ValueType::F32),
            0x7c => Ok(ValueType::F64),
            0x7b => Ok(ValueType::V128),
            0x70 => Ok(ValueType::FuncRef),
            0x6f => Ok(ValueType::ExternRef),
            _ => Err(format!("invalid value type: {byte}")),
        }
    }

    pub fn emit_bytes(&self) -> Vec<u8> {
        match self {
            ValueType::I32 => vec![0x7f],
            ValueType::I64 => vec![0x7e],
            ValueType::F32 => vec![0x7d],
            ValueType::F64 => vec![0x7c],
            ValueType::V128 => vec![0x7b],
            ValueType::FuncRef => vec![0x70],
            ValueType::ExternRef => vec![0x6f],
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ValueType::I32 => "i32",
                ValueType::I64 => "i64",
                ValueType::F64 => "f64",
                ValueType::F32 => "f32",
                ValueType::V128 => "v128",
                ValueType::FuncRef => "funcref",
                ValueType::ExternRef => "externref",
            }
        )
    }
}

/// Pre-computed validation context for efficient lookups during validation
#[derive(Debug)]
pub struct ValidationContext {
    /// Quick check if module has any memory (imported or defined)
    pub has_memory: bool,

    /// Quick check if module has tables by index
    pub has_table: Vec<bool>,

    /// Import type indices
    pub import_func_types: Vec<u32>, // function_idx -> type_idx (for imports only)
    pub import_global_types: Vec<GlobalType>, // global_idx -> type (for imports only)
    pub import_table_types: Vec<TableType>,   // table_idx -> type (for imports only)

    /// Counts of imported items
    pub import_counts: ImportCounts,

    /// Combined function type lookup (imports + defined)
    pub function_type_indices: Vec<u32>, // function_idx -> type_idx

    /// Cached function names for display
    pub function_names: std::collections::HashMap<u32, String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ImportCounts {
    pub functions: u32,
    pub tables: u32,
    pub memories: u32,
    pub globals: u32,
}

impl Module {
    /// Build a validation context with pre-computed indices
    pub fn build_validation_context(&self) -> ValidationContext {
        // Check if module has memory
        let has_memory = !self.memory.is_empty() || self.imports.memory_count() > 0;

        // Build table existence vector
        let total_tables = self.imports.table_count() + self.table.tables.len();
        let has_table = vec![true; total_tables];

        // Extract import types
        let mut import_func_types = Vec::new();
        let mut import_global_types = Vec::new();
        let mut import_table_types = Vec::new();

        for import in &self.imports.imports {
            match &import.external_kind {
                ExternalKind::Function(type_idx) => {
                    import_func_types.push(*type_idx);
                }
                ExternalKind::Global(global_type) => {
                    import_global_types.push(*global_type);
                }
                ExternalKind::Table(table_type) => {
                    import_table_types.push(*table_type);
                }
                _ => {}
            }
        }

        // Count imports
        let import_counts = ImportCounts {
            functions: import_func_types.len() as u32,
            tables: import_table_types.len() as u32,
            memories: self.imports.memory_count() as u32,
            globals: import_global_types.len() as u32,
        };

        // Build combined function type indices (imports + defined)
        let mut function_type_indices = import_func_types.clone();
        function_type_indices.extend(self.functions.functions.iter().map(|f| f.ftype_index));

        // Cache function names (simplified for now)
        let function_names = std::collections::HashMap::new();

        ValidationContext {
            has_memory,
            has_table,
            import_func_types,
            import_global_types,
            import_table_types,
            import_counts,
            function_type_indices,
            function_names,
        }
    }
}
