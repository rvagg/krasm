//! WAT Parser: S-Expression Tree -> WebAssembly Module.
//!
//! Transforms the S-expression tree from [`sexpr::read`] into a WebAssembly [`Module`].
//!
//! # Design Principles
//!
//! 1. **Grammar Correspondence**: Each parsing function corresponds to a grammar
//!    production. Comments show the grammar rule being implemented.
//!
//! 2. **No Lookahead**: Because we work on a tree, not a token stream, we can
//!    always see all children of a node. No speculative parsing needed.
//!
//! 3. **Consistent Errors**: All errors use the same pattern, with precise spans.
//!
//! 4. **Scope-Based State**: Label scopes use RAII guards, not manual push/pop.
//!
//! # Grammar Reference
//!
//! Based on the WebAssembly Text Format specification:
//! <https://webassembly.github.io/spec/core/text/index.html>

use super::sexpr::{ReadError, SExpr, SExprList};
use super::token::{FloatLit, Span, TokenKind};
use crate::parser::encoding::{
    ELEM_ACTIVE_EXPRS, ELEM_ACTIVE_FUNCS, ELEM_ACTIVE_TABLE_EXPRS, ELEM_ACTIVE_TABLE_FUNCS, ELEM_DECLARATIVE_EXPRS,
    ELEM_DECLARATIVE_FUNCS, ELEM_PASSIVE_EXPRS, ELEM_PASSIVE_FUNCS,
};
use crate::parser::instruction::{BlockType, ByteRange, Instruction, InstructionKind, MemArg, SimdOp};
use crate::parser::module::{
    CodeSection, Data, DataCountSection, DataMode, DataSection, Element, ElementMode, ElementSection, Export,
    ExportIndex, ExportSection, ExternalKind, Function, FunctionBody, FunctionSection, FunctionType, Global,
    GlobalSection, GlobalType, Import, ImportSection, Limits, Locals, Memory, MemorySection, Module, RefType,
    SectionPosition, StartSection, TableSection, TableType, TypeSection, ValueType,
};
use crate::parser::structure_builder::StructureBuilder;
use std::collections::HashMap;
use std::fmt;

// ============================================================================
// Constants
// ============================================================================

// Memory alignment constants.
//
// WebAssembly encodes alignment as log2(bytes), not raw bytes. These constants
// are the default ("natural") alignments for each operand size, used when
// the WAT source omits an explicit `align=N`.
//
// The alignment hint tells the runtime how the address will be aligned in
// memory, enabling optimised load/store paths on architectures that care.
const ALIGN_8: u32 = 0; // log2(1) - for i32.load8_s, i64.load8_u, etc.
const ALIGN_16: u32 = 1; // log2(2) - for i32.load16_s, i64.load16_u, etc.
const ALIGN_32: u32 = 2; // log2(4) - for i32.load, f32.load, i32.store, etc.
const ALIGN_64: u32 = 3; // log2(8) - for i64.load, f64.load, i64.store, etc.

// ============================================================================
// Error Type
// ============================================================================

/// An error encountered during WAT parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl ParseError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }

    /// Creates an "expected X, found Y" error.
    pub fn expected(expected: &str, found: &str, span: Span) -> Self {
        Self::new(format!("expected {}, found {}", expected, found), span)
    }

    /// Creates an "undefined X" error.
    pub fn undefined(kind: &str, name: &str, span: Span) -> Self {
        Self::new(format!("undefined {}: ${}", kind, name), span)
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at line {}, column {}",
            self.message, self.span.line, self.span.column
        )
    }
}

impl std::error::Error for ParseError {}

impl From<ReadError> for ParseError {
    fn from(e: ReadError) -> Self {
        Self {
            message: e.message,
            span: e.span,
        }
    }
}

// ============================================================================
// Namespaces
// ============================================================================

/// Index namespaces in WebAssembly.
///
/// WebAssembly has separate index spaces for different kinds of definitions.
/// Named references ($name) are resolved within their respective namespace.
///
/// Labels are not included here, they use stack-based scoping via
/// `push_label()`/`pop_label()`/`resolve_label()` instead of a flat mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Namespace {
    Type,
    Func,
    Table,
    Memory,
    Global,
    Local,
    Elem,
    Data,
}

impl Namespace {
    fn name(self) -> &'static str {
        match self {
            Namespace::Type => "type",
            Namespace::Func => "function",
            Namespace::Table => "table",
            Namespace::Memory => "memory",
            Namespace::Elem => "element",
            Namespace::Data => "data",
            Namespace::Global => "global",
            Namespace::Local => "local",
        }
    }
}

// ============================================================================
// Parse Context
// ============================================================================

/// Accumulates state during parsing.
///
/// Tracks name->index mappings for each namespace, collects definitions,
/// and manages the label stack for branch target resolution.
struct ParseContext {
    // Name tables
    type_names: HashMap<String, u32>,
    func_names: HashMap<String, u32>,
    table_names: HashMap<String, u32>,
    memory_names: HashMap<String, u32>,
    global_names: HashMap<String, u32>,
    local_names: HashMap<String, u32>,
    elem_names: HashMap<String, u32>,
    data_names: HashMap<String, u32>,

    // Counters (includes imports)
    type_count: u32,
    func_count: u32,
    table_count: u32,
    memory_count: u32,
    global_count: u32,
    local_count: u32,
    param_count: u32,
    elem_count: u32,
    data_count: u32,

    // Label stack for branch resolution
    label_stack: Vec<Option<String>>,

    // Track definitions for import ordering validation
    has_func_def: bool,
    has_table_def: bool,
    has_memory_def: bool,
    has_global_def: bool,

    // Module sections
    types: Vec<FunctionType>,
    imports: Vec<Import>,
    functions: Vec<Function>,
    tables: Vec<TableType>,
    memories: Vec<Memory>,
    globals: Vec<Global>,
    exports: Vec<Export>,
    start: Option<u32>,
    elements: Vec<Element>,
    code: Vec<FunctionBody>,
    data: Vec<Data>,
}

impl ParseContext {
    fn new() -> Self {
        Self {
            type_names: HashMap::new(),
            func_names: HashMap::new(),
            table_names: HashMap::new(),
            memory_names: HashMap::new(),
            global_names: HashMap::new(),
            local_names: HashMap::new(),
            elem_names: HashMap::new(),
            data_names: HashMap::new(),

            type_count: 0,
            func_count: 0,
            table_count: 0,
            memory_count: 0,
            global_count: 0,
            local_count: 0,
            param_count: 0,
            elem_count: 0,
            data_count: 0,

            label_stack: Vec::new(),

            has_func_def: false,
            has_table_def: false,
            has_memory_def: false,
            has_global_def: false,

            types: Vec::new(),
            imports: Vec::new(),
            functions: Vec::new(),
            tables: Vec::new(),
            memories: Vec::new(),
            globals: Vec::new(),
            exports: Vec::new(),
            start: None,
            elements: Vec::new(),
            code: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Resets per-function state (locals, labels) for a new function.
    fn reset_function(&mut self) {
        self.local_names.clear();
        self.local_count = 0;
        self.param_count = 0;
        self.label_stack.clear();
    }

    /// Pre-registers a name for forward reference resolution.
    /// Only inserts the name -> index mapping and bumps the counter.
    /// Returns an error if the name already exists in this namespace.
    fn pre_register(&mut self, ns: Namespace, name: Option<&str>, span: Span) -> Result<(), ParseError> {
        let (names, count) = self.ns_mut(ns);
        let idx = *count;
        *count += 1;
        if let Some(n) = name {
            if names.contains_key(n) {
                let label = match ns {
                    Namespace::Func => "func",
                    other => other.name(),
                };
                return Err(ParseError::new(format!("duplicate {label}"), span));
            }
            names.insert(n.to_string(), idx);
        }
        Ok(())
    }

    /// Resets counters after pre-registration, keeping names intact.
    fn reset_pre_registration(&mut self) {
        self.func_count = 0;
        self.table_count = 0;
        self.memory_count = 0;
        self.global_count = 0;
        // Type, Local, Elem, Data counters are not pre-registered
    }

    /// Returns (names, count) for the given namespace.
    fn ns_mut(&mut self, ns: Namespace) -> (&mut HashMap<String, u32>, &mut u32) {
        match ns {
            Namespace::Type => (&mut self.type_names, &mut self.type_count),
            Namespace::Func => (&mut self.func_names, &mut self.func_count),
            Namespace::Table => (&mut self.table_names, &mut self.table_count),
            Namespace::Memory => (&mut self.memory_names, &mut self.memory_count),
            Namespace::Global => (&mut self.global_names, &mut self.global_count),
            Namespace::Local => (&mut self.local_names, &mut self.local_count),
            Namespace::Elem => (&mut self.elem_names, &mut self.elem_count),
            Namespace::Data => (&mut self.data_names, &mut self.data_count),
        }
    }

    /// Registers a named or anonymous definition, returning its index.
    /// For the Local namespace, detects duplicate names and returns an error.
    /// For other namespaces, pre-registered names are already in the map
    /// (duplicate detection happens during pre_register).
    fn register(&mut self, ns: Namespace, name: Option<&str>, span: Span) -> Result<u32, ParseError> {
        let (names, count) = self.ns_mut(ns);

        let idx = *count;
        *count += 1;
        if let Some(n) = name {
            if ns == Namespace::Local {
                if names.contains_key(n) {
                    return Err(ParseError::new("duplicate local", span));
                }
                names.insert(n.to_string(), idx);
            } else {
                // Pre-registered names are already in the map; skip re-insertion
                names.entry(n.to_string()).or_insert(idx);
            }
        }
        Ok(idx)
    }

    /// Resolves a name to an index in the given namespace.
    fn resolve(&self, ns: Namespace, name: &str, span: Span) -> Result<u32, ParseError> {
        let names = match ns {
            Namespace::Type => &self.type_names,
            Namespace::Func => &self.func_names,
            Namespace::Table => &self.table_names,
            Namespace::Memory => &self.memory_names,
            Namespace::Global => &self.global_names,
            Namespace::Local => &self.local_names,
            Namespace::Elem => &self.elem_names,
            Namespace::Data => &self.data_names,
        };

        names
            .get(name)
            .copied()
            .ok_or_else(|| ParseError::undefined(ns.name(), name, span))
    }

    /// Resolves a label name to a branch depth.
    fn resolve_label(&self, name: &str, span: Span) -> Result<u32, ParseError> {
        for (depth, label) in self.label_stack.iter().rev().enumerate() {
            if label.as_deref() == Some(name) {
                return Ok(depth as u32);
            }
        }
        Err(ParseError::undefined("label", name, span))
    }

    /// Pushes a label onto the stack (for block/loop/if).
    fn push_label(&mut self, name: Option<String>) {
        self.label_stack.push(name);
    }

    /// Pops a label from the stack.
    fn pop_label(&mut self) {
        self.label_stack.pop();
    }

    /// Registers a type, deduplicating identical signatures.
    ///
    /// When a duplicate signature is found, the name (if any) is bound to the
    /// existing index. Duplicate names silently overwrite - the WAT spec forbids
    /// them but we don't enforce that.
    /// Registers a type, deduplicating against existing identical types.
    /// Used for implicit type uses in functions (no explicit `(type ...)` definition).
    fn register_type(&mut self, name: Option<&str>, func_type: FunctionType) -> u32 {
        // Check for existing identical type
        for (idx, existing) in self.types.iter().enumerate() {
            if existing.parameters == func_type.parameters && existing.return_types == func_type.return_types {
                if let Some(n) = name {
                    self.type_names.insert(n.to_string(), idx as u32);
                }
                return idx as u32;
            }
        }

        self.add_type(name, func_type)
    }

    /// Adds a type definition without deduplication.
    /// Each explicit `(type ...)` definition gets its own index per spec.
    fn add_type(&mut self, name: Option<&str>, func_type: FunctionType) -> u32 {
        let idx = self.type_count;
        self.type_count += 1;
        if let Some(n) = name {
            self.type_names.insert(n.to_string(), idx);
        }
        self.types.push(func_type);
        idx
    }

    /// Finds an existing type with the given signature, returning its index.
    fn find_type(&self, func_type: &FunctionType) -> Option<u32> {
        for (idx, existing) in self.types.iter().enumerate() {
            if existing.parameters == func_type.parameters && existing.return_types == func_type.return_types {
                return Some(idx as u32);
            }
        }
        None
    }

    fn find_or_add_type(&mut self, func_type: FunctionType) -> u32 {
        if let Some(idx) = self.find_type(&func_type) {
            return idx;
        }
        self.add_type(None, func_type)
    }

    /// Builds the final Module from accumulated state.
    fn into_module(self) -> Module {
        let start_section = if let Some(func_idx) = self.start {
            StartSection {
                start: func_idx,
                position: SectionPosition::new(1, 1), // Non-zero indicates presence
            }
        } else {
            StartSection {
                start: 0,
                position: SectionPosition::new(0, 0), // Zero indicates absence
            }
        };

        let data_count = self.data.len() as u32;

        Module {
            name: String::new(),
            magic: 0x6d736100,
            version: 1,
            types: TypeSection {
                types: self.types,
                position: SectionPosition::new(0, 0),
            },
            imports: ImportSection {
                imports: self.imports,
                position: SectionPosition::new(0, 0),
            },
            functions: FunctionSection {
                functions: self.functions,
                position: SectionPosition::new(0, 0),
            },
            table: TableSection {
                tables: self.tables,
                position: SectionPosition::new(0, 0),
            },
            memory: MemorySection {
                memory: self.memories,
                position: SectionPosition::new(0, 0),
            },
            globals: GlobalSection {
                globals: self.globals,
                position: SectionPosition::new(0, 0),
            },
            exports: ExportSection {
                exports: self.exports,
                position: SectionPosition::new(0, 0),
            },
            start: start_section,
            elements: ElementSection {
                elements: self.elements,
                position: SectionPosition::new(0, 0),
            },
            code: CodeSection {
                code: self.code,
                position: SectionPosition::new(0, 0),
            },
            data: DataSection {
                data: self.data,
                position: SectionPosition::new(0, 0),
            },
            data_count: DataCountSection {
                count: data_count,
                position: SectionPosition::new(0, 0),
            },
            custom: Vec::new(),
            validation_context: std::sync::OnceLock::new(),
        }
    }
}

// ============================================================================
// Parser
// ============================================================================

/// Parses a WAT source string into a WebAssembly Module.
///
/// # Grammar
///
/// ```text
/// module ::= '(' 'module' id? field* ')'
/// ```
///
/// # Example
///
/// ```
/// use krasm::wat::parse;
///
/// let module = parse("(module (func (result i32) (i32.const 42)))").unwrap();
/// assert_eq!(module.functions.len(), 1);
/// ```
#[must_use = "parsing result should be checked"]
pub fn parse(source: &str) -> Result<Module, ParseError> {
    let sexpr = super::sexpr::read(source)?;
    let mut ctx = ParseContext::new();
    parse_module(&sexpr, &mut ctx)?;
    let module = ctx.into_module();
    crate::parser::validate::validate_module(&module).map_err(|e| ParseError::new(e.to_string(), Span::ZERO))?;
    Ok(module)
}

/// Parses a module from its S-expression.
///
/// Grammar: `module ::= '(' 'module' id? field* ')'`
fn parse_module(sexpr: &SExpr, ctx: &mut ParseContext) -> Result<(), ParseError> {
    let list = sexpr.expect_list()?;
    list.expect_head("module")?;

    // Optional module name (ignored but validated)
    let start_idx = if list.get(1).and_then(|s| s.as_id()).is_some() {
        2
    } else {
        1
    };

    // Pre-registration: scan all fields to register names before parsing bodies.
    // This enables forward references (e.g. function A calling function B defined later).
    pre_register_fields(list, start_idx, ctx)?;

    // Pre-pass: parse all type definitions first so forward type references resolve.
    for item in list.iter_from(start_idx) {
        if let Some(inner) = item.as_list()
            && inner.head_keyword() == Some("type")
        {
            parse_type_def(inner, ctx)?;
        }
    }

    // Parse fields (type defs are idempotent due to deduplication)
    for item in list.iter_from(start_idx) {
        parse_field(item, ctx)?;
    }

    Ok(())
}

/// Pre-registers all named items so forward references resolve correctly.
/// Only registers names and bumps counters, does not parse bodies.
fn pre_register_fields(list: SExprList<'_>, start: usize, ctx: &mut ParseContext) -> Result<(), ParseError> {
    for item in list.iter_from(start) {
        let Some(inner) = item.as_list() else {
            continue;
        };
        let Some(kw) = inner.head_keyword() else {
            continue;
        };
        match kw {
            "type" => {
                let name = inner.get(1).and_then(|s| s.as_id());
                // Types are registered during parsing (deduplication), just skip
                let _ = name;
            }
            "func" => {
                let mut idx = 1;
                let name = inner.get(idx).and_then(|s| s.as_id());
                if name.is_some() {
                    idx += 1;
                }
                // Check for inline import: (func $name (import "m" "n") ...)
                let is_inline_import = inner
                    .get(idx)
                    .and_then(|s| s.as_list())
                    .is_some_and(|l| l.head_keyword() == Some("import"));
                // Skip inline exports
                while inner
                    .get(idx)
                    .and_then(|s| s.as_list())
                    .is_some_and(|l| l.head_keyword() == Some("export"))
                {
                    idx += 1;
                }
                let _ = is_inline_import;
                ctx.pre_register(Namespace::Func, name, inner.span)?;
            }
            "table" => {
                let name = inner.get(1).and_then(|s| s.as_id());
                ctx.pre_register(Namespace::Table, name, inner.span)?;
            }
            "memory" => {
                let name = inner.get(1).and_then(|s| s.as_id());
                ctx.pre_register(Namespace::Memory, name, inner.span)?;
            }
            "global" => {
                let name = inner.get(1).and_then(|s| s.as_id());
                ctx.pre_register(Namespace::Global, name, inner.span)?;
            }
            "import" => {
                // (import "mod" "name" (func|table|memory|global $name? ...))
                if let Some(desc) = inner.get(3).and_then(|s| s.as_list()) {
                    let ns = match desc.head_keyword() {
                        Some("func") => Some(Namespace::Func),
                        Some("table") => Some(Namespace::Table),
                        Some("memory") => Some(Namespace::Memory),
                        Some("global") => Some(Namespace::Global),
                        _ => None,
                    };
                    if let Some(ns) = ns {
                        let name = desc.get(1).and_then(|s| s.as_id());
                        ctx.pre_register(ns, name, desc.span)?;
                    }
                }
            }
            _ => {}
        }
    }
    // Reset counters, actual registration happens during parsing
    ctx.reset_pre_registration();
    Ok(())
}

/// Parses a module field.
///
/// Grammar: `field ::= type | import | func | table | memory | global | export | start | elem | data`
fn parse_field(sexpr: &SExpr, ctx: &mut ParseContext) -> Result<(), ParseError> {
    let list = sexpr.expect_list()?;
    let keyword = list
        .head_keyword()
        .ok_or_else(|| ParseError::new("expected field keyword", sexpr.span()))?;

    match keyword {
        "type" => Ok(()), // Already parsed in the type pre-pass
        "import" => parse_import(list, ctx),
        "func" => parse_func(list, ctx),
        "table" => parse_table(list, ctx),
        "memory" => parse_memory(list, ctx),
        "global" => parse_global(list, ctx),
        "export" => parse_export(list, ctx),
        "start" => parse_start(list, ctx),
        "elem" => parse_elem(list, ctx),
        "data" => parse_data(list, ctx),
        _ => Err(ParseError::new(format!("unknown field type: {}", keyword), list.span)),
    }
}

// ============================================================================
// Type Definitions
// ============================================================================

/// Parses a type definition.
///
/// Grammar: `type ::= '(' 'type' id? '(' 'func' functype ')' ')'`
fn parse_type_def(list: SExprList<'_>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    let mut idx = 1;
    let name = take_optional_name(list, &mut idx);

    // Expect (func ...)
    let func_sexpr = list
        .get(idx)
        .ok_or_else(|| ParseError::new("expected function type", list.span))?;
    let func_list = func_sexpr.expect_list()?;
    func_list.expect_head("func")?;

    let func_type = parse_func_type(func_list)?;
    ctx.add_type(name, func_type);

    Ok(())
}

/// Parses a function type signature.
///
/// Grammar: `functype ::= '(' 'func' param* result* ')'`
fn parse_func_type(list: SExprList<'_>) -> Result<FunctionType, ParseError> {
    let mut parameters = Vec::new();
    let mut return_types = Vec::new();
    let mut seen_result = false;

    for item in list.iter_from(1) {
        let inner = item.expect_list()?;
        match inner.head_keyword() {
            Some("param") => {
                if seen_result {
                    return Err(ParseError::new("result before parameter", inner.span));
                }
                for param_item in inner.iter_from(1) {
                    // Skip optional parameter names
                    if param_item.as_id().is_some() {
                        continue;
                    }
                    parameters.push(parse_valtype(param_item)?);
                }
            }
            Some("result") => {
                seen_result = true;
                for result_item in inner.iter_from(1) {
                    return_types.push(parse_valtype(result_item)?);
                }
            }
            Some(kw) => {
                return Err(ParseError::new(
                    format!("expected 'param' or 'result', found '{}'", kw),
                    inner.span,
                ));
            }
            None => return Err(ParseError::new("expected keyword in type", inner.span)),
        }
    }

    Ok(FunctionType {
        parameters,
        return_types,
    })
}

/// Parses a value type.
///
/// Grammar: `valtype ::= numtype | vectype | reftype`
fn parse_valtype(sexpr: &SExpr) -> Result<ValueType, ParseError> {
    match sexpr.as_keyword() {
        Some("i32") => Ok(ValueType::I32),
        Some("i64") => Ok(ValueType::I64),
        Some("f32") => Ok(ValueType::F32),
        Some("f64") => Ok(ValueType::F64),
        Some("v128") => Ok(ValueType::V128),
        Some("funcref") => Ok(ValueType::FuncRef),
        Some("externref") => Ok(ValueType::ExternRef),
        Some(kw) => Err(ParseError::expected("value type", &format!("'{}'", kw), sexpr.span())),
        None => Err(ParseError::expected("value type", "list", sexpr.span())),
    }
}

// ============================================================================
// Imports
// ============================================================================

/// Parses an import.
///
/// Grammar: `import ::= '(' 'import' name name importdesc ')'`
fn parse_import(list: SExprList<'_>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    // Validate import ordering: imports must come before definitions
    for (has_def, kind) in [
        (ctx.has_func_def, "function"),
        (ctx.has_table_def, "table"),
        (ctx.has_memory_def, "memory"),
        (ctx.has_global_def, "global"),
    ] {
        if has_def {
            return Err(ParseError::new(format!("import after {kind}"), list.span));
        }
    }

    let module_name = parse_string(
        list.get(1)
            .ok_or_else(|| ParseError::new("expected module name string", list.span))?,
    )?;

    let field_name = parse_string(
        list.get(2)
            .ok_or_else(|| ParseError::new("expected field name string", list.span))?,
    )?;

    let desc_sexpr = list
        .get(3)
        .ok_or_else(|| ParseError::new("expected import descriptor", list.span))?;
    let desc_list = desc_sexpr.expect_list()?;

    let external_kind = match desc_list.head_keyword() {
        Some("func") => {
            let name = desc_list.get(1).and_then(|s| s.as_id());
            ctx.register(Namespace::Func, name, desc_list.span)?;
            let start = if name.is_some() { 2 } else { 1 };
            let (type_idx, _) = parse_type_use(desc_list, start, false, ctx)?;
            ExternalKind::Function(type_idx)
        }
        Some("memory") => {
            let name = desc_list.get(1).and_then(|s| s.as_id());
            ctx.register(Namespace::Memory, name, desc_list.span)?;
            let start = if name.is_some() { 2 } else { 1 };
            let limits = parse_limits(desc_list, start)?;
            ExternalKind::Memory(limits)
        }
        Some("global") => {
            let name = desc_list.get(1).and_then(|s| s.as_id());
            ctx.register(Namespace::Global, name, desc_list.span)?;
            let start = if name.is_some() { 2 } else { 1 };
            let global_type = parse_global_type(desc_list, start)?;
            ExternalKind::Global(global_type)
        }
        Some("table") => {
            let name = desc_list.get(1).and_then(|s| s.as_id());
            ctx.register(Namespace::Table, name, desc_list.span)?;
            let start = if name.is_some() { 2 } else { 1 };
            let limits = parse_limits(desc_list, start)?;
            let ref_type = parse_reftype(desc_list.get(start + limits_len(&limits)))?;
            ExternalKind::Table(TableType { ref_type, limits })
        }
        Some(kw) => return Err(ParseError::new(format!("unknown import kind: {}", kw), desc_list.span)),
        None => return Err(ParseError::new("expected import kind", desc_list.span)),
    };

    ctx.imports.push(Import {
        module: module_name,
        name: field_name,
        external_kind,
    });

    Ok(())
}

/// Parses a type use (type reference or inline params/results).
///
/// Grammar: `typeuse ::= '(' 'type' idx ')' | param* result*`
///
/// When `register_locals` is true, parameter names are registered in the
/// local namespace (for function definitions). Returns the type index and
/// the position after the last type-related item.
fn parse_type_use(
    list: SExprList<'_>,
    start: usize,
    register_locals: bool,
    ctx: &mut ParseContext,
) -> Result<(u32, usize), ParseError> {
    let mut params = Vec::new();
    let mut results = Vec::new();
    let mut explicit_idx = None;
    let mut idx = start;
    let mut seen_param = false;
    let mut seen_result = false;

    while let Some(item) = list.get(idx) {
        let Some(inner) = item.as_list() else {
            break;
        };

        match inner.head_keyword() {
            Some("type") => {
                if seen_param || seen_result {
                    return Err(ParseError::new("unexpected token", inner.span));
                }
                explicit_idx = Some(parse_index(inner.get(1), Namespace::Type, ctx)?);
                idx += 1;
            }
            Some("param") => {
                if seen_result {
                    return Err(ParseError::new("unexpected token", inner.span));
                }
                seen_param = true;
                // Grammar: param ::= '(' 'param' id valtype ')' | '(' 'param' valtype* ')'
                let first = inner.get(1);
                let name = first.and_then(|s| s.as_id());
                if let Some(name) = name {
                    // Named form: exactly one valtype
                    let ty_item = inner
                        .get(2)
                        .ok_or_else(|| ParseError::new("expected valtype after param name", inner.span))?;
                    let ty = parse_valtype(ty_item)?;
                    if register_locals {
                        ctx.register(Namespace::Local, Some(name), inner.span)?;
                        ctx.param_count += 1;
                    }
                    params.push(ty);
                } else {
                    // Anonymous form: zero or more valtypes
                    for p in inner.iter_from(1) {
                        let ty = parse_valtype(p)?;
                        if register_locals {
                            ctx.register(Namespace::Local, None, inner.span)?;
                            ctx.param_count += 1;
                        }
                        params.push(ty);
                    }
                }
                idx += 1;
            }
            Some("result") => {
                seen_result = true;
                for r in inner.iter_from(1) {
                    results.push(parse_valtype(r)?);
                }
                idx += 1;
            }
            _ => break,
        }
    }

    let type_idx = if let Some(eidx) = explicit_idx {
        // Validate inline params/results match the referenced type
        if seen_param || seen_result {
            if let Some(ft) = ctx.types.get(eidx as usize)
                && (ft.parameters != params || ft.return_types != results)
            {
                return Err(ParseError::new(
                    "inline function type",
                    list.get(start).map(|s| s.span()).unwrap_or(list.span),
                ));
            }
        } else if register_locals {
            // Type reference without inline params, register anonymous locals
            // for the type's parameters so local indices are correct
            let param_count = ctx.types.get(eidx as usize).map_or(0, |ft| ft.parameters.len());
            for _ in 0..param_count {
                ctx.register(Namespace::Local, None, list.span)?;
                ctx.param_count += 1;
            }
        }
        eidx
    } else {
        ctx.register_type(
            None,
            FunctionType {
                parameters: params,
                return_types: results,
            },
        )
    };

    Ok((type_idx, idx))
}

// ============================================================================
// Functions
// ============================================================================

/// Parses a function definition.
///
/// Grammar: `func ::= '(' 'func' id? typeuse local* instr* ')'`
///          `func ::= '(' 'func' id? '(' 'import' name name ')' typeuse ')'`
fn parse_func(list: SExprList<'_>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    ctx.reset_function();

    let mut idx = 1;

    let name = take_optional_name(list, &mut idx);
    ctx.register(Namespace::Func, name, list.span)?;

    idx = collect_inline_exports(list, idx, ExportIndex::Function(ctx.func_count - 1), ctx)?;

    // Check for inline import: (func $name (import "mod" "name") ...)
    if let Some((module_name, field_name)) = parse_inline_import(list, idx)? {
        idx += 1;
        let (type_idx, _) = parse_type_use(list, idx, false, ctx)?;
        ctx.imports.push(Import {
            module: module_name,
            name: field_name,
            external_kind: ExternalKind::Function(type_idx),
        });
        return Ok(());
    }

    ctx.has_func_def = true;

    // Parse type use (type reference and/or inline params/results)
    let (type_idx, body_start) = parse_type_use(list, idx, true, ctx)?;

    // Parse locals and body (locals must precede all instructions)
    let mut locals = Vec::new();
    let mut body = Vec::new();
    let mut body_idx = body_start;
    let mut seen_body = false;

    while let Some(item) = list.get(body_idx) {
        if item.is_list_headed_by("local") {
            if seen_body {
                return Err(ParseError::new("unexpected token", item.span()));
            }
            parse_local(item.as_list().unwrap(), &mut locals, ctx)?;
            body_idx += 1;
        } else {
            seen_body = true;
            body_idx = parse_body_item(list, body_idx, &mut body, ctx)?;
        }
    }

    // Build function
    let func_type = ctx
        .types
        .get(type_idx as usize)
        .cloned()
        .unwrap_or_else(|| FunctionType {
            parameters: Vec::new(),
            return_types: Vec::new(),
        });

    let structured = StructureBuilder::build_function(&body, ctx.local_count as usize, func_type.return_types.clone())
        .map_err(|e| ParseError::new(e.to_string(), list.span))?;

    ctx.functions.push(Function { ftype_index: type_idx });
    ctx.code.push(FunctionBody {
        locals: Locals::new(compress_locals(&locals)),
        body: structured,
        position: SectionPosition::new(0, 0),
    });

    Ok(())
}

/// Parses a local declaration.
///
/// Grammar: `local ::= '(' 'local' id valtype ')' | '(' 'local' valtype* ')'`
fn parse_local(list: SExprList<'_>, locals: &mut Vec<ValueType>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    let mut idx = 1;

    let name = take_optional_name(list, &mut idx);
    if name.is_some() {
        // Named form: exactly one valtype
        let item = list
            .get(idx)
            .ok_or_else(|| ParseError::new("expected valtype after local name", list.span))?;
        let ty = parse_valtype(item)?;
        ctx.register(Namespace::Local, name, list.span)?;
        locals.push(ty);
        if list.get(idx + 1).is_some() {
            return Err(ParseError::new("named local must have exactly one type", list.span));
        }
    } else {
        // Anonymous form: zero or more valtypes
        while let Some(item) = list.get(idx) {
            let ty = parse_valtype(item)?;
            ctx.register(Namespace::Local, None, list.span)?;
            locals.push(ty);
            idx += 1;
        }
    }

    Ok(())
}

// ============================================================================
// Tables, Memories, Globals
// ============================================================================

/// Parses a table definition.
///
/// Grammar: `table ::= '(' 'table' id? limits reftype ')'`
fn parse_table(list: SExprList<'_>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    let mut idx = 1;
    let name = take_optional_name(list, &mut idx);
    ctx.register(Namespace::Table, name, list.span)?;
    let table_index = ctx.table_count - 1;
    idx = collect_inline_exports(list, idx, ExportIndex::Table(table_index), ctx)?;

    // Check for inline import: (table $name (import "mod" "name") limits reftype)
    if let Some((module_name, field_name)) = parse_inline_import(list, idx)? {
        idx += 1;
        let limits = parse_limits(list, idx)?;
        let ref_idx = idx + limits_len(&limits);
        let ref_type = parse_reftype(list.get(ref_idx))?;
        ctx.imports.push(Import {
            module: module_name,
            name: field_name,
            external_kind: ExternalKind::Table(TableType { ref_type, limits }),
        });
        return Ok(());
    }

    ctx.has_table_def = true;

    // Check for inline element segment form: (table reftype (elem ...))
    let next = list.get(idx);
    let is_reftype_keyword = next
        .and_then(|s| s.as_keyword())
        .is_some_and(|kw| kw == "funcref" || kw == "externref");

    if is_reftype_keyword {
        let has_elem = list
            .get(idx + 1)
            .and_then(|s| s.as_list())
            .is_some_and(|inner| inner.head_keyword() == Some("elem"));

        if has_elem {
            let ref_type = parse_reftype(list.get(idx))?;
            let elem_list = list.get(idx + 1).unwrap().as_list().unwrap();
            let init = parse_elem_func_indices(elem_list, 1, ctx)?;
            let n = init.len() as u32;
            ctx.tables.push(TableType {
                limits: Limits { min: n, max: Some(n) },
                ref_type,
            });
            ctx.elements.push(Element {
                flags: 0,
                ref_type,
                init,
                mode: ElementMode::Active {
                    table_index,
                    offset: vec![make_instruction(InstructionKind::I32Const { value: 0 }), make_end()],
                },
            });
            return Ok(());
        }
    }

    let limits = parse_limits(list, idx)?;
    let ref_idx = idx + limits_len(&limits);
    let ref_type = parse_reftype(list.get(ref_idx))?;

    ctx.tables.push(TableType { limits, ref_type });
    Ok(())
}

/// Parses a memory definition.
///
/// Grammar: `memory ::= '(' 'memory' id? limits ')'`
fn parse_memory(list: SExprList<'_>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    let mut idx = 1;
    let name = take_optional_name(list, &mut idx);
    ctx.register(Namespace::Memory, name, list.span)?;
    idx = collect_inline_exports(list, idx, ExportIndex::Memory(ctx.memory_count - 1), ctx)?;

    // Check for inline import: (memory $name (import "mod" "name") limits)
    if let Some((module_name, field_name)) = parse_inline_import(list, idx)? {
        idx += 1;
        let limits = parse_limits(list, idx)?;
        ctx.imports.push(Import {
            module: module_name,
            name: field_name,
            external_kind: ExternalKind::Memory(limits),
        });
        return Ok(());
    }

    ctx.has_memory_def = true;

    // Check for inline data segment: (memory (data "..."))
    if let Some(item) = list.get(idx)
        && item.is_list_headed_by("data")
    {
        let data_list = item.as_list().unwrap();
        let mut bytes = Vec::new();
        for i in 1.. {
            let Some(s) = data_list.get(i) else { break };
            bytes.extend_from_slice(&parse_byte_string(s)?);
        }
        let page_count = bytes.len().div_ceil(65536) as u32;
        ctx.memories.push(Memory {
            limits: Limits {
                min: page_count,
                max: Some(page_count),
            },
        });
        ctx.data.push(Data {
            mode: DataMode::Active {
                memory_index: ctx.memory_count - 1,
                offset: vec![make_instruction(InstructionKind::I32Const { value: 0 }), make_end()],
            },
            init: bytes,
        });
        return Ok(());
    }

    let limits = parse_limits(list, idx)?;
    ctx.memories.push(Memory { limits });
    Ok(())
}

/// Parses a global definition.
///
/// Grammar: `global ::= '(' 'global' id? globaltype expr ')'`
fn parse_global(list: SExprList<'_>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    let mut idx = 1;
    let name = take_optional_name(list, &mut idx);
    ctx.register(Namespace::Global, name, list.span)?;
    idx = collect_inline_exports(list, idx, ExportIndex::Global(ctx.global_count - 1), ctx)?;

    // Check for inline import: (global $name (import "mod" "name") globaltype)
    if let Some((module_name, field_name)) = parse_inline_import(list, idx)? {
        idx += 1;
        let global_type = parse_global_type(list, idx)?;
        ctx.imports.push(Import {
            module: module_name,
            name: field_name,
            external_kind: ExternalKind::Global(global_type),
        });
        return Ok(());
    }

    ctx.has_global_def = true;

    let global_type = parse_global_type(list, idx)?;
    idx += 1; // Consume the type or (mut type)

    let init_sexpr = list
        .get(idx)
        .ok_or_else(|| ParseError::new("expected init expression", list.span))?;
    // Reject extra items after the init expression
    if list.get(idx + 1).is_some() {
        return Err(ParseError::new("type mismatch", list.get(idx + 1).unwrap().span()));
    }
    let init = parse_const_expr(init_sexpr, ctx)?;

    ctx.globals.push(Global { global_type, init });
    Ok(())
}

/// Parses a global type.
///
/// Grammar: `globaltype ::= valtype | '(' 'mut' valtype ')'`
fn parse_global_type(list: SExprList<'_>, idx: usize) -> Result<GlobalType, ParseError> {
    let item = list
        .get(idx)
        .ok_or_else(|| ParseError::new("expected global type", list.span))?;

    if let Some(inner) = item.as_list() {
        inner.expect_head("mut")?;
        let value_type = parse_valtype(
            inner
                .get(1)
                .ok_or_else(|| ParseError::new("expected value type in mut", inner.span))?,
        )?;
        Ok(GlobalType {
            value_type,
            mutable: true,
        })
    } else {
        let value_type = parse_valtype(item)?;
        Ok(GlobalType {
            value_type,
            mutable: false,
        })
    }
}

// ============================================================================
// Exports, Start, Data
// ============================================================================

/// Consumes inline `(export "name")` forms from a definition, registering
/// each as an export of the given kind. Returns the adjusted index past any
/// inline exports. Used by func, table, memory, and global parsers.
fn collect_inline_exports(
    list: SExprList<'_>,
    mut idx: usize,
    index: ExportIndex,
    ctx: &mut ParseContext,
) -> Result<usize, ParseError> {
    while let Some(item) = list.get(idx) {
        if item.is_list_headed_by("export") {
            let export_list = item.as_list().unwrap();
            let name = parse_string(
                export_list
                    .get(1)
                    .ok_or_else(|| ParseError::new("expected export name", export_list.span))?,
            )?;
            ctx.exports.push(Export { name, index });
            idx += 1;
        } else {
            break;
        }
    }
    Ok(idx)
}

/// Parses an export.
///
/// Grammar: `export ::= '(' 'export' name '(' exportdesc ')' ')'`
fn parse_export(list: SExprList<'_>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    let name = parse_string(
        list.get(1)
            .ok_or_else(|| ParseError::new("expected export name", list.span))?,
    )?;

    let desc = list
        .get(2)
        .ok_or_else(|| ParseError::new("expected export descriptor", list.span))?
        .expect_list()?;

    let index = match desc.head_keyword() {
        Some("func") => ExportIndex::Function(parse_index(desc.get(1), Namespace::Func, ctx)?),
        Some("table") => ExportIndex::Table(parse_index(desc.get(1), Namespace::Table, ctx)?),
        Some("memory") => ExportIndex::Memory(parse_index(desc.get(1), Namespace::Memory, ctx)?),
        Some("global") => ExportIndex::Global(parse_index(desc.get(1), Namespace::Global, ctx)?),
        Some(kw) => return Err(ParseError::new(format!("unknown export kind: {}", kw), desc.span)),
        None => return Err(ParseError::new("expected export kind", desc.span)),
    };

    ctx.exports.push(Export { name, index });
    Ok(())
}

/// Parses a start section.
///
/// Grammar: `start ::= '(' 'start' funcidx ')'`
fn parse_start(list: SExprList<'_>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    if ctx.start.is_some() {
        return Err(ParseError::new("multiple start sections", list.span));
    }
    let idx = parse_index(list.get(1), Namespace::Func, ctx)?;
    ctx.start = Some(idx);
    Ok(())
}

/// Parses an element segment.
///
/// Grammar:
/// ```text
/// elem ::= '(' 'elem' id? elemlist ')'
///        | '(' 'elem' id? tableuse '(' 'offset' expr ')' elemlist ')'
///        | '(' 'elem' id? 'declare' elemlist ')'
///        | '(' 'elem' id? '(' 'offset' expr ')' funcidx* ')'  ;; abbreviated
///        | '(' 'elem' id? tableuse '(' 'offset' expr ')' funcidx* ')'  ;; abbreviated
/// elemlist ::= reftype elemexpr* | 'func' funcidx*
/// tableuse ::= '(' 'table' tableidx ')'
/// ```
fn parse_elem(list: SExprList<'_>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    let mut idx = 1;
    let name = take_optional_name(list, &mut idx);
    ctx.register(Namespace::Elem, name, list.span)?;

    // Check for 'declare' keyword (declarative mode)
    if list.get(idx).and_then(|s| s.as_keyword()) == Some("declare") {
        idx += 1;
        let (ref_type, init) = parse_elem_list(list, idx, ctx)?;
        let mode = ElementMode::Declarative;
        let flags = elem_flags(&mode, ref_type, &init);
        ctx.elements.push(Element {
            flags,
            ref_type,
            init,
            mode,
        });
        return Ok(());
    }

    // Check for (table ...) for explicit table index
    let mut table_index = 0u32;
    if let Some(item) = list.get(idx)
        && item.is_list_headed_by("table")
    {
        let table_list = item.as_list().unwrap();
        table_index = parse_index(table_list.get(1), Namespace::Table, ctx)?;
        idx += 1;
    }

    // Check for active mode: (offset ...) or inline expression
    if let Some(item) = list.get(idx) {
        let active_offset = if item.is_list_headed_by("offset") {
            let offset_list = item.as_list().unwrap();
            Some(parse_const_expr_multi(offset_list.iter_from(1), ctx)?)
        } else if item.as_list().is_some() && !is_elem_keyword(item) {
            Some(parse_const_expr(item, ctx)?)
        } else {
            None
        };
        if let Some(offset) = active_offset {
            idx += 1;
            let (ref_type, init) = parse_elem_list(list, idx, ctx)?;
            let mode = ElementMode::Active { table_index, offset };
            let flags = elem_flags(&mode, ref_type, &init);
            ctx.elements.push(Element {
                flags,
                ref_type,
                init,
                mode,
            });
            return Ok(());
        }
    }

    // Passive mode
    let (ref_type, init) = parse_elem_list(list, idx, ctx)?;
    let mode = ElementMode::Passive;
    let flags = elem_flags(&mode, ref_type, &init);
    ctx.elements.push(Element {
        flags,
        ref_type,
        init,
        mode,
    });
    Ok(())
}

/// Checks if an S-expression is an element-specific keyword (item), distinguishing
/// it from a potential offset expression in the active abbreviation form.
fn is_elem_keyword(sexpr: &SExpr) -> bool {
    sexpr.is_list_headed_by("item")
}

/// Checks if any init expression contains something other than ref.func
/// (e.g. ref.null), requiring expression-based encoding (flag 5/7 instead of 1/3).
fn has_non_ref_func_init(init: &[Vec<Instruction>]) -> bool {
    init.iter()
        .any(|expr| expr.iter().any(|i| matches!(i.kind, InstructionKind::RefNull { .. })))
}

/// Parses an element list.
///
/// Grammar: `elemlist ::= reftype elemexpr* | 'func' funcidx*`
fn parse_elem_list(
    list: SExprList<'_>,
    start: usize,
    ctx: &mut ParseContext,
) -> Result<(RefType, Vec<Vec<Instruction>>), ParseError> {
    let mut idx = start;
    let mut ref_type = RefType::FuncRef;

    // Check for explicit reftype or 'func' keyword
    if let Some(item) = list.get(idx) {
        match item.as_keyword() {
            Some("funcref") => {
                ref_type = RefType::FuncRef;
                idx += 1;
            }
            Some("externref") => {
                ref_type = RefType::ExternRef;
                idx += 1;
            }
            Some("func") => {
                // Shorthand: 'func $f1 $f2' is equivalent to 'funcref (ref.func $f1) (ref.func $f2)'
                idx += 1;
                let init = parse_elem_func_indices(list, idx, ctx)?;
                return Ok((RefType::FuncRef, init));
            }
            _ => {}
        }
    }

    // Parse element expressions. Each init expression includes a trailing End
    // to match the binary format representation.
    let mut init = Vec::new();
    while let Some(item) = list.get(idx) {
        if let Some(inner) = item.as_list() {
            // Unwrap optional (item ...) wrapper to find the keyword and argument position
            let is_item = inner.head_keyword() == Some("item");
            let (kw, arg_idx) = if is_item {
                (inner.get(1).and_then(|s| s.as_keyword()), 2)
            } else {
                (inner.head_keyword(), 1)
            };
            match kw {
                Some("ref.func") => {
                    let func_idx = parse_index(inner.get(arg_idx), Namespace::Func, ctx)?;
                    init.push(vec![
                        make_instruction(InstructionKind::RefFunc { func_idx }),
                        make_end(),
                    ]);
                }
                Some("ref.null") => {
                    let heap = inner.get(arg_idx).and_then(|s| s.as_keyword()).unwrap_or("func");
                    let rt = if heap == "extern" {
                        ValueType::ExternRef
                    } else {
                        ValueType::FuncRef
                    };
                    init.push(vec![
                        make_instruction(InstructionKind::RefNull { ref_type: rt }),
                        make_end(),
                    ]);
                }
                _ if is_item => {
                    init.push(parse_const_expr_multi(inner.iter_from(1), ctx)?);
                }
                _ => {
                    init.push(parse_const_expr(item, ctx)?);
                }
            }
        } else if let Some(id_or_idx) = item.as_id().or_else(|| {
            item.as_atom()
                .filter(|t| matches!(t.kind, TokenKind::Integer(_)))
                .map(|_| "")
        }) {
            let func_idx = if id_or_idx.is_empty() {
                parse_index(Some(item), Namespace::Func, ctx)?
            } else {
                ctx.resolve(Namespace::Func, id_or_idx, item.span())?
            };
            init.push(vec![
                make_instruction(InstructionKind::RefFunc { func_idx }),
                make_end(),
            ]);
        } else {
            break;
        }
        idx += 1;
    }

    Ok((ref_type, init))
}

/// Parses bare function indices as element init expressions.
///
/// Each index is wrapped in a `ref.func` instruction.
fn parse_elem_func_indices(
    list: SExprList<'_>,
    start: usize,
    ctx: &mut ParseContext,
) -> Result<Vec<Vec<Instruction>>, ParseError> {
    let mut init = Vec::new();
    let mut idx = start;

    while let Some(item) = list.get(idx) {
        if let Ok(func_idx) = parse_index(Some(item), Namespace::Func, ctx) {
            init.push(vec![
                make_instruction(InstructionKind::RefFunc { func_idx }),
                make_end(),
            ]);
            idx += 1;
        } else {
            break;
        }
    }

    Ok(init)
}

/// Parses a data segment.
///
/// Grammar:
/// ```text
/// data ::= '(' 'data' id? string* ')'                               ;; passive
///        | '(' 'data' id? memuse? '(' 'offset' expr ')' string* ')' ;; active
/// memuse ::= '(' 'memory' memidx ')'
/// ```
fn parse_data(list: SExprList<'_>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    let mut idx = 1;
    let data_name = take_optional_name(list, &mut idx);
    ctx.register(Namespace::Data, data_name, list.span)?;

    // Check for (memory ...) for explicit memory index
    let mut memory_index = 0u32;
    if let Some(item) = list.get(idx)
        && item.is_list_headed_by("memory")
    {
        let mem_list = item.as_list().unwrap();
        memory_index = parse_index(mem_list.get(1), Namespace::Memory, ctx)?;
        idx += 1;
    }

    // Check for offset expression, determines active vs passive
    let mode = if let Some(item) = list.get(idx) {
        if item.is_list_headed_by("offset") {
            // Explicit (offset expr) form
            let offset_list = item.as_list().unwrap();
            let offset = parse_const_expr_multi(offset_list.iter_from(1), ctx)?;
            idx += 1;
            DataMode::Active { memory_index, offset }
        } else if item.as_list().is_some() && !is_string_atom(item) {
            // Implicit offset expression (abbreviated form)
            let offset = parse_const_expr(item, ctx)?;
            idx += 1;
            DataMode::Active { memory_index, offset }
        } else {
            // No offset, passive mode
            DataMode::Passive
        }
    } else {
        DataMode::Passive
    };

    // Parse data strings
    let mut init = Vec::new();
    while let Some(item) = list.get(idx) {
        if let Some(token) = item.as_atom()
            && let TokenKind::String(bytes) = &token.kind
        {
            init.extend(bytes);
        }
        idx += 1;
    }

    ctx.data.push(Data { mode, init });

    Ok(())
}

/// Checks if an S-expression is a string atom.
fn is_string_atom(sexpr: &SExpr) -> bool {
    sexpr
        .as_atom()
        .map(|t| matches!(t.kind, TokenKind::String(_)))
        .unwrap_or(false)
}

// ============================================================================
// Instructions
// ============================================================================

/// Argument source for instruction parsing.
///
/// Abstracts the difference between flat and folded instruction syntax:
/// - Folded: `(i32.add (local.get 0) (local.get 1))` - args are children
/// - Flat: `local.get 0` - args are siblings in parent list
enum ArgSource<'a> {
    /// Arguments are children of the instruction S-expression.
    Folded(&'a [SExpr]),
    /// Arguments are siblings starting at given index in parent list.
    Flat { list: SExprList<'a>, start: usize },
}

impl<'a> ArgSource<'a> {
    /// Gets argument at index.
    fn get(&self, idx: usize) -> Option<&'a SExpr> {
        match self {
            ArgSource::Folded(args) => args.get(idx),
            ArgSource::Flat { list, start } => list.get(*start + idx),
        }
    }

    /// Counts consecutive arguments matching a predicate (for br_table).
    fn count_matching(&self, mut pred: impl FnMut(&SExpr) -> bool) -> usize {
        let mut count = 0;
        while let Some(s) = self.get(count) {
            if pred(s) {
                count += 1;
            } else {
                break;
            }
        }
        count
    }
}

/// Parses a body item (instruction with possible trailing arguments).
/// Returns the new index after consuming the instruction and its arguments.
fn parse_body_item(
    list: SExprList<'_>,
    idx: usize,
    out: &mut Vec<Instruction>,
    ctx: &mut ParseContext,
) -> Result<usize, ParseError> {
    let item = list
        .get(idx)
        .ok_or_else(|| ParseError::new("expected instruction", list.span))?;

    match item {
        SExpr::Atom(token) => {
            if let TokenKind::Keyword(kw) = &token.kind {
                match kw.as_str() {
                    // Flat block/loop: block $label? blocktype ... end
                    "block" | "loop" => parse_flat_block(kw, list, idx + 1, out, ctx),
                    // Flat if: if $label? blocktype ... else ... end
                    "if" => parse_flat_if(list, idx + 1, out, ctx),
                    // else and end are handled by flat block/if parsers
                    "else" | "end" => {
                        // Should not be reached directly, consumed by block/if
                        Ok(idx + 1)
                    }
                    _ => {
                        // Normal flat instruction
                        let (kind, consumed) = parse_flat_instruction(kw, list, idx + 1, token.span, ctx)?;
                        out.push(make_instruction(kind));
                        Ok(idx + 1 + consumed)
                    }
                }
            } else {
                // Not a keyword, skip it
                Ok(idx + 1)
            }
        }
        SExpr::List { .. } => {
            // Folded instruction, self-contained
            parse_instruction(item, out, ctx)?;
            Ok(idx + 1)
        }
    }
}

/// Consume an optional `$id` at `list[*idx]`, advancing the index if present.
fn take_optional_name<'a>(list: SExprList<'a>, idx: &mut usize) -> Option<&'a str> {
    let name = list.get(*idx).and_then(|s| s.as_id());
    if name.is_some() {
        *idx += 1;
    }
    name
}

/// Compute the element segment flags from mode, ref type, and init expressions.
fn elem_flags(mode: &ElementMode, ref_type: RefType, init: &[Vec<Instruction>]) -> u32 {
    let needs_expr = ref_type == RefType::ExternRef || has_non_ref_func_init(init);
    match mode {
        ElementMode::Declarative => {
            if needs_expr {
                ELEM_DECLARATIVE_EXPRS
            } else {
                ELEM_DECLARATIVE_FUNCS
            }
        }
        ElementMode::Passive => {
            if needs_expr {
                ELEM_PASSIVE_EXPRS
            } else {
                ELEM_PASSIVE_FUNCS
            }
        }
        ElementMode::Active { table_index, .. } => match (*table_index == 0, needs_expr) {
            (true, false) => ELEM_ACTIVE_FUNCS,
            (true, true) => ELEM_ACTIVE_EXPRS,
            (false, false) => ELEM_ACTIVE_TABLE_FUNCS,
            (false, true) => ELEM_ACTIVE_TABLE_EXPRS,
        },
    }
}

/// Check for an optional label after `end` or `else`, validating it matches the block label.
fn check_end_label(list: SExprList<'_>, idx: &mut usize, expected: &Option<String>) -> Result<(), ParseError> {
    if let Some(end_label) = list.get(*idx).and_then(|s| s.as_id()) {
        if expected.as_deref() != Some(end_label) {
            return Err(ParseError::new(
                format!("mismatching label: expected {:?}, got ${}", expected, end_label),
                list.get(*idx).unwrap().span(),
            ));
        }
        *idx += 1;
    }
    Ok(())
}

/// Parses a flat block/loop instruction (non-folded).
/// Consumes items until matching `end`.
fn parse_flat_block(
    kw: &str,
    list: SExprList<'_>,
    start: usize,
    out: &mut Vec<Instruction>,
    ctx: &mut ParseContext,
) -> Result<usize, ParseError> {
    let mut idx = start;

    // Optional label
    let label = list.get(idx).and_then(|s| s.as_id()).map(String::from);
    if label.is_some() {
        idx += 1;
    }

    // Collect block type items
    let mut bt_items = Vec::new();
    while let Some(item) = list.get(idx) {
        if item.is_list_headed_by("param") || item.is_list_headed_by("result") || item.is_list_headed_by("type") {
            bt_items.push(item.clone());
            idx += 1;
        } else {
            break;
        }
    }
    let (block_type, _) = parse_block_type_multi(&bt_items, ctx)?;

    ctx.push_label(label.clone());
    let block_kind = if kw == "block" {
        InstructionKind::Block { block_type }
    } else {
        InstructionKind::Loop { block_type }
    };
    out.push(make_instruction(block_kind));

    // Parse body until `end`
    while let Some(item) = list.get(idx) {
        if let Some(kw_inner) = item.as_keyword()
            && kw_inner == "end"
        {
            idx += 1;
            check_end_label(list, &mut idx, &label)?;
            break;
        }
        idx = parse_body_item(list, idx, out, ctx)?;
    }

    ctx.pop_label();
    out.push(make_end());
    Ok(idx)
}

/// Parses a flat if instruction (non-folded).
/// Consumes items until matching `end`, with optional `else`.
fn parse_flat_if(
    list: SExprList<'_>,
    start: usize,
    out: &mut Vec<Instruction>,
    ctx: &mut ParseContext,
) -> Result<usize, ParseError> {
    let mut idx = start;

    // Optional label
    let label = list.get(idx).and_then(|s| s.as_id()).map(String::from);
    if label.is_some() {
        idx += 1;
    }

    // Collect block type items
    let mut bt_items = Vec::new();
    while let Some(item) = list.get(idx) {
        if item.is_list_headed_by("param") || item.is_list_headed_by("result") || item.is_list_headed_by("type") {
            bt_items.push(item.clone());
            idx += 1;
        } else {
            break;
        }
    }
    let (block_type, _) = parse_block_type_multi(&bt_items, ctx)?;

    ctx.push_label(label.clone());
    out.push(make_instruction(InstructionKind::If { block_type }));

    // Parse then branch until `else` or `end`
    while let Some(item) = list.get(idx) {
        if let Some(kw) = item.as_keyword() {
            if kw == "else" {
                out.push(make_instruction(InstructionKind::Else));
                idx += 1;
                check_end_label(list, &mut idx, &label)?;
                // Parse else branch until `end`
                while let Some(item2) = list.get(idx) {
                    if let Some(kw2) = item2.as_keyword()
                        && kw2 == "end"
                    {
                        idx += 1;
                        check_end_label(list, &mut idx, &label)?;
                        break;
                    }
                    idx = parse_body_item(list, idx, out, ctx)?;
                }
                break;
            }
            if kw == "end" {
                idx += 1;
                check_end_label(list, &mut idx, &label)?;
                break;
            }
        }
        idx = parse_body_item(list, idx, out, ctx)?;
    }

    ctx.pop_label();
    out.push(make_end());
    Ok(idx)
}

/// Parses a flat instruction, consuming arguments from subsequent siblings.
/// Returns the instruction kind and number of sibling items consumed.
fn parse_flat_instruction(
    kw: &str,
    list: SExprList<'_>,
    arg_start: usize,
    span: Span,
    ctx: &mut ParseContext,
) -> Result<(InstructionKind, usize), ParseError> {
    let args = ArgSource::Flat { list, start: arg_start };
    parse_instruction_kind(kw, args, span, ctx)
}

/// Parses an instruction (flat or folded).
fn parse_instruction(sexpr: &SExpr, out: &mut Vec<Instruction>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    match sexpr {
        SExpr::Atom(token) => {
            if let TokenKind::Keyword(kw) = &token.kind {
                let kind = parse_folded_instruction(kw, &[], token.span, ctx)?;
                out.push(make_instruction(kind));
            }
        }
        SExpr::List { items, span } => {
            if items.is_empty() {
                return Ok(());
            }

            let head = &items[0];
            let kw = head
                .as_keyword()
                .ok_or_else(|| ParseError::expected("instruction keyword", "list", head.span()))?;

            // Handle control flow specially
            match kw {
                "block" | "loop" => parse_block_instruction(kw, &items[1..], *span, out, ctx)?,
                "if" => parse_if_instruction(&items[1..], *span, out, ctx)?,
                _ => {
                    // Parse nested arguments first (folded form)
                    for arg in &items[1..] {
                        if arg.as_list().is_some() && !is_immediate_list(arg) {
                            parse_instruction(arg, out, ctx)?;
                        }
                    }
                    // Then the instruction itself
                    let kind = parse_folded_instruction(kw, &items[1..], *span, ctx)?;
                    out.push(make_instruction(kind));
                }
            }
        }
    }
    Ok(())
}

/// Parses a block or loop instruction.
///
/// Grammar: `block ::= 'block' label? blocktype instr* 'end'`
///          `loop  ::= 'loop'  label? blocktype instr* 'end'`
fn parse_block_instruction(
    kw: &str,
    args: &[SExpr],
    _span: Span,
    out: &mut Vec<Instruction>,
    ctx: &mut ParseContext,
) -> Result<(), ParseError> {
    let mut idx = 0;

    // Optional label
    let label = args.get(idx).and_then(|s| s.as_id()).map(String::from);
    if label.is_some() {
        idx += 1;
    }

    // Optional block type (may consume multiple items for multi-value)
    let (block_type, consumed) = parse_block_type_multi(&args[idx..], ctx)?;
    idx += consumed;

    ctx.push_label(label);

    // Emit block/loop
    let block_kind = if kw == "block" {
        InstructionKind::Block { block_type }
    } else {
        InstructionKind::Loop { block_type }
    };
    out.push(make_instruction(block_kind));

    // Parse body, handles both folded and flat instructions within the block
    parse_body_from_args(&args[idx..], out, ctx)?;

    // Emit end
    ctx.pop_label();
    out.push(make_end());

    Ok(())
}

/// Parses a sequence of instructions from a slice, handling flat instruction
/// argument consumption (e.g. `i32.const 0` spans two atoms in the slice).
fn parse_body_from_args(args: &[SExpr], out: &mut Vec<Instruction>, ctx: &mut ParseContext) -> Result<(), ParseError> {
    let mut i = 0;
    while i < args.len() {
        match &args[i] {
            SExpr::List { .. } => {
                parse_instruction(&args[i], out, ctx)?;
                i += 1;
            }
            SExpr::Atom(token) => {
                if let TokenKind::Keyword(kw) = &token.kind {
                    match kw.as_str() {
                        "block" | "loop" => {
                            parse_block_instruction(kw, &args[i + 1..], token.span, out, ctx)?;
                            // Block consumes everything until the end of the slice
                            return Ok(());
                        }
                        "if" => {
                            parse_if_instruction(&args[i + 1..], token.span, out, ctx)?;
                            return Ok(());
                        }
                        _ => {
                            // Plain instruction (e.g. i32.add, local.get), parse operands from remaining args
                            let remaining = &args[i + 1..];
                            let flat_args = ArgSource::Folded(remaining);
                            let (kind, consumed) = parse_instruction_kind(kw, flat_args, token.span, ctx)?;
                            out.push(make_instruction(kind));
                            i += 1 + consumed;
                        }
                    }
                } else {
                    i += 1;
                }
            }
        }
    }
    Ok(())
}

/// Parses an if instruction.
///
/// Grammar: `if ::= 'if' label? blocktype instr* '(' 'then' instr* ')' '(' 'else' instr* ')?'`
fn parse_if_instruction(
    args: &[SExpr],
    _span: Span,
    out: &mut Vec<Instruction>,
    ctx: &mut ParseContext,
) -> Result<(), ParseError> {
    let mut idx = 0;

    // Optional label
    let label = args.get(idx).and_then(|s| s.as_id()).map(String::from);
    if label.is_some() {
        idx += 1;
    }

    // Optional block type (may consume multiple items for multi-value)
    let (block_type, consumed) = parse_block_type_multi(&args[idx..], ctx)?;
    idx += consumed;

    // Parse condition (everything before 'then'), must be folded (nested S-expressions)
    while idx < args.len() {
        let item = &args[idx];
        if item.is_list_headed_by("then") {
            break;
        }
        match item {
            SExpr::List { .. } => {
                parse_instruction(item, out, ctx)?;
                idx += 1;
            }
            SExpr::Atom(token) => {
                // Flat instructions are not valid in folded if condition
                return Err(ParseError::new("unexpected token", token.span));
            }
        }
    }

    ctx.push_label(label);
    out.push(make_instruction(InstructionKind::If { block_type }));

    // Parse then branch
    if let Some(then_item) = args.get(idx)
        && then_item.is_list_headed_by("then")
    {
        let then_list = then_item.as_list().unwrap();
        let then_items: Vec<_> = then_list.iter_from(1).cloned().collect();
        parse_body_from_args(&then_items, out, ctx)?;
        idx += 1;
    }

    // Parse else branch
    if let Some(else_item) = args.get(idx)
        && else_item.is_list_headed_by("else")
    {
        out.push(make_instruction(InstructionKind::Else));
        let else_list = else_item.as_list().unwrap();
        let else_items: Vec<_> = else_list.iter_from(1).cloned().collect();
        parse_body_from_args(&else_items, out, ctx)?;
    }

    ctx.pop_label();
    out.push(make_end());

    Ok(())
}

/// Parses a block type from a slice of S-expressions.
/// Returns (BlockType, items_consumed).
/// Supports: empty, single result, (type $t), and multi-value (param ...) (result ...).
fn parse_block_type_multi(args: &[SExpr], ctx: &mut ParseContext) -> Result<(BlockType, usize), ParseError> {
    let Some(first) = args.first() else {
        return Ok((BlockType::Empty, 0));
    };

    // Plain value type keyword
    if let Some(kw) = first.as_keyword() {
        return match kw {
            "i32" => Ok((BlockType::Value(ValueType::I32), 0)),
            "i64" => Ok((BlockType::Value(ValueType::I64), 0)),
            "f32" => Ok((BlockType::Value(ValueType::F32), 0)),
            "f64" => Ok((BlockType::Value(ValueType::F64), 0)),
            _ => Ok((BlockType::Empty, 0)),
        };
    }

    // (type $t), explicit function type reference
    if first.is_list_headed_by("type") {
        let list = first.as_list().unwrap();
        let type_idx = parse_index(list.get(1), Namespace::Type, ctx)?;
        // Also consume subsequent (param ...) and (result ...) annotations
        let mut consumed = 1;
        while args
            .get(consumed)
            .is_some_and(|s| s.is_list_headed_by("param") || s.is_list_headed_by("result"))
        {
            consumed += 1;
        }
        return Ok((BlockType::FuncType(type_idx), consumed));
    }

    // Check for (param ...) or (result ...), multi-value block type
    let has_param = first.is_list_headed_by("param");
    let has_result = first.is_list_headed_by("result");

    if has_param || has_result {
        let mut params = Vec::new();
        let mut results = Vec::new();
        let mut consumed = 0;

        while let Some(item) = args.get(consumed) {
            if item.is_list_headed_by("param") {
                let inner = item.as_list().unwrap();
                for p in inner.iter_from(1) {
                    if p.as_id().is_some() {
                        return Err(ParseError::new(
                            "unexpected token: named params in block type",
                            p.span(),
                        ));
                    }
                    params.push(parse_valtype(p)?);
                }
                consumed += 1;
            } else if item.is_list_headed_by("result") {
                let inner = item.as_list().unwrap();
                for r in inner.iter_from(1) {
                    results.push(parse_valtype(r)?);
                }
                consumed += 1;
            } else {
                break;
            }
        }

        // Simple case: no params and single result -> Value type
        if params.is_empty() && results.len() == 1 {
            return Ok((BlockType::Value(results[0]), consumed));
        }

        // Multi-value: register as function type
        if params.is_empty() && results.is_empty() {
            return Ok((BlockType::Empty, consumed));
        }

        let type_idx = ctx.register_type(
            None,
            FunctionType {
                parameters: params,
                return_types: results,
            },
        );
        return Ok((BlockType::FuncType(type_idx), consumed));
    }

    Ok((BlockType::Empty, 0))
}

/// Parses a folded instruction (arguments are children).
/// Discards the consumed count since folded form always consumes all children.
fn parse_folded_instruction(
    kw: &str,
    args: &[SExpr],
    span: Span,
    ctx: &mut ParseContext,
) -> Result<InstructionKind, ParseError> {
    let args = ArgSource::Folded(args);
    let (kind, _consumed) = parse_instruction_kind(kw, args, span, ctx)?;
    Ok(kind)
}

/// Unified instruction parser - single source of truth for all instructions.
///
/// Handles both flat and folded syntax through the ArgSource abstraction.
/// Returns (InstructionKind, args_consumed) where args_consumed is relevant
/// for flat syntax to know how many siblings were used.
fn parse_instruction_kind(
    kw: &str,
    args: ArgSource<'_>,
    span: Span,
    ctx: &mut ParseContext,
) -> Result<(InstructionKind, usize), ParseError> {
    match kw {
        // Control - no arguments
        "unreachable" => Ok((InstructionKind::Unreachable, 0)),
        "nop" => Ok((InstructionKind::Nop, 0)),
        "return" => Ok((InstructionKind::Return, 0)),

        // Branch - label argument
        "br" => Ok((
            InstructionKind::Br {
                label_idx: parse_label_index(args.get(0), ctx)?,
            },
            1,
        )),
        "br_if" => Ok((
            InstructionKind::BrIf {
                label_idx: parse_label_index(args.get(0), ctx)?,
            },
            1,
        )),
        "br_table" => {
            let count = args.count_matching(|s| parse_label_index(Some(s), ctx).is_ok());
            let mut labels = Vec::with_capacity(count);
            for i in 0..count {
                labels.push(parse_label_index(args.get(i), ctx)?);
            }
            let default = labels
                .pop()
                .ok_or_else(|| ParseError::new("br_table requires at least one label", span))?;
            Ok((InstructionKind::BrTable { labels, default }, count))
        }

        // Call - function/type index
        "call" => Ok((
            InstructionKind::Call {
                func_idx: parse_index(args.get(0), Namespace::Func, ctx)?,
            },
            1,
        )),
        "call_indirect" => {
            // Syntax: call_indirect $table? typeuse
            // typeuse = (type $t)? (param ...)* (result ...)*  |  $type_idx
            let mut consumed = 0;
            let mut table_idx = 0u32;
            let mut type_idx = 0u32;
            let mut explicit_type = false;

            // Optional table index (first arg if it's a plain index, not a list)
            if let Some(a) = args.get(0)
                && !a.is_list_headed_by("type")
                && !a.is_list_headed_by("param")
                && !a.is_list_headed_by("result")
                && let Ok(idx) = parse_index(Some(a), Namespace::Table, ctx)
            {
                table_idx = idx;
                consumed = 1;
            }

            // Parse type use: (type ...) and/or (param ...) (result ...)
            if let Some(a) = args.get(consumed) {
                if a.is_list_headed_by("type") {
                    let list = a.as_list().unwrap();
                    type_idx = parse_index(list.get(1), Namespace::Type, ctx)?;
                    consumed += 1;
                    explicit_type = true;
                } else if let Ok(tidx) = parse_index(Some(a), Namespace::Type, ctx)
                    && !a.is_list_headed_by("param")
                    && !a.is_list_headed_by("result")
                {
                    type_idx = tidx;
                    consumed += 1;
                    explicit_type = true;
                }
            }

            // Consume inline (param ...) and (result ...), build type if no explicit index
            // Reject (type ...) after (param ...) or (result ...)
            let mut params = Vec::new();
            let mut results = Vec::new();
            let mut ci_seen_result = false;
            while let Some(a) = args.get(consumed) {
                if a.is_list_headed_by("type") {
                    return Err(ParseError::new("unexpected token", a.span()));
                } else if a.is_list_headed_by("param") {
                    if ci_seen_result {
                        return Err(ParseError::new("unexpected token", a.span()));
                    }
                    let list = a.as_list().unwrap();
                    for p in list.iter_from(1) {
                        if p.as_id().is_some() {
                            return Err(ParseError::new("unexpected token", p.span()));
                        }
                        params.push(parse_valtype(p)?);
                    }
                    consumed += 1;
                } else if a.is_list_headed_by("result") {
                    ci_seen_result = true;
                    let list = a.as_list().unwrap();
                    for r in list.iter_from(1) {
                        results.push(parse_valtype(r)?);
                    }
                    consumed += 1;
                } else {
                    break;
                }
            }

            if !explicit_type && (!params.is_empty() || !results.is_empty()) {
                let target = FunctionType {
                    parameters: params,
                    return_types: results,
                };
                type_idx = ctx.find_or_add_type(target);
            }

            Ok((InstructionKind::CallIndirect { type_idx, table_idx }, consumed))
        }

        // Parametric
        "drop" => Ok((InstructionKind::Drop, 0)),
        "select" => {
            // select or select (result t)*, consume all consecutive (result ...) clauses
            let mut val_types = Vec::new();
            let mut consumed = 0;
            while let Some(r) = args.get(consumed) {
                if r.is_list_headed_by("result") {
                    let rlist = r.as_list().unwrap();
                    for item in rlist.iter_from(1) {
                        val_types.push(parse_valtype(item)?);
                    }
                    consumed += 1;
                } else {
                    break;
                }
            }
            if val_types.is_empty() {
                Ok((InstructionKind::Select, 0))
            } else {
                Ok((InstructionKind::SelectTyped { val_types }, consumed))
            }
        }

        // Reference types
        "ref.null" => {
            let rt = args.get(0).ok_or_else(|| ParseError::new("expected reftype", span))?;
            let vt = match rt.as_keyword() {
                Some("func") | Some("funcref") => ValueType::FuncRef,
                Some("extern") | Some("externref") => ValueType::ExternRef,
                Some(kw) => return Err(ParseError::new(format!("unknown reftype: {}", kw), rt.span())),
                None => return Err(ParseError::expected("reftype keyword", "list", rt.span())),
            };
            Ok((InstructionKind::RefNull { ref_type: vt }, 1))
        }
        "ref.is_null" => Ok((InstructionKind::RefIsNull, 0)),
        "ref.func" => Ok((
            InstructionKind::RefFunc {
                func_idx: parse_index(args.get(0), Namespace::Func, ctx)?,
            },
            1,
        )),

        // Variable - index argument
        "local.get" => Ok((
            InstructionKind::LocalGet {
                local_idx: parse_index(args.get(0), Namespace::Local, ctx)?,
            },
            1,
        )),
        "local.set" => Ok((
            InstructionKind::LocalSet {
                local_idx: parse_index(args.get(0), Namespace::Local, ctx)?,
            },
            1,
        )),
        "local.tee" => Ok((
            InstructionKind::LocalTee {
                local_idx: parse_index(args.get(0), Namespace::Local, ctx)?,
            },
            1,
        )),
        "global.get" => Ok((
            InstructionKind::GlobalGet {
                global_idx: parse_index(args.get(0), Namespace::Global, ctx)?,
            },
            1,
        )),
        "global.set" => Ok((
            InstructionKind::GlobalSet {
                global_idx: parse_index(args.get(0), Namespace::Global, ctx)?,
            },
            1,
        )),

        // Constants - value argument
        "i32.const" => Ok((
            InstructionKind::I32Const {
                value: parse_i32(args.get(0))?,
            },
            1,
        )),
        "i64.const" => Ok((
            InstructionKind::I64Const {
                value: parse_i64(args.get(0))?,
            },
            1,
        )),
        "f32.const" => Ok((
            InstructionKind::F32Const {
                value: parse_f32(args.get(0))?,
            },
            1,
        )),
        "f64.const" => Ok((
            InstructionKind::F64Const {
                value: parse_f64(args.get(0))?,
            },
            1,
        )),

        // Table operations (table index is optional, defaults to 0)
        "table.get" => {
            let (idx, consumed) = parse_optional_table_index(&args, ctx);
            Ok((InstructionKind::TableGet { table_idx: idx }, consumed))
        }
        "table.set" => {
            let (idx, consumed) = parse_optional_table_index(&args, ctx);
            Ok((InstructionKind::TableSet { table_idx: idx }, consumed))
        }
        "table.size" => {
            let (idx, consumed) = parse_optional_table_index(&args, ctx);
            Ok((InstructionKind::TableSize { table_idx: idx }, consumed))
        }
        "table.grow" => {
            let (idx, consumed) = parse_optional_table_index(&args, ctx);
            Ok((InstructionKind::TableGrow { table_idx: idx }, consumed))
        }
        "table.fill" => {
            let (idx, consumed) = parse_optional_table_index(&args, ctx);
            Ok((InstructionKind::TableFill { table_idx: idx }, consumed))
        }
        "table.copy" => {
            // table.copy dst src (both optional, default to 0)
            let (dst, c1) = parse_optional_table_index(&args, ctx);
            if c1 > 0 {
                let (src, c2) = parse_optional_table_index_at(&args, c1, ctx);
                Ok((
                    InstructionKind::TableCopy {
                        dst_table: dst,
                        src_table: src,
                    },
                    c1 + c2,
                ))
            } else {
                Ok((
                    InstructionKind::TableCopy {
                        dst_table: 0,
                        src_table: 0,
                    },
                    0,
                ))
            }
        }
        "table.init" => {
            // table.init $elem (table 0 implicit) or table.init $table $elem
            if let Some(a0) = args.get(0) {
                if let Some(a1) = args.get(1).filter(|s| s.as_atom().is_some()) {
                    // Two atom args: table + elem indices
                    if let Ok(t) = parse_index(Some(a0), Namespace::Table, ctx) {
                        let e = parse_index(Some(a1), Namespace::Elem, ctx)?;
                        return Ok((
                            InstructionKind::TableInit {
                                table_idx: t,
                                elem_idx: e,
                            },
                            2,
                        ));
                    }
                }
                // One arg: just elem index, table defaults to 0
                let e = parse_index(Some(a0), Namespace::Elem, ctx)?;
                Ok((
                    InstructionKind::TableInit {
                        table_idx: 0,
                        elem_idx: e,
                    },
                    1,
                ))
            } else {
                Err(ParseError::new("table.init requires element index", span))
            }
        }
        "elem.drop" => Ok((
            InstructionKind::ElemDrop {
                elem_idx: parse_index(args.get(0), Namespace::Elem, ctx)?,
            },
            1,
        )),

        // Memory size/grow
        "memory.size" => Ok((InstructionKind::MemorySize, 0)),
        "memory.grow" => Ok((InstructionKind::MemoryGrow, 0)),

        // Bulk memory operations
        "memory.copy" => Ok((InstructionKind::MemoryCopy, 0)),
        "memory.fill" => Ok((InstructionKind::MemoryFill, 0)),
        "memory.init" => Ok((
            InstructionKind::MemoryInit {
                data_idx: parse_index(args.get(0), Namespace::Data, ctx)?,
            },
            1,
        )),
        "data.drop" => Ok((
            InstructionKind::DataDrop {
                data_idx: parse_index(args.get(0), Namespace::Data, ctx)?,
            },
            1,
        )),

        // Memory load/store - memarg
        "i32.load" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_32)?;
            Ok((InstructionKind::I32Load { memarg: m }, c))
        }
        "i64.load" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_64)?;
            Ok((InstructionKind::I64Load { memarg: m }, c))
        }
        "f32.load" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_32)?;
            Ok((InstructionKind::F32Load { memarg: m }, c))
        }
        "f64.load" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_64)?;
            Ok((InstructionKind::F64Load { memarg: m }, c))
        }
        "i32.load8_s" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_8)?;
            Ok((InstructionKind::I32Load8S { memarg: m }, c))
        }
        "i32.load8_u" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_8)?;
            Ok((InstructionKind::I32Load8U { memarg: m }, c))
        }
        "i32.load16_s" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_16)?;
            Ok((InstructionKind::I32Load16S { memarg: m }, c))
        }
        "i32.load16_u" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_16)?;
            Ok((InstructionKind::I32Load16U { memarg: m }, c))
        }
        "i64.load8_s" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_8)?;
            Ok((InstructionKind::I64Load8S { memarg: m }, c))
        }
        "i64.load8_u" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_8)?;
            Ok((InstructionKind::I64Load8U { memarg: m }, c))
        }
        "i64.load16_s" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_16)?;
            Ok((InstructionKind::I64Load16S { memarg: m }, c))
        }
        "i64.load16_u" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_16)?;
            Ok((InstructionKind::I64Load16U { memarg: m }, c))
        }
        "i64.load32_s" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_32)?;
            Ok((InstructionKind::I64Load32S { memarg: m }, c))
        }
        "i64.load32_u" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_32)?;
            Ok((InstructionKind::I64Load32U { memarg: m }, c))
        }
        "i32.store" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_32)?;
            Ok((InstructionKind::I32Store { memarg: m }, c))
        }
        "i64.store" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_64)?;
            Ok((InstructionKind::I64Store { memarg: m }, c))
        }
        "f32.store" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_32)?;
            Ok((InstructionKind::F32Store { memarg: m }, c))
        }
        "f64.store" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_64)?;
            Ok((InstructionKind::F64Store { memarg: m }, c))
        }
        "i32.store8" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_8)?;
            Ok((InstructionKind::I32Store8 { memarg: m }, c))
        }
        "i32.store16" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_16)?;
            Ok((InstructionKind::I32Store16 { memarg: m }, c))
        }
        "i64.store8" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_8)?;
            Ok((InstructionKind::I64Store8 { memarg: m }, c))
        }
        "i64.store16" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_16)?;
            Ok((InstructionKind::I64Store16 { memarg: m }, c))
        }
        "i64.store32" => {
            let (m, c) = parse_memarg_from_args(&args, ALIGN_32)?;
            Ok((InstructionKind::I64Store32 { memarg: m }, c))
        }

        // i32 operations - no arguments
        "i32.add" => Ok((InstructionKind::I32Add, 0)),
        "i32.sub" => Ok((InstructionKind::I32Sub, 0)),
        "i32.mul" => Ok((InstructionKind::I32Mul, 0)),
        "i32.div_s" => Ok((InstructionKind::I32DivS, 0)),
        "i32.div_u" => Ok((InstructionKind::I32DivU, 0)),
        "i32.rem_s" => Ok((InstructionKind::I32RemS, 0)),
        "i32.rem_u" => Ok((InstructionKind::I32RemU, 0)),
        "i32.and" => Ok((InstructionKind::I32And, 0)),
        "i32.or" => Ok((InstructionKind::I32Or, 0)),
        "i32.xor" => Ok((InstructionKind::I32Xor, 0)),
        "i32.shl" => Ok((InstructionKind::I32Shl, 0)),
        "i32.shr_s" => Ok((InstructionKind::I32ShrS, 0)),
        "i32.shr_u" => Ok((InstructionKind::I32ShrU, 0)),
        "i32.rotl" => Ok((InstructionKind::I32Rotl, 0)),
        "i32.rotr" => Ok((InstructionKind::I32Rotr, 0)),
        "i32.clz" => Ok((InstructionKind::I32Clz, 0)),
        "i32.ctz" => Ok((InstructionKind::I32Ctz, 0)),
        "i32.popcnt" => Ok((InstructionKind::I32Popcnt, 0)),
        "i32.eqz" => Ok((InstructionKind::I32Eqz, 0)),
        "i32.eq" => Ok((InstructionKind::I32Eq, 0)),
        "i32.ne" => Ok((InstructionKind::I32Ne, 0)),
        "i32.lt_s" => Ok((InstructionKind::I32LtS, 0)),
        "i32.lt_u" => Ok((InstructionKind::I32LtU, 0)),
        "i32.gt_s" => Ok((InstructionKind::I32GtS, 0)),
        "i32.gt_u" => Ok((InstructionKind::I32GtU, 0)),
        "i32.le_s" => Ok((InstructionKind::I32LeS, 0)),
        "i32.le_u" => Ok((InstructionKind::I32LeU, 0)),
        "i32.ge_s" => Ok((InstructionKind::I32GeS, 0)),
        "i32.ge_u" => Ok((InstructionKind::I32GeU, 0)),
        "i32.wrap_i64" => Ok((InstructionKind::I32WrapI64, 0)),
        "i32.extend8_s" => Ok((InstructionKind::I32Extend8S, 0)),
        "i32.extend16_s" => Ok((InstructionKind::I32Extend16S, 0)),
        "i32.trunc_f32_s" => Ok((InstructionKind::I32TruncF32S, 0)),
        "i32.trunc_f32_u" => Ok((InstructionKind::I32TruncF32U, 0)),
        "i32.trunc_f64_s" => Ok((InstructionKind::I32TruncF64S, 0)),
        "i32.trunc_f64_u" => Ok((InstructionKind::I32TruncF64U, 0)),
        "i32.reinterpret_f32" => Ok((InstructionKind::I32ReinterpretF32, 0)),

        // i64 operations
        "i64.add" => Ok((InstructionKind::I64Add, 0)),
        "i64.sub" => Ok((InstructionKind::I64Sub, 0)),
        "i64.mul" => Ok((InstructionKind::I64Mul, 0)),
        "i64.div_s" => Ok((InstructionKind::I64DivS, 0)),
        "i64.div_u" => Ok((InstructionKind::I64DivU, 0)),
        "i64.rem_s" => Ok((InstructionKind::I64RemS, 0)),
        "i64.rem_u" => Ok((InstructionKind::I64RemU, 0)),
        "i64.and" => Ok((InstructionKind::I64And, 0)),
        "i64.or" => Ok((InstructionKind::I64Or, 0)),
        "i64.xor" => Ok((InstructionKind::I64Xor, 0)),
        "i64.shl" => Ok((InstructionKind::I64Shl, 0)),
        "i64.shr_s" => Ok((InstructionKind::I64ShrS, 0)),
        "i64.shr_u" => Ok((InstructionKind::I64ShrU, 0)),
        "i64.rotl" => Ok((InstructionKind::I64Rotl, 0)),
        "i64.rotr" => Ok((InstructionKind::I64Rotr, 0)),
        "i64.clz" => Ok((InstructionKind::I64Clz, 0)),
        "i64.ctz" => Ok((InstructionKind::I64Ctz, 0)),
        "i64.popcnt" => Ok((InstructionKind::I64Popcnt, 0)),
        "i64.eqz" => Ok((InstructionKind::I64Eqz, 0)),
        "i64.eq" => Ok((InstructionKind::I64Eq, 0)),
        "i64.ne" => Ok((InstructionKind::I64Ne, 0)),
        "i64.lt_s" => Ok((InstructionKind::I64LtS, 0)),
        "i64.lt_u" => Ok((InstructionKind::I64LtU, 0)),
        "i64.gt_s" => Ok((InstructionKind::I64GtS, 0)),
        "i64.gt_u" => Ok((InstructionKind::I64GtU, 0)),
        "i64.le_s" => Ok((InstructionKind::I64LeS, 0)),
        "i64.le_u" => Ok((InstructionKind::I64LeU, 0)),
        "i64.ge_s" => Ok((InstructionKind::I64GeS, 0)),
        "i64.ge_u" => Ok((InstructionKind::I64GeU, 0)),
        "i64.extend_i32_s" => Ok((InstructionKind::I64ExtendI32S, 0)),
        "i64.extend_i32_u" => Ok((InstructionKind::I64ExtendI32U, 0)),
        "i64.extend8_s" => Ok((InstructionKind::I64Extend8S, 0)),
        "i64.extend16_s" => Ok((InstructionKind::I64Extend16S, 0)),
        "i64.extend32_s" => Ok((InstructionKind::I64Extend32S, 0)),
        "i64.trunc_f32_s" => Ok((InstructionKind::I64TruncF32S, 0)),
        "i64.trunc_f32_u" => Ok((InstructionKind::I64TruncF32U, 0)),
        "i64.trunc_f64_s" => Ok((InstructionKind::I64TruncF64S, 0)),
        "i64.trunc_f64_u" => Ok((InstructionKind::I64TruncF64U, 0)),
        "i64.reinterpret_f64" => Ok((InstructionKind::I64ReinterpretF64, 0)),

        // f32 operations
        "f32.add" => Ok((InstructionKind::F32Add, 0)),
        "f32.sub" => Ok((InstructionKind::F32Sub, 0)),
        "f32.mul" => Ok((InstructionKind::F32Mul, 0)),
        "f32.div" => Ok((InstructionKind::F32Div, 0)),
        "f32.min" => Ok((InstructionKind::F32Min, 0)),
        "f32.max" => Ok((InstructionKind::F32Max, 0)),
        "f32.abs" => Ok((InstructionKind::F32Abs, 0)),
        "f32.neg" => Ok((InstructionKind::F32Neg, 0)),
        "f32.ceil" => Ok((InstructionKind::F32Ceil, 0)),
        "f32.floor" => Ok((InstructionKind::F32Floor, 0)),
        "f32.trunc" => Ok((InstructionKind::F32Trunc, 0)),
        "f32.nearest" => Ok((InstructionKind::F32Nearest, 0)),
        "f32.sqrt" => Ok((InstructionKind::F32Sqrt, 0)),
        "f32.copysign" => Ok((InstructionKind::F32Copysign, 0)),
        "f32.eq" => Ok((InstructionKind::F32Eq, 0)),
        "f32.ne" => Ok((InstructionKind::F32Ne, 0)),
        "f32.lt" => Ok((InstructionKind::F32Lt, 0)),
        "f32.gt" => Ok((InstructionKind::F32Gt, 0)),
        "f32.le" => Ok((InstructionKind::F32Le, 0)),
        "f32.ge" => Ok((InstructionKind::F32Ge, 0)),
        "f32.convert_i32_s" => Ok((InstructionKind::F32ConvertI32S, 0)),
        "f32.convert_i32_u" => Ok((InstructionKind::F32ConvertI32U, 0)),
        "f32.convert_i64_s" => Ok((InstructionKind::F32ConvertI64S, 0)),
        "f32.convert_i64_u" => Ok((InstructionKind::F32ConvertI64U, 0)),
        "f32.demote_f64" => Ok((InstructionKind::F32DemoteF64, 0)),
        "f32.reinterpret_i32" => Ok((InstructionKind::F32ReinterpretI32, 0)),

        // f64 operations
        "f64.add" => Ok((InstructionKind::F64Add, 0)),
        "f64.sub" => Ok((InstructionKind::F64Sub, 0)),
        "f64.mul" => Ok((InstructionKind::F64Mul, 0)),
        "f64.div" => Ok((InstructionKind::F64Div, 0)),
        "f64.min" => Ok((InstructionKind::F64Min, 0)),
        "f64.max" => Ok((InstructionKind::F64Max, 0)),
        "f64.abs" => Ok((InstructionKind::F64Abs, 0)),
        "f64.neg" => Ok((InstructionKind::F64Neg, 0)),
        "f64.ceil" => Ok((InstructionKind::F64Ceil, 0)),
        "f64.floor" => Ok((InstructionKind::F64Floor, 0)),
        "f64.trunc" => Ok((InstructionKind::F64Trunc, 0)),
        "f64.nearest" => Ok((InstructionKind::F64Nearest, 0)),
        "f64.sqrt" => Ok((InstructionKind::F64Sqrt, 0)),
        "f64.copysign" => Ok((InstructionKind::F64Copysign, 0)),
        "f64.eq" => Ok((InstructionKind::F64Eq, 0)),
        "f64.ne" => Ok((InstructionKind::F64Ne, 0)),
        "f64.lt" => Ok((InstructionKind::F64Lt, 0)),
        "f64.gt" => Ok((InstructionKind::F64Gt, 0)),
        "f64.le" => Ok((InstructionKind::F64Le, 0)),
        "f64.ge" => Ok((InstructionKind::F64Ge, 0)),
        "f64.convert_i32_s" => Ok((InstructionKind::F64ConvertI32S, 0)),
        "f64.convert_i32_u" => Ok((InstructionKind::F64ConvertI32U, 0)),
        "f64.convert_i64_s" => Ok((InstructionKind::F64ConvertI64S, 0)),
        "f64.convert_i64_u" => Ok((InstructionKind::F64ConvertI64U, 0)),
        "f64.promote_f32" => Ok((InstructionKind::F64PromoteF32, 0)),
        "f64.reinterpret_i64" => Ok((InstructionKind::F64ReinterpretI64, 0)),

        // Saturating truncation
        "i32.trunc_sat_f32_s" => Ok((InstructionKind::I32TruncSatF32S, 0)),
        "i32.trunc_sat_f32_u" => Ok((InstructionKind::I32TruncSatF32U, 0)),
        "i32.trunc_sat_f64_s" => Ok((InstructionKind::I32TruncSatF64S, 0)),
        "i32.trunc_sat_f64_u" => Ok((InstructionKind::I32TruncSatF64U, 0)),
        "i64.trunc_sat_f32_s" => Ok((InstructionKind::I64TruncSatF32S, 0)),
        "i64.trunc_sat_f32_u" => Ok((InstructionKind::I64TruncSatF32U, 0)),
        "i64.trunc_sat_f64_s" => Ok((InstructionKind::I64TruncSatF64S, 0)),
        "i64.trunc_sat_f64_u" => Ok((InstructionKind::I64TruncSatF64U, 0)),

        _ => {
            // Try SIMD instruction
            if let Some(result) = parse_simd_instruction_kind(kw, &args, span)? {
                Ok(result)
            } else {
                Err(ParseError::new(format!("unknown instruction: {}", kw), span))
            }
        }
    }
}

/// Parses a SIMD instruction from its mnemonic.
/// Returns None if not a recognised SIMD mnemonic.
fn parse_simd_instruction_kind(
    kw: &str,
    args: &ArgSource<'_>,
    span: Span,
) -> Result<Option<(InstructionKind, usize)>, ParseError> {
    use SimdOp::*;

    let simd = |op: SimdOp| -> Option<(InstructionKind, usize)> { Some((InstructionKind::Simd(op), 0)) };

    let result = match kw {
        // v128 memory operations
        "v128.load" => {
            let (m, c) = parse_memarg_from_args(args, 4 /* 16-byte natural align = log2(16) */)?;
            Some((InstructionKind::Simd(V128Load { memarg: m }), c))
        }
        "v128.load8x8_s" => {
            let (m, c) = parse_memarg_from_args(args, 3)?;
            Some((InstructionKind::Simd(V128Load8x8S { memarg: m }), c))
        }
        "v128.load8x8_u" => {
            let (m, c) = parse_memarg_from_args(args, 3)?;
            Some((InstructionKind::Simd(V128Load8x8U { memarg: m }), c))
        }
        "v128.load16x4_s" => {
            let (m, c) = parse_memarg_from_args(args, 3)?;
            Some((InstructionKind::Simd(V128Load16x4S { memarg: m }), c))
        }
        "v128.load16x4_u" => {
            let (m, c) = parse_memarg_from_args(args, 3)?;
            Some((InstructionKind::Simd(V128Load16x4U { memarg: m }), c))
        }
        "v128.load32x2_s" => {
            let (m, c) = parse_memarg_from_args(args, 3)?;
            Some((InstructionKind::Simd(V128Load32x2S { memarg: m }), c))
        }
        "v128.load32x2_u" => {
            let (m, c) = parse_memarg_from_args(args, 3)?;
            Some((InstructionKind::Simd(V128Load32x2U { memarg: m }), c))
        }
        "v128.load8_splat" => {
            let (m, c) = parse_memarg_from_args(args, 0)?;
            Some((InstructionKind::Simd(V128Load8Splat { memarg: m }), c))
        }
        "v128.load16_splat" => {
            let (m, c) = parse_memarg_from_args(args, 1)?;
            Some((InstructionKind::Simd(V128Load16Splat { memarg: m }), c))
        }
        "v128.load32_splat" => {
            let (m, c) = parse_memarg_from_args(args, 2)?;
            Some((InstructionKind::Simd(V128Load32Splat { memarg: m }), c))
        }
        "v128.load64_splat" => {
            let (m, c) = parse_memarg_from_args(args, 3)?;
            Some((InstructionKind::Simd(V128Load64Splat { memarg: m }), c))
        }
        "v128.load32_zero" => {
            let (m, c) = parse_memarg_from_args(args, 2)?;
            Some((InstructionKind::Simd(V128Load32Zero { memarg: m }), c))
        }
        "v128.load64_zero" => {
            let (m, c) = parse_memarg_from_args(args, 3)?;
            Some((InstructionKind::Simd(V128Load64Zero { memarg: m }), c))
        }
        "v128.store" => {
            let (m, c) = parse_memarg_from_args(args, 4)?;
            Some((InstructionKind::Simd(V128Store { memarg: m }), c))
        }

        // v128 lane load/store (memarg + lane index)
        "v128.load8_lane" => {
            let (m, mc) = parse_memarg_from_args(args, 0)?;
            let li = parse_lane_index(args, mc, span)?;
            Some((InstructionKind::Simd(V128Load8Lane { memarg: m, lane: li }), mc + 1))
        }
        "v128.load16_lane" => {
            let (m, mc) = parse_memarg_from_args(args, 1)?;
            let li = parse_lane_index(args, mc, span)?;
            Some((InstructionKind::Simd(V128Load16Lane { memarg: m, lane: li }), mc + 1))
        }
        "v128.load32_lane" => {
            let (m, mc) = parse_memarg_from_args(args, 2)?;
            let li = parse_lane_index(args, mc, span)?;
            Some((InstructionKind::Simd(V128Load32Lane { memarg: m, lane: li }), mc + 1))
        }
        "v128.load64_lane" => {
            let (m, mc) = parse_memarg_from_args(args, 3)?;
            let li = parse_lane_index(args, mc, span)?;
            Some((InstructionKind::Simd(V128Load64Lane { memarg: m, lane: li }), mc + 1))
        }
        "v128.store8_lane" => {
            let (m, mc) = parse_memarg_from_args(args, 0)?;
            let li = parse_lane_index(args, mc, span)?;
            Some((InstructionKind::Simd(V128Store8Lane { memarg: m, lane: li }), mc + 1))
        }
        "v128.store16_lane" => {
            let (m, mc) = parse_memarg_from_args(args, 1)?;
            let li = parse_lane_index(args, mc, span)?;
            Some((InstructionKind::Simd(V128Store16Lane { memarg: m, lane: li }), mc + 1))
        }
        "v128.store32_lane" => {
            let (m, mc) = parse_memarg_from_args(args, 2)?;
            let li = parse_lane_index(args, mc, span)?;
            Some((InstructionKind::Simd(V128Store32Lane { memarg: m, lane: li }), mc + 1))
        }
        "v128.store64_lane" => {
            let (m, mc) = parse_memarg_from_args(args, 3)?;
            let li = parse_lane_index(args, mc, span)?;
            Some((InstructionKind::Simd(V128Store64Lane { memarg: m, lane: li }), mc + 1))
        }

        // v128.const lane_type values...
        "v128.const" => {
            let (value, consumed) = parse_v128_const(args, span)?;
            Some((InstructionKind::Simd(V128Const { value }), consumed))
        }

        // i8x16.shuffle takes 16 lane indices
        "i8x16.shuffle" => {
            let mut lanes = [0u8; 16];
            for (i, lane) in lanes.iter_mut().enumerate() {
                let s = args
                    .get(i)
                    .ok_or_else(|| ParseError::new("invalid lane length", span))?;
                let token = match s.as_atom() {
                    Some(t) => t,
                    None => return Err(ParseError::new("invalid lane length", span)),
                };
                if let TokenKind::Integer(sv) = &token.kind {
                    *lane = sv
                        .to_u64()
                        .and_then(|n| u8::try_from(n).ok())
                        .ok_or_else(|| ParseError::new("lane index out of range", token.span))?;
                } else {
                    return Err(ParseError::new("invalid lane length", span));
                }
            }
            // Reject extra lane indices beyond 16
            if let Some(extra) = args.get(16)
                && extra.as_atom().is_some_and(|t| matches!(t.kind, TokenKind::Integer(_)))
            {
                return Err(ParseError::new("invalid lane length", extra.span()));
            }
            Some((InstructionKind::Simd(I8x16Shuffle { lanes }), 16))
        }

        // Splat ops (no args)
        "i8x16.splat" => simd(I8x16Splat),
        "i16x8.splat" => simd(I16x8Splat),
        "i32x4.splat" => simd(I32x4Splat),
        "i64x2.splat" => simd(I64x2Splat),
        "f32x4.splat" => simd(F32x4Splat),
        "f64x2.splat" => simd(F64x2Splat),
        "i8x16.swizzle" => simd(I8x16Swizzle),

        // Lane extract/replace (1 lane index argument)
        "i8x16.extract_lane_s" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(I8x16ExtractLaneS { lane: li }), 1))
        }
        "i8x16.extract_lane_u" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(I8x16ExtractLaneU { lane: li }), 1))
        }
        "i8x16.replace_lane" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(I8x16ReplaceLane { lane: li }), 1))
        }
        "i16x8.extract_lane_s" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(I16x8ExtractLaneS { lane: li }), 1))
        }
        "i16x8.extract_lane_u" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(I16x8ExtractLaneU { lane: li }), 1))
        }
        "i16x8.replace_lane" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(I16x8ReplaceLane { lane: li }), 1))
        }
        "i32x4.extract_lane" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(I32x4ExtractLane { lane: li }), 1))
        }
        "i32x4.replace_lane" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(I32x4ReplaceLane { lane: li }), 1))
        }
        "i64x2.extract_lane" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(I64x2ExtractLane { lane: li }), 1))
        }
        "i64x2.replace_lane" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(I64x2ReplaceLane { lane: li }), 1))
        }
        "f32x4.extract_lane" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(F32x4ExtractLane { lane: li }), 1))
        }
        "f32x4.replace_lane" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(F32x4ReplaceLane { lane: li }), 1))
        }
        "f64x2.extract_lane" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(F64x2ExtractLane { lane: li }), 1))
        }
        "f64x2.replace_lane" => {
            let li = parse_lane_index(args, 0, span)?;
            Some((InstructionKind::Simd(F64x2ReplaceLane { lane: li }), 1))
        }

        // All remaining SIMD ops take no arguments
        "i8x16.eq" => simd(I8x16Eq),
        "i8x16.ne" => simd(I8x16Ne),
        "i8x16.lt_s" => simd(I8x16LtS),
        "i8x16.lt_u" => simd(I8x16LtU),
        "i8x16.gt_s" => simd(I8x16GtS),
        "i8x16.gt_u" => simd(I8x16GtU),
        "i8x16.le_s" => simd(I8x16LeS),
        "i8x16.le_u" => simd(I8x16LeU),
        "i8x16.ge_s" => simd(I8x16GeS),
        "i8x16.ge_u" => simd(I8x16GeU),
        "i16x8.eq" => simd(I16x8Eq),
        "i16x8.ne" => simd(I16x8Ne),
        "i16x8.lt_s" => simd(I16x8LtS),
        "i16x8.lt_u" => simd(I16x8LtU),
        "i16x8.gt_s" => simd(I16x8GtS),
        "i16x8.gt_u" => simd(I16x8GtU),
        "i16x8.le_s" => simd(I16x8LeS),
        "i16x8.le_u" => simd(I16x8LeU),
        "i16x8.ge_s" => simd(I16x8GeS),
        "i16x8.ge_u" => simd(I16x8GeU),
        "i32x4.eq" => simd(I32x4Eq),
        "i32x4.ne" => simd(I32x4Ne),
        "i32x4.lt_s" => simd(I32x4LtS),
        "i32x4.lt_u" => simd(I32x4LtU),
        "i32x4.gt_s" => simd(I32x4GtS),
        "i32x4.gt_u" => simd(I32x4GtU),
        "i32x4.le_s" => simd(I32x4LeS),
        "i32x4.le_u" => simd(I32x4LeU),
        "i32x4.ge_s" => simd(I32x4GeS),
        "i32x4.ge_u" => simd(I32x4GeU),
        "i64x2.eq" => simd(I64x2Eq),
        "i64x2.ne" => simd(I64x2Ne),
        "i64x2.lt_s" => simd(I64x2LtS),
        "i64x2.gt_s" => simd(I64x2GtS),
        "i64x2.le_s" => simd(I64x2LeS),
        "i64x2.ge_s" => simd(I64x2GeS),
        "f32x4.eq" => simd(F32x4Eq),
        "f32x4.ne" => simd(F32x4Ne),
        "f32x4.lt" => simd(F32x4Lt),
        "f32x4.gt" => simd(F32x4Gt),
        "f32x4.le" => simd(F32x4Le),
        "f32x4.ge" => simd(F32x4Ge),
        "f64x2.eq" => simd(F64x2Eq),
        "f64x2.ne" => simd(F64x2Ne),
        "f64x2.lt" => simd(F64x2Lt),
        "f64x2.gt" => simd(F64x2Gt),
        "f64x2.le" => simd(F64x2Le),
        "f64x2.ge" => simd(F64x2Ge),

        // v128 bitwise
        "v128.not" => simd(V128Not),
        "v128.and" => simd(V128And),
        "v128.andnot" => simd(V128AndNot),
        "v128.or" => simd(V128Or),
        "v128.xor" => simd(V128Xor),
        "v128.bitselect" => simd(V128Bitselect),
        "v128.any_true" => simd(V128AnyTrue),

        // i8x16 arithmetic
        "i8x16.abs" => simd(I8x16Abs),
        "i8x16.neg" => simd(I8x16Neg),
        "i8x16.popcnt" => simd(I8x16Popcnt),
        "i8x16.all_true" => simd(I8x16AllTrue),
        "i8x16.bitmask" => simd(I8x16Bitmask),
        "i8x16.narrow_i16x8_s" => simd(I8x16NarrowI16x8S),
        "i8x16.narrow_i16x8_u" => simd(I8x16NarrowI16x8U),
        "i8x16.shl" => simd(I8x16Shl),
        "i8x16.shr_s" => simd(I8x16ShrS),
        "i8x16.shr_u" => simd(I8x16ShrU),
        "i8x16.add" => simd(I8x16Add),
        "i8x16.add_sat_s" => simd(I8x16AddSatS),
        "i8x16.add_sat_u" => simd(I8x16AddSatU),
        "i8x16.sub" => simd(I8x16Sub),
        "i8x16.sub_sat_s" => simd(I8x16SubSatS),
        "i8x16.sub_sat_u" => simd(I8x16SubSatU),
        "i8x16.min_s" => simd(I8x16MinS),
        "i8x16.min_u" => simd(I8x16MinU),
        "i8x16.max_s" => simd(I8x16MaxS),
        "i8x16.max_u" => simd(I8x16MaxU),
        "i8x16.avgr_u" => simd(I8x16AvgrU),

        // i16x8 arithmetic
        "i16x8.abs" => simd(I16x8Abs),
        "i16x8.neg" => simd(I16x8Neg),
        "i16x8.q15mulr_sat_s" => simd(I16x8Q15MulrSatS),
        "i16x8.all_true" => simd(I16x8AllTrue),
        "i16x8.bitmask" => simd(I16x8Bitmask),
        "i16x8.narrow_i32x4_s" => simd(I16x8NarrowI32x4S),
        "i16x8.narrow_i32x4_u" => simd(I16x8NarrowI32x4U),
        "i16x8.extend_low_i8x16_s" => simd(I16x8ExtendLowI8x16S),
        "i16x8.extend_high_i8x16_s" => simd(I16x8ExtendHighI8x16S),
        "i16x8.extend_low_i8x16_u" => simd(I16x8ExtendLowI8x16U),
        "i16x8.extend_high_i8x16_u" => simd(I16x8ExtendHighI8x16U),
        "i16x8.shl" => simd(I16x8Shl),
        "i16x8.shr_s" => simd(I16x8ShrS),
        "i16x8.shr_u" => simd(I16x8ShrU),
        "i16x8.add" => simd(I16x8Add),
        "i16x8.add_sat_s" => simd(I16x8AddSatS),
        "i16x8.add_sat_u" => simd(I16x8AddSatU),
        "i16x8.sub" => simd(I16x8Sub),
        "i16x8.sub_sat_s" => simd(I16x8SubSatS),
        "i16x8.sub_sat_u" => simd(I16x8SubSatU),
        "i16x8.mul" => simd(I16x8Mul),
        "i16x8.min_s" => simd(I16x8MinS),
        "i16x8.min_u" => simd(I16x8MinU),
        "i16x8.max_s" => simd(I16x8MaxS),
        "i16x8.max_u" => simd(I16x8MaxU),
        "i16x8.avgr_u" => simd(I16x8AvgrU),
        "i16x8.extmul_low_i8x16_s" => simd(I16x8ExtMulLowI8x16S),
        "i16x8.extmul_high_i8x16_s" => simd(I16x8ExtMulHighI8x16S),
        "i16x8.extmul_low_i8x16_u" => simd(I16x8ExtMulLowI8x16U),
        "i16x8.extmul_high_i8x16_u" => simd(I16x8ExtMulHighI8x16U),
        "i16x8.extadd_pairwise_i8x16_s" => simd(I16x8ExtAddPairwiseI8x16S),
        "i16x8.extadd_pairwise_i8x16_u" => simd(I16x8ExtAddPairwiseI8x16U),

        // i32x4 arithmetic
        "i32x4.abs" => simd(I32x4Abs),
        "i32x4.neg" => simd(I32x4Neg),
        "i32x4.all_true" => simd(I32x4AllTrue),
        "i32x4.bitmask" => simd(I32x4Bitmask),
        "i32x4.extend_low_i16x8_s" => simd(I32x4ExtendLowI16x8S),
        "i32x4.extend_high_i16x8_s" => simd(I32x4ExtendHighI16x8S),
        "i32x4.extend_low_i16x8_u" => simd(I32x4ExtendLowI16x8U),
        "i32x4.extend_high_i16x8_u" => simd(I32x4ExtendHighI16x8U),
        "i32x4.shl" => simd(I32x4Shl),
        "i32x4.shr_s" => simd(I32x4ShrS),
        "i32x4.shr_u" => simd(I32x4ShrU),
        "i32x4.add" => simd(I32x4Add),
        "i32x4.sub" => simd(I32x4Sub),
        "i32x4.mul" => simd(I32x4Mul),
        "i32x4.min_s" => simd(I32x4MinS),
        "i32x4.min_u" => simd(I32x4MinU),
        "i32x4.max_s" => simd(I32x4MaxS),
        "i32x4.max_u" => simd(I32x4MaxU),
        "i32x4.dot_i16x8_s" => simd(I32x4DotI16x8S),
        "i32x4.extmul_low_i16x8_s" => simd(I32x4ExtMulLowI16x8S),
        "i32x4.extmul_high_i16x8_s" => simd(I32x4ExtMulHighI16x8S),
        "i32x4.extmul_low_i16x8_u" => simd(I32x4ExtMulLowI16x8U),
        "i32x4.extmul_high_i16x8_u" => simd(I32x4ExtMulHighI16x8U),
        "i32x4.extadd_pairwise_i16x8_s" => simd(I32x4ExtAddPairwiseI16x8S),
        "i32x4.extadd_pairwise_i16x8_u" => simd(I32x4ExtAddPairwiseI16x8U),

        // i64x2 arithmetic
        "i64x2.abs" => simd(I64x2Abs),
        "i64x2.neg" => simd(I64x2Neg),
        "i64x2.all_true" => simd(I64x2AllTrue),
        "i64x2.bitmask" => simd(I64x2Bitmask),
        "i64x2.extend_low_i32x4_s" => simd(I64x2ExtendLowI32x4S),
        "i64x2.extend_high_i32x4_s" => simd(I64x2ExtendHighI32x4S),
        "i64x2.extend_low_i32x4_u" => simd(I64x2ExtendLowI32x4U),
        "i64x2.extend_high_i32x4_u" => simd(I64x2ExtendHighI32x4U),
        "i64x2.shl" => simd(I64x2Shl),
        "i64x2.shr_s" => simd(I64x2ShrS),
        "i64x2.shr_u" => simd(I64x2ShrU),
        "i64x2.add" => simd(I64x2Add),
        "i64x2.sub" => simd(I64x2Sub),
        "i64x2.mul" => simd(I64x2Mul),
        "i64x2.extmul_low_i32x4_s" => simd(I64x2ExtMulLowI32x4S),
        "i64x2.extmul_high_i32x4_s" => simd(I64x2ExtMulHighI32x4S),
        "i64x2.extmul_low_i32x4_u" => simd(I64x2ExtMulLowI32x4U),
        "i64x2.extmul_high_i32x4_u" => simd(I64x2ExtMulHighI32x4U),

        // f32x4 arithmetic
        "f32x4.abs" => simd(F32x4Abs),
        "f32x4.neg" => simd(F32x4Neg),
        "f32x4.sqrt" => simd(F32x4Sqrt),
        "f32x4.add" => simd(F32x4Add),
        "f32x4.sub" => simd(F32x4Sub),
        "f32x4.mul" => simd(F32x4Mul),
        "f32x4.div" => simd(F32x4Div),
        "f32x4.min" => simd(F32x4Min),
        "f32x4.max" => simd(F32x4Max),
        "f32x4.pmin" => simd(F32x4PMin),
        "f32x4.pmax" => simd(F32x4PMax),
        "f32x4.ceil" => simd(F32x4Ceil),
        "f32x4.floor" => simd(F32x4Floor),
        "f32x4.trunc" => simd(F32x4Trunc),
        "f32x4.nearest" => simd(F32x4Nearest),

        // f64x2 arithmetic
        "f64x2.abs" => simd(F64x2Abs),
        "f64x2.neg" => simd(F64x2Neg),
        "f64x2.sqrt" => simd(F64x2Sqrt),
        "f64x2.add" => simd(F64x2Add),
        "f64x2.sub" => simd(F64x2Sub),
        "f64x2.mul" => simd(F64x2Mul),
        "f64x2.div" => simd(F64x2Div),
        "f64x2.min" => simd(F64x2Min),
        "f64x2.max" => simd(F64x2Max),
        "f64x2.pmin" => simd(F64x2PMin),
        "f64x2.pmax" => simd(F64x2PMax),
        "f64x2.ceil" => simd(F64x2Ceil),
        "f64x2.floor" => simd(F64x2Floor),
        "f64x2.trunc" => simd(F64x2Trunc),
        "f64x2.nearest" => simd(F64x2Nearest),

        // Conversion operations
        "i32x4.trunc_sat_f32x4_s" => simd(I32x4TruncSatF32x4S),
        "i32x4.trunc_sat_f32x4_u" => simd(I32x4TruncSatF32x4U),
        "f32x4.convert_i32x4_s" => simd(F32x4ConvertI32x4S),
        "f32x4.convert_i32x4_u" => simd(F32x4ConvertI32x4U),
        "i32x4.trunc_sat_f64x2_s_zero" => simd(I32x4TruncSatF64x2SZero),
        "i32x4.trunc_sat_f64x2_u_zero" => simd(I32x4TruncSatF64x2UZero),
        "f64x2.convert_low_i32x4_s" => simd(F64x2ConvertLowI32x4S),
        "f64x2.convert_low_i32x4_u" => simd(F64x2ConvertLowI32x4U),
        "f32x4.demote_f64x2_zero" => simd(F32x4DemoteF64x2Zero),
        "f64x2.promote_low_f32x4" => simd(F64x2PromoteLowF32x4),

        _ => None,
    };

    Ok(result)
}

/// Parses a lane index from args at the given offset.
fn parse_lane_index(args: &ArgSource<'_>, offset: usize, span: Span) -> Result<u8, ParseError> {
    let s = args
        .get(offset)
        .ok_or_else(|| ParseError::new("expected lane index", span))?;
    let token = s.expect_atom()?;
    if let TokenKind::Integer(sv) = &token.kind {
        if sv.has_sign {
            return Err(ParseError::new("unexpected token", token.span));
        }
        sv.to_u64()
            .and_then(|n| u8::try_from(n).ok())
            .ok_or_else(|| ParseError::new("lane index out of range", token.span))
    } else {
        Err(ParseError::expected(
            "integer",
            &format!("{:?}", token.kind),
            token.span,
        ))
    }
}

/// Parses a v128.const value.
/// Syntax: v128.const lane_type val val val ... (e.g. v128.const i32x4 1 2 3 4)
fn parse_v128_const(args: &ArgSource<'_>, span: Span) -> Result<([u8; 16], usize), ParseError> {
    let lane_type = args.get(0).and_then(|s| s.as_keyword()).ok_or_else(|| {
        ParseError::new(
            "v128.const requires lane type (i8x16, i16x8, i32x4, i64x2, f32x4, f64x2)",
            span,
        )
    })?;

    let mut bytes = [0u8; 16];
    let consumed = match lane_type {
        "i8x16" => {
            for (i, byte) in bytes.iter_mut().enumerate() {
                let val = parse_v128_lane_int(args.get(1 + i), span)?;
                if !(-0x80..=0xFF).contains(&val) {
                    return Err(ParseError::new("constant out of range", span));
                }
                *byte = val as u8;
            }
            17 // lane_type + 16 values
        }
        "i16x8" => {
            for i in 0..8 {
                let val = parse_v128_lane_int(args.get(1 + i), span)?;
                if !(-0x8000..=0xFFFF).contains(&val) {
                    return Err(ParseError::new("constant out of range", span));
                }
                let le = (val as i16).to_le_bytes();
                bytes[i * 2] = le[0];
                bytes[i * 2 + 1] = le[1];
            }
            9
        }
        "i32x4" => {
            for i in 0..4 {
                let val = parse_v128_lane_int(args.get(1 + i), span)?;
                if !(-0x80000000..=0xFFFFFFFF).contains(&val) {
                    return Err(ParseError::new("constant out of range", span));
                }
                let le = (val as i32).to_le_bytes();
                bytes[i * 4..i * 4 + 4].copy_from_slice(&le);
            }
            5
        }
        "i64x2" => {
            for i in 0..2 {
                let val = parse_v128_lane_int(args.get(1 + i), span)?;
                let le = val.to_le_bytes();
                bytes[i * 8..i * 8 + 8].copy_from_slice(&le);
            }
            3
        }
        "f32x4" => {
            for i in 0..4 {
                let val = parse_f32(args.get(1 + i))?;
                let le = val.to_le_bytes();
                bytes[i * 4..i * 4 + 4].copy_from_slice(&le);
            }
            5
        }
        "f64x2" => {
            for i in 0..2 {
                let val = parse_f64(args.get(1 + i))?;
                let le = val.to_le_bytes();
                bytes[i * 8..i * 8 + 8].copy_from_slice(&le);
            }
            3
        }
        _ => return Err(ParseError::new(format!("unknown v128 lane type: {}", lane_type), span)),
    };

    // Check for extra lane values (too many arguments)
    if args.get(consumed).is_some() {
        return Err(ParseError::new("wrong number of lane literals", span));
    }

    Ok((bytes, consumed))
}

/// Parses a v128 lane integer value (signed or unsigned, truncated to i64).
fn parse_v128_lane_int(sexpr: Option<&SExpr>, span: Span) -> Result<i64, ParseError> {
    let s = sexpr.ok_or_else(|| ParseError::new("expected v128 lane value", span))?;
    let token = s.expect_atom()?;
    if let TokenKind::Integer(sv) = &token.kind {
        if sv.negative {
            sv.to_i64()
                .ok_or_else(|| ParseError::new("v128 lane value out of range", token.span))
        } else {
            Ok(sv
                .to_u64()
                .ok_or_else(|| ParseError::new("v128 lane value out of range", token.span))? as i64)
        }
    } else {
        Err(ParseError::expected(
            "integer",
            &format!("{:?}", token.kind),
            token.span,
        ))
    }
}

/// Parses optional `offset=N` and `align=N` for memory instructions.
///
/// WAT uses bytes for alignment (`align=4`), but the binary format uses log2
/// (`align=2`). This function converts: `trailing_zeros(4) = 2`.
///
/// If omitted, `offset` defaults to 0 and `align` defaults to `natural_align`
/// (the operand size of the instruction, e.g. ALIGN_32 for i32.load).
fn parse_memarg_from_args(args: &ArgSource<'_>, natural_align: u32) -> Result<(MemArg, usize), ParseError> {
    let mut offset = 0u32;
    let mut align = natural_align;
    let mut consumed = 0;

    while let Some(s) = args.get(consumed) {
        if let Some(kw) = s.as_keyword() {
            if let Some(val) = kw.strip_prefix("offset=") {
                offset = parse_u32_string(val).map_err(|_| ParseError::new("invalid offset", s.span()))?;
                consumed += 1;
            } else if let Some(val) = kw.strip_prefix("align=") {
                // WAT: align=4 (bytes) -> binary: align=2 (log2)
                let bytes: u32 = parse_u32_string(val).map_err(|_| ParseError::new("invalid align", s.span()))?;
                if bytes == 0 || (bytes & (bytes - 1)) != 0 {
                    return Err(ParseError::new("alignment must be a power of two", s.span()));
                }
                align = bytes.trailing_zeros();
                consumed += 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    Ok((MemArg { align, offset }, consumed))
}

/// Parses a u32 from a string, supporting hex (0x...) and decimal notation.
fn parse_u32_string(s: &str) -> Result<u32, std::num::ParseIntError> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16)
    } else {
        s.parse()
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Creates an Instruction with empty position.
/// Parses a constant expression and appends a trailing `End` instruction.
///
/// The binary format terminates constant expressions with 0x0B (End). The
/// binary parser includes this in the instruction vector; the WAT parser must
/// do the same so the runtime evaluator sees a consistent representation.
fn parse_const_expr(sexpr: &SExpr, ctx: &mut ParseContext) -> Result<Vec<Instruction>, ParseError> {
    let mut instrs = Vec::new();
    parse_instruction(sexpr, &mut instrs, ctx)?;
    instrs.push(make_end());
    Ok(instrs)
}

/// Parses multiple S-expressions as a constant expression with trailing `End`.
fn parse_const_expr_multi<'a>(
    items: impl Iterator<Item = &'a SExpr>,
    ctx: &mut ParseContext,
) -> Result<Vec<Instruction>, ParseError> {
    let mut instrs = Vec::new();
    for item in items {
        parse_instruction(item, &mut instrs, ctx)?;
    }
    instrs.push(make_end());
    Ok(instrs)
}

fn make_instruction(kind: InstructionKind) -> Instruction {
    Instruction {
        kind,
        position: ByteRange { offset: 0, length: 0 },
        original_bytes: Vec::new(),
    }
}

fn make_end() -> Instruction {
    make_instruction(InstructionKind::End)
}

/// Checks if an S-expression is an immediate (not a nested instruction).
fn is_immediate_list(sexpr: &SExpr) -> bool {
    sexpr.is_list_headed_by("type") || sexpr.is_list_headed_by("result") || sexpr.is_list_headed_by("param")
}

/// Parses an index (numeric or named).
///
/// Grammar: `idx ::= u32 | id`
fn parse_index(sexpr: Option<&SExpr>, ns: Namespace, ctx: &ParseContext) -> Result<u32, ParseError> {
    let s = sexpr.ok_or_else(|| ParseError::new(format!("expected {} index", ns.name()), Span::ZERO))?;

    if let Some(name) = s.as_id() {
        return ctx.resolve(ns, name, s.span());
    }

    if let Some(token) = s.as_atom()
        && let TokenKind::Integer(sv) = &token.kind
    {
        return sv
            .to_u64()
            .map(|n| n as u32)
            .ok_or_else(|| ParseError::new("index out of range", token.span));
    }

    Err(ParseError::expected("index", "other", s.span()))
}

/// Parses a label index (numeric depth or named).
///
/// Grammar: `labelidx ::= u32 | id`
fn parse_label_index(sexpr: Option<&SExpr>, ctx: &ParseContext) -> Result<u32, ParseError> {
    let s = sexpr.ok_or_else(|| ParseError::new("expected label", Span::ZERO))?;

    if let Some(name) = s.as_id() {
        return ctx.resolve_label(name, s.span());
    }

    if let Some(token) = s.as_atom()
        && let TokenKind::Integer(sv) = &token.kind
    {
        return sv
            .to_u64()
            .map(|n| n as u32)
            .ok_or_else(|| ParseError::new("label out of range", token.span));
    }

    Err(ParseError::expected("label", "other", s.span()))
}

/// Parses a string literal.
///
/// Grammar: `string ::= '"' stringchar* '"'`
fn parse_string(sexpr: &SExpr) -> Result<String, ParseError> {
    let token = sexpr.expect_atom()?;
    if let TokenKind::String(bytes) = &token.kind {
        String::from_utf8(bytes.clone()).map_err(|_| ParseError::new("malformed UTF-8 encoding", token.span))
    } else {
        Err(ParseError::expected("string", &format!("{:?}", token.kind), token.span))
    }
}

/// Try to parse an inline import `(import "mod" "name")` at `list[idx]`.
///
/// Returns `Some((module, field))` if found, `None` otherwise.
fn parse_inline_import(list: SExprList<'_>, idx: usize) -> Result<Option<(String, String)>, ParseError> {
    if let Some(item) = list.get(idx)
        && item.is_list_headed_by("import")
    {
        let imp_list = item.as_list().unwrap();
        let module_name = parse_string(
            imp_list
                .get(1)
                .ok_or_else(|| ParseError::new("expected module name", imp_list.span))?,
        )?;
        let field_name = parse_string(
            imp_list
                .get(2)
                .ok_or_else(|| ParseError::new("expected field name", imp_list.span))?,
        )?;
        Ok(Some((module_name, field_name)))
    } else {
        Ok(None)
    }
}

/// Parses a string literal as raw bytes.
fn parse_byte_string(sexpr: &SExpr) -> Result<Vec<u8>, ParseError> {
    let token = sexpr.expect_atom()?;
    if let TokenKind::String(bytes) = &token.kind {
        Ok(bytes.clone())
    } else {
        Err(ParseError::expected("string", &format!("{:?}", token.kind), token.span))
    }
}

/// Parses limits (min, optional max).
///
/// Grammar: `limits ::= u32 | u32 u32`
fn parse_limits(list: SExprList<'_>, start: usize) -> Result<Limits, ParseError> {
    let min_sexpr = list
        .get(start)
        .ok_or_else(|| ParseError::new("expected minimum limit", list.span))?;
    let min = parse_u32(min_sexpr)?;

    let max = if let Some(s) = list.get(start + 1)
        && s.as_atom()
            .map(|t| matches!(t.kind, TokenKind::Integer(_)))
            .unwrap_or(false)
    {
        Some(parse_u32(s)?)
    } else {
        None
    };

    Ok(Limits { min, max })
}

/// Returns how many items the limits consumed.
fn limits_len(limits: &Limits) -> usize {
    if limits.max.is_some() { 2 } else { 1 }
}

/// Parses a reference type.
///
/// Grammar: `reftype ::= 'funcref' | 'externref'`
fn parse_reftype(sexpr: Option<&SExpr>) -> Result<RefType, ParseError> {
    let s = sexpr.ok_or_else(|| ParseError::new("expected reference type", Span::ZERO))?;

    match s.as_keyword() {
        Some("funcref") => Ok(RefType::FuncRef),
        Some("externref") => Ok(RefType::ExternRef),
        Some(kw) => Err(ParseError::expected("reference type", kw, s.span())),
        None => Err(ParseError::expected("reference type", "list", s.span())),
    }
}

/// Parses an i32 value.
/// Parses an optional table index, defaulting to 0 if not present or not an index.
fn parse_optional_table_index(args: &ArgSource<'_>, ctx: &ParseContext) -> (u32, usize) {
    parse_optional_table_index_at(args, 0, ctx)
}

/// Parses an optional table index at a given offset.
fn parse_optional_table_index_at(args: &ArgSource<'_>, offset: usize, ctx: &ParseContext) -> (u32, usize) {
    if let Some(s) = args.get(offset)
        && let Ok(idx) = parse_index(Some(s), Namespace::Table, ctx)
    {
        return (idx, 1);
    }
    (0, 0)
}

fn parse_i32(sexpr: Option<&SExpr>) -> Result<i32, ParseError> {
    let s = sexpr.ok_or_else(|| ParseError::new("expected i32", Span::ZERO))?;
    let token = s.expect_atom()?;

    if let TokenKind::Integer(sv) = &token.kind {
        sv.to_i64()
            .and_then(|n| i32::try_from(n).ok())
            .or_else(|| sv.to_u64().and_then(|n| u32::try_from(n).ok().map(|u| u as i32)))
            .ok_or_else(|| ParseError::new("constant out of range", token.span))
    } else {
        Err(ParseError::expected("i32", &format!("{:?}", token.kind), token.span))
    }
}

/// Parses an i64 value.
fn parse_i64(sexpr: Option<&SExpr>) -> Result<i64, ParseError> {
    let s = sexpr.ok_or_else(|| ParseError::new("expected i64", Span::ZERO))?;
    let token = s.expect_atom()?;

    if let TokenKind::Integer(sv) = &token.kind {
        sv.to_i64()
            .or_else(|| sv.to_u64().map(|u| u as i64))
            .ok_or_else(|| ParseError::new("i64 out of range", token.span))
    } else {
        Err(ParseError::expected("i64", &format!("{:?}", token.kind), token.span))
    }
}

/// Generates a float parsing function for f32 or f64.
///
/// Parameters: function name, float type, unsigned-int type, label string,
/// max NaN payload, exponent-only bits, canonical NaN bits, sign bit, to_fN method.
macro_rules! parse_float {
    ($name:ident, $fty:ty, $uty:ty, $label:expr,
     $max_payload:expr, $exp_bits:expr, $canon_nan:expr, $sign_bit:expr,
     $to_float:ident) => {
        fn $name(sexpr: Option<&SExpr>) -> Result<$fty, ParseError> {
            let s = sexpr.ok_or_else(|| ParseError::new(concat!("expected ", $label), Span::ZERO))?;
            let token = s.expect_atom()?;
            match &token.kind {
                TokenKind::Float(fl) => match fl {
                    FloatLit::Nan { negative, payload } => {
                        let mut bits: $uty = match payload {
                            Some(p) => {
                                if *p == 0 || *p > $max_payload {
                                    return Err(ParseError::expected(
                                        "constant out of range",
                                        &format!("{:?}", token.kind),
                                        token.span,
                                    ));
                                }
                                $exp_bits | (*p as $uty)
                            }
                            None => $canon_nan,
                        };
                        if *negative {
                            bits |= $sign_bit;
                        }
                        Ok(<$fty>::from_bits(bits))
                    }
                    FloatLit::Inf { .. } => Ok(fl.$to_float()),
                    _ => {
                        let val = fl.$to_float();
                        if val.is_infinite() {
                            return Err(ParseError::expected(
                                "constant out of range",
                                &format!("{:?}", token.kind),
                                token.span,
                            ));
                        }
                        Ok(val)
                    }
                },
                TokenKind::Integer(sv) => {
                    let val = if sv.negative {
                        -(sv.value as f64)
                    } else {
                        sv.value as f64
                    };
                    Ok(val as $fty)
                }
                _ => Err(ParseError::expected(
                    $label,
                    &format!("{:?}", token.kind),
                    token.span,
                )),
            }
        }
    };
}

parse_float!(
    parse_f32,
    f32,
    u32,
    "f32",
    0x7FFFFFu64,
    0x7F800000u32,
    0x7FC00000u32,
    0x80000000u32,
    to_f32
);
parse_float!(
    parse_f64,
    f64,
    u64,
    "f64",
    0xFFFFFFFFFFFFFu64,
    0x7FF0000000000000u64,
    0x7FF8000000000000u64,
    0x8000000000000000u64,
    to_f64
);

/// Parses a u32 value.
fn parse_u32(sexpr: &SExpr) -> Result<u32, ParseError> {
    let token = sexpr.expect_atom()?;
    if let TokenKind::Integer(sv) = &token.kind {
        sv.to_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| ParseError::new("constant out of range", token.span))
    } else {
        Err(ParseError::expected("u32", &format!("{:?}", token.kind), token.span))
    }
}

/// Compresses consecutive identical types.
fn compress_locals(locals: &[ValueType]) -> Vec<(u32, ValueType)> {
    let mut result = Vec::new();
    let mut iter = locals.iter().peekable();

    while let Some(&ty) = iter.next() {
        let mut count = 1u32;
        while iter.peek() == Some(&&ty) {
            iter.next();
            count += 1;
        }
        result.push((count, ty));
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_module() {
        let module = parse("(module)").unwrap();
        assert_eq!(module.functions.len(), 0);
    }

    #[test]
    fn simple_function() {
        let wat = "(module (func (result i32) (i32.const 42)))";
        let module = parse(wat).unwrap();
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.code.code.len(), 1);
    }

    #[test]
    fn function_with_params() {
        let wat = "(module (func (param i32 i32) (result i32) (i32.add (local.get 0) (local.get 1))))";
        let module = parse(wat).unwrap();
        let func_type = &module.types.types[module.functions.functions[0].ftype_index as usize];
        assert_eq!(func_type.parameters.len(), 2);
        assert_eq!(func_type.return_types.len(), 1);
    }

    #[test]
    fn named_function() {
        let wat = "(module (func $add (param $a i32) (param $b i32) (result i32) \
                   (i32.add (local.get $a) (local.get $b))))";
        let module = parse(wat).unwrap();
        assert_eq!(module.functions.len(), 1);
    }

    #[test]
    fn nested_blocks() {
        let wat = "(module (func (block $outer (block $inner (br $outer)))))";
        let module = parse(wat).unwrap();
        assert_eq!(module.functions.len(), 1);
    }

    #[test]
    fn if_then_else() {
        let wat = "(module (func (param i32) (result i32) \
                   (if (result i32) (local.get 0) (then (i32.const 1)) (else (i32.const 0)))))";
        let module = parse(wat).unwrap();
        assert_eq!(module.functions.len(), 1);
    }

    #[test]
    fn import_function() {
        let wat = r#"(module (import "env" "log" (func (param i32))))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.imports.imports.len(), 1);
    }

    #[test]
    fn export_function() {
        let wat = r#"(module (func $main (result i32) (i32.const 0)) (export "main" (func $main)))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.exports.exports.len(), 1);
    }

    #[test]
    fn memory() {
        let wat = "(module (memory 1 4))";
        let module = parse(wat).unwrap();
        assert_eq!(module.memory.memory.len(), 1);
        assert_eq!(module.memory.memory[0].limits.min, 1);
        assert_eq!(module.memory.memory[0].limits.max, Some(4));
    }

    #[test]
    fn global() {
        let wat = "(module (global $g (mut i32) (i32.const 42)))";
        let module = parse(wat).unwrap();
        assert_eq!(module.globals.globals.len(), 1);
        assert!(module.globals.globals[0].global_type.mutable);
    }

    #[test]
    fn start_section() {
        use crate::parser::module::Positional;
        let wat = "(module (func $main) (start $main))";
        let module = parse(wat).unwrap();
        assert!(module.start.has_position());
    }

    #[test]
    fn type_definition() {
        let wat = "(module (type $sig (func (param i32) (result i32))))";
        let module = parse(wat).unwrap();
        assert_eq!(module.types.types.len(), 1);
    }

    #[test]
    fn error_unknown_instruction() {
        let result = parse("(module (func (not.an.instruction)))");
        assert!(result.is_err());
    }

    #[test]
    fn br_table_multiple_labels() {
        let wat = r#"(module
            (func (param i32) (result i32)
                (block $b0
                    (block $b1
                        (block $b2
                            (br_table $b0 $b1 $b2 (local.get 0)))
                        (return (i32.const 2)))
                    (return (i32.const 1)))
                (i32.const 0)))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.functions.len(), 1);
    }

    #[test]
    fn flat_instruction_syntax() {
        let wat = r#"(module
            (func (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.functions.len(), 1);
    }

    #[test]
    fn memory_with_offset_align() {
        let wat = r#"(module
            (memory 1)
            (func (param i32) (result i32)
                (i32.load offset=4 align=4 (local.get 0))))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.functions.len(), 1);
    }

    #[test]
    fn flat_memory_with_offset_align() {
        let wat = r#"(module
            (memory 1)
            (func (param i32) (result i32)
                local.get 0
                i32.load offset=8 align=2))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.functions.len(), 1);
    }

    #[test]
    fn multiple_functions_cross_reference() {
        let wat = r#"(module
            (func $helper (param i32) (result i32)
                (i32.add (local.get 0) (i32.const 1)))
            (func $main (result i32)
                (call $helper (i32.const 41))))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.functions.len(), 2);
    }

    #[test]
    fn mixed_named_numeric_indices() {
        let wat = r#"(module
            (func $first (result i32) (i32.const 1))
            (func $second (result i32) (i32.const 2))
            (func (result i32)
                (i32.add (call $first) (call 1))))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.functions.len(), 3);
    }

    #[test]
    fn import_ordering() {
        let wat = r#"(module
            (import "env" "mem" (memory 1))
            (import "env" "log" (func (param i32)))
            (func $main (call 0 (i32.const 42))))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.imports.imports.len(), 2);
        assert_eq!(module.functions.len(), 1);
    }

    #[test]
    fn reference_types_in_params() {
        let wat = "(module (func (param funcref externref)))";
        let module = parse(wat).unwrap();
        let func_type = &module.types.types[0];
        assert_eq!(func_type.parameters.len(), 2);
        assert_eq!(func_type.parameters[0], ValueType::FuncRef);
        assert_eq!(func_type.parameters[1], ValueType::ExternRef);
    }

    // Element segment tests

    #[test]
    fn elem_active_abbreviated() {
        let wat = r#"(module
            (table 2 funcref)
            (func $f1)
            (func $f2)
            (elem (i32.const 0) $f1 $f2))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.elements.elements.len(), 1);
        assert!(matches!(
            module.elements.elements[0].mode,
            ElementMode::Active { table_index: 0, .. }
        ));
        assert_eq!(module.elements.elements[0].init.len(), 2);
    }

    #[test]
    fn elem_active_with_table() {
        let wat = r#"(module
            (table $t 2 funcref)
            (func $f1)
            (elem (table $t) (offset (i32.const 0)) funcref (ref.func $f1)))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.elements.elements.len(), 1);
        assert!(matches!(
            module.elements.elements[0].mode,
            ElementMode::Active { table_index: 0, .. }
        ));
    }

    #[test]
    fn elem_passive() {
        let wat = r#"(module
            (func $f1)
            (func $f2)
            (elem funcref (ref.func $f1) (ref.func $f2)))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.elements.elements.len(), 1);
        assert!(matches!(module.elements.elements[0].mode, ElementMode::Passive));
        assert_eq!(module.elements.elements[0].init.len(), 2);
    }

    #[test]
    fn elem_declarative() {
        let wat = r#"(module
            (func $f1)
            (elem declare funcref (ref.func $f1)))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.elements.elements.len(), 1);
        assert!(matches!(module.elements.elements[0].mode, ElementMode::Declarative));
    }

    #[test]
    fn elem_with_func_keyword() {
        let wat = r#"(module
            (table 2 funcref)
            (func $f1)
            (func $f2)
            (elem (i32.const 0) func $f1 $f2))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.elements.elements.len(), 1);
        assert_eq!(module.elements.elements[0].init.len(), 2);
    }

    // Data segment tests

    #[test]
    fn data_active_abbreviated() {
        let wat = r#"(module
            (memory 1)
            (data (i32.const 0) "hello"))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.data.data.len(), 1);
        assert!(matches!(
            module.data.data[0].mode,
            DataMode::Active { memory_index: 0, .. }
        ));
        assert_eq!(module.data.data[0].init, b"hello");
    }

    #[test]
    fn data_active_with_memory() {
        let wat = r#"(module
            (memory $m 1)
            (data (memory $m) (offset (i32.const 0)) "hello"))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.data.data.len(), 1);
        assert!(matches!(
            module.data.data[0].mode,
            DataMode::Active { memory_index: 0, .. }
        ));
    }

    #[test]
    fn data_passive() {
        let wat = r#"(module (data "hello world"))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.data.data.len(), 1);
        assert!(matches!(module.data.data[0].mode, DataMode::Passive));
        assert_eq!(module.data.data[0].init, b"hello world");
    }

    #[test]
    fn data_multiple_strings() {
        let wat = r#"(module
            (memory 1)
            (data (i32.const 0) "hello" " " "world"))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.data.data[0].init, b"hello world");
    }

    // ====================================================================
    // Inline exports
    // ====================================================================

    #[test]
    fn inline_export_func() {
        let wat = r#"(module (func (export "f") (result i32) (i32.const 1)))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.exports.exports.len(), 1);
        assert_eq!(module.exports.exports[0].name, "f");
        assert!(matches!(module.exports.exports[0].index, ExportIndex::Function(0)));
    }

    #[test]
    fn inline_export_memory() {
        let wat = r#"(module (memory (export "mem") 1))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.exports.exports.len(), 1);
        assert_eq!(module.exports.exports[0].name, "mem");
        assert!(matches!(module.exports.exports[0].index, ExportIndex::Memory(0)));
    }

    #[test]
    fn inline_export_table() {
        let wat = r#"(module (table (export "t") 1 funcref))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.exports.exports.len(), 1);
        assert_eq!(module.exports.exports[0].name, "t");
        assert!(matches!(module.exports.exports[0].index, ExportIndex::Table(0)));
    }

    #[test]
    fn inline_export_global() {
        let wat = r#"(module (global (export "g") i32 (i32.const 42)))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.exports.exports.len(), 1);
        assert_eq!(module.exports.exports[0].name, "g");
        assert!(matches!(module.exports.exports[0].index, ExportIndex::Global(0)));
    }

    #[test]
    fn inline_export_multiple_on_one_item() {
        let wat = r#"(module (memory (export "m1") (export "m2") 1))"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.exports.exports.len(), 2);
        assert_eq!(module.exports.exports[0].name, "m1");
        assert_eq!(module.exports.exports[1].name, "m2");
    }

    #[test]
    fn inline_export_named_item() {
        let wat = r#"(module
            (memory $mem (export "memory") 1)
            (func $f (export "_start") (nop))
        )"#;
        let module = parse(wat).unwrap();
        assert_eq!(module.exports.exports.len(), 2);
        assert_eq!(module.exports.exports[0].name, "memory");
        assert!(matches!(module.exports.exports[0].index, ExportIndex::Memory(0)));
        assert_eq!(module.exports.exports[1].name, "_start");
        assert!(matches!(module.exports.exports[1].index, ExportIndex::Function(0)));
    }

    // ====================================================================
    // Structured control flow via StructureBuilder
    // ====================================================================

    #[test]
    fn block_produces_structured_instruction() {
        use crate::parser::structured::StructuredInstruction;
        let wat = "(module (func (block (nop))))";
        let module = parse(wat).unwrap();
        let body = &module.code.code[0].body.body;
        assert!(
            matches!(body[0], StructuredInstruction::Block { .. }),
            "expected Block, got {:?}",
            body[0]
        );
    }

    #[test]
    fn nested_blocks_structured() {
        use crate::parser::structured::StructuredInstruction;
        let wat = "(module (func (result i32)
            (block $outer (result i32)
                (block $inner (result i32)
                    (i32.const 42)
                    (br $outer)
                )
            )
        ))";
        let module = parse(wat).unwrap();
        let body = &module.code.code[0].body.body;
        assert!(matches!(body[0], StructuredInstruction::Block { .. }));
        if let StructuredInstruction::Block { body: inner, .. } = &body[0] {
            assert!(matches!(inner[0], StructuredInstruction::Block { .. }));
        }
    }

    #[test]
    fn if_else_structured() {
        use crate::parser::structured::StructuredInstruction;
        let wat = "(module (func (param i32) (result i32)
            (if (result i32) (local.get 0)
                (then (i32.const 1))
                (else (i32.const 0))
            )
        ))";
        let module = parse(wat).unwrap();
        let body = &module.code.code[0].body.body;
        // body[0] is the condition (local.get 0), body[1] is the If
        assert!(matches!(body[0], StructuredInstruction::Plain(_)));
        assert!(matches!(
            body[1],
            StructuredInstruction::If {
                else_branch: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn loop_structured() {
        use crate::parser::structured::StructuredInstruction;
        let wat = "(module (func (loop $l (br $l))))";
        let module = parse(wat).unwrap();
        let body = &module.code.code[0].body.body;
        assert!(matches!(body[0], StructuredInstruction::Loop { .. }));
    }
}
