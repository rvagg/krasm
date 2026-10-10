//! Encodes a [`Module`] to WebAssembly binary format (`.wasm`).
//!
//! This is the conceptual inverse of [`crate::parser::parse`]. Given a parsed
//! Module (from either the binary parser or the WAT text parser), it produces
//! the binary encoding as specified by the WebAssembly 2.0 specification.
//!
//! # Binary format overview
//!
//! A WebAssembly binary begins with a magic number (`\0asm`) and version (1),
//! followed by sections in a fixed order. Each section is encoded as:
//!
//! ```text
//! section_id: u8 | byte_length: vu32 | contents: byte*
//! ```
//!
//! Sections are emitted only when present (non-empty). The encoder produces
//! minimal LEB128 encoding for all integer values.
//!
//! # Example
//!
//! ```
//! use krasm::encoder;
//! use krasm::wat;
//!
//! let module = wat::parse("(module (func))").unwrap();
//! let bytes = encoder::encode(&module).unwrap();
//! assert_eq!(&bytes[0..4], b"\0asm");
//! ```

use std::fmt;

use crate::parser::encoding::{
    DATA_ACTIVE, DATA_ACTIVE_EXPLICIT, DATA_PASSIVE, DESC_FUNC, DESC_GLOBAL, DESC_MEMORY, DESC_TABLE,
    ELEM_ACTIVE_EXPRS, ELEM_ACTIVE_FUNCS, ELEM_ACTIVE_TABLE_EXPRS, ELEM_ACTIVE_TABLE_FUNCS, ELEM_DECLARATIVE_EXPRS,
    ELEM_DECLARATIVE_FUNCS, ELEM_PASSIVE_EXPRS, ELEM_PASSIVE_FUNCS, ELEMKIND_FUNCREF, OP_END, SECTION_CODE,
    SECTION_CUSTOM, SECTION_DATA, SECTION_DATA_COUNT, SECTION_ELEMENT, SECTION_EXPORT, SECTION_FUNCTION,
    SECTION_GLOBAL, SECTION_IMPORT, SECTION_MEMORY, SECTION_START, SECTION_TABLE, SECTION_TYPE, TYPE_FUNC, write_vu1,
    write_vu32,
};
use crate::parser::instruction::{Instruction, InstructionKind};
use crate::parser::module::{
    CustomSection, DataMode, ElementMode, ExportIndex, ExternalKind, Module, Positional, RefType,
};
use crate::parser::structured::StructuredInstruction;

// ===========================================================================
// Error type
// ===========================================================================

/// Errors that can occur during binary encoding.
#[derive(Debug)]
pub enum EncodeError {
    /// Element segment has flags outside the valid range (0-7).
    InvalidElementFlags(u32),
    /// Unexpected state encountered during encoding.
    InvalidState(String),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::InvalidElementFlags(flags) => {
                write!(f, "invalid element segment flags: {flags}")
            }
            EncodeError::InvalidState(msg) => write!(f, "invalid state: {msg}"),
        }
    }
}

impl std::error::Error for EncodeError {}

// ===========================================================================
// Public API
// ===========================================================================

/// Encodes a WebAssembly module to binary format.
///
/// The inverse of [`crate::parser::parse`].
pub fn encode(module: &Module) -> Result<Vec<u8>, EncodeError> {
    let mut buf = Vec::new();

    // Magic number: \0asm
    buf.extend_from_slice(b"\0asm");
    // Version: 1
    buf.extend_from_slice(&1u32.to_le_bytes());

    // Sections in wire order
    encode_type_section(&mut buf, module);
    encode_import_section(&mut buf, module);
    encode_function_section(&mut buf, module);
    encode_table_section(&mut buf, module);
    encode_memory_section(&mut buf, module);
    encode_global_section(&mut buf, module);
    encode_export_section(&mut buf, module);
    encode_start_section(&mut buf, module);
    encode_element_section(&mut buf, module)?;
    encode_data_count_section(&mut buf, module);
    encode_code_section(&mut buf, module);
    encode_data_section(&mut buf, module);
    encode_custom_sections(&mut buf, module);

    Ok(buf)
}

// ===========================================================================
// Section encoders (in wire order)
// ===========================================================================

/// Type section (id 1): function signatures.
///
/// ```text
/// typesec ::= section_1(vec(functype))
/// functype ::= 0x60 vec(valtype) vec(valtype)
/// ```
fn encode_type_section(buf: &mut Vec<u8>, module: &Module) {
    let types = &module.types.types;
    if types.is_empty() {
        return;
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, types.len() as u32);
    for ft in types {
        contents.push(TYPE_FUNC);
        write_vu32(&mut contents, ft.parameters.len() as u32);
        for p in &ft.parameters {
            contents.extend(p.emit_bytes());
        }
        write_vu32(&mut contents, ft.return_types.len() as u32);
        for r in &ft.return_types {
            contents.extend(r.emit_bytes());
        }
    }
    emit_section(buf, SECTION_TYPE, &contents);
}

/// Import section (id 2): imported functions, tables, memories, globals.
///
/// ```text
/// importsec ::= section_2(vec(import))
/// import    ::= module:name name:name importdesc
/// importdesc ::= 0x00 typeidx | 0x01 tabletype | 0x02 memtype | 0x03 globaltype
/// ```
fn encode_import_section(buf: &mut Vec<u8>, module: &Module) {
    let imports = &module.imports.imports;
    if imports.is_empty() {
        return;
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, imports.len() as u32);
    for imp in imports {
        emit_name(&mut contents, &imp.module);
        emit_name(&mut contents, &imp.name);
        match &imp.external_kind {
            ExternalKind::Function(type_idx) => {
                contents.push(DESC_FUNC);
                write_vu32(&mut contents, *type_idx);
            }
            ExternalKind::Table(table_type) => {
                contents.push(DESC_TABLE);
                emit_ref_type(&mut contents, &table_type.ref_type);
                emit_limits(&mut contents, &table_type.limits);
            }
            ExternalKind::Memory(limits) => {
                contents.push(DESC_MEMORY);
                emit_limits(&mut contents, limits);
            }
            ExternalKind::Global(global_type) => {
                contents.push(DESC_GLOBAL);
                contents.extend(global_type.value_type.emit_bytes());
                write_vu1(&mut contents, global_type.mutable);
            }
        }
    }
    emit_section(buf, SECTION_IMPORT, &contents);
}

/// Function section (id 3): type index per local function.
///
/// ```text
/// funcsec ::= section_3(vec(typeidx))
/// ```
fn encode_function_section(buf: &mut Vec<u8>, module: &Module) {
    let functions = &module.functions.functions;
    if functions.is_empty() {
        return;
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, functions.len() as u32);
    for func in functions {
        write_vu32(&mut contents, func.ftype_index);
    }
    emit_section(buf, SECTION_FUNCTION, &contents);
}

/// Table section (id 4): table declarations.
///
/// ```text
/// tablesec  ::= section_4(vec(table))
/// table     ::= tabletype
/// tabletype ::= reftype limits
/// ```
fn encode_table_section(buf: &mut Vec<u8>, module: &Module) {
    let tables = &module.table.tables;
    if tables.is_empty() {
        return;
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, tables.len() as u32);
    for table in tables {
        emit_ref_type(&mut contents, &table.ref_type);
        emit_limits(&mut contents, &table.limits);
    }
    emit_section(buf, SECTION_TABLE, &contents);
}

/// Memory section (id 5): memory declarations.
///
/// ```text
/// memsec  ::= section_5(vec(mem))
/// mem     ::= memtype
/// memtype ::= limits
/// ```
fn encode_memory_section(buf: &mut Vec<u8>, module: &Module) {
    let memories = &module.memory.memory;
    if memories.is_empty() {
        return;
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, memories.len() as u32);
    for mem in memories {
        emit_limits(&mut contents, &mem.limits);
    }
    emit_section(buf, SECTION_MEMORY, &contents);
}

/// Global section (id 6): global declarations with init expressions.
///
/// ```text
/// globalsec  ::= section_6(vec(global))
/// global     ::= globaltype expr
/// globaltype ::= valtype mut
/// mut        ::= 0x00 (const) | 0x01 (var)
/// ```
fn encode_global_section(buf: &mut Vec<u8>, module: &Module) {
    let globals = &module.globals.globals;
    if globals.is_empty() {
        return;
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, globals.len() as u32);
    for global in globals {
        contents.extend(global.global_type.value_type.emit_bytes());
        write_vu1(&mut contents, global.global_type.mutable);
        emit_expression(&mut contents, &global.init);
    }
    emit_section(buf, SECTION_GLOBAL, &contents);
}

/// Export section (id 7): exported items.
///
/// ```text
/// exportsec  ::= section_7(vec(export))
/// export     ::= name exportdesc
/// exportdesc ::= 0x00 funcidx | 0x01 tableidx | 0x02 memidx | 0x03 globalidx
/// ```
fn encode_export_section(buf: &mut Vec<u8>, module: &Module) {
    let exports = &module.exports.exports;
    if exports.is_empty() {
        return;
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, exports.len() as u32);
    for export in exports {
        emit_name(&mut contents, &export.name);
        match &export.index {
            ExportIndex::Function(idx) => {
                contents.push(DESC_FUNC);
                write_vu32(&mut contents, *idx);
            }
            ExportIndex::Table(idx) => {
                contents.push(DESC_TABLE);
                write_vu32(&mut contents, *idx);
            }
            ExportIndex::Memory(idx) => {
                contents.push(DESC_MEMORY);
                write_vu32(&mut contents, *idx);
            }
            ExportIndex::Global(idx) => {
                contents.push(DESC_GLOBAL);
                write_vu32(&mut contents, *idx);
            }
        }
    }
    emit_section(buf, SECTION_EXPORT, &contents);
}

/// Start section (id 8): start function index.
///
/// ```text
/// startsec ::= section_8(funcidx)
/// ```
fn encode_start_section(buf: &mut Vec<u8>, module: &Module) {
    if !module.start.has_position() {
        return;
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, module.start.start);
    emit_section(buf, SECTION_START, &contents);
}

/// Element section (id 9): table initialisation segments.
///
/// The encoding depends on `Element.flags` (0-7), which determines the
/// combination of mode (active/passive/declarative), element kind encoding
/// (elemkind vs reftype), and init encoding (func indices vs expressions).
///
/// ```text
/// elemsec ::= section_9(vec(elem))
/// ```
fn encode_element_section(buf: &mut Vec<u8>, module: &Module) -> Result<(), EncodeError> {
    let elements = &module.elements.elements;
    if elements.is_empty() {
        return Ok(());
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, elements.len() as u32);

    for elem in elements {
        write_vu32(&mut contents, elem.flags);

        match elem.flags {
            // Active, implicit table 0, elemkind=funcref, func indices
            ELEM_ACTIVE_FUNCS => {
                if let ElementMode::Active { offset, .. } = &elem.mode {
                    emit_expression(&mut contents, offset);
                }
                write_vu32(&mut contents, elem.init.len() as u32);
                for init in &elem.init {
                    emit_func_index_from_expr(&mut contents, init)?;
                }
            }
            // Passive, elemkind byte, func indices
            ELEM_PASSIVE_FUNCS => {
                contents.push(ELEMKIND_FUNCREF);
                write_vu32(&mut contents, elem.init.len() as u32);
                for init in &elem.init {
                    emit_func_index_from_expr(&mut contents, init)?;
                }
            }
            // Active, explicit table, elemkind byte, func indices
            ELEM_ACTIVE_TABLE_FUNCS => {
                if let ElementMode::Active { table_index, offset } = &elem.mode {
                    write_vu32(&mut contents, *table_index);
                    emit_expression(&mut contents, offset);
                }
                contents.push(ELEMKIND_FUNCREF);
                write_vu32(&mut contents, elem.init.len() as u32);
                for init in &elem.init {
                    emit_func_index_from_expr(&mut contents, init)?;
                }
            }
            // Declarative, elemkind byte, func indices
            ELEM_DECLARATIVE_FUNCS => {
                contents.push(ELEMKIND_FUNCREF);
                write_vu32(&mut contents, elem.init.len() as u32);
                for init in &elem.init {
                    emit_func_index_from_expr(&mut contents, init)?;
                }
            }
            // Active, implicit table 0, expression init
            ELEM_ACTIVE_EXPRS => {
                if let ElementMode::Active { offset, .. } = &elem.mode {
                    emit_expression(&mut contents, offset);
                }
                write_vu32(&mut contents, elem.init.len() as u32);
                for init in &elem.init {
                    emit_expression(&mut contents, init);
                }
            }
            // Passive, reftype, expression init
            ELEM_PASSIVE_EXPRS => {
                emit_ref_type(&mut contents, &elem.ref_type);
                write_vu32(&mut contents, elem.init.len() as u32);
                for init in &elem.init {
                    emit_expression(&mut contents, init);
                }
            }
            // Active, explicit table, reftype, expression init
            ELEM_ACTIVE_TABLE_EXPRS => {
                if let ElementMode::Active { table_index, offset } = &elem.mode {
                    write_vu32(&mut contents, *table_index);
                    emit_expression(&mut contents, offset);
                }
                emit_ref_type(&mut contents, &elem.ref_type);
                write_vu32(&mut contents, elem.init.len() as u32);
                for init in &elem.init {
                    emit_expression(&mut contents, init);
                }
            }
            // Declarative, reftype, expression init
            ELEM_DECLARATIVE_EXPRS => {
                emit_ref_type(&mut contents, &elem.ref_type);
                write_vu32(&mut contents, elem.init.len() as u32);
                for init in &elem.init {
                    emit_expression(&mut contents, init);
                }
            }
            _ => return Err(EncodeError::InvalidElementFlags(elem.flags)),
        }
    }
    emit_section(buf, SECTION_ELEMENT, &contents);
    Ok(())
}

/// DataCount section (id 12): count of data segments.
///
/// Must precede the Code section in wire order. Required only when
/// `memory.init` or `data.drop` instructions are present in the code
/// section (spec §3.4.10). Omitted otherwise to match reference tooling.
///
/// ```text
/// datacountsec ::= section_12(u32)
/// ```
fn encode_data_count_section(buf: &mut Vec<u8>, module: &Module) {
    if module.data.data.is_empty() {
        return;
    }
    if !code_requires_data_count(module) {
        return;
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, module.data.data.len() as u32);
    emit_section(buf, SECTION_DATA_COUNT, &contents);
}

/// Returns true if any function body contains `memory.init` or `data.drop`,
/// which require the DataCount section to be present.
fn code_requires_data_count(module: &Module) -> bool {
    fn scan_structured(instructions: &[StructuredInstruction]) -> bool {
        for inst in instructions {
            match inst {
                StructuredInstruction::Plain(i) => {
                    if matches!(
                        i.kind,
                        InstructionKind::MemoryInit { .. } | InstructionKind::DataDrop { .. }
                    ) {
                        return true;
                    }
                }
                StructuredInstruction::Block { body, .. } | StructuredInstruction::Loop { body, .. } => {
                    if scan_structured(body) {
                        return true;
                    }
                }
                StructuredInstruction::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    if scan_structured(then_branch) {
                        return true;
                    }
                    if let Some(else_body) = else_branch
                        && scan_structured(else_body)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    for body in &module.code.code {
        if scan_structured(&body.body.body) {
            return true;
        }
    }
    false
}

/// Code section (id 10): function bodies (locals + instructions).
///
/// ```text
/// codesec ::= section_10(vec(code))
/// code    ::= size:u32 func
/// func    ::= vec(locals) expr
/// locals  ::= n:u32 t:valtype
/// ```
fn encode_code_section(buf: &mut Vec<u8>, module: &Module) {
    let bodies = &module.code.code;
    if bodies.is_empty() {
        return;
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, bodies.len() as u32);

    for body in bodies {
        let mut func_buf = Vec::new();

        // Locals: already in compressed (count, type) pairs
        let locals: Vec<_> = body.locals.iter().collect();
        write_vu32(&mut func_buf, locals.len() as u32);
        for (count, vt) in &locals {
            write_vu32(&mut func_buf, *count);
            func_buf.extend(vt.emit_bytes());
        }

        // Instructions: flatten structured tree back to linear sequence.
        // Binary-parsed bodies include a trailing End via StructuredFunction's
        // end_instruction field; WAT-parsed bodies set it to None. Encode all
        // flattened instructions, then append the function body terminator
        // if the structured representation didn't include one.
        let instructions = body.body.flatten();
        for inst in &instructions {
            func_buf.extend(inst.kind.encode());
        }
        if body.body.end_instruction.is_none() {
            func_buf.push(OP_END);
        }

        // Length-prefixed function body
        write_vu32(&mut contents, func_buf.len() as u32);
        contents.extend(func_buf);
    }
    emit_section(buf, SECTION_CODE, &contents);
}

/// Data section (id 11): memory initialisation segments.
///
/// ```text
/// datasec ::= section_11(vec(data))
/// data    ::= 0x00 expr vec(byte)         (active, memory 0)
///           | 0x01 vec(byte)              (passive)
///           | 0x02 memidx expr vec(byte)  (active, explicit memory)
/// ```
fn encode_data_section(buf: &mut Vec<u8>, module: &Module) {
    let data = &module.data.data;
    if data.is_empty() {
        return;
    }

    let mut contents = Vec::new();
    write_vu32(&mut contents, data.len() as u32);

    for seg in data {
        match &seg.mode {
            DataMode::Active { memory_index, offset } => {
                if *memory_index == 0 {
                    write_vu32(&mut contents, DATA_ACTIVE);
                    emit_expression(&mut contents, offset);
                } else {
                    write_vu32(&mut contents, DATA_ACTIVE_EXPLICIT);
                    write_vu32(&mut contents, *memory_index);
                    emit_expression(&mut contents, offset);
                }
            }
            DataMode::Passive => {
                write_vu32(&mut contents, DATA_PASSIVE);
            }
        }
        write_vu32(&mut contents, seg.init.len() as u32);
        contents.extend_from_slice(&seg.init);
    }
    emit_section(buf, SECTION_DATA, &contents);
}

/// Custom sections (id 0): emitted after all standard sections.
///
/// ```text
/// customsec ::= section_0(name byte*)
/// ```
///
/// The spec allows custom sections anywhere in the binary, but emitting them
/// at the end is simplest and matches convention (the name section is
/// typically last).
fn encode_custom_sections(buf: &mut Vec<u8>, module: &Module) {
    for custom in &module.custom {
        encode_one_custom_section(buf, custom);
    }
}

fn encode_one_custom_section(buf: &mut Vec<u8>, custom: &CustomSection) {
    let mut contents = Vec::new();
    emit_name(&mut contents, &custom.name);
    contents.extend_from_slice(&custom.data);
    emit_section(buf, SECTION_CUSTOM, &contents);
}

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Wraps section contents with a section ID and length prefix.
fn emit_section(buf: &mut Vec<u8>, id: u8, contents: &[u8]) {
    buf.push(id);
    write_vu32(buf, contents.len() as u32);
    buf.extend_from_slice(contents);
}

/// Encodes a UTF-8 name as a length-prefixed byte vector.
fn emit_name(buf: &mut Vec<u8>, name: &str) {
    let bytes = name.as_bytes();
    write_vu32(buf, bytes.len() as u32);
    buf.extend_from_slice(bytes);
}

/// Encodes limits (min, optional max).
///
/// ```text
/// limits ::= 0x00 min:u32 | 0x01 min:u32 max:u32
/// ```
fn emit_limits(buf: &mut Vec<u8>, limits: &crate::parser::module::Limits) {
    match limits.max {
        Some(max) => {
            write_vu1(buf, true);
            write_vu32(buf, limits.min);
            write_vu32(buf, max);
        }
        None => {
            write_vu1(buf, false);
            write_vu32(buf, limits.min);
        }
    }
}

/// Encodes a reference type.
fn emit_ref_type(buf: &mut Vec<u8>, rt: &RefType) {
    buf.push(rt.wire_byte());
}

/// Encodes an instruction sequence followed by the end marker.
///
/// Binary-parsed expressions include a trailing End instruction; WAT-parsed
/// expressions do not. This function handles both by encoding all non-End
/// instructions, then unconditionally appending the end marker.
fn emit_expression(buf: &mut Vec<u8>, instructions: &[Instruction]) {
    for inst in instructions {
        if matches!(inst.kind, InstructionKind::End) {
            continue;
        }
        buf.extend(inst.kind.encode());
    }
    buf.push(OP_END);
}

/// Extracts a function index from a single-instruction init expression.
///
/// Element segments with func-index encoding store each entry as a
/// `[ref.func $idx, end]` expression. This extracts the index directly.
fn emit_func_index_from_expr(buf: &mut Vec<u8>, init: &[Instruction]) -> Result<(), EncodeError> {
    for inst in init {
        if let InstructionKind::RefFunc { func_idx } = &inst.kind {
            write_vu32(buf, *func_idx);
            return Ok(());
        }
    }
    Err(EncodeError::InvalidState(
        "element init expression missing ref.func".to_string(),
    ))
}
