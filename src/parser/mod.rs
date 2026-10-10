//! WebAssembly binary format parser.
//!
//! Decodes `.wasm` bytes into a [`module::Module`] representation. The parser
//! reads sections in wire order (§5.5.2), validates structural constraints
//! inline (section ordering, count consistency, index bounds), and builds
//! a structured instruction tree for each function body.
//!
//! The entry point is [`parse()`], which consumes a [`reader::Reader`] and
//! returns a fully decoded `Module`.

pub(crate) mod encoding;
pub mod index_spaces;
pub mod instruction;
pub(crate) mod limits;
pub mod module;
pub mod reader;
pub(crate) mod structure_builder;
pub mod structured;
pub(crate) mod validate;

pub use validate::{ValidatedModule, ValidationError};

#[cfg(test)]
mod utf8_tests;

use std::collections::HashMap;
use std::io;

use self::module::Positional;

/// Decode a WebAssembly binary module from the given byte reader.
///
/// Reads the 8-byte header (magic number + version), then iterates through
/// sections in wire order. Each section is validated for correct ordering,
/// length consistency, and structural invariants (e.g. function/code count
/// match, start function signature, data count agreement).
///
/// The `module_registry` provides previously parsed modules for import
/// resolution during linking.
///
/// # Errors
///
/// Returns [`instruction::DecodeError`] on malformed input: bad magic number,
/// unknown section IDs, section length mismatches, invalid LEB128 encoding,
/// out-of-range indices, or type validation failures.
pub fn parse(
    module_registry: &HashMap<String, module::Module>,
    name: &str,
    bytes: &mut reader::Reader,
) -> Result<ValidatedModule, instruction::DecodeError> {
    let mut unit = module::Module::new(name);

    read_header(bytes, &mut unit.magic, &mut unit.version)?;

    let mut last_section_id = 0;

    loop {
        if !bytes.has_at_least(1) {
            break;
        }

        let sec_id = bytes
            .read_byte()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "malformed section id"))?;
        if sec_id != encoding::SECTION_CUSTOM {
            // DataCount (12) precedes Code (10) in wire order; remap to monotonic sequence for ordering validation
            let mapped_section_id = match sec_id {
                encoding::SECTION_CODE => encoding::SECTION_DATA,
                encoding::SECTION_DATA => encoding::SECTION_DATA_COUNT,
                encoding::SECTION_DATA_COUNT => encoding::SECTION_CODE,
                _ => sec_id,
            };
            if mapped_section_id <= last_section_id {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected content after last section").into());
            }
            last_section_id = mapped_section_id;
        }
        let sec_len = bytes.read_vu32()?;

        if !bytes.has_at_least(sec_len as usize) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "length out of bounds").into());
        }
        let start_pos = bytes.pos();

        read_sections(sec_id, bytes, module_registry, &mut unit, sec_len).map_err(
            |e: instruction::DecodeError| match e {
                instruction::DecodeError::Io(err) => match err.kind() {
                    io::ErrorKind::UnexpectedEof => {
                        io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected end of section or function").into()
                    }
                    _ => instruction::DecodeError::Io(err),
                },
                _ => e,
            },
        )?;
        let end_pos = bytes.pos();
        let actual_len = end_pos - start_pos;
        if actual_len != sec_len as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("section length mismatch, expected {sec_len}, got {actual_len}"),
            )
            .into());
        }
    }

    if unit.functions.functions.len() != unit.code.code.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "function and code section have inconsistent lengths",
        )
        .into());
    }

    // Validate data count matches actual data segments
    if unit.data_count.has_position() && unit.data_count.count as usize != unit.data.data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "data count and data section have inconsistent lengths",
        )
        .into());
    }

    // Validate start section if present
    if unit.start.has_position() {
        let total_functions = unit.imports.function_count() + unit.functions.functions.len();
        if unit.start.start >= total_functions as u32 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unknown function").into());
        }

        // Check if the start function has the correct signature (no params, no returns)
        let func_idx = unit.start.start;
        let type_idx = if func_idx < unit.imports.function_count() as u32 {
            // It's an imported function
            unit.imports
                .get_function_type_index(func_idx)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unknown function"))?
        } else {
            // It's a local function
            let local_idx = func_idx - unit.imports.function_count() as u32;
            unit.functions
                .functions
                .get(local_idx as usize)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unknown function"))?
                .ftype_index
        };

        let func_type = unit
            .types
            .types
            .get(type_idx as usize)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unknown type"))?;

        if !func_type.parameters.is_empty() || !func_type.return_types.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "start function").into());
        }
    }

    // Every section and function body was validated as it was decoded.
    Ok(ValidatedModule::from_validated(unit))
}

/// Dispatch a single section by ID, delegating to the appropriate section reader.
fn read_sections(
    sec_id: u8,
    bytes: &mut reader::Reader,
    module_registry: &HashMap<String, module::Module>,
    unit: &mut module::Module,
    section_len: u32,
) -> Result<(), instruction::DecodeError> {
    let start_pos: usize = bytes.pos();
    let section_end = start_pos + section_len as usize;
    let positional: &mut dyn module::Positional = match sec_id {
        encoding::SECTION_TYPE => {
            read_section_type(bytes, &mut unit.types, section_end)?;
            &mut unit.types as &mut dyn module::Positional
        }
        encoding::SECTION_IMPORT => {
            read_section_import(
                bytes,
                module_registry,
                &mut unit.imports,
                &unit.types,
                &unit.memory,
                &unit.table,
                section_end,
            )?;
            &mut unit.imports as &mut dyn module::Positional
        }
        encoding::SECTION_FUNCTION => {
            read_section_function(bytes, &mut unit.functions, &unit.types, section_end)?;
            &mut unit.functions as &mut dyn module::Positional
        }
        encoding::SECTION_TABLE => {
            read_section_table(bytes, &mut unit.table, section_end)?;
            &mut unit.table as &mut dyn module::Positional
        }
        encoding::SECTION_MEMORY => {
            read_section_memory(bytes, &mut unit.memory, &unit.imports, section_end)?;
            &mut unit.memory as &mut dyn module::Positional
        }
        encoding::SECTION_GLOBAL => {
            read_section_global(
                bytes,
                &mut unit.globals,
                &unit.imports,
                (unit.imports.function_count() + unit.functions.functions.len()) as u32,
                section_end,
            )?;
            &mut unit.globals as &mut dyn module::Positional
        }
        encoding::SECTION_EXPORT => {
            read_section_export(
                bytes,
                &mut unit.exports,
                &unit.imports,
                &unit.functions,
                &unit.globals,
                &unit.table,
                &unit.memory,
                section_end,
            )?;
            &mut unit.exports as &mut dyn module::Positional
        }
        encoding::SECTION_START => {
            read_section_start(bytes, &mut unit.start)?;
            &mut unit.start as &mut dyn module::Positional
        }
        encoding::SECTION_ELEMENT => {
            read_section_elements(
                bytes,
                &unit.imports,
                &unit.table,
                &mut unit.elements,
                (unit.imports.function_count() + unit.functions.functions.len()) as u32,
                section_end,
            )?;
            &mut unit.elements as &mut dyn module::Positional
        }
        encoding::SECTION_CODE => {
            read_section_code(bytes, unit, section_end)?;
            &mut unit.code as &mut dyn module::Positional
        }
        encoding::SECTION_DATA => {
            read_section_data(
                bytes,
                &unit.imports,
                &mut unit.data,
                if unit.data_count.has_position() {
                    Some(unit.data_count.count)
                } else {
                    None
                },
                unit.memory.memory.len() as u32,
                section_end,
            )?;
            &mut unit.data as &mut dyn module::Positional
        }
        encoding::SECTION_DATA_COUNT => {
            read_section_datacount(bytes, &mut unit.data_count)?;
            &mut unit.data_count as &mut dyn module::Positional
        }
        encoding::SECTION_CUSTOM => {
            let custom = read_section_custom(bytes, section_len, &mut unit.custom)?;
            custom as &mut dyn module::Positional
        }
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "malformed section id").into()),
    };

    if bytes.pos() != (start_pos + section_len as usize) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "section size mismatch").into());
    }

    let end = bytes.pos();
    positional.set_position(start_pos as u32, end as u32);

    Ok(())
}

fn read_header(bytes: &mut reader::Reader, magic: &mut u32, version: &mut u32) -> Result<(), io::Error> {
    *magic = bytes.read_u32().and_then(|value| {
        if value == 0x6d736100 {
            Ok(value)
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidData, "magic header not detected"))
        }
    })?;
    *version = bytes.read_u32().and_then(|value| {
        if value == 0x1 {
            Ok(value)
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidData, "unknown binary version"))
        }
    })?;
    Ok(())
}

/* SECTION READERS ************************************************/

fn read_value_types_list(bytes: &mut reader::Reader, max_count: u32) -> Result<Vec<module::ValueType>, io::Error> {
    let mut rt: Vec<module::ValueType> = vec![];
    let count = bytes.read_vu32()?;
    if count > max_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "function type count exceeds implementation limit",
        ));
    }
    bytes.validate_item_count(count)?;
    for _ in 0..count {
        let value = module::ValueType::decode(bytes.read_byte()?).map_err(std::io::Error::other)?;
        rt.push(value);
    }
    Ok(rt)
}

fn read_section_type(
    bytes: &mut reader::Reader,
    types: &mut module::TypeSection,
    section_end: usize,
) -> Result<(), io::Error> {
    let count = bytes.read_vu32()?;
    if count > limits::MAX_TYPES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "type count exceeds implementation limit",
        ));
    }
    bytes.validate_item_count_in_section(count, section_end)?;

    for _ in 0..count {
        if bytes.read_byte()? != encoding::TYPE_FUNC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                // NOTE: binary-leb128.wast asserts this, for some reason, by pushing through a 0xe0 byte
                // and expecting this error string, however the spec says to expect an 0x60 byte always
                "integer representation too long",
                // "expected 0x60 to lead function type, got 0x{:02x} for func #{} / #{}", byt, ii, count
            ));
        }
        let parameters = read_value_types_list(bytes, limits::MAX_FUNCTION_PARAMS)?;
        let return_types = read_value_types_list(bytes, limits::MAX_FUNCTION_RETURNS)?;

        types.push(module::FunctionType {
            parameters,
            return_types,
        });
    }

    Ok(())
}

fn read_section_import(
    bytes: &mut reader::Reader,
    module_registry: &HashMap<String, module::Module>,
    imports: &mut module::ImportSection,
    types: &module::TypeSection,
    memory: &module::MemorySection,
    tables: &module::TableSection,
    section_end: usize,
) -> Result<(), instruction::DecodeError> {
    let count = bytes.read_vu32()?;
    if count > limits::MAX_IMPORTS {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "import count exceeds implementation limit").into());
    }
    bytes.validate_item_count_in_section(count, section_end)?;

    let mut memory_import_count = 0;

    for _ in 0..count {
        let import = module::Import::decode(bytes, module_registry, types, memory, tables, section_end)?;

        // Validate that we don't have multiple memory imports
        if matches!(import.external_kind, module::ExternalKind::Memory(_)) {
            memory_import_count += 1;
            if memory_import_count > 1 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "multiple memories").into());
            }
        }

        imports.push(import);
    }

    Ok(())
}

impl module::Import {
    pub fn decode(
        reader: &mut reader::Reader,
        _module_registry: &HashMap<String, module::Module>,
        types: &module::TypeSection,
        _memory: &module::MemorySection,
        _tables: &module::TableSection,
        section_end: usize,
    ) -> Result<module::Import, instruction::DecodeError> {
        let module = reader.read_string_bounded(section_end)?;
        let name = reader.read_string_bounded(section_end)?;
        let external_kind = match module::ExternalKind::decode(reader, types.len() as u32) {
            Ok(kind) => kind,
            Err(e) => {
                return Err(e);
            }
        };
        // Import validation is deferred to instantiation time
        // This allows modules to parse successfully even if imports are not available
        // during parsing. The spec requires that we validate structure during parsing
        // but not existence or type compatibility of imports.
        Ok(module::Import {
            external_kind,
            module,
            name,
        })
    }
}

fn read_section_function(
    bytes: &mut reader::Reader,
    functions: &mut module::FunctionSection,
    types: &module::TypeSection,
    section_end: usize,
) -> Result<(), io::Error> {
    let count = bytes.read_vu32()?;
    if count > limits::MAX_FUNCTIONS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "function count exceeds implementation limit",
        ));
    }
    bytes.validate_item_count_in_section(count, section_end)?;

    for _ in 0..count {
        let ftype_index = bytes.read_vu32()?;
        if ftype_index >= types.len() as u32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unknown type; function type index out of range, expected < {}, got {}",
                    types.len(),
                    ftype_index
                ),
            ));
        }

        functions.push(module::Function { ftype_index })
    }

    Ok(())
}

fn read_section_memory(
    bytes: &mut reader::Reader,
    memory: &mut module::MemorySection,
    imports: &module::ImportSection,
    section_end: usize,
) -> Result<(), instruction::DecodeError> {
    let count = bytes.read_vu32()?;
    bytes.validate_item_count_in_section(count, section_end)?;

    // Check if we already have imported memories
    let imported_memory_count = imports
        .imports
        .iter()
        .filter(|imp| matches!(imp.external_kind, module::ExternalKind::Memory(_)))
        .count();

    // WASM 1.0 spec: at most one memory total
    if imported_memory_count + count as usize > 1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "multiple memories").into());
    }

    for _ in 0..count {
        let limits = module::Limits::decode_memory_limits(bytes)?;
        memory.push(module::Memory { limits })
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn read_section_export(
    bytes: &mut reader::Reader,
    exports: &mut module::ExportSection,
    imports: &module::ImportSection,
    functions: &module::FunctionSection,
    globals: &module::GlobalSection,
    tables: &module::TableSection,
    memory: &module::MemorySection,
    section_end: usize,
) -> Result<(), io::Error> {
    let count = bytes.read_vu32()?;
    if count > limits::MAX_EXPORTS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "export count exceeds implementation limit",
        ));
    }
    bytes.validate_item_count_in_section(count, section_end)?;

    for _ in 0..count {
        let name = bytes.read_string_bounded(section_end)?;
        if exports.exports.iter().any(|e| e.name == name) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("duplicate export name: {name}"),
            ));
        }
        let typ = bytes.read_byte()?;
        let idx = bytes.read_vu32()?;
        let index = module::ExportIndex::decode(typ, idx)?;
        match index {
            module::ExportIndex::Function(idx) => {
                if idx >= (functions.functions.len() + imports.function_count()) as u32 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown function index for export {idx}"),
                    ));
                }
            }
            module::ExportIndex::Global(idx) => {
                if idx >= (globals.globals.len() + imports.global_count()) as u32 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown global index for export {idx}"),
                    ));
                }
            }
            module::ExportIndex::Table(idx) => {
                if idx >= (tables.tables.len() + imports.table_count()) as u32 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown table index for export {idx}"),
                    ));
                }
            }
            module::ExportIndex::Memory(idx) => {
                let total_memories = (memory.memory.len() + imports.memory_count()) as u32;

                if idx >= total_memories {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown memory index for export {idx}"),
                    ));
                }
            }
        }
        exports.push(module::Export { index, name })
    }

    Ok(())
}

fn read_section_code(
    bytes: &mut reader::Reader,
    module: &mut module::Module,
    section_end: usize,
) -> Result<(), instruction::DecodeError> {
    let count = bytes.read_vu32()?;
    bytes.validate_item_count_in_section(count, section_end)?;

    if count != module.functions.functions.len() as u32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "function and code section have inconsistent lengths",
        )
        .into());
    }

    for ii in 0..count {
        let size = bytes.read_vu32()?;
        if size > limits::MAX_FUNCTION_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "function body size exceeds implementation limit",
            )
            .into());
        }
        let size = size as usize;
        let start_pos = bytes.pos();
        let end_pos = start_pos + size;
        let locals_count = bytes.read_vu32()?;
        bytes.validate_item_count(locals_count)?;

        let mut nts = Vec::new();
        for _ in 0..locals_count {
            let local_n = bytes.read_vu32()?;
            let b = bytes.read_byte()?;
            let value = module::ValueType::decode(b).map_err(std::io::Error::other)?;
            nts.push((local_n, value));
        }

        let locals = module::Locals::new(nts);
        if locals.len() > u64::from(limits::MAX_FUNCTION_LOCALS) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "too many locals").into());
        }

        // Directly decode to structured representation without buffering
        let body = instruction::decode_function(bytes, module, &locals, module.code.len() as u32).map_err(|err| {
            if bytes.pos() > end_pos {
                io::Error::new(io::ErrorKind::InvalidData, "end opcode expected").into()
            } else if ii == count - 1 && bytes.pos() == end_pos {
                match err {
                    instruction::DecodeError::Validation(_) => err,
                    _ => io::Error::new(io::ErrorKind::InvalidData, "unexpected end of section or function").into(),
                }
            } else {
                err
            }
        })?;

        if bytes.pos() < end_pos {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected end of section or function").into());
        }

        let function_body = module::FunctionBody {
            locals,
            body,
            position: module::SectionPosition {
                start: start_pos as u32,
                end: (start_pos + size) as u32,
            },
        };
        module.code.code.push(function_body);
    }

    Ok(())
}

impl module::Data {
    pub fn decode(
        bytes: &mut reader::Reader,
        imports: &module::ImportSection,
        memory_count: u32,
        section_end: usize,
    ) -> Result<Self, instruction::DecodeError> {
        let mut mode = module::DataMode::Passive;
        let typ = bytes.read_vu32()?;
        match typ {
            encoding::DATA_ACTIVE => {
                mode = module::DataMode::Active {
                    memory_index: 0,
                    offset: instruction::decode_constant_expression(bytes, imports, module::ValueType::I32)?,
                };
            }
            encoding::DATA_PASSIVE => {}
            encoding::DATA_ACTIVE_EXPLICIT => {
                mode = module::DataMode::Active {
                    memory_index: bytes.read_vu32()?,
                    offset: instruction::decode_constant_expression(bytes, imports, module::ValueType::I32)?,
                };
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown data type: 0x{typ:02x}"),
            ))?,
        };

        if let module::DataMode::Active {
            memory_index,
            ref offset,
        } = mode
        {
            // Check if memory exists and memory_index is valid
            let total_memory_count = imports.memory_count() as u32 + memory_count;
            if memory_index >= total_memory_count {
                return Err(
                    io::Error::new(io::ErrorKind::InvalidData, format!("unknown memory {memory_index}")).into(),
                );
            }
            // we expect a constant and an End
            if offset.len() != 2 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "type mismatch").into());
            }
        }

        Ok(module::Data {
            init: bytes.read_u8vec_bounded(section_end)?,
            mode,
        })
    }
}

fn read_section_data(
    bytes: &mut reader::Reader,
    imports: &module::ImportSection,
    datas: &mut module::DataSection,
    data_count: Option<u32>,
    memory_count: u32,
    section_end: usize,
) -> Result<(), instruction::DecodeError> {
    let count = bytes.read_vu32()?;
    if count > limits::MAX_DATA_SEGMENTS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "data segment count exceeds implementation limit",
        )
        .into());
    }
    bytes.validate_item_count_in_section(count, section_end)?;

    if let Some(expected_count) = data_count
        && expected_count != count
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "data count and data section have inconsistent lengths",
        )
        .into());
    }

    for _ in 0..count {
        let data = module::Data::decode(bytes, imports, memory_count, section_end)?;
        datas.data.push(data);
    }

    Ok(())
}

fn read_section_custom<'a>(
    bytes: &mut reader::Reader,
    read_len: u32,
    customs: &'a mut Vec<module::CustomSection>,
) -> Result<&'a mut module::CustomSection, instruction::DecodeError> {
    let mut custom = module::CustomSection::new();
    let start_pos = bytes.pos();
    let section_end = start_pos + read_len as usize;
    custom.name = bytes.read_string_bounded(section_end)?;
    if bytes.pos() >= start_pos && read_len as usize >= (bytes.pos() - start_pos) {
        custom.data = bytes.read_bytes(read_len as usize - (bytes.pos() - start_pos))?;
        customs.push(custom);
        if let Some(last_custom) = customs.last_mut() {
            Ok(last_custom)
        } else {
            panic!("custom section not found");
        }
    } else {
        Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected end").into())
    }
}

fn read_section_datacount(
    bytes: &mut reader::Reader,
    data_count: &mut module::DataCountSection,
) -> Result<(), instruction::DecodeError> {
    data_count.count = bytes.read_vu32()?;
    Ok(())
}

impl module::ExternalKind {
    pub fn decode(
        bytes: &mut reader::Reader,
        type_count: u32,
    ) -> Result<module::ExternalKind, instruction::DecodeError> {
        let byte = bytes.read_byte()?;
        match byte {
            encoding::DESC_FUNC => {
                let typeidx = bytes.read_vu32()?;
                if typeidx >= type_count {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown type index {typeidx} for imported function"),
                    )
                    .into());
                }
                Ok(module::ExternalKind::Function(typeidx))
            }
            encoding::DESC_TABLE => {
                let tabletype = module::TableType::decode(bytes)?;
                Ok(module::ExternalKind::Table(tabletype))
            }
            encoding::DESC_MEMORY => {
                let memtype = module::Limits::decode_memory_limits(bytes)?;
                Ok(module::ExternalKind::Memory(memtype))
            }
            encoding::DESC_GLOBAL => {
                let globaltype = module::GlobalType::decode(bytes)?;
                Ok(module::ExternalKind::Global(globaltype))
            }
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "malformed import kind").into()),
        }
    }
}

impl module::GlobalType {
    pub fn decode(bytes: &mut reader::Reader) -> Result<Self, io::Error> {
        let value_type = module::ValueType::decode(bytes.read_byte()?).map_err(std::io::Error::other)?;
        let mutable = match bytes.read_byte()? {
            0x00 => false,
            0x01 => true,
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "malformed mutability"))?,
        };
        Ok(module::GlobalType { value_type, mutable })
    }
}

impl module::Global {
    pub fn decode(
        bytes: &mut reader::Reader,
        imports: &module::ImportSection,
        total_functions: u32,
    ) -> Result<Self, instruction::DecodeError> {
        let global_type = module::GlobalType::decode(bytes)?;
        Ok(module::Global {
            global_type,
            init: instruction::decode_constant_expression_with_ref_func(
                bytes,
                imports,
                global_type.value_type,
                total_functions,
            )?,
        })
    }
}

fn read_section_global(
    bytes: &mut reader::Reader,
    globals: &mut module::GlobalSection,
    imports: &module::ImportSection,
    total_functions: u32,
    section_end: usize,
) -> Result<(), instruction::DecodeError> {
    let count = bytes.read_vu32()?;
    if count > limits::MAX_GLOBALS {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "global count exceeds implementation limit").into());
    }
    bytes.validate_item_count_in_section(count, section_end)?;

    for _ in 0..count {
        let data = module::Global::decode(bytes, imports, total_functions)?;
        globals.globals.push(data);
    }

    Ok(())
}

impl module::RefType {
    pub fn decode(bytes: &mut reader::Reader) -> Result<Self, instruction::DecodeError> {
        let byte = bytes.read_byte()?;
        module::RefType::from_wire_byte(byte).map_err(|e| e.into())
    }
}

impl module::Limits {
    pub fn decode(bytes: &mut reader::Reader) -> Result<Self, instruction::DecodeError> {
        let has_max = bytes.read_vu1()?;
        let min = bytes.read_vu32()?;
        let max = if has_max { Some(bytes.read_vu32()?) } else { None };

        if let Some(max_val) = max
            && min > max_val
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "size minimum must not be greater than maximum",
            )
            .into());
        }

        Ok(module::Limits { min, max })
    }

    pub fn decode_memory_limits(bytes: &mut reader::Reader) -> Result<Self, instruction::DecodeError> {
        let has_max = bytes.read_vu1()?;
        let min = bytes.read_vu32()?;
        let max = if has_max {
            let max_val = bytes.read_vu32()?;
            // Validate max when explicitly specified
            if max_val > limits::MAX_MEMORY_PAGES_32 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "memory size must be at most 65536 pages (4GiB)",
                )
                .into());
            }
            Some(max_val)
        } else {
            None
        };

        if let Some(max_val) = max
            && min > max_val
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "size minimum must not be greater than maximum",
            )
            .into());
        }

        // Check page limit (4GiB) for memory min
        if min > limits::MAX_MEMORY_PAGES_32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "memory size must be at most 65536 pages (4GiB)",
            )
            .into());
        }

        Ok(module::Limits { min, max })
    }

    pub fn decode_table_limits(bytes: &mut reader::Reader) -> Result<Self, instruction::DecodeError> {
        let has_max = bytes.read_vu1()?;
        let min = bytes.read_vu32()?;
        let max = if has_max { Some(bytes.read_vu32()?) } else { None };

        // Spec validation: min must not exceed max
        if let Some(max_val) = max
            && min > max_val
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "size minimum must not be greater than maximum",
            )
            .into());
        }

        // Implementation limit: check the initial (minimum) size.
        // The declared maximum is just a limit and doesn't require allocation.
        if min > limits::MAX_TABLE_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "table size exceeds implementation limit").into());
        }

        Ok(module::Limits { min, max })
    }
}

impl module::TableType {
    pub fn decode(bytes: &mut reader::Reader) -> Result<Self, instruction::DecodeError> {
        // reftype then limits
        let ref_type = module::RefType::decode(bytes)?;
        let limits = module::Limits::decode_table_limits(bytes)?;
        Ok(module::TableType { ref_type, limits })
    }
}

fn read_section_table(
    bytes: &mut reader::Reader,
    table: &mut module::TableSection,
    section_end: usize,
) -> Result<(), instruction::DecodeError> {
    let count = bytes.read_vu32()?;
    if count > limits::MAX_TABLES {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "table count exceeds implementation limit").into());
    }
    bytes.validate_item_count_in_section(count, section_end)?;

    for _ in 0..count {
        let table_type = module::TableType::decode(bytes)?;
        table.tables.push(table_type);
    }

    Ok(())
}

fn read_section_start(bytes: &mut reader::Reader, start: &mut module::StartSection) -> Result<(), io::Error> {
    start.start = bytes.read_vu32()?;
    Ok(())
}

impl module::Element {
    pub fn decode(
        bytes: &mut reader::Reader,
        imports: &module::ImportSection,
        table: &module::TableSection,
        function_count: u32,
    ) -> Result<Self, instruction::DecodeError> {
        let read_type = |bytes: &mut reader::Reader| -> Result<module::RefType, io::Error> {
            let byte = bytes.read_byte()?;
            module::RefType::from_wire_byte(byte)
        };
        let read_element_kind = |bytes: &mut reader::Reader| -> Result<module::RefType, io::Error> {
            let byte = bytes.read_byte()?;
            match byte {
                encoding::ELEMKIND_FUNCREF => Ok(module::RefType::FuncRef),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown element kind: 0x{byte:02x}"),
                )),
            }
        };
        let read_table_index = |bytes: &mut reader::Reader| -> Result<u32, instruction::DecodeError> {
            bytes.read_vu32().map_err(|e| e.into())
        };
        let read_init = |bytes: &mut reader::Reader,
                         return_type: module::ValueType|
         -> Result<Vec<Vec<instruction::Instruction>>, instruction::DecodeError> {
            let count = bytes.read_vu32()?;
            if count > limits::MAX_TABLE_INIT_ENTRIES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "table init entries exceed implementation limit",
                )
                .into());
            }
            bytes.validate_item_count(count)?;
            let mut init: Vec<Vec<instruction::Instruction>> = vec![];
            for _ in 0..count {
                init.push(instruction::decode_constant_expression_with_ref_func(
                    bytes,
                    imports,
                    return_type,
                    function_count,
                )?);
            }
            Ok(init)
        };
        let read_func_indexes = |bytes: &mut reader::Reader| -> Result<Vec<u32>, io::Error> {
            let count = bytes.read_vu32()?;
            if count > limits::MAX_TABLE_INIT_ENTRIES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "table init entries exceed implementation limit",
                ));
            }
            bytes.validate_item_count(count)?;
            let mut func_indexes: Vec<u32> = vec![];
            for _ in 0..count {
                let func_index = bytes.read_vu32()?;
                if func_index >= function_count {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "unknown function"));
                }
                func_indexes.push(func_index);
            }
            Ok(func_indexes)
        };
        let read_func_index_init =
            |bytes: &mut reader::Reader| -> Result<Vec<Vec<instruction::Instruction>>, io::Error> {
                let fi = read_func_indexes(bytes)?;
                Ok(fi
                    .into_iter()
                    .map(|func_index| {
                        vec![
                            instruction::Instruction {
                                kind: instruction::InstructionKind::RefFunc { func_idx: func_index },
                                position: instruction::ByteRange { offset: 0, length: 0 },
                                original_bytes: Vec::new(),
                            },
                            instruction::Instruction {
                                kind: instruction::InstructionKind::End,
                                position: instruction::ByteRange { offset: 0, length: 0 },
                                original_bytes: Vec::new(),
                            },
                        ]
                    })
                    .collect())
            };

        let flags = bytes.read_vu32()?;
        let element = match flags {
            // 0:u32 𝑒:expr 𝑦*:vec(funcidx) => {type funcref, init ((ref.func 𝑦) end)*, mode active {table 0, offset 𝑒}}
            encoding::ELEM_ACTIVE_FUNCS => module::Element {
                flags,
                ref_type: module::RefType::FuncRef,
                mode: module::ElementMode::Active {
                    table_index: 0,
                    offset: instruction::decode_constant_expression(bytes, imports, module::ValueType::I32)?,
                },
                init: read_func_index_init(bytes)?,
            },
            // 1:u32 et : elemkind 𝑦*:vec(funcidx) => {type et, init ((ref.func 𝑦) end)*, mode passive}
            encoding::ELEM_PASSIVE_FUNCS => module::Element {
                flags,
                ref_type: read_element_kind(bytes)?,
                mode: module::ElementMode::Passive,
                init: read_func_index_init(bytes)?,
            },
            // 2:u32 𝑥:tableidx 𝑒:expr et : elemkind 𝑦*:vec(funcidx) => {type et, init ((ref.func 𝑦) end)*, mode active {table 𝑥, offset 𝑒}}
            encoding::ELEM_ACTIVE_TABLE_FUNCS => module::Element {
                flags,
                mode: module::ElementMode::Active {
                    table_index: read_table_index(bytes)?,
                    offset: instruction::decode_constant_expression(bytes, imports, module::ValueType::I32)?,
                },
                ref_type: read_element_kind(bytes)?,
                init: read_func_index_init(bytes)?,
            },
            // 3:u32 et : elemkind 𝑦*:vec(funcidx) => {type et, init ((ref.func 𝑦) end)*, mode declarative}
            encoding::ELEM_DECLARATIVE_FUNCS => module::Element {
                flags,
                ref_type: read_element_kind(bytes)?,
                mode: module::ElementMode::Declarative,
                init: read_func_index_init(bytes)?,
            },
            // 4:u32 𝑒:expr el *:vec(expr) => {type funcref, init el *, mode active {table 0, offset 𝑒}}
            encoding::ELEM_ACTIVE_EXPRS => module::Element {
                flags,
                ref_type: module::RefType::FuncRef,
                mode: module::ElementMode::Active {
                    table_index: 0,
                    offset: instruction::decode_constant_expression(bytes, imports, module::ValueType::I32)?,
                },
                init: read_init(bytes, module::ValueType::FuncRef)?,
            },
            // 5:u32 et : reftype el *:vec(expr) => {type 𝑒𝑡, init el *, mode passive}
            encoding::ELEM_PASSIVE_EXPRS => {
                let ref_type = read_type(bytes)?;
                let init = read_init(bytes, ref_type.into())?;
                module::Element {
                    flags,
                    ref_type,
                    mode: module::ElementMode::Passive,
                    init,
                }
            }
            // 6:u32 𝑥:tableidx 𝑒:expr et : reftype el *:vec(expr) => {type 𝑒𝑡, init el *, mode active {table 𝑥, offset 𝑒}}
            encoding::ELEM_ACTIVE_TABLE_EXPRS => {
                let table_index = read_table_index(bytes)?;
                let offset = instruction::decode_constant_expression(bytes, imports, module::ValueType::I32)?;
                let ref_type = read_type(bytes)?;
                let init = read_init(bytes, ref_type.into())?;
                module::Element {
                    flags,
                    mode: module::ElementMode::Active { table_index, offset },
                    ref_type,
                    init,
                }
            }
            // 7:u32 et : reftype el *:vec(expr) => {type 𝑒𝑡, init el *, mode declarative}
            encoding::ELEM_DECLARATIVE_EXPRS => {
                let ref_type = read_type(bytes)?;
                let init = read_init(bytes, ref_type.into())?;
                module::Element {
                    flags,
                    ref_type,
                    mode: module::ElementMode::Declarative,
                    init,
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown element type: 0x{flags:02x}"),
                )
                .into());
            }
        };
        if let module::ElementMode::Active { table_index, .. } = element.mode {
            let ref_type: Option<&module::TableType> = if table_index < imports.table_count() as u32 {
                imports.get_table(table_index)
            } else if (table_index - imports.table_count() as u32) < table.tables.len() as u32 {
                table.tables.get((table_index - imports.table_count() as u32) as usize)
            } else {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown table {table_index}")).into());
            };
            if let Some(ref_type) = ref_type
                && element.ref_type != ref_type.ref_type
            {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "element type mismatch").into());
            }
        };
        Ok(element)
    }
}

fn read_section_elements(
    bytes: &mut reader::Reader,
    imports: &module::ImportSection,
    table: &module::TableSection,
    elements: &mut module::ElementSection,
    function_count: u32,
    section_end: usize,
) -> Result<(), instruction::DecodeError> {
    let count = bytes.read_vu32()?;
    if count > limits::MAX_ELEMENT_SEGMENTS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "element segment count exceeds implementation limit",
        )
        .into());
    }
    bytes.validate_item_count_in_section(count, section_end)?;

    for _ in 0..count {
        let element = module::Element::decode(bytes, imports, table, function_count)?;
        elements.push(element);
    }

    Ok(())
}
