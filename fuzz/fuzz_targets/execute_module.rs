#![no_main]

use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;
use std::sync::Arc;

use krasm::parser::module::{ExportIndex, Positional, ValueType};
use krasm::parser::{self, reader::Reader};
use krasm::runtime::{EngineKind, Store, Value};

const MAX_MEMORY_PAGES: u32 = 16;
const MAX_TABLE_ELEMENTS: u32 = 1024;

/// Generate a Value of the specified type from fuzz data
fn generate_value(typ: &ValueType, data: &mut &[u8]) -> Value {
    match typ {
        ValueType::I32 => {
            let val = if data.len() >= 4 {
                let bytes = [data[0], data[1], data[2], data[3]];
                *data = &data[4..];
                i32::from_le_bytes(bytes)
            } else {
                0
            };
            Value::I32(val)
        }
        ValueType::I64 => {
            let val = if data.len() >= 8 {
                let bytes = [data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]];
                *data = &data[8..];
                i64::from_le_bytes(bytes)
            } else {
                0
            };
            Value::I64(val)
        }
        ValueType::F32 => {
            let val = if data.len() >= 4 {
                let bytes = [data[0], data[1], data[2], data[3]];
                *data = &data[4..];
                f32::from_le_bytes(bytes)
            } else {
                0.0
            };
            Value::F32(val)
        }
        ValueType::F64 => {
            let val = if data.len() >= 8 {
                let bytes = [data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]];
                *data = &data[8..];
                f64::from_le_bytes(bytes)
            } else {
                0.0
            };
            Value::F64(val)
        }
        ValueType::FuncRef => Value::FuncRef(None),
        ValueType::ExternRef => Value::ExternRef(None),
        ValueType::V128 => {
            let mut bytes = [0; 16];
            let len = data.len().min(bytes.len());
            bytes[..len].copy_from_slice(&data[..len]);
            *data = &data[len..];
            Value::V128(bytes)
        }
    }
}

fuzz_target!(|data: &[u8]| {
    // Need enough bytes for a minimal wasm module plus some fuzz data for arguments
    if data.len() < 8 {
        return;
    }

    // Split data: most for the wasm module, tail for argument generation
    let split_point = data.len().saturating_sub(64).max(8);
    let wasm_data = &data[..split_point];

    // Parse the module
    let mut reader = Reader::new(wasm_data.to_vec());
    let mut module = match parser::parse(&HashMap::new(), "fuzz", &mut reader) {
        Ok(m) => m,
        Err(_) => return,
    };

    // Instantiation does not expose a start-function budget.
    if module.start.has_position() || module.table.tables.len() > 16 {
        return;
    }
    // Bound allocation independently of the instruction budget, including growth.
    for memory in &mut module.memory.memory {
        if memory.limits.min > MAX_MEMORY_PAGES {
            return;
        }
        memory.limits.max = Some(memory.limits.max.unwrap_or(MAX_MEMORY_PAGES).min(MAX_MEMORY_PAGES));
    }
    for table in &mut module.table.tables {
        if table.limits.min > MAX_TABLE_ELEMENTS {
            return;
        }
        table.limits.max = Some(table.limits.max.unwrap_or(MAX_TABLE_ELEMENTS).min(MAX_TABLE_ELEMENTS));
    }
    let module = Arc::new(module);
    for engine in [EngineKind::Structured, EngineKind::Flat] {
        let mut store = Store::new();
        store.set_engine(engine);
        let instance_id = match store.create_instance(Arc::clone(&module), None) {
            Ok(id) => id,
            Err(_) => continue,
        };
        let mut arg_data = &data[split_point..];
        for export in &module.exports.exports {
            if let ExportIndex::Function(func_idx) = export.index {
                if let Some(func_type) = module.get_function_type_by_idx(func_idx) {
                    let args = func_type
                        .parameters
                        .iter()
                        .map(|typ| generate_value(typ, &mut arg_data))
                        .collect();
                    // Engine-specific budgets bound execution, not equivalent instruction counts.
                    let _ = store.invoke_export(instance_id, &export.name, args, Some(100_000));
                }
            }
        }
    }
});
