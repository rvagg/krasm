//! Standard spectest host module for the WebAssembly spec test suite.
//!
//! The spectest module provides the standard host imports that `.wast` spec
//! tests expect: no-op print functions, globals initialised to 666, a memory
//! (1 page min, 2 max), and a funcref table (10 min, 20 max).

use crate::parser::module::{
    Export, ExportIndex, Function, FunctionBody, FunctionType, Global, GlobalType, Limits, Locals, Memory, Module,
    RefType, SectionPosition, TableType, ValueType,
};
use crate::parser::structured::StructuredFunction;
use crate::runtime::{ImportObject, Memory as RuntimeMemory, Store, Table, Value};

/// Build a [`Module`] representing the spectest host module.
///
/// This is used by the binary parser's module registry so that modules
/// importing from `"spectest"` can resolve their import types.
pub fn create_spectest_module() -> Module {
    let mut m = Module::new("spectest");

    add_func(&mut m, "print", vec![], vec![]);
    add_func(&mut m, "print_i32", vec![ValueType::I32], vec![]);
    add_func(&mut m, "print_i64", vec![ValueType::I64], vec![]);
    add_func(&mut m, "print_f32", vec![ValueType::F32], vec![]);
    add_func(&mut m, "print_f64", vec![ValueType::F64], vec![]);
    add_func(&mut m, "print_i32_f32", vec![ValueType::I32, ValueType::F32], vec![]);
    add_func(&mut m, "print_f64_f64", vec![ValueType::F64, ValueType::F64], vec![]);

    add_global(&mut m, "global_i32", ValueType::I32);
    add_global(&mut m, "global_i64", ValueType::I64);
    add_global(&mut m, "global_f32", ValueType::F32);
    add_global(&mut m, "global_f64", ValueType::F64);

    m.table.tables.push(TableType {
        ref_type: RefType::FuncRef,
        limits: Limits { min: 10, max: None },
    });
    m.exports.exports.push(Export {
        index: ExportIndex::Table((m.table.tables.len() - 1) as u32),
        name: "table".to_string(),
    });

    m.memory.memory.push(Memory {
        limits: Limits { min: 1, max: None },
    });
    m.exports.exports.push(Export {
        index: ExportIndex::Memory((m.memory.memory.len() - 1) as u32),
        name: "memory".to_string(),
    });

    m
}

/// Create a spectest [`ImportObject`] with live allocations in the [`Store`].
///
/// Allocates globals (`i32`/`i64`/`f32`/`f64` = 666), no-op print functions,
/// a memory (1 page min, 2 max), and a funcref table (10 min, 20 max).
pub fn create_spectest_imports<T>(store: &mut Store<T>) -> ImportObject {
    let mut imports = ImportObject::new();

    let g_i32 = store.allocate_global(Value::I32(666), false);
    let g_i64 = store.allocate_global(Value::I64(666), false);
    let g_f32 = store.allocate_global(Value::F32(666.6), false);
    let g_f64 = store.allocate_global(Value::F64(666.6), false);
    imports.add_global("spectest", "global_i32", g_i32);
    imports.add_global("spectest", "global_i64", g_i64);
    imports.add_global("spectest", "global_f32", g_f32);
    imports.add_global("spectest", "global_f64", g_f64);

    imports.add_function("spectest", "print", store.wrap(|| {}));
    imports.add_function("spectest", "print_i32", store.wrap(|_: i32| {}));
    imports.add_function("spectest", "print_i64", store.wrap(|_: i64| {}));
    imports.add_function("spectest", "print_f32", store.wrap(|_: f32| {}));
    imports.add_function("spectest", "print_f64", store.wrap(|_: f64| {}));
    imports.add_function("spectest", "print_i32_f32", store.wrap(|_: i32, _: f32| {}));
    imports.add_function("spectest", "print_f64_f64", store.wrap(|_: f64, _: f64| {}));

    let spectest_memory = RuntimeMemory::new(1, Some(2)).unwrap();
    let mem_addr = store.allocate_memory(spectest_memory);
    imports.add_memory("spectest", "memory", mem_addr);

    let spectest_table = Table::new(RefType::FuncRef, Limits { min: 10, max: Some(20) }).unwrap();
    let table_addr = store.allocate_table(spectest_table);
    imports.add_table("spectest", "table", table_addr);

    imports
}

fn add_func(m: &mut Module, name: &str, parameters: Vec<ValueType>, return_types: Vec<ValueType>) {
    let ftype = FunctionType {
        parameters,
        return_types,
    };
    let typeidx = match m.types.find(&ftype) {
        Some(idx) => idx,
        None => {
            m.types.types.push(ftype);
            (m.types.types.len() - 1) as u32
        }
    };
    m.functions.functions.push(Function { ftype_index: typeidx });
    m.code.code.push(FunctionBody {
        locals: Locals::new(vec![]),
        body: StructuredFunction {
            body: vec![],
            local_count: 0,
            return_types: vec![],
            end_instruction: None,
        },
        position: SectionPosition { start: 0, end: 0 },
    });
    let funcidx = (m.functions.functions.len() - 1) as u32;
    m.exports.exports.push(Export {
        index: ExportIndex::Function(funcidx),
        name: name.to_string(),
    });
}

fn add_global(m: &mut Module, name: &str, value_type: ValueType) {
    m.globals.globals.push(Global {
        global_type: GlobalType {
            value_type,
            mutable: false,
        },
        init: vec![],
    });
    let idx = (m.globals.globals.len() - 1) as u32;
    m.exports.exports.push(Export {
        index: ExportIndex::Global(idx),
        name: name.to_string(),
    });
}
