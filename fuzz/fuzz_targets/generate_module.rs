#![no_main]

//! Structure-aware WebAssembly module fuzzer
//!
//! Instead of mutating raw bytes, this generates syntactically valid WebAssembly
//! modules from structured fuzz input. This is much more effective at finding
//! runtime bugs because generated modules pass parsing and execute interesting code.

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;
use std::sync::Arc;

use krasm::parser::module::{ExportIndex, ValueType};
use krasm::parser::{self, reader::Reader};
use krasm::runtime::{EngineKind, Store, Value};

/// Configuration for generated module
#[derive(Debug, Arbitrary)]
struct ModuleConfig {
    /// Number of i32 parameters for the main function (0-4)
    num_i32_params: u8,
    /// Number of i64 parameters (0-2)
    num_i64_params: u8,
    /// Number of local variables (0-10)
    num_locals: u8,
    /// Whether to have a memory
    has_memory: bool,
    /// Initial memory pages (0-4)
    memory_pages: u8,
    /// Instructions to generate
    instructions: Vec<Instruction>,
    /// Argument values for execution
    args: Vec<i64>,
}

/// Simplified instruction set for generation
#[derive(Debug, Clone, Arbitrary)]
enum Instruction {
    // Constants
    I32Const(i32),
    I64Const(i64),

    // Local operations
    LocalGet(u8),
    LocalSet(u8),
    LocalTee(u8),

    // Arithmetic i32
    I32Add,
    I32Sub,
    I32Mul,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrU,
    I32ShrS,
    I32Clz,
    I32Ctz,
    I32Popcnt,

    // Arithmetic i64
    I64Add,
    I64Sub,
    I64Mul,
    I64And,
    I64Or,
    I64Xor,

    // Comparison
    I32Eqz,
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I64Eqz,

    // Conversion
    I32WrapI64,
    I64ExtendI32S,
    I64ExtendI32U,

    // Control flow
    Drop,
    Select,
    Nop,

    // Memory operations (if memory exists)
    I32Load(u8),  // offset
    I32Store(u8), // offset
    I64Load(u8),
    I64Store(u8),
    MemorySize,
    MemoryGrow,

    // Simple block
    Block(Vec<Instruction>),
    Loop(Vec<Instruction>),
    If(Vec<Instruction>, Vec<Instruction>),
    Br(u8),
    BrIf(u8),
}

impl Instruction {
    fn encode(&self, out: &mut Vec<u8>, has_memory: bool, depth: u8) {
        match self {
            Instruction::I32Const(v) => {
                out.push(0x41);
                encode_sleb128(*v as i64, out);
            }
            Instruction::I64Const(v) => {
                out.push(0x42);
                encode_sleb128(*v, out);
            }
            Instruction::LocalGet(idx) => {
                out.push(0x20);
                out.push(*idx);
            }
            Instruction::LocalSet(idx) => {
                out.push(0x21);
                out.push(*idx);
            }
            Instruction::LocalTee(idx) => {
                out.push(0x22);
                out.push(*idx);
            }
            Instruction::I32Add => out.push(0x6a),
            Instruction::I32Sub => out.push(0x6b),
            Instruction::I32Mul => out.push(0x6c),
            Instruction::I32And => out.push(0x71),
            Instruction::I32Or => out.push(0x72),
            Instruction::I32Xor => out.push(0x73),
            Instruction::I32Shl => out.push(0x74),
            Instruction::I32ShrU => out.push(0x76),
            Instruction::I32ShrS => out.push(0x75),
            Instruction::I32Clz => out.push(0x67),
            Instruction::I32Ctz => out.push(0x68),
            Instruction::I32Popcnt => out.push(0x69),
            Instruction::I64Add => out.push(0x7c),
            Instruction::I64Sub => out.push(0x7d),
            Instruction::I64Mul => out.push(0x7e),
            Instruction::I64And => out.push(0x83),
            Instruction::I64Or => out.push(0x84),
            Instruction::I64Xor => out.push(0x85),
            Instruction::I32Eqz => out.push(0x45),
            Instruction::I32Eq => out.push(0x46),
            Instruction::I32Ne => out.push(0x47),
            Instruction::I32LtS => out.push(0x48),
            Instruction::I32LtU => out.push(0x49),
            Instruction::I32GtS => out.push(0x4a),
            Instruction::I32GtU => out.push(0x4b),
            Instruction::I64Eqz => out.push(0x50),
            Instruction::I32WrapI64 => out.push(0xa7),
            Instruction::I64ExtendI32S => out.push(0xac),
            Instruction::I64ExtendI32U => out.push(0xad),
            Instruction::Drop => out.push(0x1a),
            Instruction::Select => out.push(0x1b),
            Instruction::Nop => out.push(0x01),
            Instruction::I32Load(offset) if has_memory => {
                out.push(0x28);
                out.push(0x02); // align
                encode_uleb128(*offset as u64, out);
            }
            Instruction::I32Store(offset) if has_memory => {
                out.push(0x36);
                out.push(0x02); // align
                encode_uleb128(*offset as u64, out);
            }
            Instruction::I64Load(offset) if has_memory => {
                out.push(0x29);
                out.push(0x03); // align
                encode_uleb128(*offset as u64, out);
            }
            Instruction::I64Store(offset) if has_memory => {
                out.push(0x37);
                out.push(0x03); // align
                encode_uleb128(*offset as u64, out);
            }
            Instruction::MemorySize if has_memory => {
                out.push(0x3f);
                out.push(0x00);
            }
            Instruction::MemoryGrow if has_memory => {
                out.push(0x40);
                out.push(0x00);
            }
            Instruction::Block(body) if depth < 3 => {
                out.push(0x02); // block
                out.push(0x40); // void return
                for instr in body.iter().take(10) {
                    instr.encode(out, has_memory, depth + 1);
                }
                out.push(0x0b); // end
            }
            Instruction::Loop(body) if depth < 3 => {
                out.push(0x03); // loop
                out.push(0x40); // void return
                for instr in body.iter().take(10) {
                    instr.encode(out, has_memory, depth + 1);
                }
                out.push(0x0b); // end
            }
            Instruction::If(then_body, else_body) if depth < 3 => {
                // Need a condition on stack
                out.push(0x41); // i32.const
                out.push(0x01); // 1
                out.push(0x04); // if
                out.push(0x40); // void return
                for instr in then_body.iter().take(5) {
                    instr.encode(out, has_memory, depth + 1);
                }
                if !else_body.is_empty() {
                    out.push(0x05); // else
                    for instr in else_body.iter().take(5) {
                        instr.encode(out, has_memory, depth + 1);
                    }
                }
                out.push(0x0b); // end
            }
            Instruction::Br(label) if depth > 0 => {
                out.push(0x0c); // br
                out.push((*label).min(depth - 1));
            }
            Instruction::BrIf(label) if depth > 0 => {
                // Need a condition
                out.push(0x41); // i32.const
                out.push(0x00); // 0 (won't branch)
                out.push(0x0d); // br_if
                out.push((*label).min(depth - 1));
            }
            // Skip instructions that don't apply
            _ => {}
        }
    }
}

fn encode_uleb128(mut val: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (val & 0x7f) as u8;
        val >>= 7;
        if val == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

fn encode_sleb128(mut val: i64, out: &mut Vec<u8>) {
    loop {
        let byte = (val & 0x7f) as u8;
        val >>= 7;
        let done = (val == 0 && (byte & 0x40) == 0) || (val == -1 && (byte & 0x40) != 0);
        if done {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

fn generate_module(config: &ModuleConfig) -> Vec<u8> {
    let mut module = Vec::new();

    // Magic and version
    module.extend_from_slice(b"\x00asm\x01\x00\x00\x00");

    // Calculate parameters
    let num_i32 = (config.num_i32_params % 5) as usize;
    let num_i64 = (config.num_i64_params % 3) as usize;
    let num_locals = (config.num_locals % 11) as usize;
    let has_memory = config.has_memory;
    let memory_pages = (config.memory_pages % 5) as u32;

    // Type section - function signature
    let mut types = Vec::new();
    types.push(0x01); // 1 type
    types.push(0x60); // func type

    // Parameters
    types.push((num_i32 + num_i64) as u8);
    for _ in 0..num_i32 {
        types.push(0x7f); // i32
    }
    for _ in 0..num_i64 {
        types.push(0x7e); // i64
    }

    // Return type - i32
    types.push(0x01);
    types.push(0x7f);

    module.push(0x01); // type section
    encode_uleb128(types.len() as u64, &mut module);
    module.extend_from_slice(&types);

    // Function section
    module.push(0x03); // function section
    module.push(0x02); // size
    module.push(0x01); // 1 function
    module.push(0x00); // type index 0

    // Memory section (if enabled)
    if has_memory {
        module.push(0x05); // memory section
        module.push(0x04); // size
        module.push(0x01); // 1 memory
        module.push(0x01); // minimum and maximum
        encode_uleb128(memory_pages as u64, &mut module);
        encode_uleb128(4, &mut module); // Bound memory.grow as well as initial allocation.
    }

    // Export section
    module.push(0x07); // export section
    module.push(0x08); // size
    module.push(0x01); // 1 export
    module.push(0x04); // name length
    module.extend_from_slice(b"main");
    module.push(0x00); // func export
    module.push(0x00); // func index 0

    // Code section
    let mut code = Vec::new();
    code.push(0x01); // 1 function

    let mut func_body = Vec::new();

    // Local declarations
    if num_locals > 0 {
        func_body.push(0x01); // 1 local decl
        func_body.push(num_locals as u8);
        func_body.push(0x7f); // i32
    } else {
        func_body.push(0x00); // no locals
    }

    let total_locals = num_i32 + num_i64 + num_locals;

    // Generate instructions
    for instr in config.instructions.iter().take(50) {
        // Clamp local indices
        let clamped = match instr {
            Instruction::LocalGet(idx) | Instruction::LocalSet(idx) | Instruction::LocalTee(idx)
                if total_locals > 0 =>
            {
                let new_idx = (*idx as usize % total_locals) as u8;
                match instr {
                    Instruction::LocalGet(_) => Instruction::LocalGet(new_idx),
                    Instruction::LocalSet(_) => Instruction::LocalSet(new_idx),
                    Instruction::LocalTee(_) => Instruction::LocalTee(new_idx),
                    _ => unreachable!(),
                }
            }
            // Skip local ops if no locals
            Instruction::LocalGet(_) | Instruction::LocalSet(_) | Instruction::LocalTee(_) => {
                continue;
            }
            other => other.clone(),
        };
        clamped.encode(&mut func_body, has_memory, 0);
    }

    // Ensure return value on stack
    func_body.push(0x41); // i32.const
    func_body.push(0x00); // 0
    func_body.push(0x0b); // end

    // Function body with size prefix
    let mut func_with_size = Vec::new();
    encode_uleb128(func_body.len() as u64, &mut func_with_size);
    func_with_size.extend_from_slice(&func_body);

    code.extend_from_slice(&func_with_size);

    module.push(0x0a); // code section
    encode_uleb128(code.len() as u64, &mut module);
    module.extend_from_slice(&code);

    module
}

fuzz_target!(|data: &[u8]| {
    // Parse arbitrary input into module config
    let mut u = Unstructured::new(data);
    let config: ModuleConfig = match u.arbitrary() {
        Ok(c) => c,
        Err(_) => return,
    };

    // Generate the module bytes
    let wasm_bytes = generate_module(&config);

    // Parse and execute
    let mut reader = Reader::new(wasm_bytes);
    let module = match parser::parse(&HashMap::new(), "fuzz", &mut reader) {
        Ok(m) => m,
        Err(_) => return, // Generated module was invalid (shouldn't happen often)
    };

    let module = Arc::new(module);
    for engine in [EngineKind::Structured, EngineKind::Flat] {
        let mut store = Store::new();
        store.set_engine(engine);
        let instance_id = match store.create_instance(Arc::clone(&module), None) {
            Ok(id) => id,
            Err(_) => continue,
        };

        for export in &module.exports.exports {
            if let ExportIndex::Function(func_idx) = export.index {
                if let Some(func_type) = module.get_function_type_by_idx(func_idx) {
                    let args = func_type
                        .parameters
                        .iter()
                        .enumerate()
                        .map(|(i, param_type)| {
                            let arg = config.args.get(i).copied().unwrap_or(0);
                            match param_type {
                                ValueType::I32 => Value::I32(arg as i32),
                                ValueType::I64 => Value::I64(arg),
                                _ => unreachable!("generated parameters are integers"),
                            }
                        })
                        .collect();
                    let _ = store.invoke_export(instance_id, &export.name, args, Some(100_000));
                }
            }
        }
    }
});
