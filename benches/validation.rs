use criterion::{Criterion, criterion_group, criterion_main};
use krasm::parser;
use std::collections::HashMap;
use std::fs;
use std::hint::black_box;

fn load_test_module(test_name: &str) -> Vec<u8> {
    // Load test from JSON fixture
    let json_path = format!("tests/spec/{}.json", test_name);
    let json_content = fs::read_to_string(&json_path).unwrap_or_else(|_| panic!("Failed to read {}", json_path));

    let test_data: serde_json::Value = serde_json::from_str(&json_content).expect("Failed to parse JSON");

    // Try both formats - first check for bin section (first module wins)
    if let Some(bin) = test_data["bin"].as_object()
        && let Some((_, value)) = bin.iter().next()
        && let Some(binary) = value.as_str()
    {
        return base64::Engine::decode(&base64::engine::general_purpose::STANDARD, binary)
            .expect("Failed to decode base64");
    }

    // Otherwise check commands array
    if let Some(commands) = test_data["commands"].as_array() {
        for command in commands {
            if command["type"] == "module"
                && let Some(binary) = command["binary"].as_str()
            {
                return base64::Engine::decode(&base64::engine::general_purpose::STANDARD, binary)
                    .expect("Failed to decode base64");
            }
        }
    }

    panic!("No module found in test file");
}

fn count_memory_instructions(module: &parser::module::Module) -> usize {
    let mut count = 0;
    for code in &module.code.code {
        for inst in code.body.flatten() {
            use parser::instruction::InstructionKind::*;
            match &inst.kind {
                I32Load { .. }
                | I64Load { .. }
                | F32Load { .. }
                | F64Load { .. }
                | I32Store { .. }
                | I64Store { .. }
                | F32Store { .. }
                | F64Store { .. }
                | I32Load8S { .. }
                | I32Load8U { .. }
                | I32Load16S { .. }
                | I32Load16U { .. }
                | I64Load8S { .. }
                | I64Load8U { .. }
                | I64Load16S { .. }
                | I64Load16U { .. }
                | I64Load32S { .. }
                | I64Load32U { .. }
                | I32Store8 { .. }
                | I32Store16 { .. }
                | I64Store8 { .. }
                | I64Store16 { .. }
                | I64Store32 { .. }
                | MemorySize
                | MemoryGrow
                | MemoryFill
                | MemoryCopy
                | MemoryInit { .. } => {
                    count += 1;
                }
                _ => {}
            }
        }
    }
    count
}

fn count_call_instructions(module: &parser::module::Module) -> usize {
    let mut count = 0;
    for code in &module.code.code {
        for inst in code.body.flatten() {
            use parser::instruction::InstructionKind::*;
            match &inst.kind {
                Call { .. } | CallIndirect { .. } => count += 1,
                _ => {}
            }
        }
    }
    count
}

fn benchmark_validation(c: &mut Criterion) {
    // Test with different module characteristics
    let test_cases = vec![
        ("memory_intensive", "memory_copy"), // Lots of memory operations
        ("call_intensive", "call_indirect"), // Lots of function calls
        ("mixed", "float_exprs"),            // Mix of operations
    ];

    // First, analyse what we're testing
    println!("\nModule analysis:");
    for (name, test_name) in &test_cases {
        let wasm_bytes = load_test_module(test_name);
        let mut reader = parser::reader::Reader::new(wasm_bytes.clone());
        let module_registry = HashMap::new();
        let module = parser::parse(&module_registry, test_name, &mut reader).unwrap();

        let import_count = module.imports.imports.len();
        let func_count = module.functions.functions.len();
        let memory_ops = count_memory_instructions(&module);
        let call_ops = count_call_instructions(&module);

        println!(
            "{}: {} imports, {} functions, {} memory ops, {} calls",
            name, import_count, func_count, memory_ops, call_ops
        );
    }

    // Benchmark full parsing (includes validation)
    let mut group = c.benchmark_group("full_parse_validate");
    for (name, test_name) in &test_cases {
        let wasm_bytes = load_test_module(test_name);
        let module_size = wasm_bytes.len();

        group.bench_function(format!("{}_{}kb", name, module_size / 1024), |b| {
            b.iter(|| {
                let mut reader = parser::reader::Reader::new(wasm_bytes.clone());
                let module_registry = HashMap::new();
                let module = parser::parse(&module_registry, test_name, &mut reader).expect("Failed to parse module");
                black_box(module);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_validation);
criterion_main!(benches);
