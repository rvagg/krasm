//! Execution benchmarks for the WebAssembly interpreter.
//!
//! These benchmarks measure instruction dispatch, memory operations,
//! and overall execution throughput.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use krasm::{Module, Store, Value};
use std::hint::black_box;
use std::sync::Arc;

/// Load and parse a WAT module from benches/modules/
fn load_module(name: &str) -> Module {
    let wat_path = format!("benches/modules/{}.wat", name);
    let wat_source = std::fs::read_to_string(&wat_path).expect(&format!("Failed to read {}", wat_path));
    krasm::wat::parse(&wat_source).expect(&format!("Failed to parse WAT: {}", name))
}

/// Create a store and instantiate a module
fn instantiate(module: Arc<Module>) -> (Store, usize) {
    let mut store = Store::new();
    let instance_id = store.create_instance(module, None).expect("Failed to instantiate");
    (store, instance_id)
}

/// Execute a function and return the result
fn execute(
    store: &mut Store,
    instance_id: usize,
    func: &str,
    args: Vec<Value>,
) -> Result<Vec<Value>, krasm::runtime::RuntimeError> {
    store.invoke_export(instance_id, func, args, None)
}

/// Verify module correctness before benchmarking
fn verify_modules() {
    // noop_loop: run(n) should return n
    {
        let module = Arc::new(load_module("noop_loop"));
        let (mut store, instance_id) = instantiate(Arc::clone(&module));
        let result = execute(&mut store, instance_id, "run", vec![Value::I32(1000)]).unwrap();
        assert_eq!(result, vec![Value::I32(1000)], "noop_loop(1000) should be 1000");
    }

    // fib_iterative: verify known values
    {
        let module = Arc::new(load_module("fib_iterative"));
        let (mut store, instance_id) = instantiate(Arc::clone(&module));

        let cases = [(0, 0), (1, 1), (10, 55), (20, 6765), (40, 102334155)];
        for (n, expected) in cases {
            let result = execute(&mut store, instance_id, "fib", vec![Value::I32(n)]).unwrap();
            assert_eq!(
                result,
                vec![Value::I32(expected)],
                "fib_iterative({}) should be {}",
                n,
                expected
            );
        }
    }

    // fib_recursive: verify known values
    {
        let module = Arc::new(load_module("fib_recursive"));
        let (mut store, instance_id) = instantiate(Arc::clone(&module));

        let cases = [(0, 0), (1, 1), (10, 55), (20, 6765)];
        for (n, expected) in cases {
            let result = execute(&mut store, instance_id, "fib", vec![Value::I32(n)]).unwrap();
            assert_eq!(
                result,
                vec![Value::I32(expected)],
                "fib_recursive({}) should be {}",
                n,
                expected
            );
        }
    }

    // memcpy: fill, copy, verify
    {
        let module = Arc::new(load_module("memcpy"));
        let (mut store, instance_id) = instantiate(Arc::clone(&module));

        // Fill source with pattern
        let result = execute(
            &mut store,
            instance_id,
            "fill",
            vec![Value::I32(0), Value::I32(1000), Value::I32(0x42)],
        )
        .unwrap();
        assert_eq!(result, vec![Value::I32(1000)], "fill should return 1000");

        // Copy to destination
        let result = execute(
            &mut store,
            instance_id,
            "copy",
            vec![Value::I32(0), Value::I32(4096), Value::I32(1000)],
        )
        .unwrap();
        assert_eq!(result, vec![Value::I32(1000)], "copy should return 1000");

        // Verify copy
        let result = execute(
            &mut store,
            instance_id,
            "verify",
            vec![Value::I32(0), Value::I32(4096), Value::I32(1000)],
        )
        .unwrap();
        assert_eq!(result, vec![Value::I32(1)], "verify should return 1 (success)");
    }

    // primes: verify known counts
    {
        let module = Arc::new(load_module("primes"));

        let cases = [(10, 4), (100, 25), (1000, 168), (10000, 1229)];
        for (limit, expected) in cases {
            // Need fresh instance since sieve modifies memory
            let (mut store, instance_id) = instantiate(Arc::clone(&module));
            let result = execute(&mut store, instance_id, "count_primes", vec![Value::I32(limit)]).unwrap();
            assert_eq!(
                result,
                vec![Value::I32(expected)],
                "count_primes({}) should be {}",
                limit,
                expected
            );
        }
    }

    println!("All module correctness checks passed.");
}

fn bench_noop_loop(c: &mut Criterion) {
    let module = Arc::new(load_module("noop_loop"));

    let mut group = c.benchmark_group("dispatch");
    for iterations in [1_000, 10_000, 100_000, 1_000_000] {
        group.bench_with_input(BenchmarkId::new("noop_loop", iterations), &iterations, |b, &n| {
            let (mut store, instance_id) = instantiate(Arc::clone(&module));
            b.iter(|| {
                let result = execute(&mut store, instance_id, "run", vec![Value::I32(n)]).unwrap();
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_fib_iterative(c: &mut Criterion) {
    let module = Arc::new(load_module("fib_iterative"));

    let mut group = c.benchmark_group("compute");
    for n in [10, 20, 30, 40, 46] {
        group.bench_with_input(BenchmarkId::new("fib_iterative", n), &n, |b, &n| {
            let (mut store, instance_id) = instantiate(Arc::clone(&module));
            b.iter(|| {
                let result = execute(&mut store, instance_id, "fib", vec![Value::I32(n)]).unwrap();
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_fib_recursive(c: &mut Criterion) {
    let module = Arc::new(load_module("fib_recursive"));

    let mut group = c.benchmark_group("call_overhead");
    // Note: n=30 makes ~2.7M calls, n=35 makes ~29M calls
    // Keep n small for reasonable benchmark times
    for n in [10, 15, 20, 25] {
        group.bench_with_input(BenchmarkId::new("fib_recursive", n), &n, |b, &n| {
            let (mut store, instance_id) = instantiate(Arc::clone(&module));
            b.iter(|| {
                let result = execute(&mut store, instance_id, "fib", vec![Value::I32(n)]).unwrap();
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_memcpy(c: &mut Criterion) {
    let module = Arc::new(load_module("memcpy"));

    let mut group = c.benchmark_group("memory");
    for size in [100, 1000, 4000] {
        group.bench_with_input(BenchmarkId::new("memcpy", size), &size, |b, &size| {
            let (mut store, instance_id) = instantiate(Arc::clone(&module));
            // Pre-fill source buffer
            execute(
                &mut store,
                instance_id,
                "fill",
                vec![Value::I32(0), Value::I32(size), Value::I32(0x42)],
            )
            .unwrap();
            b.iter(|| {
                let result = execute(
                    &mut store,
                    instance_id,
                    "copy",
                    vec![Value::I32(0), Value::I32(4096), Value::I32(size)],
                )
                .unwrap();
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_primes(c: &mut Criterion) {
    let module = Arc::new(load_module("primes"));

    let mut group = c.benchmark_group("mixed");
    for limit in [1000, 10000, 50000] {
        group.bench_with_input(BenchmarkId::new("primes", limit), &limit, |b, &limit| {
            b.iter(|| {
                // Fresh instance each time (sieve modifies memory)
                let (mut store, instance_id) = instantiate(Arc::clone(&module));
                let result = execute(&mut store, instance_id, "count_primes", vec![Value::I32(limit)]).unwrap();
                black_box(result)
            });
        });
    }
    group.finish();
}

// -- Flat bytecode executor benchmarks --

fn compile_func(module: &Module, func_index: usize) -> krasm::runtime::bytecode::CompiledFunction {
    let func = &module.code.code[func_index];
    let ftype_idx = module.functions.functions[func_index].ftype_index;
    let ftype = module.types.get(ftype_idx).unwrap();
    krasm::runtime::compiler::compile(&func.body, ftype.parameters.len() as u32)
}

fn bench_flat_noop_loop(c: &mut Criterion) {
    let module = load_module("noop_loop");
    let compiled = compile_func(&module, 0);

    let mut group = c.benchmark_group("flat_dispatch");
    for iterations in [1_000, 10_000, 100_000, 1_000_000] {
        group.bench_with_input(BenchmarkId::new("noop_loop", iterations), &iterations, |b, &n| {
            b.iter(|| {
                let result = krasm::runtime::flat_executor::execute_flat(&compiled, &[Value::I32(n)]).unwrap();
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_flat_fib_iterative(c: &mut Criterion) {
    let module = load_module("fib_iterative");
    let compiled = compile_func(&module, 0);

    let mut group = c.benchmark_group("flat_compute");
    for n in [10, 20, 30, 40, 46] {
        group.bench_with_input(BenchmarkId::new("fib_iterative", n), &n, |b, &n| {
            b.iter(|| {
                let result = krasm::runtime::flat_executor::execute_flat(&compiled, &[Value::I32(n)]).unwrap();
                black_box(result)
            });
        });
    }
    group.finish();
}

// Run verification before benchmarks
fn verify_and_bench(c: &mut Criterion) {
    verify_modules();
    bench_noop_loop(c);
    bench_fib_iterative(c);
    bench_fib_recursive(c);
    bench_memcpy(c);
    bench_primes(c);
    bench_flat_noop_loop(c);
    bench_flat_fib_iterative(c);
}

criterion_group!(benches, verify_and_bench);
criterion_main!(benches);
