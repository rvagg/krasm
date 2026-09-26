//! Execution benchmarks for the WebAssembly interpreter.
//!
//! These benchmarks measure instruction dispatch, memory operations,
//! and overall execution throughput.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use krasm::{EngineKind, Module, Store, Value};
use std::hint::black_box;
use std::sync::Arc;

/// Load and parse a WAT module from benches/modules/
fn load_module(name: &str) -> Module {
    let wat_path = format!("benches/modules/{}.wat", name);
    let wat_source = std::fs::read_to_string(&wat_path).unwrap_or_else(|_| panic!("Failed to read {}", wat_path));
    krasm::wat::parse(&wat_source).unwrap_or_else(|e| panic!("Failed to parse WAT {}: {}", name, e))
}

/// Create a store and instantiate a module
fn instantiate(module: Arc<Module>, engine: EngineKind) -> (Store, usize) {
    let mut store = Store::new();
    store.set_engine(engine);
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
fn verify_modules(engine: EngineKind) {
    // noop_loop: run(n) should return n
    {
        let module = Arc::new(load_module("noop_loop"));
        let (mut store, instance_id) = instantiate(Arc::clone(&module), engine);
        let result = execute(&mut store, instance_id, "run", vec![Value::I32(1000)]).unwrap();
        assert_eq!(result, vec![Value::I32(1000)], "noop_loop(1000) should be 1000");
    }

    // fib_iterative: verify known values
    {
        let module = Arc::new(load_module("fib_iterative"));
        let (mut store, instance_id) = instantiate(Arc::clone(&module), engine);

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
        let (mut store, instance_id) = instantiate(Arc::clone(&module), engine);

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
        let (mut store, instance_id) = instantiate(Arc::clone(&module), engine);

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
        let (mut store, instance_id) = instantiate(module, engine);

        let cases = [(10, 4), (100, 25), (1000, 168), (10000, 1229)];
        for (limit, expected) in cases {
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

    println!("{engine:?}: all module correctness checks passed.");
}

fn bench_noop_loop(c: &mut Criterion, engine: EngineKind) {
    let module = Arc::new(load_module("noop_loop"));
    let mut group = c.benchmark_group(format!("dispatch/{engine:?}"));
    for iterations in [1_000, 10_000, 100_000, 1_000_000] {
        group.bench_with_input(BenchmarkId::new("noop_loop", iterations), &iterations, |b, &n| {
            let (mut store, instance_id) = instantiate(Arc::clone(&module), engine);
            b.iter(|| {
                let result = execute(&mut store, instance_id, "run", vec![Value::I32(n)]).unwrap();
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_fib_iterative(c: &mut Criterion, engine: EngineKind) {
    let module = Arc::new(load_module("fib_iterative"));
    let mut group = c.benchmark_group(format!("compute/{engine:?}"));
    for n in [10, 20, 30, 40, 46] {
        group.bench_with_input(BenchmarkId::new("fib_iterative", n), &n, |b, &n| {
            let (mut store, instance_id) = instantiate(Arc::clone(&module), engine);
            b.iter(|| {
                let result = execute(&mut store, instance_id, "fib", vec![Value::I32(n)]).unwrap();
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_fib_recursive(c: &mut Criterion, engine: EngineKind) {
    let module = Arc::new(load_module("fib_recursive"));
    let mut group = c.benchmark_group(format!("call_overhead/{engine:?}"));
    for n in [10, 15, 20, 25] {
        group.bench_with_input(BenchmarkId::new("fib_recursive", n), &n, |b, &n| {
            let (mut store, instance_id) = instantiate(Arc::clone(&module), engine);
            b.iter(|| {
                let result = execute(&mut store, instance_id, "fib", vec![Value::I32(n)]).unwrap();
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_memcpy(c: &mut Criterion, engine: EngineKind) {
    let module = Arc::new(load_module("memcpy"));
    let mut group = c.benchmark_group(format!("memory/{engine:?}"));
    for size in [100, 1000, 4000] {
        group.bench_with_input(BenchmarkId::new("memcpy", size), &size, |b, &size| {
            let (mut store, instance_id) = instantiate(Arc::clone(&module), engine);
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

fn bench_primes(c: &mut Criterion, engine: EngineKind) {
    let module = Arc::new(load_module("primes"));
    let mut group = c.benchmark_group(format!("mixed/{engine:?}"));
    for limit in [1000, 10000, 50000] {
        group.bench_with_input(BenchmarkId::new("primes", limit), &limit, |b, &limit| {
            // count_primes clears its sieve on every invocation.
            let (mut store, instance_id) = instantiate(Arc::clone(&module), engine);
            b.iter(|| {
                let result = execute(&mut store, instance_id, "count_primes", vec![Value::I32(limit)]).unwrap();
                black_box(result)
            });
        });
    }
    group.finish();
}

fn bench_instantiation(c: &mut Criterion, engine: EngineKind) {
    let mut group = c.benchmark_group(format!("instantiation/{engine:?}"));
    for name in ["noop_loop", "fib_iterative", "fib_recursive", "memcpy", "primes"] {
        let module = Arc::new(load_module(name));
        group.bench_function(name, |b| b.iter(|| black_box(instantiate(Arc::clone(&module), engine))));
    }
    group.finish();
}

fn verify_and_bench(c: &mut Criterion) {
    for engine in [EngineKind::Structured, EngineKind::Flat] {
        verify_modules(engine);
        bench_noop_loop(c, engine);
        bench_fib_iterative(c, engine);
        bench_fib_recursive(c, engine);
        bench_memcpy(c, engine);
        bench_primes(c, engine);
        bench_instantiation(c, engine);
    }
}

criterion_group!(benches, verify_and_bench);
criterion_main!(benches);
