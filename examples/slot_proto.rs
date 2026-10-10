//! Run one export on the flat engine or the slot prototype, for `perf stat`.
//!
//! ```text
//! cargo run --release --features slot-proto --example slot_proto -- slot benches/modules/fib_iterative.wat fib 100000000
//! ```

use krasm::runtime::slot_proto;
use krasm::{Store, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, engine, path, export, arg] = args.as_slice() else {
        eprintln!("usage: slot_proto <flat|slot> <file.wat> <export> <i32 arg>");
        std::process::exit(2);
    };
    let source = std::fs::read_to_string(path).expect("read module");
    let module = krasm::wat::parse(&source).expect("parse module");
    let n: i32 = arg.parse().expect("i32 argument");
    let result = match engine.as_str() {
        "slot" => slot_proto::invoke(&module, export, &[n as u32 as u64]).map(|r| r[0] as u32 as i32),
        "flat" => {
            let mut store = Store::new();
            let instance = store.create_instance(&module, None).expect("instantiate");
            store
                .invoke_export(instance, export, vec![Value::I32(n)], None)
                .map(|r| match r[0] {
                    Value::I32(v) => v,
                    ref other => panic!("unexpected result {other:?}"),
                })
        }
        other => panic!("unknown engine {other}"),
    };
    println!("{}", result.expect("run"));
}
