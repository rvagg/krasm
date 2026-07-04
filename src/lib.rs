//! WebAssembly 2.0 runtime, parser, and toolkit.
//!
//! krasm is a complete WebAssembly implementation: binary parser, text format
//! (WAT) parser, interpreter, binary encoder, and WASI preview1 host. It
//! targets the W3C Working Group 2.0 specification including bulk memory
//! operations, reference types, and the full SIMD instruction set.
//!
//! # Quick start
//!
//! Parse a WAT module, instantiate it, and call an exported function:
//!
//! ```
//! use krasm::{Store, Value};
//! use std::sync::Arc;
//!
//! let module = krasm::wat::parse(r#"
//!     (module
//!         (func (export "add") (param i32 i32) (result i32)
//!             local.get 0
//!             local.get 1
//!             i32.add))
//! "#).unwrap();
//!
//! let mut store = Store::new();
//! let id = store.create_instance(Arc::new(module), None).unwrap();
//! let result = store.invoke_export(id, "add", vec![Value::I32(2), Value::I32(3)], None).unwrap();
//! assert_eq!(result, vec![Value::I32(5)]);
//! ```
//!
//! # Host functions
//!
//! Register typed Rust closures as WebAssembly imports:
//!
//! ```
//! use krasm::{Store, Value, ImportObject};
//! use std::sync::Arc;
//!
//! let mut store = Store::new();
//! let double = store.wrap(|x: i32| -> i32 { x * 2 });
//!
//! let mut imports = ImportObject::new();
//! imports.add_function("env", "double", double);
//!
//! let module = krasm::wat::parse(r#"
//!     (module
//!         (import "env" "double" (func $double (param i32) (result i32)))
//!         (func (export "quad") (param i32) (result i32)
//!             local.get 0
//!             call $double
//!             call $double))
//! "#).unwrap();
//!
//! let id = store.create_instance(Arc::new(module), Some(&imports)).unwrap();
//! let result = store.invoke_export(id, "quad", vec![Value::I32(5)], None).unwrap();
//! assert_eq!(result, vec![Value::I32(20)]);
//! ```
//!
//! Host functions can also access the calling instance's memory and
//! embedder-defined state via [`Caller`]:
//!
//! ```
//! use krasm::{Store, Caller};
//!
//! let mut store = Store::with_data(0u32);
//! let _addr = store.wrap_with_caller(|caller: &mut Caller<'_, u32>, x: i32| {
//!     *caller.data_mut() += x as u32;
//! });
//! ```
//!
//! # Submodules
//!
//! - [`parser`] — Binary format decoder (`.wasm` bytes to [`Module`])
//! - [`wat`] — Text format parser (WAT source to [`Module`])
//! - [`encoder`] — Binary encoder ([`Module`] to `.wasm` bytes)
//! - [`wasi`] — WASI preview1 host (stdin/stdout, args, environ, proc_exit)
//! - [`wast`] — Spec test file parser (for `.wast` compliance testing)

pub mod encoder;
pub mod parser;
pub mod runtime;
pub mod wast;
pub mod wat;

// ---------------------------------------------------------------------------
// Top-level re-exports
// ---------------------------------------------------------------------------

// Core runtime types
pub use runtime::store::{Caller, FuncAddr, GlobalAddr, MemoryAddr, Store, TableAddr};
pub use runtime::value::Value;
pub use runtime::{EngineKind, ImportObject, Memory, RuntimeError, Table};

// Typed host function traits
pub use runtime::host::{WasmResult, WasmType};

// Parser types used in public API signatures
pub use parser::module::{FunctionType, Limits, Module, RefType, ValueType};

// Error types
pub use encoder::EncodeError;
pub use parser::instruction::DecodeError;
pub use wat::ParseError;

/// WASI preview1 support.
///
/// Provides the [`WasiContext`](wasi::WasiContext) builder and convenience
/// functions for instantiating WASI-enabled WebAssembly modules.
///
/// ```no_run
/// use krasm::{Store, Value};
/// use krasm::wasi::{WasiContext, create_wasi_instance};
/// use std::sync::Arc;
///
/// let module = krasm::wat::parse(r#"(module
///     (import "wasi_snapshot_preview1" "proc_exit" (func (param i32)))
///     (memory (export "memory") 1)
///     (func (export "_start"))
/// )"#).unwrap();
///
/// let ctx = Arc::new(WasiContext::builder()
///     .args(["prog", "arg1"])
///     .build());
///
/// let mut store = Store::new();
/// let instance_id = create_wasi_instance(&mut store, Arc::new(module), ctx, true).unwrap();
/// store.invoke_export(instance_id, "_start", vec![], None).unwrap();
/// ```
pub mod wasi {
    pub use crate::runtime::wasi::{
        FileDescriptor, WasiContext, WasiContextBuilder, add_assemblyscript_imports, create_wasi_imports,
        create_wasi_instance,
    };
}
