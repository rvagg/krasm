//! AssemblyScript runtime support
//!
//! AssemblyScript is a TypeScript-like language that compiles to WebAssembly.
//! Unlike pure WASI modules, AssemblyScript requires additional host functions
//! from the "env" module that are not part of the WASI specification.
//!
//! This module provides these AssemblyScript-specific imports. They are kept
//! separate from WASI functions to clearly distinguish between standard WASI
//! and AssemblyScript's runtime requirements.
//!
//! # Background
//!
//! When running in a browser or Node.js, AssemblyScript modules are loaded with
//! JavaScript glue code that provides these functions. When running in a WASI
//! runtime like krasm, we must provide them ourselves.
//!
//! # Usage
//!
//! ```ignore
//! use krasm::runtime::wasi::{create_wasi_imports, add_assemblyscript_imports};
//!
//! let mut imports = create_wasi_imports(&mut store, ctx.clone());
//! add_assemblyscript_imports(&mut store, &mut imports, ctx.clone());
//! ```

use super::WasiContext;
use crate::runtime::store::{Caller, Store};
use crate::runtime::{ImportObject, Memory, RuntimeError};
use std::sync::Arc;

/// Add AssemblyScript-specific imports to an ImportObject.
///
/// This registers the `env.abort` function required by AssemblyScript's runtime.
/// Call this after `create_wasi_imports` if your module is compiled from AssemblyScript.
pub fn add_assemblyscript_imports<T: 'static>(store: &mut Store<T>, imports: &mut ImportObject, ctx: Arc<WasiContext>) {
    let abort_addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>,
              message_ptr: i32,
              filename_ptr: i32,
              line: i32,
              column: i32|
              -> Result<(), RuntimeError> { env_abort(&ctx, caller, message_ptr, filename_ptr, line, column) },
    );
    imports.add_function("env", "abort", abort_addr);
}

/// Read an AssemblyScript string from memory.
///
/// AssemblyScript strings are stored as:
/// - ptr-4: length in bytes (u32)
/// - ptr: UTF-16LE encoded string data
///
/// Returns None if the pointer is 0 (null) or if reading fails.
fn read_as_string(memory: &Memory, ptr: u32) -> Option<String> {
    if ptr == 0 {
        return None;
    }

    // Read length from ptr-4
    let len_bytes = memory.read_u32(ptr.wrapping_sub(4)).ok()?;

    // Sanity check: length should be reasonable and even (UTF-16)
    if len_bytes == 0 || len_bytes > 10000 || len_bytes % 2 != 0 {
        return None;
    }

    // Read UTF-16LE bytes
    let bytes = memory.read_bytes(ptr, len_bytes as usize).ok()?;

    // Convert from UTF-16LE to String
    let utf16: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    String::from_utf16(&utf16).ok()
}

/// AssemblyScript abort function.
///
/// Called when an assertion fails or when `abort()` is called explicitly in
/// AssemblyScript code. The message and filename pointers reference UTF-16
/// encoded strings in linear memory (AssemblyScript's native string format).
fn env_abort<T>(
    ctx: &WasiContext,
    caller: &mut Caller<'_, T>,
    message_ptr: i32,
    filename_ptr: i32,
    line: i32,
    column: i32,
) -> Result<(), RuntimeError> {
    // Try to read the message and filename from memory (best-effort)
    let (message, filename) = match caller.memory() {
        Some(memory) => (
            read_as_string(memory, message_ptr as u32),
            read_as_string(memory, filename_ptr as u32),
        ),
        None => (None, None),
    };

    // Set exit code to 1 (abnormal termination)
    ctx.set_exit_code(1);

    // Build error message with available information
    let error_msg = match (message, filename) {
        (Some(msg), Some(file)) => {
            format!("abort: {} at {}:{}:{}", msg, file, line, column)
        }
        (Some(msg), None) => {
            format!("abort: {} at line {}, column {}", msg, line, column)
        }
        (None, Some(file)) => {
            format!("abort at {}:{}:{}", file, line, column)
        }
        (None, None) => {
            format!("abort at line {}, column {}", line, column)
        }
    };

    Err(RuntimeError::Trap(error_msg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Memory;

    #[test]
    fn test_add_assemblyscript_imports() {
        let mut store = crate::runtime::store::Store::new();
        let ctx = Arc::new(WasiContext::builder().build());
        let mut imports = ImportObject::new();

        add_assemblyscript_imports(&mut store, &mut imports, ctx);

        assert!(imports.get_function("env", "abort").is_ok());
    }

    #[test]
    fn test_env_abort_sets_exit_code() {
        let ctx = WasiContext::builder().build();

        let mut memory = Memory::new(1, None).unwrap();
        let mut data = ();
        let mut caller = Caller::for_test(Some(&mut memory), &mut data);
        let result = env_abort(&ctx, &mut caller, 0, 0, 42, 10);

        assert!(result.is_err());
        assert_eq!(ctx.exit_code(), Some(1));

        if let Err(RuntimeError::Trap(msg)) = result {
            assert!(msg.contains("line 42"));
            assert!(msg.contains("column 10"));
        } else {
            panic!("Expected Trap error");
        }
    }

    #[test]
    fn test_env_abort_reads_strings() {
        let ctx = WasiContext::builder().build();

        let mut memory = Memory::new(1, None).unwrap();

        // Write an AssemblyScript string "Error!" at address 100
        // Format: length at ptr-4, UTF-16LE data at ptr
        // "Error!" = 6 chars = 12 bytes in UTF-16
        let message = "Error!";
        let utf16: Vec<u16> = message.encode_utf16().collect();
        let len_bytes = (utf16.len() * 2) as u32;

        // Write length at address 96 (100 - 4)
        memory.write_u32(96, len_bytes).unwrap();

        // Write UTF-16LE data at address 100
        let mut utf16_bytes = Vec::new();
        for ch in &utf16 {
            utf16_bytes.extend_from_slice(&ch.to_le_bytes());
        }
        memory.write_bytes(100, &utf16_bytes).unwrap();

        // Write filename "test.ts" at address 200
        let filename = "test.ts";
        let utf16_fn: Vec<u16> = filename.encode_utf16().collect();
        let len_fn = (utf16_fn.len() * 2) as u32;

        memory.write_u32(196, len_fn).unwrap();
        let mut utf16_fn_bytes = Vec::new();
        for ch in &utf16_fn {
            utf16_fn_bytes.extend_from_slice(&ch.to_le_bytes());
        }
        memory.write_bytes(200, &utf16_fn_bytes).unwrap();

        let mut data = ();
        let mut caller = Caller::for_test(Some(&mut memory), &mut data);
        let result = env_abort(&ctx, &mut caller, 100, 200, 42, 10);

        assert!(result.is_err());
        if let Err(RuntimeError::Trap(msg)) = result {
            assert!(msg.contains("Error!"), "Expected 'Error!' in: {}", msg);
            assert!(msg.contains("test.ts"), "Expected 'test.ts' in: {}", msg);
            assert!(msg.contains("42"), "Expected line 42 in: {}", msg);
            assert!(msg.contains("10"), "Expected column 10 in: {}", msg);
        } else {
            panic!("Expected Trap error");
        }
    }

    #[test]
    fn test_read_as_string() {
        let mut memory = Memory::new(1, None).unwrap();

        // Null pointer returns None
        assert_eq!(read_as_string(&memory, 0), None);

        // Write "Hello" as AssemblyScript string at address 104 (length at 100)
        let text = "Hello";
        let utf16: Vec<u16> = text.encode_utf16().collect();
        let len_bytes = (utf16.len() * 2) as u32;

        memory.write_u32(100, len_bytes).unwrap();
        let mut utf16_bytes = Vec::new();
        for ch in &utf16 {
            utf16_bytes.extend_from_slice(&ch.to_le_bytes());
        }
        memory.write_bytes(104, &utf16_bytes).unwrap();

        assert_eq!(read_as_string(&memory, 104), Some("Hello".to_string()));
    }
}
