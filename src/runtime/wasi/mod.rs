//! WASI (WebAssembly System Interface) preview1 implementation
//!
//! This module provides WASI preview1 support for stdin, stdout, stderr, and argv
//! to enable CLI-style WebAssembly programs.
//!
//! # Specification
//!
//! This implementation follows the WASI preview1 specification:
//! <https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md>
//!
//! # Architecture
//!
//! WASI functions are closures that capture `Arc<WasiContext>`. Linear memory is
//! accessed through `Caller<T>` at each host function invocation — the context
//! itself does not hold a memory reference. The functions are generic over `T`
//! (the Store's user data type) but do not use it.
//!
//! # Usage
//!
//! ```ignore
//! use krasm::runtime::wasi::{WasiContext, create_wasi_imports};
//! use std::sync::Arc;
//!
//! let ctx = Arc::new(WasiContext::builder()
//!     .args(["prog", "arg1"])
//!     .stdin(Box::new(std::io::stdin()))
//!     .stdout(Box::new(std::io::stdout()))
//!     .stderr(Box::new(std::io::stderr()))
//!     .build());
//!
//! let imports = create_wasi_imports(&mut store, ctx.clone());
//! let instance_id = store.create_instance(module, Some(&imports))?;
//! store.invoke_export(instance_id, "_start", vec![], None)?;
//! ```

pub mod assemblyscript;
mod clock;
pub mod context;
mod dir;
mod fd;
mod path;
mod process;
pub mod types;

pub use assemblyscript::add_assemblyscript_imports;
pub use context::{DEFAULT_MAX_FDS, DEFAULT_MAX_IOVECS, FileDescriptor, WasiContext, WasiContextBuilder};
pub use types::WasiErrno;

use crate::parser::ValidatedModule;
use crate::runtime::store::{Caller, Store};
use crate::runtime::{ImportObject, Memory, RuntimeError};
use std::sync::Arc;

/// Create a WASI instance with all necessary setup.
///
/// This is a convenience function that:
/// 1. Creates WASI imports
/// 2. Optionally adds AssemblyScript imports
/// 3. Creates the module instance
///
/// # Arguments
///
/// * `store` - The WebAssembly store
/// * `module` - The validated WebAssembly module
/// * `ctx` - The WASI context (wrapped in Arc for sharing)
/// * `include_assemblyscript` - Whether to include AssemblyScript's env.abort
///
/// # Returns
///
/// The instance ID on success, or a RuntimeError on failure.
pub fn create_wasi_instance<T: 'static>(
    store: &mut Store<T>,
    module: &ValidatedModule,
    ctx: Arc<WasiContext>,
    include_assemblyscript: bool,
) -> Result<usize, RuntimeError> {
    let mut imports = create_wasi_imports(store, ctx.clone());

    if include_assemblyscript {
        add_assemblyscript_imports(store, &mut imports, ctx.clone());
    }

    let instance_id = store.create_instance(module, Some(&imports))?;

    Ok(instance_id)
}

/// Helper: extract memory from Caller, returning a RuntimeError if absent
fn require_memory<'a, T>(caller: &'a mut Caller<'_, T>) -> Result<&'a mut Memory, RuntimeError> {
    caller
        .memory_mut()
        .ok_or_else(|| RuntimeError::MemoryError("wasi: no linear memory available".to_string()))
}

/// Create WASI import functions and register them in the store
///
/// Returns an ImportObject with all WASI functions registered.
pub fn create_wasi_imports<T: 'static>(store: &mut Store<T>, ctx: Arc<WasiContext>) -> ImportObject {
    let mut imports = ImportObject::new();

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>,
              fd: i32,
              iovs_ptr: i32,
              iovs_len: i32,
              nwritten_ptr: i32|
              -> Result<i32, RuntimeError> {
            fd::wasi_fd_write(&c, caller, fd, iovs_ptr, iovs_len, nwritten_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "fd_write", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>,
              fd: i32,
              iovs_ptr: i32,
              iovs_len: i32,
              nread_ptr: i32|
              -> Result<i32, RuntimeError> { fd::wasi_fd_read(&c, caller, fd, iovs_ptr, iovs_len, nread_ptr) },
    );
    imports.add_function("wasi_snapshot_preview1", "fd_read", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(move |_caller: &mut Caller<'_, T>, fd: i32| -> i32 { fd::wasi_fd_close(&c, fd) });
    imports.add_function("wasi_snapshot_preview1", "fd_close", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, argc_ptr: i32, argv_buf_size_ptr: i32| -> Result<i32, RuntimeError> {
            process::wasi_args_sizes_get(&c, caller, argc_ptr, argv_buf_size_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "args_sizes_get", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, argv_ptr: i32, argv_buf_ptr: i32| -> Result<i32, RuntimeError> {
            process::wasi_args_get(&c, caller, argv_ptr, argv_buf_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "args_get", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, environc_ptr: i32, environ_buf_size_ptr: i32| -> Result<i32, RuntimeError> {
            process::wasi_environ_sizes_get(&c, caller, environc_ptr, environ_buf_size_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "environ_sizes_get", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, environ_ptr: i32, environ_buf_ptr: i32| -> Result<i32, RuntimeError> {
            process::wasi_environ_get(&c, caller, environ_ptr, environ_buf_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "environ_get", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, fd: i32, buf_ptr: i32| -> Result<i32, RuntimeError> {
            dir::wasi_fd_prestat_get(&c, caller, fd, buf_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "fd_prestat_get", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, fd: i32, path_ptr: i32, path_len: i32| -> Result<i32, RuntimeError> {
            dir::wasi_fd_prestat_dir_name(&c, caller, fd, path_ptr, path_len)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "fd_prestat_dir_name", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>,
              dir_fd: i32,
              dirflags: i32,
              path_ptr: i32,
              path_len: i32,
              oflags: i32,
              rights_base: i64,
              rights_inheriting: i64,
              fdflags: i32,
              opened_fd_ptr: i32|
              -> Result<i32, RuntimeError> {
            path::wasi_path_open(
                &c,
                caller,
                dir_fd,
                dirflags,
                path_ptr,
                path_len,
                oflags,
                rights_base,
                rights_inheriting,
                fdflags,
                opened_fd_ptr,
            )
        },
    );
    imports.add_function("wasi_snapshot_preview1", "path_open", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, fd: i32, buf_ptr: i32| -> Result<i32, RuntimeError> {
            fd::wasi_fd_fdstat_get(&c, caller, fd, buf_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "fd_fdstat_get", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>,
              fd: i32,
              offset: i64,
              whence: i32,
              newoffset_ptr: i32|
              -> Result<i32, RuntimeError> { fd::wasi_fd_seek(&c, caller, fd, offset, whence, newoffset_ptr) },
    );
    imports.add_function("wasi_snapshot_preview1", "fd_seek", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, fd: i32, offset_ptr: i32| -> Result<i32, RuntimeError> {
            fd::wasi_fd_tell(&c, caller, fd, offset_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "fd_tell", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, fd: i32, buf_ptr: i32| -> Result<i32, RuntimeError> {
            fd::wasi_fd_filestat_get(&c, caller, fd, buf_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "fd_filestat_get", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(move |_caller: &mut Caller<'_, T>, fd: i32, flags: i32| -> i32 {
        fd::wasi_fd_fdstat_set_flags(&c, fd, flags)
    });
    imports.add_function("wasi_snapshot_preview1", "fd_fdstat_set_flags", addr);

    let c = ctx.clone();
    #[allow(clippy::too_many_arguments)]
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>,
              dir_fd: i32,
              flags: i32,
              path_ptr: i32,
              path_len: i32,
              buf_ptr: i32|
              -> Result<i32, RuntimeError> {
            path::wasi_path_filestat_get(&c, caller, dir_fd, flags, path_ptr, path_len, buf_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "path_filestat_get", addr);

    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, clockid: i32, precision: i64, time_ptr: i32| -> Result<i32, RuntimeError> {
            clock::wasi_clock_time_get(&c, caller, clockid, precision, time_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "clock_time_get", addr);

    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, clockid: i32, resolution_ptr: i32| -> Result<i32, RuntimeError> {
            clock::wasi_clock_res_get(caller, clockid, resolution_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "clock_res_get", addr);

    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, buf_ptr: i32, buf_len: i32| -> Result<i32, RuntimeError> {
            clock::wasi_random_get(caller, buf_ptr, buf_len)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "random_get", addr);

    let addr = store.wrap_with_caller(move |_caller: &mut Caller<'_, T>| -> i32 { clock::wasi_sched_yield() });
    imports.add_function("wasi_snapshot_preview1", "sched_yield", addr);

    // fd_readdir
    let c = ctx.clone();
    #[allow(clippy::too_many_arguments)]
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>,
              fd: i32,
              buf_ptr: i32,
              buf_len: i32,
              cookie: i64,
              bufused_ptr: i32|
              -> Result<i32, RuntimeError> {
            dir::wasi_fd_readdir(&c, caller, fd, buf_ptr, buf_len, cookie, bufused_ptr)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "fd_readdir", addr);

    // path_create_directory
    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, fd: i32, path_ptr: i32, path_len: i32| -> Result<i32, RuntimeError> {
            path::wasi_path_create_directory(&c, caller, fd, path_ptr, path_len)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "path_create_directory", addr);

    // path_remove_directory
    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, fd: i32, path_ptr: i32, path_len: i32| -> Result<i32, RuntimeError> {
            path::wasi_path_remove_directory(&c, caller, fd, path_ptr, path_len)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "path_remove_directory", addr);

    // path_unlink_file
    let c = ctx.clone();
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>, fd: i32, path_ptr: i32, path_len: i32| -> Result<i32, RuntimeError> {
            path::wasi_path_unlink_file(&c, caller, fd, path_ptr, path_len)
        },
    );
    imports.add_function("wasi_snapshot_preview1", "path_unlink_file", addr);

    // path_rename
    let c = ctx.clone();
    #[allow(clippy::too_many_arguments)]
    let addr = store.wrap_with_caller(
        move |caller: &mut Caller<'_, T>,
              old_fd: i32,
              old_path_ptr: i32,
              old_path_len: i32,
              new_fd: i32,
              new_path_ptr: i32,
              new_path_len: i32|
              -> Result<i32, RuntimeError> {
            path::wasi_path_rename(
                &c,
                caller,
                old_fd,
                old_path_ptr,
                old_path_len,
                new_fd,
                new_path_ptr,
                new_path_len,
            )
        },
    );
    imports.add_function("wasi_snapshot_preview1", "path_rename", addr);

    // NOSYS stubs — all remaining WASI preview1 functions return ENOSYS.
    // The macro generates a typed closure matching each function's wasm signature.
    macro_rules! nosys {
        ($name:expr, $($p:ty),*) => {{
            let nosys = WasiErrno::NoSys.as_i32();
            #[allow(clippy::too_many_arguments)]
            let addr = store.wrap_with_caller(
                move |_c: &mut Caller<'_, T>, $(_: $p),*| -> i32 { nosys },
            );
            imports.add_function("wasi_snapshot_preview1", $name, addr);
        }};
    }

    nosys!("fd_advise", i32, i64, i64, i32);
    nosys!("fd_allocate", i32, i64, i64);
    nosys!("fd_datasync", i32);
    nosys!("fd_sync", i32);
    nosys!("fd_renumber", i32, i32);
    nosys!("fd_pread", i32, i32, i32, i64, i32);
    nosys!("fd_pwrite", i32, i32, i32, i64, i32);
    nosys!("fd_filestat_set_size", i32, i64);
    nosys!("fd_filestat_set_times", i32, i64, i64, i32);
    nosys!("fd_fdstat_set_rights", i32, i64, i64);
    nosys!("path_filestat_set_times", i32, i32, i32, i32, i64, i64, i32);
    nosys!("path_link", i32, i32, i32, i32, i32, i32, i32);
    nosys!("path_readlink", i32, i32, i32, i32, i32, i32);
    nosys!("path_symlink", i32, i32, i32, i32, i32);
    nosys!("poll_oneoff", i32, i32, i32, i32);
    nosys!("proc_raise", i32);
    nosys!("sock_accept", i32, i32, i32);
    nosys!("sock_recv", i32, i32, i32, i32, i32, i32);
    nosys!("sock_send", i32, i32, i32, i32, i32);
    nosys!("sock_shutdown", i32, i32);

    // proc_exit (must be last — consumes ctx)
    let c = ctx;
    let addr = store.wrap_with_caller(
        move |_caller: &mut Caller<'_, T>, rval: i32| -> Result<(), RuntimeError> { process::wasi_proc_exit(&c, rval) },
    );
    imports.add_function("wasi_snapshot_preview1", "proc_exit", addr);

    imports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_wasi_imports() {
        let mut store: Store<()> = Store::new();
        let ctx = Arc::new(WasiContext::builder().build());

        let imports = create_wasi_imports(&mut store, ctx);

        assert!(imports.get_function("wasi_snapshot_preview1", "fd_write").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_read").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_close").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "args_sizes_get").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "args_get").is_ok());
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "environ_sizes_get")
                .is_ok()
        );
        assert!(imports.get_function("wasi_snapshot_preview1", "environ_get").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_prestat_get").is_ok());
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "fd_prestat_dir_name")
                .is_ok()
        );
        assert!(imports.get_function("wasi_snapshot_preview1", "path_open").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_fdstat_get").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_seek").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_tell").is_ok());
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "fd_filestat_get")
                .is_ok()
        );
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "fd_fdstat_set_flags")
                .is_ok()
        );
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "path_filestat_get")
                .is_ok()
        );
        assert!(imports.get_function("wasi_snapshot_preview1", "clock_time_get").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "clock_res_get").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "random_get").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "sched_yield").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_readdir").is_ok());
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "path_create_directory")
                .is_ok()
        );
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "path_remove_directory")
                .is_ok()
        );
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "path_unlink_file")
                .is_ok()
        );
        assert!(imports.get_function("wasi_snapshot_preview1", "path_rename").is_ok());
        // NOSYS stubs
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_advise").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_allocate").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_datasync").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_sync").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_renumber").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_pread").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "fd_pwrite").is_ok());
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "fd_filestat_set_size")
                .is_ok()
        );
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "fd_filestat_set_times")
                .is_ok()
        );
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "fd_fdstat_set_rights")
                .is_ok()
        );
        assert!(
            imports
                .get_function("wasi_snapshot_preview1", "path_filestat_set_times")
                .is_ok()
        );
        assert!(imports.get_function("wasi_snapshot_preview1", "path_link").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "path_readlink").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "path_symlink").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "poll_oneoff").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "proc_raise").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "sock_accept").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "sock_recv").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "sock_send").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "sock_shutdown").is_ok());
        assert!(imports.get_function("wasi_snapshot_preview1", "proc_exit").is_ok());

        // env.abort should NOT be included in pure WASI imports
        assert!(imports.get_function("env", "abort").is_err());
    }

    #[test]
    fn test_create_wasi_imports_with_assemblyscript() {
        let mut store: Store<()> = Store::new();
        let ctx = Arc::new(WasiContext::builder().build());

        let mut imports = create_wasi_imports(&mut store, ctx.clone());
        add_assemblyscript_imports(&mut store, &mut imports, ctx);

        assert!(imports.get_function("env", "abort").is_ok());
    }

    #[test]
    fn test_args_sizes_get() {
        let ctx = WasiContext::builder().args(["prog", "hello", "world"]).build();

        let mut memory = Memory::new(1, None).unwrap();
        let mut data = ();
        let mut caller = Caller::for_test(Some(&mut memory), &mut data);
        let result = process::wasi_args_sizes_get(&ctx, &mut caller, 0, 4).unwrap();

        assert_eq!(result, 0); // Success
        assert_eq!(memory.read_u32(0).unwrap(), 3);
        // argv_buf_size = 5 + 6 + 6 = 17 ("prog\0" + "hello\0" + "world\0")
        assert_eq!(memory.read_u32(4).unwrap(), 17);
    }

    #[test]
    fn test_args_get() {
        let ctx = WasiContext::builder().args(["prog", "hello"]).build();

        let mut memory = Memory::new(1, None).unwrap();
        let mut data = ();
        let mut caller = Caller::for_test(Some(&mut memory), &mut data);
        let result = process::wasi_args_get(&ctx, &mut caller, 0, 100).unwrap();

        assert_eq!(result, 0); // Success
        assert_eq!(memory.read_u32(0).unwrap(), 100);
        assert_eq!(memory.read_u32(4).unwrap(), 105);

        let prog = memory.read_bytes(100, 5).unwrap();
        assert_eq!(prog, b"prog\0");
        let hello = memory.read_bytes(105, 6).unwrap();
        assert_eq!(hello, b"hello\0");
    }

    #[test]
    fn test_proc_exit() {
        let ctx = WasiContext::builder().build();

        let result = process::wasi_proc_exit(&ctx, 42);

        assert!(result.is_err());
        assert_eq!(ctx.exit_code(), Some(42));
    }

    #[test]
    fn test_wasi_function_without_memory() {
        let ctx = WasiContext::builder().args(["prog"]).build();

        let mut data = ();
        let mut caller = Caller::for_test(None, &mut data);
        let result = process::wasi_args_sizes_get(&ctx, &mut caller, 0, 4);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("no linear memory"), "Expected memory error, got: {err}");
    }
}
