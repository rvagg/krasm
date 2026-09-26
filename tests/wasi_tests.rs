//! Integration tests for WASI functionality

mod common;

use common::{CapturedWriter, temp_dir};
use krasm::parser;
use krasm::parser::reader::Reader;
use krasm::wasi::{WasiContext, add_assemblyscript_imports, create_wasi_imports};
use krasm::{EngineKind, Module, Store};
use rstest::rstest;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

/// Helper to parse WAT source into a Module
fn parse_wat(wat: &str) -> Module {
    krasm::wat::parse(wat).expect("Failed to parse WAT")
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_hello_wasi(#[case] engine: EngineKind) {
    let wat = r#"
    ;; Smoke test for fd_write - prints "Hello from WASI!" to stdout
    (module
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (memory (export "memory") 1)

      ;; "Hello from WASI!\n" at offset 8 (17 bytes)
      (data (i32.const 8) "Hello from WASI!\n")

      ;; iovec at offset 0: {ptr: 8, len: 17}
      (data (i32.const 0) "\08\00\00\00\11\00\00\00")

      (func (export "_start")
        (drop (call $fd_write
          (i32.const 1)   ;; fd = stdout
          (i32.const 0)   ;; iovs ptr
          (i32.const 1)   ;; iovs len
          (i32.const 100) ;; nwritten ptr
        ))
      )
    )
    "#;
    let module = parse_wat(wat);

    // Create captured stdout
    let stdout_buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let stdout = CapturedWriter(stdout_buffer.clone());

    // Create WASI context
    let ctx = Arc::new(
        WasiContext::builder()
            .args(["hello.wasm"])
            .stdout(Box::new(stdout))
            .build(),
    );

    // Create store and imports
    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());

    // Create instance
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    // Execute _start
    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    // Check output
    let output = stdout_buffer.lock().unwrap();
    assert_eq!(&*output, b"Hello from WASI!\n");
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_fd_read_stdin(#[case] engine: EngineKind) {
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_read"
        (func $fd_read (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (memory (export "memory") 1)

      ;; Read buffer at offset 100 (64 bytes)
      ;; iovec at offset 0: {ptr: 100, len: 64}
      (data (i32.const 0) "\64\00\00\00\40\00\00\00")

      (func (export "_start")
        ;; Read from stdin
        (drop (call $fd_read
          (i32.const 0)   ;; fd = stdin
          (i32.const 0)   ;; iovs ptr
          (i32.const 1)   ;; iovs len
          (i32.const 200) ;; nread ptr
        ))

        ;; Update iovec to point to read data with actual length
        (i32.store (i32.const 4) (i32.load (i32.const 200)))

        ;; Write to stdout
        (drop (call $fd_write
          (i32.const 1)   ;; fd = stdout
          (i32.const 0)   ;; iovs ptr
          (i32.const 1)   ;; iovs len
          (i32.const 204) ;; nwritten ptr
        ))
      )
    )
    "#;

    let module = parse_wat(wat);

    // Create captured stdout and mock stdin
    let stdout_buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let stdout = CapturedWriter(stdout_buffer.clone());
    let stdin = Cursor::new(b"test input\n".to_vec());

    let ctx = Arc::new(
        WasiContext::builder()
            .stdin(Box::new(stdin))
            .stdout(Box::new(stdout))
            .build(),
    );

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    // Check that input was echoed to output
    let output = stdout_buffer.lock().unwrap();
    assert_eq!(&*output, b"test input\n");
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_args(#[case] engine: EngineKind) {
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "args_sizes_get"
        (func $args_sizes_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "args_get"
        (func $args_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (memory (export "memory") 1)

      (global $argc (mut i32) (i32.const 0))
      (global $argv_buf_size (mut i32) (i32.const 0))

      (func (export "_start")
        ;; Get arg sizes (store at 0, 4)
        (drop (call $args_sizes_get (i32.const 0) (i32.const 4)))
        (global.set $argc (i32.load (i32.const 0)))
        (global.set $argv_buf_size (i32.load (i32.const 4)))

        ;; Get args: argv at 100, buf at 200
        (drop (call $args_get (i32.const 100) (i32.const 200)))

        ;; Print first arg (program name)
        ;; Create iovec at 300: {ptr from argv[0], len till null}
        (i32.store (i32.const 300) (i32.load (i32.const 100))) ;; ptr
        (i32.store (i32.const 304) (i32.const 4)) ;; len = "prog" = 4 bytes

        (drop (call $fd_write
          (i32.const 1)
          (i32.const 300)
          (i32.const 1)
          (i32.const 400)
        ))
      )
    )
    "#;

    let module = parse_wat(wat);

    let stdout_buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let stdout = CapturedWriter(stdout_buffer.clone());

    let ctx = Arc::new(
        WasiContext::builder()
            .args(["prog", "arg1", "arg2"])
            .stdout(Box::new(stdout))
            .build(),
    );

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    // Check that "prog" was printed
    let output = stdout_buffer.lock().unwrap();
    assert_eq!(&*output, b"prog");
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_assemblyscript_hello(#[case] engine: EngineKind) {
    // Test the pre-built AssemblyScript hello world example
    let wasm = std::fs::read("examples/assemblyscript/build/release.wasm")
        .expect("Failed to read AssemblyScript wasm - run `npm run asbuild` in examples/assemblyscript first");

    let module = parser::parse(&std::collections::HashMap::new(), "test", &mut Reader::new(wasm))
        .expect("Failed to parse module");

    let stdout_buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let stdout = CapturedWriter(stdout_buffer.clone());

    let ctx = Arc::new(
        WasiContext::builder()
            .args(["hello.wasm"])
            .stdout(Box::new(stdout))
            .build(),
    );

    let mut store = Store::new();
    store.set_engine(engine);
    let mut imports = create_wasi_imports(&mut store, ctx.clone());
    // AssemblyScript requires env.abort
    add_assemblyscript_imports(&mut store, &mut imports, ctx.clone());

    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    let output = stdout_buffer.lock().unwrap();
    let output_str = String::from_utf8_lossy(&output);
    assert!(
        output_str.contains("Hello from AssemblyScript"),
        "Expected 'Hello from AssemblyScript' in output, got: {}",
        output_str
    );
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_proc_exit(#[case] engine: EngineKind) {
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "proc_exit"
        (func $proc_exit (param i32)))
      (memory (export "memory") 1)

      (func (export "_start")
        (call $proc_exit (i32.const 42))
      )
    )
    "#;

    let module = parse_wat(wat);

    let ctx = Arc::new(WasiContext::builder().build());

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    // proc_exit should trap
    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_err());

    // Exit code should be 42
    assert_eq!(ctx.exit_code(), Some(42));
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_multiple_iovecs(#[case] engine: EngineKind) {
    // Test scatter/gather I/O with multiple iovec entries
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (memory (export "memory") 1)

      ;; Three strings at different offsets
      (data (i32.const 100) "Hello")
      (data (i32.const 110) ", ")
      (data (i32.const 120) "World!")

      ;; Three iovecs starting at offset 0
      ;; iovec[0]: {ptr: 100, len: 5} - "Hello"
      ;; iovec[1]: {ptr: 110, len: 2} - ", "
      ;; iovec[2]: {ptr: 120, len: 6} - "World!"
      (data (i32.const 0) "\64\00\00\00\05\00\00\00")   ;; {100, 5}
      (data (i32.const 8) "\6e\00\00\00\02\00\00\00")   ;; {110, 2}
      (data (i32.const 16) "\78\00\00\00\06\00\00\00")  ;; {120, 6}

      (func (export "_start")
        (drop (call $fd_write
          (i32.const 1)   ;; fd = stdout
          (i32.const 0)   ;; iovs ptr
          (i32.const 3)   ;; iovs len = 3 iovecs
          (i32.const 200) ;; nwritten ptr
        ))
      )
    )
    "#;

    let module = parse_wat(wat);

    let stdout_buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let stdout = CapturedWriter(stdout_buffer.clone());

    let ctx = Arc::new(WasiContext::builder().stdout(Box::new(stdout)).build());

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    let output = stdout_buffer.lock().unwrap();
    assert_eq!(&*output, b"Hello, World!");
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_fd_read_eof(#[case] engine: EngineKind) {
    // Test that reading from empty stdin returns 0 bytes (EOF)
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_read"
        (func $fd_read (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (memory (export "memory") 1)

      ;; iovec at offset 0: {ptr: 100, len: 64}
      (data (i32.const 0) "\64\00\00\00\40\00\00\00")

      (func (export "_start") (result i32)
        ;; Read from stdin (which is empty)
        (drop (call $fd_read
          (i32.const 0)   ;; fd = stdin
          (i32.const 0)   ;; iovs ptr
          (i32.const 1)   ;; iovs len
          (i32.const 200) ;; nread ptr
        ))
        ;; Return bytes read (should be 0 for EOF)
        (i32.load (i32.const 200))
      )
    )
    "#;

    let module = parse_wat(wat);

    // Empty stdin
    let stdin = Cursor::new(Vec::<u8>::new());

    let ctx = Arc::new(WasiContext::builder().stdin(Box::new(stdin)).build());

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    // Function returns nread, which should be 0 for EOF
    let values = result.unwrap();
    assert_eq!(values.len(), 1);
    if let krasm::runtime::Value::I32(nread) = values[0] {
        assert_eq!(nread, 0, "Expected 0 bytes read for EOF");
    } else {
        panic!("Expected i32 return value");
    }
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_environ(#[case] engine: EngineKind) {
    // Test environment variable access - get sizes, fetch values, print them
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "environ_sizes_get"
        (func $environ_sizes_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "environ_get"
        (func $environ_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (memory (export "memory") 1)

      ;; Memory layout:
      ;; 0-3: environc
      ;; 4-7: environ_buf_size
      ;; 100-119: environ pointers (up to 5 env vars)
      ;; 200-399: environ string buffer
      ;; 400-407: iovec for writing
      ;; 408-411: nwritten

      (func (export "_start") (result i32)
        (local $i i32)
        (local $ptr i32)
        (local $len i32)

        ;; Get environ sizes
        (drop (call $environ_sizes_get (i32.const 0) (i32.const 4)))

        ;; Get environ data: pointers at 100, strings at 200
        (drop (call $environ_get (i32.const 100) (i32.const 200)))

        ;; Print each environment variable
        (local.set $i (i32.const 0))
        (block $done
          (loop $loop
            ;; Check if we've printed all env vars
            (br_if $done (i32.ge_u (local.get $i) (i32.load (i32.const 0))))

            ;; Get pointer to this env var string
            (local.set $ptr (i32.load (i32.add (i32.const 100) (i32.mul (local.get $i) (i32.const 4)))))

            ;; Calculate string length (scan for null terminator)
            (local.set $len (i32.const 0))
            (block $len_done
              (loop $len_loop
                (br_if $len_done (i32.eqz (i32.load8_u (i32.add (local.get $ptr) (local.get $len)))))
                (local.set $len (i32.add (local.get $len) (i32.const 1)))
                (br $len_loop)
              )
            )

            ;; Set up iovec at 400: {ptr, len}
            (i32.store (i32.const 400) (local.get $ptr))
            (i32.store (i32.const 404) (local.get $len))

            ;; Write env var to stdout
            (drop (call $fd_write (i32.const 1) (i32.const 400) (i32.const 1) (i32.const 408)))

            ;; Write newline
            (i32.store8 (i32.const 450) (i32.const 10))
            (i32.store (i32.const 400) (i32.const 450))
            (i32.store (i32.const 404) (i32.const 1))
            (drop (call $fd_write (i32.const 1) (i32.const 400) (i32.const 1) (i32.const 408)))

            ;; Next env var
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $loop)
          )
        )

        ;; Return environc
        (i32.load (i32.const 0))
      )
    )
    "#;

    let module = parse_wat(wat);

    let stdout_buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let stdout = CapturedWriter(stdout_buffer.clone());

    let ctx = Arc::new(
        WasiContext::builder()
            .env(["FOO=bar", "PATH=/usr/bin"])
            .stdout(Box::new(stdout))
            .build(),
    );

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    // Check environc return value
    let values = result.unwrap();
    assert_eq!(values.len(), 1);
    if let krasm::runtime::Value::I32(environc) = values[0] {
        assert_eq!(environc, 2, "Expected 2 environment variables");
    } else {
        panic!("Expected i32 return value");
    }

    // Check printed output
    let output = stdout_buffer.lock().unwrap();
    let output_str = String::from_utf8_lossy(&output);
    assert!(
        output_str.contains("FOO=bar"),
        "Expected 'FOO=bar' in output, got: {}",
        output_str
    );
    assert!(
        output_str.contains("PATH=/usr/bin"),
        "Expected 'PATH=/usr/bin' in output, got: {}",
        output_str
    );
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_fd_prestat_returns_badf(#[case] engine: EngineKind) {
    // Test that fd_prestat_get returns EBADF (no preopened directories)
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (memory (export "memory") 1)

      (func (export "_start") (result i32)
        ;; Try to get prestat for fd 3 (first possible preopened fd)
        (call $fd_prestat_get
          (i32.const 3)   ;; fd
          (i32.const 0)   ;; buf ptr
        )
      )
    )
    "#;

    let module = parse_wat(wat);

    let ctx = Arc::new(WasiContext::builder().build());

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    // Function returns errno, which should be EBADF (8)
    let values = result.unwrap();
    assert_eq!(values.len(), 1);
    if let krasm::runtime::Value::I32(errno) = values[0] {
        assert_eq!(errno, 8, "Expected EBADF (8) for no preopened fds");
    } else {
        panic!("Expected i32 return value");
    }
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_preopen_enumerate(#[case] engine: EngineKind) {
    // fd_prestat_get(3) succeeds with a preopen, fd_prestat_get(4) returns EBADF
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_prestat_dir_name"
        (func $fd_prestat_dir_name (param i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (memory (export "memory") 1)

      (func (export "_start") (result i32)
        (local $err i32)
        (local $name_len i32)

        ;; Try fd_prestat_get(3, buf=0)
        (local.set $err (call $fd_prestat_get (i32.const 3) (i32.const 0)))

        ;; If error, return it
        (if (local.get $err)
          (then (return (local.get $err)))
        )

        ;; Read name_len from prestat struct (offset 4)
        (local.set $name_len (i32.load (i32.const 4)))

        ;; Get the dir name into buffer at 100
        (local.set $err (call $fd_prestat_dir_name (i32.const 3) (i32.const 100) (local.get $name_len)))
        (if (local.get $err)
          (then (return (i32.const 999)))
        )

        ;; Write the dir name to stdout
        (i32.store (i32.const 200) (i32.const 100))  ;; iovec ptr
        (i32.store (i32.const 204) (local.get $name_len)) ;; iovec len
        (drop (call $fd_write (i32.const 1) (i32.const 200) (i32.const 1) (i32.const 300)))

        ;; Check that fd 4 returns EBADF
        (call $fd_prestat_get (i32.const 4) (i32.const 0))
      )
    )
    "#;

    let module = parse_wat(wat);

    let stdout_buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let stdout = CapturedWriter(stdout_buffer.clone());

    let temp = temp_dir();
    let dir = temp.path();
    let dir_str = dir.to_str().unwrap();

    let ctx = Arc::new(
        WasiContext::builder()
            .stdout(Box::new(stdout))
            .preopen_dir(dir_str, dir_str)
            .build(),
    );

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    // Return value is fd_prestat_get(4) which should be EBADF (8)
    let values = result.unwrap();
    if let krasm::runtime::Value::I32(errno) = values[0] {
        assert_eq!(errno, 8, "Expected EBADF for fd 4");
    } else {
        panic!("Expected i32 return value");
    }

    // Stdout should contain the preopen dir name
    let output = stdout_buffer.lock().unwrap();
    let output_str = String::from_utf8_lossy(&output);
    assert_eq!(output_str, dir_str);
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_path_open_and_read(#[case] engine: EngineKind) {
    // Open a file via path_open, read its contents, write to stdout
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "path_open"
        (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_read"
        (func $fd_read (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_close"
        (func $fd_close (param i32) (result i32)))
      (memory (export "memory") 1)

      ;; Filename at offset 500 (written by test setup via data segment won't work,
      ;; we'll use args instead). We'll hardcode "test_read.txt" = 13 bytes.
      (data (i32.const 500) "test_read.txt")

      (func (export "_start") (result i32)
        (local $err i32)
        (local $file_fd i32)
        (local $nread i32)

        ;; Verify preopen exists
        (local.set $err (call $fd_prestat_get (i32.const 3) (i32.const 0)))
        (if (local.get $err)
          (then (return (i32.const 100)))
        )

        ;; path_open(dir_fd=3, dirflags=0, path=500, path_len=13, oflags=0,
        ;;   rights_base=all, rights_inheriting=all, fdflags=0, opened_fd_ptr=400)
        (local.set $err (call $path_open
          (i32.const 3)           ;; dir fd
          (i32.const 0)           ;; dirflags
          (i32.const 500)         ;; path ptr
          (i32.const 13)          ;; path len
          (i32.const 0)           ;; oflags
          (i64.const 0x1fffffff)  ;; rights_base
          (i64.const 0x1fffffff)  ;; rights_inheriting
          (i32.const 0)           ;; fdflags
          (i32.const 400)         ;; opened_fd ptr
        ))
        (if (local.get $err)
          (then (return (i32.add (i32.const 200) (local.get $err))))
        )

        (local.set $file_fd (i32.load (i32.const 400)))

        ;; Read from file: iovec at 0, buffer at 100
        (i32.store (i32.const 0) (i32.const 100))  ;; buf ptr
        (i32.store (i32.const 4) (i32.const 256))   ;; buf len
        (local.set $err (call $fd_read
          (local.get $file_fd)
          (i32.const 0)    ;; iovs ptr
          (i32.const 1)    ;; iovs count
          (i32.const 300)  ;; nread ptr
        ))
        (if (local.get $err)
          (then (return (i32.const 300)))
        )

        (local.set $nread (i32.load (i32.const 300)))

        ;; Write read data to stdout
        (i32.store (i32.const 0) (i32.const 100))        ;; buf ptr
        (i32.store (i32.const 4) (local.get $nread))       ;; buf len
        (drop (call $fd_write
          (i32.const 1)    ;; stdout
          (i32.const 0)    ;; iovs ptr
          (i32.const 1)    ;; iovs count
          (i32.const 304)  ;; nwritten ptr
        ))

        ;; Close the file
        (drop (call $fd_close (local.get $file_fd)))

        ;; Return 0 for success
        (i32.const 0)
      )
    )
    "#;

    let module = parse_wat(wat);

    // Create a temp file with known content
    let temp = temp_dir();
    let dir = temp.path();
    let file_path = dir.join("test_read.txt");
    std::fs::write(&file_path, "file contents here").expect("Failed to write test file");

    let stdout_buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let stdout = CapturedWriter(stdout_buffer.clone());
    let dir_str = dir.to_str().unwrap();

    let ctx = Arc::new(
        WasiContext::builder()
            .stdout(Box::new(stdout))
            .preopen_dir(dir_str, dir_str)
            .build(),
    );

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    let values = result.unwrap();
    if let krasm::runtime::Value::I32(ret) = values[0] {
        assert_eq!(ret, 0, "Expected success (0), got {}", ret);
    }

    let output = stdout_buffer.lock().unwrap();
    assert_eq!(&*output, b"file contents here");
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_path_open_write(#[case] engine: EngineKind) {
    // Open a file for writing via path_open with O_CREAT
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "path_open"
        (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_close"
        (func $fd_close (param i32) (result i32)))
      (memory (export "memory") 1)

      ;; Filename
      (data (i32.const 500) "test_write.txt")
      ;; Data to write
      (data (i32.const 600) "written by wasm")

      (func (export "_start") (result i32)
        (local $err i32)
        (local $file_fd i32)

        ;; Verify preopen
        (local.set $err (call $fd_prestat_get (i32.const 3) (i32.const 0)))
        (if (local.get $err) (then (return (i32.const 100))))

        ;; path_open with O_CREAT (1) | O_TRUNC (8) = 9
        (local.set $err (call $path_open
          (i32.const 3) (i32.const 0) (i32.const 500) (i32.const 14)
          (i32.const 9)
          (i64.const 0x1fffffff) (i64.const 0x1fffffff)
          (i32.const 0) (i32.const 400)
        ))
        (if (local.get $err) (then (return (i32.add (i32.const 200) (local.get $err)))))

        (local.set $file_fd (i32.load (i32.const 400)))

        ;; Write data to file
        (i32.store (i32.const 0) (i32.const 600))   ;; buf ptr
        (i32.store (i32.const 4) (i32.const 15))     ;; len = "written by wasm"
        (local.set $err (call $fd_write
          (local.get $file_fd)
          (i32.const 0) (i32.const 1) (i32.const 300)
        ))
        (if (local.get $err) (then (return (i32.const 300))))

        ;; Close
        (drop (call $fd_close (local.get $file_fd)))

        (i32.const 0)
      )
    )
    "#;

    let module = parse_wat(wat);

    let temp = temp_dir();
    let dir = temp.path();
    let file_path = dir.join("test_write.txt");

    let dir_str = dir.to_str().unwrap();
    let ctx = Arc::new(WasiContext::builder().preopen_dir(dir_str, dir_str).build());

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    let values = result.unwrap();
    if let krasm::runtime::Value::I32(ret) = values[0] {
        assert_eq!(ret, 0, "Expected success (0), got {}", ret);
    }

    // Verify file was written on host
    let contents = std::fs::read_to_string(&file_path).expect("Failed to read written file");
    assert_eq!(contents, "written by wasm");
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_fd_close_then_read_fails(#[case] engine: EngineKind) {
    // Open a file, close it, then attempt to read (should return EBADF)
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "path_open"
        (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_read"
        (func $fd_read (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_close"
        (func $fd_close (param i32) (result i32)))
      (memory (export "memory") 1)

      (data (i32.const 500) "test_close.txt")

      (func (export "_start") (result i32)
        (local $file_fd i32)
        (local $err i32)

        ;; Verify preopen
        (local.set $err (call $fd_prestat_get (i32.const 3) (i32.const 0)))
        (if (local.get $err) (then (return (i32.const 100))))

        ;; Open file
        (local.set $err (call $path_open
          (i32.const 3) (i32.const 0) (i32.const 500) (i32.const 14)
          (i32.const 0) (i64.const 0x1fffffff) (i64.const 0x1fffffff)
          (i32.const 0) (i32.const 400)
        ))
        (if (local.get $err) (then (return (i32.const 200))))

        (local.set $file_fd (i32.load (i32.const 400)))

        ;; Close the file
        (drop (call $fd_close (local.get $file_fd)))

        ;; Try to read from closed fd
        (i32.store (i32.const 0) (i32.const 100))
        (i32.store (i32.const 4) (i32.const 64))
        (call $fd_read (local.get $file_fd) (i32.const 0) (i32.const 1) (i32.const 300))
        ;; Should return EBADF (8)
      )
    )
    "#;

    let module = parse_wat(wat);

    let temp = temp_dir();
    let dir = temp.path();
    let file_path = dir.join("test_close.txt");
    std::fs::write(&file_path, "data").expect("Failed to write test file");

    let dir_str = dir.to_str().unwrap();
    let ctx = Arc::new(WasiContext::builder().preopen_dir(dir_str, dir_str).build());

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    let values = result.unwrap();
    if let krasm::runtime::Value::I32(errno) = values[0] {
        assert_eq!(errno, 8, "Expected EBADF (8) for closed fd, got {}", errno);
    }
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_path_traversal_blocked(#[case] engine: EngineKind) {
    // Attempt to open ../../../etc/passwd should return EACCES
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "path_open"
        (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
      (memory (export "memory") 1)

      ;; Traversal path
      (data (i32.const 500) "../../../etc/passwd")

      (func (export "_start") (result i32)
        (local $err i32)

        ;; Verify preopen
        (local.set $err (call $fd_prestat_get (i32.const 3) (i32.const 0)))
        (if (local.get $err) (then (return (i32.const 100))))

        ;; Try path traversal
        (call $path_open
          (i32.const 3) (i32.const 0) (i32.const 500) (i32.const 19)
          (i32.const 0) (i64.const 0x1fffffff) (i64.const 0x1fffffff)
          (i32.const 0) (i32.const 400)
        )
        ;; Should return EACCES (2)
      )
    )
    "#;

    let module = parse_wat(wat);

    let temp = temp_dir();
    let dir = temp.path();
    let dir_str = dir.to_str().unwrap();
    let ctx = Arc::new(WasiContext::builder().preopen_dir(dir_str, dir_str).build());

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    let values = result.unwrap();
    if let krasm::runtime::Value::I32(errno) = values[0] {
        // Should be EACCES (2) for path traversal
        assert_eq!(errno, 2, "Expected EACCES (2) for path traversal, got {}", errno);
    }
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_path_open_nonexistent_file(#[case] engine: EngineKind) {
    // Opening a file that doesn't exist without O_CREAT returns ENOENT
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "path_open"
        (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
      (memory (export "memory") 1)

      (data (i32.const 500) "no_such_file.txt")

      (func (export "_start") (result i32)
        (local $err i32)
        (local.set $err (call $fd_prestat_get (i32.const 3) (i32.const 0)))
        (if (local.get $err) (then (return (i32.const 100))))

        (call $path_open
          (i32.const 3) (i32.const 0) (i32.const 500) (i32.const 16)
          (i32.const 0) (i64.const 0x1fffffff) (i64.const 0x1fffffff)
          (i32.const 0) (i32.const 400)
        )
      )
    )
    "#;

    let module = parse_wat(wat);

    let temp = temp_dir();
    let dir = temp.path();
    let dir_str = dir.to_str().unwrap();
    let ctx = Arc::new(WasiContext::builder().preopen_dir(dir_str, dir_str).build());

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    let values = result.unwrap();
    if let krasm::runtime::Value::I32(errno) = values[0] {
        assert_eq!(errno, 44, "Expected ENOENT (44), got {}", errno);
    }
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_fd_seek(#[case] engine: EngineKind) {
    // Open a file, seek to offset 6, read from there
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "path_open"
        (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_read"
        (func $fd_read (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_seek"
        (func $fd_seek (param i32 i64 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_close"
        (func $fd_close (param i32) (result i32)))
      (memory (export "memory") 1)

      (data (i32.const 500) "test_seek.txt")

      (func (export "_start") (result i32)
        (local $err i32)
        (local $file_fd i32)
        (local $nread i32)

        (local.set $err (call $fd_prestat_get (i32.const 3) (i32.const 0)))
        (if (local.get $err) (then (return (i32.const 100))))

        ;; Open file
        (local.set $err (call $path_open
          (i32.const 3) (i32.const 0) (i32.const 500) (i32.const 13)
          (i32.const 0) (i64.const 0x1fffffff) (i64.const 0x1fffffff)
          (i32.const 0) (i32.const 400)
        ))
        (if (local.get $err) (then (return (i32.add (i32.const 200) (local.get $err)))))
        (local.set $file_fd (i32.load (i32.const 400)))

        ;; Seek to offset 6 (SEEK_SET = 0)
        (local.set $err (call $fd_seek
          (local.get $file_fd)
          (i64.const 6)
          (i32.const 0)    ;; SEEK_SET
          (i32.const 410)  ;; newoffset ptr
        ))
        (if (local.get $err) (then (return (i32.const 300))))

        ;; Read remaining bytes
        (i32.store (i32.const 0) (i32.const 100))
        (i32.store (i32.const 4) (i32.const 256))
        (local.set $err (call $fd_read
          (local.get $file_fd) (i32.const 0) (i32.const 1) (i32.const 420)
        ))
        (if (local.get $err) (then (return (i32.const 400))))
        (local.set $nread (i32.load (i32.const 420)))

        ;; Write to stdout
        (i32.store (i32.const 0) (i32.const 100))
        (i32.store (i32.const 4) (local.get $nread))
        (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 430)))

        (drop (call $fd_close (local.get $file_fd)))
        (i32.const 0)
      )
    )
    "#;

    let module = parse_wat(wat);

    let temp = temp_dir();
    let dir = temp.path();
    let file_path = dir.join("test_seek.txt");
    std::fs::write(&file_path, "Hello,World!").expect("Failed to write test file");

    let stdout_buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
    let stdout = CapturedWriter(stdout_buffer.clone());
    let dir_str = dir.to_str().unwrap();

    let ctx = Arc::new(
        WasiContext::builder()
            .stdout(Box::new(stdout))
            .preopen_dir(dir_str, dir_str)
            .build(),
    );

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    let values = result.unwrap();
    if let krasm::runtime::Value::I32(ret) = values[0] {
        assert_eq!(ret, 0, "Expected success (0), got {}", ret);
    }

    // After seeking to offset 6, reading "Hello,World!" should give "World!"
    let output = stdout_buffer.lock().unwrap();
    assert_eq!(&*output, b"World!");
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_fd_fdstat_get(#[case] engine: EngineKind) {
    // Verify fd_fdstat_get returns correct file types
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_fdstat_get"
        (func $fd_fdstat_get (param i32 i32) (result i32)))
      (memory (export "memory") 1)

      (func (export "_start") (result i32)
        (local $err i32)

        ;; Verify preopen exists
        (local.set $err (call $fd_prestat_get (i32.const 3) (i32.const 0)))
        (if (local.get $err) (then (return (i32.const 100))))

        ;; fd_fdstat_get on stdout (fd 1) -> CharacterDevice (2)
        (local.set $err (call $fd_fdstat_get (i32.const 1) (i32.const 200)))
        (if (local.get $err) (then (return (i32.const 200))))

        ;; fd_fdstat_get on preopen dir (fd 3) -> Directory (3)
        (local.set $err (call $fd_fdstat_get (i32.const 3) (i32.const 300)))
        (if (local.get $err) (then (return (i32.const 300))))

        ;; Return: (stdout_type << 8) | dir_type
        ;; So we can check both in one return value
        (i32.or
          (i32.shl (i32.load8_u (i32.const 200)) (i32.const 8))
          (i32.load8_u (i32.const 300))
        )
      )
    )
    "#;

    let module = parse_wat(wat);

    let temp = temp_dir();
    let dir = temp.path();
    let dir_str = dir.to_str().unwrap();
    let ctx = Arc::new(
        WasiContext::builder()
            .stdout(Box::new(std::io::sink()))
            .preopen_dir(dir_str, dir_str)
            .build(),
    );

    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx.clone());
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    let values = result.unwrap();
    if let krasm::runtime::Value::I32(combined) = values[0] {
        let stdout_type = (combined >> 8) & 0xff;
        let dir_type = combined & 0xff;
        assert_eq!(
            stdout_type, 2,
            "Expected CharacterDevice (2) for stdout, got {}",
            stdout_type
        );
        assert_eq!(dir_type, 3, "Expected Directory (3) for preopen, got {}", dir_type);
    } else {
        panic!("Expected i32 return value");
    }
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_return_from_nested_block_preserves_caller(#[case] engine: EngineKind) {
    // This tests the fix for a bug where returning from within a nested block
    // did not properly pop the call frame, causing the caller's execution
    // context to be corrupted.
    //
    // The WAT creates two functions:
    // - helper: contains nested blocks and returns 42 from inside them
    // - main: stores 100 in local, calls helper, returns its local
    //
    // Before the fix, main would return garbage because its execution context
    // was corrupted by helper's return not cleaning up properly.

    let wat = r#"
    (module
      ;; Helper function: returns 42 from within nested blocks
      (func $helper (result i32)
        (block
          (block
            (i32.const 42)
            (return)
          )
        )
        (i32.const 0)  ;; unreachable
      )

      ;; Main function: stores value in local, calls helper, returns local
      (func (export "_start") (result i32)
        (local $value i32)
        (local.set $value (i32.const 100))
        (drop (call $helper))
        (local.get $value)
      )
    )
    "#;

    let module = parse_wat(wat);

    let mut store = Store::new();
    store.set_engine(engine);
    let instance_id = store
        .create_instance(Arc::new(module), None)
        .expect("Failed to create instance");

    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "Execution failed: {:?}", result);

    let values = result.unwrap();
    if let krasm::runtime::Value::I32(v) = values[0] {
        assert_eq!(v, 100, "Expected caller's local (100) to be preserved, got {}", v);
    } else {
        panic!("Expected i32 return value");
    }
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_clock_time_get(#[case] engine: EngineKind) {
    // clock_time_get should return success (0) for CLOCK_REALTIME (0)
    // and write a non-zero timestamp
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "clock_time_get"
        (func $clock_time_get (param i32 i64 i32) (result i32)))
      (memory (export "memory") 1)

      (func (export "_start")
        (local $err i32)

        ;; clock_time_get(CLOCK_REALTIME=0, precision=1000, time_ptr=100)
        (local.set $err (call $clock_time_get
          (i32.const 0)
          (i64.const 1000)
          (i32.const 100)
        ))
        (if (local.get $err) (then unreachable))

        ;; Timestamp should be non-zero (low 32 bits at offset 100)
        (if (i32.eqz (i32.or
          (i32.load (i32.const 100))
          (i32.load (i32.const 104))
        )) (then unreachable))
      )
    )
    "#;

    let module = parse_wat(wat);
    let ctx = Arc::new(WasiContext::builder().build());
    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx);
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");
    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "clock_time_get failed: {:?}", result);
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_random_get(#[case] engine: EngineKind) {
    // random_get should fill a buffer with non-zero bytes (probabilistically)
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "random_get"
        (func $random_get (param i32 i32) (result i32)))
      (memory (export "memory") 1)

      ;; Zero-initialise 32 bytes at offset 100, then fill with random
      (data (i32.const 100) "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00")

      (func (export "_start")
        (local $err i32)
        (local $sum i32)
        (local $i i32)

        ;; random_get(buf=100, len=32)
        (local.set $err (call $random_get (i32.const 100) (i32.const 32)))
        (if (local.get $err) (then unreachable))

        ;; Sum all 32 bytes; astronomically unlikely to be all zero
        (local.set $i (i32.const 0))
        (block $done
          (loop $loop
            (br_if $done (i32.ge_u (local.get $i) (i32.const 32)))
            (local.set $sum (i32.add
              (local.get $sum)
              (i32.load8_u (i32.add (i32.const 100) (local.get $i)))
            ))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $loop)
          )
        )

        ;; Sum should be non-zero
        (if (i32.eqz (local.get $sum)) (then unreachable))
      )
    )
    "#;

    let module = parse_wat(wat);
    let ctx = Arc::new(WasiContext::builder().build());
    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx);
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");
    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "random_get failed: {:?}", result);
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_fd_tell(#[case] engine: EngineKind) {
    // Open a file, write to it, then fd_tell should report the current position
    let temp = temp_dir();
    let dir = temp.path();
    std::fs::write(dir.join("tell_test.txt"), "Hello, World!").unwrap();

    let dir_str = dir.to_str().unwrap();

    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "path_open"
        (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_read"
        (func $fd_read (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_tell"
        (func $fd_tell (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_close"
        (func $fd_close (param i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (memory (export "memory") 1)

      ;; "tell_test.txt" at offset 200 (13 bytes)
      (data (i32.const 200) "tell_test.txt")

      (func (export "_start") (result i32)
        (local $file_fd i32)
        (local $err i32)

        ;; Verify preopen exists
        (if (call $fd_prestat_get (i32.const 3) (i32.const 500)) (then (return (i32.const 1))))

        ;; Open the file
        (local.set $err (call $path_open
          (i32.const 3) (i32.const 0)
          (i32.const 200) (i32.const 13)
          (i32.const 0) (i64.const 0x1fffffff) (i64.const 0x1fffffff)
          (i32.const 0) (i32.const 300)
        ))
        (if (local.get $err) (then (return (i32.const 2))))
        (local.set $file_fd (i32.load (i32.const 300)))

        ;; Read 5 bytes
        (i32.store (i32.const 0) (i32.const 100))  ;; iov.buf = 100
        (i32.store (i32.const 4) (i32.const 5))    ;; iov.len = 5
        (local.set $err (call $fd_read
          (local.get $file_fd) (i32.const 0) (i32.const 1) (i32.const 310)
        ))
        (if (local.get $err) (then (return (i32.const 3))))

        ;; fd_tell should report position 5
        (local.set $err (call $fd_tell (local.get $file_fd) (i32.const 400)))
        (if (local.get $err) (then (return (i32.const 4))))

        ;; Check low 32 bits of offset == 5
        (if (i32.ne (i32.load (i32.const 400)) (i32.const 5))
          (then (return (i32.const 5))))

        (drop (call $fd_close (local.get $file_fd)))
        (i32.const 0)
      )
    )
    "#;

    let module = parse_wat(wat);
    let ctx = Arc::new(WasiContext::builder().preopen_dir(dir_str, dir_str).build());
    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx);
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");
    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "fd_tell test execution failed: {:?}", result);
    let values = result.unwrap();
    assert_eq!(
        values[0],
        krasm::runtime::Value::I32(0),
        "fd_tell test returned non-zero error code"
    );
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_fd_readdir(#[case] engine: EngineKind) {
    // Create a directory with known files, open it, and read entries via fd_readdir
    let temp = temp_dir();
    let dir = temp.path();
    std::fs::write(dir.join("alpha.txt"), "a").unwrap();
    std::fs::write(dir.join("beta.txt"), "b").unwrap();
    std::fs::create_dir(dir.join("gamma")).unwrap();

    let dir_str = dir.to_str().unwrap();

    // This module opens the preopened dir itself (fd 3 is the directory),
    // calls fd_readdir, and counts entries. It returns the count.
    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_readdir"
        (func $fd_readdir (param i32 i32 i32 i64 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (memory (export "memory") 1)

      (func (export "_start") (result i32)
        (local $err i32)
        (local $bufused i32)
        (local $count i32)
        (local $offset i32)
        (local $namelen i32)

        ;; Verify preopen exists
        (if (call $fd_prestat_get (i32.const 3) (i32.const 500)) (then (return (i32.const -1))))

        ;; fd_readdir(fd=3, buf=1000, buf_len=4096, cookie=0, bufused_ptr=900)
        (local.set $err (call $fd_readdir
          (i32.const 3) (i32.const 1000) (i32.const 4096) (i64.const 0) (i32.const 900)
        ))
        (if (local.get $err) (then (return (i32.const -2))))

        (local.set $bufused (i32.load (i32.const 900)))

        ;; Walk entries: each dirent is 24-byte header + variable-length name
        (local.set $offset (i32.const 0))
        (block $done
          (loop $loop
            ;; Need at least 24 bytes for the header
            (br_if $done (i32.gt_u
              (i32.add (local.get $offset) (i32.const 24))
              (local.get $bufused)
            ))
            ;; Read namelen at offset+16
            (local.set $namelen (i32.load (i32.add (i32.const 1000) (i32.add (local.get $offset) (i32.const 16)))))
            ;; Check we have the full name
            (br_if $done (i32.gt_u
              (i32.add (i32.add (local.get $offset) (i32.const 24)) (local.get $namelen))
              (local.get $bufused)
            ))
            ;; Count this entry
            (local.set $count (i32.add (local.get $count) (i32.const 1)))
            ;; Advance past header + name
            (local.set $offset (i32.add
              (i32.add (local.get $offset) (i32.const 24))
              (local.get $namelen)
            ))
            (br $loop)
          )
        )

        ;; Return count (should be 3: alpha.txt, beta.txt, gamma)
        (local.get $count)
      )
    )
    "#;

    let module = parse_wat(wat);
    let ctx = Arc::new(WasiContext::builder().preopen_dir(dir_str, dir_str).build());
    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx);
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");
    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "fd_readdir test failed: {:?}", result);
    let values = result.unwrap();
    assert_eq!(
        values[0],
        krasm::runtime::Value::I32(3),
        "Expected 3 directory entries, got {:?}",
        values[0]
    );
}

#[rstest]
#[case::structured(EngineKind::Structured)]
#[case::flat(EngineKind::Flat)]
fn test_path_open_directory(#[case] engine: EngineKind) {
    // Open a subdirectory via path_open, then fd_readdir on the opened fd
    let temp = temp_dir();
    let dir = temp.path();
    let sub = dir.join("mydir");
    std::fs::create_dir(&sub).unwrap();
    std::fs::write(sub.join("file1.txt"), "one").unwrap();
    std::fs::write(sub.join("file2.txt"), "two").unwrap();

    let dir_str = dir.to_str().unwrap();

    let wat = r#"
    (module
      (import "wasi_snapshot_preview1" "path_open"
        (func $path_open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_readdir"
        (func $fd_readdir (param i32 i32 i32 i64 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_close"
        (func $fd_close (param i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_prestat_get"
        (func $fd_prestat_get (param i32 i32) (result i32)))
      (memory (export "memory") 1)

      ;; "mydir" at offset 200
      (data (i32.const 200) "mydir")

      (func (export "_start") (result i32)
        (local $dir_fd i32)
        (local $err i32)
        (local $bufused i32)
        (local $count i32)
        (local $offset i32)
        (local $namelen i32)

        (if (call $fd_prestat_get (i32.const 3) (i32.const 500)) (then (return (i32.const -1))))

        ;; path_open with O_DIRECTORY (oflags=2)
        (local.set $err (call $path_open
          (i32.const 3) (i32.const 0)
          (i32.const 200) (i32.const 5)
          (i32.const 2) (i64.const 0x1fffffff) (i64.const 0x1fffffff)
          (i32.const 0) (i32.const 300)
        ))
        (if (local.get $err) (then (return (i32.const -2))))
        (local.set $dir_fd (i32.load (i32.const 300)))

        ;; fd_readdir on the opened directory
        (local.set $err (call $fd_readdir
          (local.get $dir_fd) (i32.const 1000) (i32.const 4096) (i64.const 0) (i32.const 900)
        ))
        (if (local.get $err) (then (return (i32.const -3))))

        (local.set $bufused (i32.load (i32.const 900)))

        ;; Count entries
        (local.set $offset (i32.const 0))
        (block $done
          (loop $loop
            (br_if $done (i32.gt_u
              (i32.add (local.get $offset) (i32.const 24))
              (local.get $bufused)
            ))
            (local.set $namelen (i32.load (i32.add (i32.const 1000) (i32.add (local.get $offset) (i32.const 16)))))
            (br_if $done (i32.gt_u
              (i32.add (i32.add (local.get $offset) (i32.const 24)) (local.get $namelen))
              (local.get $bufused)
            ))
            (local.set $count (i32.add (local.get $count) (i32.const 1)))
            (local.set $offset (i32.add
              (i32.add (local.get $offset) (i32.const 24))
              (local.get $namelen)
            ))
            (br $loop)
          )
        )

        (drop (call $fd_close (local.get $dir_fd)))

        ;; Should have 2 entries: file1.txt, file2.txt
        (local.get $count)
      )
    )
    "#;

    let module = parse_wat(wat);
    let ctx = Arc::new(WasiContext::builder().preopen_dir(dir_str, dir_str).build());
    let mut store = Store::new();
    store.set_engine(engine);
    let imports = create_wasi_imports(&mut store, ctx);
    let instance_id = store
        .create_instance(Arc::new(module), Some(&imports))
        .expect("Failed to create instance");
    let result = store.invoke_export(instance_id, "_start", vec![], None);
    assert!(result.is_ok(), "path_open directory test failed: {:?}", result);
    let values = result.unwrap();
    assert_eq!(
        values[0],
        krasm::runtime::Value::I32(2),
        "Expected 2 entries in subdir, got {:?}",
        values[0]
    );
}
