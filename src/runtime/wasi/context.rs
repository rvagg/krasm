//! WASI context for managing WASI state
//!
//! This module provides the `WasiContext` which holds all WASI-related state
//! including file descriptors, arguments, and environment variables.
//!
//! Memory access is provided by the calling instance via `Caller` — the context
//! itself does not hold a reference to linear memory.

use super::types::{WasiErrno, WasiFileType};
use crate::runtime::Memory;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

/// A file descriptor entry for WASI
pub struct FileDescriptor {
    /// The underlying reader (if readable)
    reader: Option<Box<dyn Read + Send>>,
    /// The underlying writer (if writable)
    writer: Option<Box<dyn Write + Send>>,
    /// The underlying seeker (if seekable)
    seeker: Option<Box<dyn Seek + Send>>,
    /// Whether this fd is readable
    pub readable: bool,
    /// Whether this fd is writable
    pub writable: bool,
    /// The WASI file type
    pub file_type: WasiFileType,
    /// Host filesystem path (for metadata queries)
    pub host_path: Option<PathBuf>,
}

impl FileDescriptor {
    /// Create a new FileDescriptor for a reader (e.g. stdin)
    pub fn new_reader(reader: Box<dyn Read + Send>) -> Self {
        FileDescriptor {
            reader: Some(reader),
            writer: None,
            seeker: None,
            readable: true,
            writable: false,
            file_type: WasiFileType::CharacterDevice,
            host_path: None,
        }
    }

    /// Create a new FileDescriptor for a writer (e.g. stdout/stderr)
    pub fn new_writer(writer: Box<dyn Write + Send>) -> Self {
        FileDescriptor {
            reader: None,
            writer: Some(writer),
            seeker: None,
            readable: false,
            writable: true,
            file_type: WasiFileType::CharacterDevice,
            host_path: None,
        }
    }

    /// Create a new FileDescriptor for a directory (preopens)
    pub fn new_directory(host_path: PathBuf) -> Self {
        FileDescriptor {
            reader: None,
            writer: None,
            seeker: None,
            readable: false,
            writable: false,
            file_type: WasiFileType::Directory,
            host_path: Some(host_path),
        }
    }

    /// Create a new FileDescriptor for a regular file
    ///
    /// Uses the same underlying std::fs::File for reading, writing, and seeking.
    pub fn new_file(
        file: std::fs::File,
        readable: bool,
        writable: bool,
        host_path: Option<PathBuf>,
    ) -> Result<Self, std::io::Error> {
        let reader: Option<Box<dyn Read + Send>> = if readable {
            Some(Box::new(file.try_clone()?))
        } else {
            None
        };

        let writer: Option<Box<dyn Write + Send>> = if writable {
            Some(Box::new(file.try_clone()?))
        } else {
            None
        };

        let seeker: Option<Box<dyn Seek + Send>> = Some(Box::new(file));

        Ok(FileDescriptor {
            reader,
            writer,
            seeker,
            readable,
            writable,
            file_type: WasiFileType::RegularFile,
            host_path,
        })
    }

    /// Read from this file descriptor
    pub fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match &mut self.reader {
            Some(reader) => reader.read(buf),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "fd is not readable",
            )),
        }
    }

    /// Write to this file descriptor
    pub fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match &mut self.writer {
            Some(writer) => writer.write(buf),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "fd is not writable",
            )),
        }
    }

    /// Flush this file descriptor
    pub fn flush(&mut self) -> std::io::Result<()> {
        match &mut self.writer {
            Some(writer) => writer.flush(),
            None => Ok(()),
        }
    }

    /// Seek within this file descriptor
    pub fn seek(&mut self, pos: SeekFrom) -> Result<u64, std::io::Error> {
        match &mut self.seeker {
            Some(seeker) => seeker.seek(pos),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "fd is not seekable",
            )),
        }
    }
}

impl std::fmt::Debug for FileDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileDescriptor")
            .field("readable", &self.readable)
            .field("writable", &self.writable)
            .field("file_type", &self.file_type)
            .finish()
    }
}

/// WASI context holding all WASI-related state
///
/// The context is shared between all WASI functions via `Arc<WasiContext>`.
/// Linear memory is accessed through `Caller` at each host function invocation,
/// not stored in the context.
/// Default maximum number of file descriptors a guest can have open.
pub const DEFAULT_MAX_FDS: u32 = 1024;

/// Default maximum number of iovecs per read/write call.
pub const DEFAULT_MAX_IOVECS: u32 = 1024;

pub struct WasiContext {
    /// File descriptors (0=stdin, 1=stdout, 2=stderr, 3+=preopens/files).
    /// `Mutex` rather than `RefCell` keeps the context `Sync`, so
    /// `Arc<WasiContext>` is genuinely shareable across threads.
    fds: Mutex<Vec<Option<FileDescriptor>>>,
    /// Preopened directories: (fd number, guest path, host path)
    preopens: Vec<(u32, String, PathBuf)>,
    /// Command line arguments
    args: Vec<String>,
    /// Environment variables (name=value pairs)
    env: Vec<String>,
    /// Process exit code (set by proc_exit)
    exit_code: Mutex<Option<i32>>,
    /// Monotonic clock baseline (captured at context construction)
    monotonic_epoch: Instant,
    /// Maximum number of open file descriptors (prevents resource exhaustion)
    max_fds: u32,
    /// Maximum iovec count per read/write call (prevents allocation bombs)
    max_iovecs: u32,
}

impl WasiContext {
    /// Create a new builder for WasiContext
    pub fn builder() -> WasiContextBuilder {
        WasiContextBuilder::new()
    }

    /// Get the exit code if proc_exit was called
    pub fn exit_code(&self) -> Option<i32> {
        *self.exit_code.lock().unwrap()
    }

    /// Return the monotonic clock baseline
    pub fn monotonic_epoch(&self) -> Instant {
        self.monotonic_epoch
    }

    /// Set the exit code (called by proc_exit)
    pub fn set_exit_code(&self, code: i32) {
        *self.exit_code.lock().unwrap() = Some(code);
    }

    /// Get a reference to the args
    pub fn args(&self) -> &[String] {
        &self.args
    }

    /// Get a reference to the environment variables
    pub fn env(&self) -> &[String] {
        &self.env
    }

    /// Lock the file descriptors for reading
    pub fn fds_ref(&self) -> std::sync::MutexGuard<'_, Vec<Option<FileDescriptor>>> {
        self.fds.lock().unwrap()
    }

    // === Preopen methods ===

    /// Get the guest path for a preopened fd
    pub fn get_preopen(&self, fd: u32) -> Option<&str> {
        self.preopens
            .iter()
            .find(|(preopen_fd, _, _)| *preopen_fd == fd)
            .map(|(_, path, _)| path.as_str())
    }

    /// Get the list of preopens as (fd, guest_path) pairs
    pub fn preopens(&self) -> Vec<(u32, &str)> {
        self.preopens
            .iter()
            .map(|(fd, guest, _)| (*fd, guest.as_str()))
            .collect()
    }

    /// Resolve a guest path relative to a preopened directory.
    ///
    /// Canonicalises the result and verifies it does not escape the preopen
    /// directory (symlink traversal, `..` components). For paths that do not
    /// yet exist (e.g. a file about to be created), the parent directory is
    /// canonicalised instead and the filename is appended.
    /// Resolve a guest-provided path relative to a preopened directory.
    ///
    /// Returns the canonicalised host path on success, ensuring it falls within
    /// the preopened directory tree. For non-existent files (e.g. O_CREAT), the
    /// parent directory is canonicalised and the final component is appended.
    ///
    /// # Security
    ///
    /// This check is subject to TOCTOU (time-of-check-time-of-use): between
    /// canonicalisation and the subsequent filesystem operation, a symlink
    /// could be swapped in. Full mitigation requires `openat()`-style
    /// fd-relative operations, which `std::fs` does not expose. The current
    /// approach is sufficient when the embedder controls the host filesystem
    /// (the typical case) but should not be relied upon for sandboxing
    /// untrusted code with concurrent host filesystem access.
    pub fn resolve_path(&self, dir_fd: u32, path: &str) -> Result<PathBuf, WasiErrno> {
        let (_, _, host_dir) = self
            .preopens
            .iter()
            .find(|(fd, _, _)| *fd == dir_fd)
            .ok_or(WasiErrno::BadF)?;

        let resolved = host_dir.join(path);

        let canonical_dir = host_dir.canonicalize().map_err(|_| WasiErrno::NoEnt)?;

        // Return the canonicalised path to close the TOCTOU window as
        // tightly as possible. For non-existent files, canonicalise the
        // parent and re-append the final component.
        let canonical = if resolved.exists() {
            resolved.canonicalize().map_err(|_| WasiErrno::NoEnt)?
        } else {
            let parent = resolved.parent().ok_or(WasiErrno::NoEnt)?;
            let canonical_parent = parent.canonicalize().map_err(|_| WasiErrno::NoEnt)?;
            let filename = resolved.file_name().ok_or(WasiErrno::Inval)?;
            canonical_parent.join(filename)
        };

        if !canonical.starts_with(&canonical_dir) {
            return Err(WasiErrno::Access);
        }

        Ok(canonical)
    }

    // === File descriptor management ===

    /// Allocate a new file descriptor, returning its number.
    ///
    /// Reuses closed (None) slots before appending to avoid unbounded growth.
    /// Allocate a new file descriptor, returning its number.
    ///
    /// Reuses closed (None) slots before appending. Returns ENFILE if the
    /// configured `max_fds` limit would be exceeded.
    pub fn allocate_fd(&self, fd: FileDescriptor) -> Result<u32, WasiErrno> {
        let mut fds = self.fds.lock().unwrap();
        // Reuse a closed slot if available (skip stdin/stdout/stderr)
        for (i, slot) in fds.iter_mut().enumerate().skip(3) {
            if slot.is_none() {
                *slot = Some(fd);
                return Ok(i as u32);
            }
        }
        if fds.len() as u32 >= self.max_fds {
            return Err(WasiErrno::Nfile);
        }
        let idx = fds.len();
        fds.push(Some(fd));
        Ok(idx as u32)
    }

    /// Close a file descriptor.
    ///
    /// Note: this allows closing stdin/stdout/stderr (fds 0-2), consistent
    /// with the WASI spec. The slot is set to None; a subsequent path_open
    /// will not reuse it (allocation starts from fd 3).
    pub fn close_fd(&self, fd: u32) -> Result<(), WasiErrno> {
        let mut fds = self.fds.lock().unwrap();
        let entry = fds.get_mut(fd as usize).ok_or(WasiErrno::BadF)?;
        if entry.is_none() {
            return Err(WasiErrno::BadF);
        }
        *entry = None;
        Ok(())
    }

    /// Seek within a file descriptor
    pub fn fd_seek(&self, fd: u32, offset: i64, whence: u32) -> Result<u64, WasiErrno> {
        let seek_from = match whence {
            0 => {
                if offset < 0 {
                    return Err(WasiErrno::Inval);
                }
                SeekFrom::Start(offset as u64)
            }
            1 => SeekFrom::Current(offset),
            2 => SeekFrom::End(offset),
            _ => return Err(WasiErrno::Inval),
        };

        let mut fds = self.fds.lock().unwrap();
        let fd_entry = fds
            .get_mut(fd as usize)
            .ok_or(WasiErrno::BadF)?
            .as_mut()
            .ok_or(WasiErrno::BadF)?;

        fd_entry.seek(seek_from).map_err(|_| WasiErrno::Spipe)
    }

    // === File descriptor I/O methods ===

    /// Read from a file descriptor into memory
    ///
    /// Returns the number of bytes read or a WASI errno.
    pub fn fd_read(&self, memory: &mut Memory, fd: u32, iovs_ptr: u32, iovs_len: u32) -> Result<usize, WasiErrno> {
        let iovecs = self.read_iovecs(memory, iovs_ptr, iovs_len)?;

        let mut fds = self.fds.lock().unwrap();
        let fd_entry = fds
            .get_mut(fd as usize)
            .ok_or(WasiErrno::BadF)?
            .as_mut()
            .ok_or(WasiErrno::BadF)?;

        if !fd_entry.readable {
            return Err(WasiErrno::BadF);
        }

        let mut total_read = 0;
        for (buf_ptr, buf_len) in iovecs {
            if buf_len == 0 {
                continue;
            }

            let mut temp_buf = vec![0u8; buf_len as usize];
            match fd_entry.read(&mut temp_buf) {
                Ok(0) => break, // EOF
                Ok(n) => {
                    memory
                        .write_bytes(buf_ptr, &temp_buf[..n])
                        .map_err(|_| WasiErrno::Fault)?;
                    total_read += n;
                    if n < buf_len as usize {
                        break; // Partial read
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => return Err(WasiErrno::Io),
            }
        }

        Ok(total_read)
    }

    /// Write from memory to a file descriptor
    ///
    /// Returns the number of bytes written or a WASI errno.
    pub fn fd_write(&self, memory: &Memory, fd: u32, iovs_ptr: u32, iovs_len: u32) -> Result<usize, WasiErrno> {
        let iovecs = self.read_iovecs(memory, iovs_ptr, iovs_len)?;

        let mut fds = self.fds.lock().unwrap();
        let fd_entry = fds
            .get_mut(fd as usize)
            .ok_or(WasiErrno::BadF)?
            .as_mut()
            .ok_or(WasiErrno::BadF)?;

        if !fd_entry.writable {
            return Err(WasiErrno::BadF);
        }

        let mut total_written = 0;
        for (buf_ptr, buf_len) in iovecs {
            if buf_len == 0 {
                continue;
            }

            let data = memory
                .read_bytes(buf_ptr, buf_len as usize)
                .map_err(|_| WasiErrno::Fault)?;

            match fd_entry.write(&data) {
                Ok(n) => {
                    total_written += n;
                    if n < buf_len as usize {
                        break; // Partial write
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => return Err(WasiErrno::Io),
            }
        }

        // Flush after writing — propagate errors so callers know if data
        // did not reach the underlying device.
        fd_entry.flush().map_err(|_| WasiErrno::Io)?;

        Ok(total_written)
    }

    /// Read iovec structures from memory.
    ///
    /// Each iovec is 8 bytes: 4-byte pointer (buf) + 4-byte length (buf_len).
    /// Capped by `max_iovecs` to prevent allocation bombs from malicious guests.
    ///
    /// See: <https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md#iovec>
    fn read_iovecs(&self, memory: &Memory, ptr: u32, len: u32) -> Result<Vec<(u32, u32)>, WasiErrno> {
        if len > self.max_iovecs {
            return Err(WasiErrno::Inval);
        }
        let mut iovecs = Vec::with_capacity(len as usize);
        for i in 0..len {
            let base = ptr
                .checked_add(i.checked_mul(8).ok_or(WasiErrno::Inval)?)
                .ok_or(WasiErrno::Fault)?;
            let buf_ptr = memory.read_u32(base).map_err(|_| WasiErrno::Fault)?;
            let buf_len = memory.read_u32(base + 4).map_err(|_| WasiErrno::Fault)?;
            iovecs.push((buf_ptr, buf_len));
        }
        Ok(iovecs)
    }
}

impl std::fmt::Debug for WasiContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasiContext")
            .field("args", &self.args)
            .field("env", &self.env)
            .field("exit_code", &self.exit_code)
            .finish()
    }
}

/// Builder for WasiContext
pub struct WasiContextBuilder {
    args: Vec<String>,
    env: Vec<String>,
    stdin: Option<Box<dyn Read + Send>>,
    stdout: Option<Box<dyn Write + Send>>,
    stderr: Option<Box<dyn Write + Send>>,
    preopens: Vec<(String, String)>,
    max_fds: u32,
    max_iovecs: u32,
}

impl WasiContextBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            args: Vec::new(),
            env: Vec::new(),
            stdin: None,
            stdout: None,
            stderr: None,
            preopens: Vec::new(),
            max_fds: DEFAULT_MAX_FDS,
            max_iovecs: DEFAULT_MAX_IOVECS,
        }
    }

    /// Set command line arguments
    pub fn args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args = args.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Set environment variables (as "NAME=value" strings)
    pub fn env(mut self, env: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.env = env.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Set stdin
    pub fn stdin(mut self, stdin: Box<dyn Read + Send>) -> Self {
        self.stdin = Some(stdin);
        self
    }

    /// Set stdout
    pub fn stdout(mut self, stdout: Box<dyn Write + Send>) -> Self {
        self.stdout = Some(stdout);
        self
    }

    /// Set stderr
    pub fn stderr(mut self, stderr: Box<dyn Write + Send>) -> Self {
        self.stderr = Some(stderr);
        self
    }

    /// Add a preopened directory
    ///
    /// `host_path` is the path on the host filesystem.
    /// `guest_path` is the path the WASI program sees.
    pub fn preopen_dir(mut self, host_path: impl Into<String>, guest_path: impl Into<String>) -> Self {
        self.preopens.push((host_path.into(), guest_path.into()));
        self
    }

    /// Set the maximum number of open file descriptors (default: 1024).
    ///
    /// Prevents a malicious guest from exhausting host memory by opening
    /// files in a loop. Returns ENFILE when the limit is reached.
    pub fn max_fds(mut self, max: u32) -> Self {
        self.max_fds = max;
        self
    }

    /// Set the maximum iovec count per read/write call (default: 1024).
    ///
    /// Prevents a malicious guest from triggering a huge allocation by
    /// passing a large iovs_len to fd_read/fd_write. Returns EINVAL when exceeded.
    pub fn max_iovecs(mut self, max: u32) -> Self {
        self.max_iovecs = max;
        self
    }

    /// Build the WasiContext
    pub fn build(self) -> WasiContext {
        let mut fds: Vec<Option<FileDescriptor>> = vec![
            // fd 0 = stdin
            self.stdin.map(FileDescriptor::new_reader),
            // fd 1 = stdout
            self.stdout.map(FileDescriptor::new_writer),
            // fd 2 = stderr
            self.stderr.map(FileDescriptor::new_writer),
        ];

        let mut preopens = Vec::new();
        for (host_path, guest_path) in &self.preopens {
            let path = PathBuf::from(host_path);
            if path.is_dir() {
                let fd_num = fds.len() as u32;
                fds.push(Some(FileDescriptor::new_directory(path.clone())));
                preopens.push((fd_num, guest_path.clone(), path));
            }
        }

        WasiContext {
            fds: Mutex::new(fds),
            preopens,
            args: self.args,
            env: self.env,
            exit_code: Mutex::new(None),
            monotonic_epoch: Instant::now(),
            max_fds: self.max_fds,
            max_iovecs: self.max_iovecs,
        }
    }
}

impl Default for WasiContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Memory;
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_builder_creates_context_with_args() {
        let ctx = WasiContext::builder().args(["program", "arg1", "arg2"]).build();

        assert_eq!(ctx.args(), ["program", "arg1", "arg2"]);
    }

    #[test]
    fn test_builder_creates_context_with_env() {
        let ctx = WasiContext::builder().env(["PATH=/usr/bin", "HOME=/home/user"]).build();

        assert_eq!(ctx.env(), ["PATH=/usr/bin", "HOME=/home/user"]);
    }

    #[test]
    fn test_read_write_u32() {
        let mut memory = Memory::new(1, None).unwrap();

        memory.write_u32(100, 0x12345678).unwrap();
        let value = memory.read_u32(100).unwrap();
        assert_eq!(value, 0x12345678);
    }

    #[test]
    fn test_read_write_bytes() {
        let mut memory = Memory::new(1, None).unwrap();

        let data = b"Hello, WASI!";
        memory.write_bytes(200, data).unwrap();
        let read_data = memory.read_bytes(200, data.len()).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_exit_code() {
        let ctx = WasiContext::builder().build();

        assert!(ctx.exit_code().is_none());

        ctx.set_exit_code(42);
        assert_eq!(ctx.exit_code(), Some(42));
    }

    #[test]
    fn test_fd_write_to_captured_stdout() {
        let stdout_buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let stdout_clone = stdout_buffer.clone();

        struct CapturedWriter(Arc<Mutex<Vec<u8>>>);
        impl Write for CapturedWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let ctx = WasiContext::builder()
            .stdout(Box::new(CapturedWriter(stdout_clone)))
            .build();

        let mut memory = Memory::new(1, None).unwrap();

        // Set up iovec in memory: {ptr: 100, len: 13}
        let message = b"Hello, World!";
        memory.write_bytes(100, message).unwrap();
        memory.write_u32(0, 100).unwrap(); // iovec.ptr
        memory.write_u32(4, 13).unwrap(); // iovec.len

        let result = ctx.fd_write(&memory, 1, 0, 1);
        assert_eq!(result, Ok(13));

        let output = stdout_buffer.lock().unwrap();
        assert_eq!(&*output, b"Hello, World!");
    }

    #[test]
    fn test_fd_read_from_mock_stdin() {
        let input = Cursor::new(b"test input\n".to_vec());

        let ctx = WasiContext::builder().stdin(Box::new(input)).build();

        let mut memory = Memory::new(1, None).unwrap();

        // Set up iovec in memory: {ptr: 100, len: 64}
        memory.write_u32(0, 100).unwrap(); // iovec.ptr
        memory.write_u32(4, 64).unwrap(); // iovec.len

        let result = ctx.fd_read(&mut memory, 0, 0, 1);
        assert_eq!(result, Ok(11)); // "test input\n" is 11 bytes

        let data = memory.read_bytes(100, 11).unwrap();
        assert_eq!(&data, b"test input\n");
    }

    #[test]
    fn test_fd_read_eof() {
        let input = Cursor::new(Vec::<u8>::new());

        let ctx = WasiContext::builder().stdin(Box::new(input)).build();

        let mut memory = Memory::new(1, None).unwrap();

        memory.write_u32(0, 100).unwrap();
        memory.write_u32(4, 64).unwrap();

        let result = ctx.fd_read(&mut memory, 0, 0, 1);
        assert_eq!(result, Ok(0));
    }

    #[test]
    fn test_bad_fd_returns_error() {
        let ctx = WasiContext::builder().build();

        let mut memory = Memory::new(1, None).unwrap();

        memory.write_u32(0, 100).unwrap();
        memory.write_u32(4, 64).unwrap();

        let result = ctx.fd_read(&mut memory, 99, 0, 1);
        assert_eq!(result, Err(WasiErrno::BadF));
    }

    #[test]
    fn test_max_fds_limit() {
        // Set limit to 5 (stdin/stdout/stderr = 3, so only 2 more allowed)
        let ctx = WasiContext::builder().max_fds(5).build();

        let fd1 = ctx.allocate_fd(FileDescriptor::new_reader(Box::new(std::io::empty())));
        assert!(fd1.is_ok(), "First allocation should succeed");

        let fd2 = ctx.allocate_fd(FileDescriptor::new_reader(Box::new(std::io::empty())));
        assert!(fd2.is_ok(), "Second allocation should succeed");

        let fd3 = ctx.allocate_fd(FileDescriptor::new_reader(Box::new(std::io::empty())));
        assert_eq!(fd3, Err(WasiErrno::Nfile), "Third allocation should hit limit");
    }

    #[test]
    fn test_max_fds_reuse_after_close() {
        let ctx = WasiContext::builder().max_fds(5).build();

        let fd1 = ctx
            .allocate_fd(FileDescriptor::new_reader(Box::new(std::io::empty())))
            .unwrap();
        let _fd2 = ctx
            .allocate_fd(FileDescriptor::new_reader(Box::new(std::io::empty())))
            .unwrap();

        // At limit now. Close fd1, then allocate again -- should reuse the slot.
        ctx.close_fd(fd1).unwrap();
        let fd3 = ctx.allocate_fd(FileDescriptor::new_reader(Box::new(std::io::empty())));
        assert!(fd3.is_ok(), "Should reuse closed slot");
        assert_eq!(fd3.unwrap(), fd1, "Should reuse the same fd number");
    }

    #[test]
    fn test_max_iovecs_limit() {
        let ctx = WasiContext::builder()
            .max_iovecs(2)
            .stdin(Box::new(std::io::empty()))
            .build();

        let mut memory = Memory::new(1, None).unwrap();
        // Set up 3 iovecs (over the limit of 2)
        for i in 0..3u32 {
            memory.write_u32(i * 8, 100).unwrap();
            memory.write_u32(i * 8 + 4, 10).unwrap();
        }

        let result = ctx.fd_read(&mut memory, 0, 0, 3);
        assert_eq!(result, Err(WasiErrno::Inval), "Should reject excessive iovec count");

        // 2 iovecs should be fine
        let result = ctx.fd_read(&mut memory, 0, 0, 2);
        assert!(result.is_ok(), "2 iovecs should be within limit");
    }

    #[test]
    fn test_iovec_pointer_overflow() {
        let ctx = WasiContext::builder().stdin(Box::new(std::io::empty())).build();

        let mut memory = Memory::new(1, None).unwrap();
        // Attempt to read iovecs from near the end of u32 space
        // ptr = u32::MAX - 4, len = 2 would overflow ptr + i*8
        let result = ctx.fd_read(&mut memory, 0, u32::MAX - 4, 2);
        assert!(result.is_err(), "Should detect pointer arithmetic overflow");
    }
}
