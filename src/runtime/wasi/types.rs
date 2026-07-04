//! WASI type definitions, error codes, and constants
//!
//! This module defines WASI preview1 error codes, file types, flag constants,
//! and struct sizes.
//!
//! # Specification
//!
//! See: <https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md>

// === Rights (64-bit flags) ===
// See: https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md#rights

pub const RIGHT_FD_DATASYNC: u64 = 1 << 0;
pub const RIGHT_FD_READ: u64 = 1 << 1;
pub const RIGHT_FD_SEEK: u64 = 1 << 2;
pub const RIGHT_FD_FDSTAT_SET_FLAGS: u64 = 1 << 3;
pub const RIGHT_FD_SYNC: u64 = 1 << 4;
pub const RIGHT_FD_TELL: u64 = 1 << 5;
pub const RIGHT_FD_WRITE: u64 = 1 << 6;
pub const RIGHT_FD_ADVISE: u64 = 1 << 7;
pub const RIGHT_FD_ALLOCATE: u64 = 1 << 8;
pub const RIGHT_PATH_CREATE_DIRECTORY: u64 = 1 << 9;
pub const RIGHT_PATH_CREATE_FILE: u64 = 1 << 10;
pub const RIGHT_PATH_LINK_SOURCE: u64 = 1 << 11;
pub const RIGHT_PATH_LINK_TARGET: u64 = 1 << 12;
pub const RIGHT_PATH_OPEN: u64 = 1 << 13;
pub const RIGHT_FD_READDIR: u64 = 1 << 14;
pub const RIGHT_PATH_READLINK: u64 = 1 << 15;
pub const RIGHT_PATH_RENAME_SOURCE: u64 = 1 << 16;
pub const RIGHT_PATH_RENAME_TARGET: u64 = 1 << 17;
pub const RIGHT_PATH_FILESTAT_GET: u64 = 1 << 18;
pub const RIGHT_PATH_FILESTAT_SET_SIZE: u64 = 1 << 19;
pub const RIGHT_PATH_FILESTAT_SET_TIMES: u64 = 1 << 20;
pub const RIGHT_FD_FILESTAT_GET: u64 = 1 << 21;
pub const RIGHT_FD_FILESTAT_SET_SIZE: u64 = 1 << 22;
pub const RIGHT_FD_FILESTAT_SET_TIMES: u64 = 1 << 23;
pub const RIGHT_PATH_SYMLINK: u64 = 1 << 24;
pub const RIGHT_PATH_REMOVE_DIRECTORY: u64 = 1 << 25;
pub const RIGHT_PATH_UNLINK_FILE: u64 = 1 << 26;
pub const RIGHT_POLL_FD_READWRITE: u64 = 1 << 27;
pub const RIGHT_SOCK_SHUTDOWN: u64 = 1 << 28;
pub const RIGHT_SOCK_ACCEPT: u64 = 1 << 29;

/// All defined rights (bits 0-29)
pub const RIGHTS_ALL: u64 = (1 << 30) - 1;

// === oflags (16-bit flags) ===
// See: https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md#oflags

/// Create file if it does not exist
pub const OFLAG_CREAT: u16 = 1 << 0;
/// Fail if not a directory
pub const OFLAG_DIRECTORY: u16 = 1 << 1;
/// Fail if file already exists
pub const OFLAG_EXCL: u16 = 1 << 2;
/// Truncate file to size 0
pub const OFLAG_TRUNC: u16 = 1 << 3;

// === fdflags (16-bit flags) ===
// See: https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md#fdflags

/// Append mode
pub const FDFLAG_APPEND: u16 = 1 << 0;
/// Synchronised I/O data integrity
pub const FDFLAG_DSYNC: u16 = 1 << 1;
/// Non-blocking mode
pub const FDFLAG_NONBLOCK: u16 = 1 << 2;
/// Synchronised read I/O
pub const FDFLAG_RSYNC: u16 = 1 << 3;
/// Synchronised I/O file integrity
pub const FDFLAG_SYNC: u16 = 1 << 4;

// === lookupflags (32-bit flags) ===
// See: https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md#lookupflags

/// Follow symbolic links when resolving paths
pub const LOOKUP_SYMLINK_FOLLOW: u32 = 1 << 0;

// === fstflags (16-bit flags) ===
// See: https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md#fstflags

/// Set access time to explicit value
pub const FSTFLAG_ATIM: u16 = 1 << 0;
/// Set access time to current time
pub const FSTFLAG_ATIM_NOW: u16 = 1 << 1;
/// Set modification time to explicit value
pub const FSTFLAG_MTIM: u16 = 1 << 2;
/// Set modification time to current time
pub const FSTFLAG_MTIM_NOW: u16 = 1 << 3;

// === whence ===
// See: https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md#whence

/// Seek relative to start of file
pub const WHENCE_SET: u32 = 0;
/// Seek relative to current position
pub const WHENCE_CUR: u32 = 1;
/// Seek relative to end of file
pub const WHENCE_END: u32 = 2;

// === clockid ===
// See: https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md#clockid

/// Wall-clock time (epoch: 1970-01-01T00:00:00Z)
pub const CLOCKID_REALTIME: u32 = 0;
/// Monotonic time (arbitrary epoch, not settable)
pub const CLOCKID_MONOTONIC: u32 = 1;
/// Process CPU time
pub const CLOCKID_PROCESS_CPUTIME: u32 = 2;
/// Thread CPU time
pub const CLOCKID_THREAD_CPUTIME: u32 = 3;

// === preopentype ===
// See: https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md#preopentype

/// Preopened directory
pub const PREOPENTYPE_DIR: u32 = 0;

// === Struct sizes and field offsets ===

/// Size of the prestat struct (variant tag + prestat_dir)
pub const PRESTAT_SIZE: u32 = 8;

/// Size of the fdstat struct
pub const FDSTAT_SIZE: u32 = 24;
// fdstat field offsets:
//   0: fs_filetype (u8)
//   2: fs_flags (u16, fdflags)
//   8: fs_rights_base (u64)
//  16: fs_rights_inheriting (u64)

/// Size of the filestat struct
pub const FILESTAT_SIZE: u32 = 64;
// filestat field offsets:
//   0: dev (u64)
//   8: ino (u64)
//  16: filetype (u8, padded to 8 bytes)
//  24: nlink (u64)
//  32: size (u64)
//  40: atim (u64, nanoseconds)
//  48: mtim (u64, nanoseconds)
//  56: ctim (u64, nanoseconds)

/// Size of the dirent header (before the variable-length name)
pub const DIRENT_HEADER_SIZE: u32 = 24;
// dirent field offsets:
//   0: d_next (u64, cookie of next entry)
//   8: d_ino (u64)
//  16: d_namlen (u32)
//  20: d_type (u8, filetype)
// followed by d_namlen bytes of name (NOT null-terminated)

/// WASI error codes (errno values)
///
/// These are the standard WASI error codes as defined in the WASI preview1 specification.
///
/// See: <https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md#errno>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WasiErrno {
    /// No error occurred
    Success = 0,
    /// Argument list too long
    TooBig = 1,
    /// Permission denied
    Access = 2,
    /// Address in use
    AddrInUse = 3,
    /// Address not available
    AddrNotAvail = 4,
    /// Address family not supported
    AfNoSupport = 5,
    /// Resource unavailable, or operation would block
    Again = 6,
    /// Connection already in progress
    Already = 7,
    /// Bad file descriptor
    BadF = 8,
    /// Bad message
    BadMsg = 9,
    /// Device or resource busy
    Busy = 10,
    /// Operation cancelled
    Cancelled = 11,
    /// No child processes
    Child = 12,
    /// Connection aborted
    ConnAborted = 13,
    /// Connection refused
    ConnRefused = 14,
    /// Connection reset
    ConnReset = 15,
    /// Resource deadlock would occur
    Deadlk = 16,
    /// Destination address required
    DestAddrReq = 17,
    /// Mathematics argument out of domain of function
    Dom = 18,
    /// Reserved
    Dquot = 19,
    /// File exists
    Exist = 20,
    /// Bad address
    Fault = 21,
    /// File too large
    Fbig = 22,
    /// Host is unreachable
    HostUnreach = 23,
    /// Identifier removed
    Idrm = 24,
    /// Illegal byte sequence
    Ilseq = 25,
    /// Operation in progress
    InProgress = 26,
    /// Interrupted function
    Intr = 27,
    /// Invalid argument
    Inval = 28,
    /// I/O error
    Io = 29,
    /// Socket is connected
    IsConn = 30,
    /// Is a directory
    IsDir = 31,
    /// Too many levels of symbolic links
    Loop = 32,
    /// File descriptor value too large
    Mfile = 33,
    /// Too many links
    Mlink = 34,
    /// Message too large
    MsgSize = 35,
    /// Reserved
    Multihop = 36,
    /// Filename too long
    NameTooLong = 37,
    /// Network is down
    NetDown = 38,
    /// Connection aborted by network
    NetReset = 39,
    /// Network unreachable
    NetUnreach = 40,
    /// Too many files open in system
    Nfile = 41,
    /// No buffer space available
    NoBufs = 42,
    /// No such device
    NoDev = 43,
    /// No such file or directory
    NoEnt = 44,
    /// Executable file format error
    NoExec = 45,
    /// No locks available
    NoLck = 46,
    /// Reserved
    NoLink = 47,
    /// Not enough space
    NoMem = 48,
    /// No message of the desired type
    NoMsg = 49,
    /// Protocol not available
    NoProtoOpt = 50,
    /// No space left on device
    NoSpc = 51,
    /// Function not supported
    NoSys = 52,
    /// The socket is not connected
    NotConn = 53,
    /// Not a directory or a symbolic link to a directory
    NotDir = 54,
    /// Directory not empty
    NotEmpty = 55,
    /// State not recoverable
    NotRecoverable = 56,
    /// Not a socket
    NotSock = 57,
    /// Not supported, or operation not supported on socket
    NotSup = 58,
    /// Inappropriate I/O control operation
    NoTty = 59,
    /// No such device or address
    Nxio = 60,
    /// Value too large to be stored in data type
    Overflow = 61,
    /// Previous owner died
    OwnerDead = 62,
    /// Operation not permitted
    Perm = 63,
    /// Broken pipe
    Pipe = 64,
    /// Protocol error
    Proto = 65,
    /// Protocol not supported
    ProtoNoSupport = 66,
    /// Protocol wrong type for socket
    Prototype = 67,
    /// Result too large
    Range = 68,
    /// Read-only file system
    Rofs = 69,
    /// Invalid seek
    Spipe = 70,
    /// No such process
    Srch = 71,
    /// Reserved
    Stale = 72,
    /// Connection timed out
    TimedOut = 73,
    /// Text file busy
    TxtBsy = 74,
    /// Cross-device link
    Xdev = 75,
    /// Extension: capabilities insufficient
    NotCapable = 76,
}

/// WASI file types
///
/// See: <https://github.com/WebAssembly/WASI/blob/wasi-0.1/preview1/docs.md#filetype>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WasiFileType {
    /// Unknown or unrecognised file type
    Unknown = 0,
    /// Block device
    BlockDevice = 1,
    /// Character device (e.g. stdin/stdout/stderr)
    CharacterDevice = 2,
    /// Directory
    Directory = 3,
    /// Regular file
    RegularFile = 4,
    /// Datagram socket
    SocketDgram = 5,
    /// Stream socket
    SocketStream = 6,
    /// Symbolic link
    SymbolicLink = 7,
}

impl WasiErrno {
    /// Convert to i32 for WASI return values
    pub fn as_i32(self) -> i32 {
        self as u32 as i32
    }

    /// Convert to u32 for WASI return values
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

impl From<WasiErrno> for i32 {
    fn from(errno: WasiErrno) -> i32 {
        errno as u32 as i32
    }
}

impl From<WasiErrno> for u32 {
    fn from(errno: WasiErrno) -> u32 {
        errno as u32
    }
}

/// Map a std::io::Error to the closest WasiErrno
pub fn errno_from_io_error(e: &std::io::Error) -> WasiErrno {
    match e.kind() {
        std::io::ErrorKind::NotFound => WasiErrno::NoEnt,
        std::io::ErrorKind::PermissionDenied => WasiErrno::Access,
        std::io::ErrorKind::AlreadyExists => WasiErrno::Exist,
        std::io::ErrorKind::InvalidInput => WasiErrno::Inval,
        std::io::ErrorKind::BrokenPipe => WasiErrno::Pipe,
        _ => WasiErrno::Io,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_errno_values() {
        assert_eq!(WasiErrno::Success.as_u32(), 0);
        assert_eq!(WasiErrno::BadF.as_u32(), 8);
        assert_eq!(WasiErrno::Inval.as_u32(), 28);
        assert_eq!(WasiErrno::NoSys.as_u32(), 52);
    }

    #[test]
    fn test_errno_as_i32() {
        assert_eq!(WasiErrno::Success.as_i32(), 0);
        assert_eq!(WasiErrno::BadF.as_i32(), 8);
    }

    #[test]
    fn test_errno_into_u32() {
        let errno: u32 = WasiErrno::Success.into();
        assert_eq!(errno, 0);
    }

    #[test]
    fn test_errno_into_i32() {
        let errno: i32 = WasiErrno::BadF.into();
        assert_eq!(errno, 8);
    }

    #[test]
    fn test_filetype_values() {
        assert_eq!(WasiFileType::Unknown as u8, 0);
        assert_eq!(WasiFileType::BlockDevice as u8, 1);
        assert_eq!(WasiFileType::CharacterDevice as u8, 2);
        assert_eq!(WasiFileType::Directory as u8, 3);
        assert_eq!(WasiFileType::RegularFile as u8, 4);
        assert_eq!(WasiFileType::SocketDgram as u8, 5);
        assert_eq!(WasiFileType::SocketStream as u8, 6);
        assert_eq!(WasiFileType::SymbolicLink as u8, 7);
    }

    #[test]
    fn test_rights_all_covers_all_bits() {
        // RIGHTS_ALL should include all 30 defined right bits (0-29)
        assert_eq!(RIGHTS_ALL, 0x3fff_ffff);
        assert_eq!(RIGHTS_ALL & RIGHT_SOCK_ACCEPT, RIGHT_SOCK_ACCEPT); // bit 29, the highest
        assert_eq!(RIGHTS_ALL & (1 << 30), 0); // bit 30 should not be set
    }

    #[test]
    fn test_oflags() {
        assert_eq!(OFLAG_CREAT, 1);
        assert_eq!(OFLAG_DIRECTORY, 2);
        assert_eq!(OFLAG_EXCL, 4);
        assert_eq!(OFLAG_TRUNC, 8);
    }

    #[test]
    fn test_fdflags() {
        assert_eq!(FDFLAG_APPEND, 1);
        assert_eq!(FDFLAG_DSYNC, 2);
        assert_eq!(FDFLAG_NONBLOCK, 4);
        assert_eq!(FDFLAG_RSYNC, 8);
        assert_eq!(FDFLAG_SYNC, 16);
    }

    #[test]
    fn test_io_error_mapping() {
        let not_found = std::io::Error::new(std::io::ErrorKind::NotFound, "");
        assert_eq!(errno_from_io_error(&not_found), WasiErrno::NoEnt);

        let perm = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "");
        assert_eq!(errno_from_io_error(&perm), WasiErrno::Access);

        let exists = std::io::Error::new(std::io::ErrorKind::AlreadyExists, "");
        assert_eq!(errno_from_io_error(&exists), WasiErrno::Exist);

        let other = std::io::Error::other("");
        assert_eq!(errno_from_io_error(&other), WasiErrno::Io);
    }
}
