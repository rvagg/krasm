//! WebAssembly linear memory implementation
//!
//! This module provides safe, bounds-checked access to WebAssembly linear memory.
//! Security is the primary concern - all access operations are carefully validated.
//!
//! Memory layout follows the WebAssembly specification:
//! - Page size: 64KB (65,536 bytes)
//! - Address space: 32-bit (max 4GB)
//! - Bounds checking: Required for all accesses
//! - Out-of-bounds access: Traps (runtime error)

use super::{RuntimeError, Trap};

/// WebAssembly page size in bytes (64KB)
pub const PAGE_SIZE: usize = 65536;

/// Maximum number of pages (2^16 = 64K pages = 4GB total)
pub const MAX_PAGES: u32 = 65536;

/// Maximum memory size in bytes (4GB)
pub const MAX_MEMORY_SIZE: usize = MAX_PAGES as usize * PAGE_SIZE;

/// A WebAssembly linear memory instance
///
/// Security considerations:
/// - All access is bounds-checked before touching the underlying data
/// - Growth operations validate against maximum limits
/// - No unsafe code or unchecked access
#[derive(Debug)]
pub struct Memory {
    /// The actual memory data
    data: Vec<u8>,

    /// Current size in pages
    current_pages: u32,

    /// Maximum size in pages (None = default max)
    max_pages: Option<u32>,
}

impl Memory {
    /// Create a new memory instance with the given limits
    ///
    /// # Arguments
    /// * `initial_pages` - Initial size in pages
    /// * `max_pages` - Optional maximum size in pages
    ///
    /// # Errors
    /// - Initial pages exceeds maximum
    /// - Initial pages exceeds system maximum
    pub fn new(initial_pages: u32, max_pages: Option<u32>) -> Result<Self, RuntimeError> {
        // Validate initial size
        if initial_pages > MAX_PAGES {
            return Err(RuntimeError::MemoryError(format!(
                "Initial memory size {initial_pages} pages exceeds maximum {MAX_PAGES} pages"
            )));
        }

        // Validate max against initial
        if let Some(max) = max_pages {
            if initial_pages > max {
                return Err(RuntimeError::MemoryError(format!(
                    "Initial size {initial_pages} pages exceeds specified maximum {max} pages"
                )));
            }
            if max > MAX_PAGES {
                return Err(RuntimeError::MemoryError(format!(
                    "Maximum size {max} pages exceeds system maximum {MAX_PAGES} pages"
                )));
            }
        }

        // Calculate initial size in bytes
        let initial_bytes = initial_pages as usize * PAGE_SIZE;

        // Allocate memory - initialised to zero per WebAssembly spec
        let data = vec![0u8; initial_bytes];

        Ok(Memory {
            data,
            current_pages: initial_pages,
            max_pages,
        })
    }

    /// Get the current memory size in pages
    pub fn size(&self) -> u32 {
        self.current_pages
    }

    /// Get the maximum memory size in pages (None = unbounded)
    pub fn max_pages(&self) -> Option<u32> {
        self.max_pages
    }

    /// Grow memory by the specified number of pages
    ///
    /// Returns the previous size in pages, or -1 if growth fails
    ///
    /// # Security
    /// - Validates against maximum limits
    /// - Checks for integer overflow
    /// - Ensures new allocation doesn't exceed system limits
    pub fn grow(&mut self, delta_pages: u32) -> i32 {
        let current = self.current_pages;

        // Check for integer overflow
        let Some(new_pages) = current.checked_add(delta_pages) else {
            return -1; // Growth would overflow
        };

        // Check against maximum pages
        let effective_max = self.max_pages.unwrap_or(MAX_PAGES);
        if new_pages > effective_max {
            return -1; // Would exceed maximum
        }

        // Calculate new size in bytes
        let new_bytes = new_pages as usize * PAGE_SIZE;

        // Attempt to grow the vector
        // This can fail if system is out of memory
        match self.data.try_reserve(new_bytes - self.data.len()) {
            Ok(()) => {
                // Resize and zero-initialise new pages
                self.data.resize(new_bytes, 0);
                self.current_pages = new_pages;
                current as i32
            }
            Err(_) => -1, // Out of memory
        }
    }

    /// Check if an access at the given address with the given size is valid
    ///
    /// # Security
    /// This is the critical bounds checking function. It must:
    /// - Check for integer overflow in address calculation
    /// - Ensure the entire access range is within bounds
    /// - Handle edge cases (e.g., size = 0)
    #[inline]
    fn check_bounds(&self, addr: u32, size: usize) -> Result<usize, Trap> {
        let addr = addr as usize;

        // Check for integer overflow in the access range
        let end = addr.checked_add(size).ok_or(Trap::MemoryOutOfBounds)?;

        // Check if access is within bounds
        if end > self.data.len() {
            return Err(Trap::MemoryOutOfBounds);
        }

        Ok(addr)
    }

    /// Read a u8 from memory
    pub fn read_u8(&self, addr: u32) -> Result<u8, Trap> {
        let addr = self.check_bounds(addr, 1)?;
        Ok(self.data[addr])
    }

    /// Read a u16 from memory (little-endian)
    pub fn read_u16(&self, addr: u32) -> Result<u16, Trap> {
        let addr = self.check_bounds(addr, 2)?;
        Ok(u16::from_le_bytes([self.data[addr], self.data[addr + 1]]))
    }

    /// Read a u32 from memory (little-endian)
    pub fn read_u32(&self, addr: u32) -> Result<u32, Trap> {
        let addr = self.check_bounds(addr, 4)?;
        Ok(u32::from_le_bytes([
            self.data[addr],
            self.data[addr + 1],
            self.data[addr + 2],
            self.data[addr + 3],
        ]))
    }

    /// Read a u64 from memory (little-endian)
    pub fn read_u64(&self, addr: u32) -> Result<u64, Trap> {
        let addr = self.check_bounds(addr, 8)?;
        Ok(u64::from_le_bytes([
            self.data[addr],
            self.data[addr + 1],
            self.data[addr + 2],
            self.data[addr + 3],
            self.data[addr + 4],
            self.data[addr + 5],
            self.data[addr + 6],
            self.data[addr + 7],
        ]))
    }

    /// Read an i8 from memory
    pub fn read_i8(&self, addr: u32) -> Result<i8, Trap> {
        Ok(self.read_u8(addr)? as i8)
    }

    /// Read an i16 from memory (little-endian)
    pub fn read_i16(&self, addr: u32) -> Result<i16, Trap> {
        Ok(self.read_u16(addr)? as i16)
    }

    /// Read an i32 from memory (little-endian)
    pub fn read_i32(&self, addr: u32) -> Result<i32, Trap> {
        Ok(self.read_u32(addr)? as i32)
    }

    /// Read an i64 from memory (little-endian)
    pub fn read_i64(&self, addr: u32) -> Result<i64, Trap> {
        Ok(self.read_u64(addr)? as i64)
    }

    /// Read an f32 from memory (little-endian)
    pub fn read_f32(&self, addr: u32) -> Result<f32, Trap> {
        Ok(f32::from_le_bytes(self.read_u32(addr)?.to_le_bytes()))
    }

    /// Read an f64 from memory (little-endian)
    pub fn read_f64(&self, addr: u32) -> Result<f64, Trap> {
        Ok(f64::from_le_bytes(self.read_u64(addr)?.to_le_bytes()))
    }

    /// Write a u8 to memory
    pub fn write_u8(&mut self, addr: u32, value: u8) -> Result<(), Trap> {
        let addr = self.check_bounds(addr, 1)?;
        self.data[addr] = value;
        Ok(())
    }

    /// Write a u16 to memory (little-endian)
    pub fn write_u16(&mut self, addr: u32, value: u16) -> Result<(), Trap> {
        let addr = self.check_bounds(addr, 2)?;
        let bytes = value.to_le_bytes();
        self.data[addr] = bytes[0];
        self.data[addr + 1] = bytes[1];
        Ok(())
    }

    /// Write a u32 to memory (little-endian)
    pub fn write_u32(&mut self, addr: u32, value: u32) -> Result<(), Trap> {
        let addr = self.check_bounds(addr, 4)?;
        let bytes = value.to_le_bytes();
        self.data[addr] = bytes[0];
        self.data[addr + 1] = bytes[1];
        self.data[addr + 2] = bytes[2];
        self.data[addr + 3] = bytes[3];
        Ok(())
    }

    /// Write a u64 to memory (little-endian)
    pub fn write_u64(&mut self, addr: u32, value: u64) -> Result<(), Trap> {
        let addr = self.check_bounds(addr, 8)?;
        let bytes = value.to_le_bytes();
        for (i, &byte) in bytes.iter().enumerate() {
            self.data[addr + i] = byte;
        }
        Ok(())
    }

    /// Write an i8 to memory
    pub fn write_i8(&mut self, addr: u32, value: i8) -> Result<(), Trap> {
        self.write_u8(addr, value as u8)
    }

    /// Write an i16 to memory (little-endian)
    pub fn write_i16(&mut self, addr: u32, value: i16) -> Result<(), Trap> {
        self.write_u16(addr, value as u16)
    }

    /// Write an i32 to memory (little-endian)
    pub fn write_i32(&mut self, addr: u32, value: i32) -> Result<(), Trap> {
        self.write_u32(addr, value as u32)
    }

    /// Write an i64 to memory (little-endian)
    pub fn write_i64(&mut self, addr: u32, value: i64) -> Result<(), Trap> {
        self.write_u64(addr, value as u64)
    }

    /// Write an f32 to memory (little-endian)
    pub fn write_f32(&mut self, addr: u32, value: f32) -> Result<(), Trap> {
        self.write_u32(addr, u32::from_le_bytes(value.to_le_bytes()))
    }

    /// Write an f64 to memory (little-endian)
    pub fn write_f64(&mut self, addr: u32, value: f64) -> Result<(), Trap> {
        self.write_u64(addr, u64::from_le_bytes(value.to_le_bytes()))
    }

    /// Read a slice of bytes from memory
    ///
    /// # Security
    /// - Bounds checks the entire range
    /// - Returns a copy to prevent aliasing issues
    pub fn read_bytes(&self, addr: u32, len: usize) -> Result<Vec<u8>, Trap> {
        let addr = self.check_bounds(addr, len)?;
        Ok(self.data[addr..addr + len].to_vec())
    }

    /// Write a slice of bytes to memory
    ///
    /// # Security
    /// - Bounds checks the entire range
    /// - Copies data to prevent aliasing issues
    pub fn write_bytes(&mut self, addr: u32, bytes: &[u8]) -> Result<(), Trap> {
        let addr = self.check_bounds(addr, bytes.len())?;
        self.data[addr..addr + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_creation() {
        // Valid creation
        let mem = Memory::new(1, None).unwrap();
        assert_eq!(mem.size(), 1);
        assert_eq!(mem.data.len(), PAGE_SIZE);

        // With max pages
        let mem = Memory::new(1, Some(10)).unwrap();
        assert_eq!(mem.size(), 1);
        assert_eq!(mem.max_pages, Some(10));
    }

    #[test]
    fn test_memory_creation_errors() {
        // Initial exceeds max
        assert!(Memory::new(10, Some(5)).is_err());

        // Initial exceeds system max
        assert!(Memory::new(MAX_PAGES + 1, None).is_err());

        // Max exceeds system max
        assert!(Memory::new(1, Some(MAX_PAGES + 1)).is_err());
    }

    #[test]
    fn test_memory_grow() {
        let mut mem = Memory::new(1, Some(10)).unwrap();

        // Successful grow
        assert_eq!(mem.grow(2), 1); // Returns previous size
        assert_eq!(mem.size(), 3);
        assert_eq!(mem.data.len(), 3 * PAGE_SIZE);

        // Grow to max
        assert_eq!(mem.grow(7), 3);
        assert_eq!(mem.size(), 10);

        // Attempt to grow beyond max
        assert_eq!(mem.grow(1), -1);
        assert_eq!(mem.size(), 10); // Unchanged
    }

    #[test]
    fn test_memory_grow_overflow() {
        let mut mem = Memory::new(1, None).unwrap();

        // Attempt to grow by u32::MAX
        assert_eq!(mem.grow(u32::MAX), -1);
        assert_eq!(mem.size(), 1); // Unchanged
    }

    #[test]
    fn test_bounds_checking() {
        let mem = Memory::new(1, None).unwrap();

        // Valid accesses
        assert!(mem.check_bounds(0, 1).is_ok());
        assert!(mem.check_bounds(PAGE_SIZE as u32 - 1, 1).is_ok());
        assert!(mem.check_bounds(0, PAGE_SIZE).is_ok());

        // Invalid accesses
        assert!(mem.check_bounds(PAGE_SIZE as u32, 1).is_err());
        assert!(mem.check_bounds(PAGE_SIZE as u32 - 1, 2).is_err());
        assert!(mem.check_bounds(u32::MAX, 1).is_err());

        // Overflow checks
        assert!(mem.check_bounds(u32::MAX - 1, 2).is_err());
        assert!(mem.check_bounds(u32::MAX, usize::MAX).is_err());
    }

    #[test]
    fn test_read_write_u8() {
        let mut mem = Memory::new(1, None).unwrap();

        // Write and read back
        mem.write_u8(100, 42).unwrap();
        assert_eq!(mem.read_u8(100).unwrap(), 42);

        // Boundary test
        mem.write_u8(PAGE_SIZE as u32 - 1, 255).unwrap();
        assert_eq!(mem.read_u8(PAGE_SIZE as u32 - 1).unwrap(), 255);

        // Out of bounds
        assert!(mem.write_u8(PAGE_SIZE as u32, 1).is_err());
        assert!(mem.read_u8(PAGE_SIZE as u32).is_err());
    }

    #[test]
    fn test_read_write_u32() {
        let mut mem = Memory::new(1, None).unwrap();

        // Write and read back
        mem.write_u32(100, 0x12345678).unwrap();
        assert_eq!(mem.read_u32(100).unwrap(), 0x12345678);

        // Check little-endian storage
        assert_eq!(mem.read_u8(100).unwrap(), 0x78);
        assert_eq!(mem.read_u8(101).unwrap(), 0x56);
        assert_eq!(mem.read_u8(102).unwrap(), 0x34);
        assert_eq!(mem.read_u8(103).unwrap(), 0x12);

        // Boundary test
        mem.write_u32(PAGE_SIZE as u32 - 4, 0xDEADBEEF).unwrap();
        assert_eq!(mem.read_u32(PAGE_SIZE as u32 - 4).unwrap(), 0xDEADBEEF);

        // Out of bounds
        assert!(mem.write_u32(PAGE_SIZE as u32 - 3, 1).is_err());
        assert!(mem.read_u32(PAGE_SIZE as u32 - 3).is_err());
    }

    #[test]
    fn test_read_write_u64() {
        let mut mem = Memory::new(1, None).unwrap();

        // Write and read back
        mem.write_u64(100, 0x123456789ABCDEF0).unwrap();
        assert_eq!(mem.read_u64(100).unwrap(), 0x123456789ABCDEF0);

        // Boundary test
        mem.write_u64(PAGE_SIZE as u32 - 8, u64::MAX).unwrap();
        assert_eq!(mem.read_u64(PAGE_SIZE as u32 - 8).unwrap(), u64::MAX);

        // Out of bounds
        assert!(mem.write_u64(PAGE_SIZE as u32 - 7, 1).is_err());
        assert!(mem.read_u64(PAGE_SIZE as u32 - 7).is_err());
    }

    #[test]
    fn test_signed_integers() {
        let mut mem = Memory::new(1, None).unwrap();

        // i8
        mem.write_i8(0, -128).unwrap();
        assert_eq!(mem.read_i8(0).unwrap(), -128);
        mem.write_i8(1, 127).unwrap();
        assert_eq!(mem.read_i8(1).unwrap(), 127);

        // i32
        mem.write_i32(10, -2147483648).unwrap();
        assert_eq!(mem.read_i32(10).unwrap(), -2147483648);
        mem.write_i32(20, 2147483647).unwrap();
        assert_eq!(mem.read_i32(20).unwrap(), 2147483647);
    }

    #[test]
    fn test_floating_point() {
        let mut mem = Memory::new(1, None).unwrap();

        // f32
        mem.write_f32(0, std::f32::consts::PI).unwrap();
        assert_eq!(mem.read_f32(0).unwrap(), std::f32::consts::PI);

        // f64
        mem.write_f64(10, std::f64::consts::E).unwrap();
        assert_eq!(mem.read_f64(10).unwrap(), std::f64::consts::E);

        // Special values
        mem.write_f32(20, f32::INFINITY).unwrap();
        assert!(mem.read_f32(20).unwrap().is_infinite());
        mem.write_f32(24, f32::NAN).unwrap();
        assert!(mem.read_f32(24).unwrap().is_nan());
    }

    #[test]
    fn test_bytes_operations() {
        let mut mem = Memory::new(1, None).unwrap();

        // Write and read bytes
        let data = vec![1, 2, 3, 4, 5];
        mem.write_bytes(100, &data).unwrap();
        let read = mem.read_bytes(100, 5).unwrap();
        assert_eq!(read, data);

        // Empty slice
        mem.write_bytes(200, &[]).unwrap();
        let read = mem.read_bytes(200, 0).unwrap();
        assert!(read.is_empty());

        // Boundary test
        let data = vec![0xFF; 10];
        mem.write_bytes(PAGE_SIZE as u32 - 10, &data).unwrap();
        let read = mem.read_bytes(PAGE_SIZE as u32 - 10, 10).unwrap();
        assert_eq!(read, data);

        // Out of bounds
        assert!(mem.write_bytes(PAGE_SIZE as u32 - 5, &[0; 10]).is_err());
        assert!(mem.read_bytes(PAGE_SIZE as u32 - 5, 10).is_err());
    }

    #[test]
    fn test_zero_initialisation() {
        let mem = Memory::new(2, None).unwrap();

        // Check that memory is zero-initialised
        for i in 0..100 {
            assert_eq!(mem.read_u8(i).unwrap(), 0);
        }

        // Check last bytes
        let last_addr = (2 * PAGE_SIZE - 1) as u32;
        assert_eq!(mem.read_u8(last_addr).unwrap(), 0);
    }

    #[test]
    fn test_grow_zero_initialisation() {
        let mut mem = Memory::new(1, None).unwrap();

        // Write some data
        mem.write_u32(0, 0xDEADBEEF).unwrap();

        // Grow memory
        mem.grow(1);

        // Original data should be preserved
        assert_eq!(mem.read_u32(0).unwrap(), 0xDEADBEEF);

        // New memory should be zero
        let new_page_start = PAGE_SIZE as u32;
        assert_eq!(mem.read_u32(new_page_start).unwrap(), 0);
        assert_eq!(mem.read_u32(new_page_start + 100).unwrap(), 0);
    }

    #[test]
    fn test_unaligned_access() {
        let mut mem = Memory::new(1, None).unwrap();

        // WebAssembly allows unaligned access
        mem.write_u32(1, 0x12345678).unwrap(); // Unaligned
        assert_eq!(mem.read_u32(1).unwrap(), 0x12345678);

        mem.write_u64(3, 0x123456789ABCDEF0).unwrap(); // Unaligned
        assert_eq!(mem.read_u64(3).unwrap(), 0x123456789ABCDEF0);
    }
}
