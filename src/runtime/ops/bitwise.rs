//! Bitwise operations for WebAssembly
//!
//! This module provides implementations of bitwise operations
//! as specified in the WebAssembly specification section 4.4.1.3 (Binary Operations).
//!
//! All shift and rotate operations mask the shift count to the appropriate bit width.

// ============================================================================
// i32 Bitwise Operations
// ============================================================================

/// i32.and - Bitwise AND
/// spec: 4.4.1.3
pub fn i32_and(a: i32, b: i32) -> i32 {
    a & b
}

/// i32.or - Bitwise OR
/// spec: 4.4.1.3
pub fn i32_or(a: i32, b: i32) -> i32 {
    a | b
}

/// i32.xor - Bitwise XOR
/// spec: 4.4.1.3
pub fn i32_xor(a: i32, b: i32) -> i32 {
    a ^ b
}

/// i32.shl - Shift left
/// spec: 4.4.1.3
/// The shift count is taken modulo 32
pub fn i32_shl(value: i32, count: i32) -> i32 {
    // WebAssembly spec: shift count is modulo 32
    let shift = (count & 31) as u32;
    value << shift
}

/// i32.shr_s - Arithmetic shift right (sign-extending)
/// spec: 4.4.1.3
/// The shift count is taken modulo 32
pub fn i32_shr_s(value: i32, count: i32) -> i32 {
    // WebAssembly spec: shift count is modulo 32
    let shift = (count & 31) as u32;
    value >> shift
}

/// i32.shr_u - Logical shift right (zero-extending)
/// spec: 4.4.1.3
/// The shift count is taken modulo 32
pub fn i32_shr_u(value: i32, count: i32) -> i32 {
    let value = value as u32;
    // WebAssembly spec: shift count is modulo 32
    let shift = (count & 31) as u32;
    (value >> shift) as i32
}

/// i32.rotl - Rotate left
/// spec: 4.4.1.3
/// The rotation count is taken modulo 32
pub fn i32_rotl(value: i32, count: i32) -> i32 {
    let value = value as u32;
    // WebAssembly spec: rotation count is modulo 32
    let rotation = (count & 31) as u32;
    let result = value.rotate_left(rotation);
    result as i32
}

/// i32.rotr - Rotate right
/// spec: 4.4.1.3
/// The rotation count is taken modulo 32
pub fn i32_rotr(value: i32, count: i32) -> i32 {
    let value = value as u32;
    // WebAssembly spec: rotation count is modulo 32
    let rotation = (count & 31) as u32;
    let result = value.rotate_right(rotation);
    result as i32
}

// ============================================================================
// i64 Bitwise Operations
// ============================================================================

/// i64.and - Bitwise AND
/// spec: 4.4.1.3
pub fn i64_and(a: i64, b: i64) -> i64 {
    a & b
}

/// i64.or - Bitwise OR
/// spec: 4.4.1.3
pub fn i64_or(a: i64, b: i64) -> i64 {
    a | b
}

/// i64.xor - Bitwise XOR
/// spec: 4.4.1.3
pub fn i64_xor(a: i64, b: i64) -> i64 {
    a ^ b
}

/// i64.shl - Shift left
/// spec: 4.4.1.3
/// The shift count is taken modulo 64
pub fn i64_shl(value: i64, count: i64) -> i64 {
    // WebAssembly spec: shift count is modulo 64
    let shift = (count & 63) as u32;
    value << shift
}

/// i64.shr_s - Arithmetic shift right (sign-extending)
/// spec: 4.4.1.3
/// The shift count is taken modulo 64
pub fn i64_shr_s(value: i64, count: i64) -> i64 {
    // WebAssembly spec: shift count is modulo 64
    let shift = (count & 63) as u32;
    value >> shift
}

/// i64.shr_u - Logical shift right (zero-extending)
/// spec: 4.4.1.3
/// The shift count is taken modulo 64
pub fn i64_shr_u(value: i64, count: i64) -> i64 {
    let value = value as u64;
    // WebAssembly spec: shift count is modulo 64
    let shift = (count & 63) as u32;
    (value >> shift) as i64
}

/// i64.rotl - Rotate left
/// spec: 4.4.1.3
/// The rotation count is taken modulo 64
pub fn i64_rotl(value: i64, count: i64) -> i64 {
    let value = value as u64;
    // WebAssembly spec: rotation count is modulo 64
    let rotation = (count & 63) as u32;
    let result = value.rotate_left(rotation);
    result as i64
}

/// i64.rotr - Rotate right
/// spec: 4.4.1.3
/// The rotation count is taken modulo 64
pub fn i64_rotr(value: i64, count: i64) -> i64 {
    let value = value as u64;
    // WebAssembly spec: rotation count is modulo 64
    let rotation = (count & 63) as u32;
    let result = value.rotate_right(rotation);
    result as i64
}

#[cfg(test)]
mod tests {
    use crate::parser::instruction::InstructionKind;
    use crate::parser::module::ValueType;
    use crate::runtime::Value;
    use crate::runtime::test_utils::test::ExecutorTest;

    // ============================================================================
    // i32 Bitwise Tests
    // ============================================================================

    #[test]
    fn test_i32_and() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0xFF00FF00u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 0x0F0F0F0F })
            .inst(InstructionKind::I32And)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x0F000F00)]);

        // Test with negative numbers
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32And)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x12345678)]);
    }

    #[test]
    fn test_i32_bitwise_masks() {
        // Clear high byte: x & 0x00FFFFFF
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32Const { value: 0x00FFFFFF })
            .inst(InstructionKind::I32And)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x00345678)]);

        // Extract nibble with AND
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32Const { value: 0x0000F000 })
            .inst(InstructionKind::I32And)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x00005000)]);

        // Set bits with OR
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x12340000 })
            .inst(InstructionKind::I32Const { value: 0x00005678 })
            .inst(InstructionKind::I32Or)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x12345678)]);

        // Toggle bits with XOR
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32Const {
                value: 0xFF00FF00u32 as i32,
            })
            .inst(InstructionKind::I32Xor)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0xED34A978u32 as i32)]);

        // XOR to toggle bits
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0xFFFFFFFFu32 as i32,
            })
            .inst(InstructionKind::I32Const {
                value: 0x80808080u32 as i32,
            })
            .inst(InstructionKind::I32Xor)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x7F7F7F7F)]);
    }

    #[test]
    fn test_i32_or() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0xFF00FF00u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 0x0F0F0F0F })
            .inst(InstructionKind::I32Or)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0xFF0FFF0Fu32 as i32)]);

        // Test with zero
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32Or)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x12345678)]);
    }

    #[test]
    fn test_i32_xor() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0xFF00FF00u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 0x0F0F0F0F })
            .inst(InstructionKind::I32Xor)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0xF00FF00Fu32 as i32)]);

        // XOR with self should be zero
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32Xor)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn test_i32_shl() {
        // Basic shift
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 4 })
            .inst(InstructionKind::I32Shl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(16)]);

        // Shift by 0
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::I32Shl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);

        // Shift by 32 (should wrap to 0)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 32 })
            .inst(InstructionKind::I32Shl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // Shift by 33 (should wrap to 1)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 33 })
            .inst(InstructionKind::I32Shl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(2)]);
    }

    #[test]
    fn test_i32_shl_overflow() {
        // Test that bits are lost when shifted out
        // 0x80000000 << 1 should be 0 (MSB lost)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000000u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Shl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);

        // 0xFFFFFFFF << 1 should be 0xFFFFFFFE (LSB becomes 0)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Shl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-2)]);

        // 0x40000000 << 1 should be 0x80000000 (becomes negative)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x40000000 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Shl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x80000000u32 as i32)]);

        // Test large shift counts with modulo
        // 0xFF << 100 = 0xFF << (100 % 32) = 0xFF << 4 = 0xFF0
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0xFF })
            .inst(InstructionKind::I32Const { value: 100 })
            .inst(InstructionKind::I32Shl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0xFF0)]);
    }

    #[test]
    fn test_i32_shr_s() {
        // Positive number
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 16 })
            .inst(InstructionKind::I32Const { value: 4 })
            .inst(InstructionKind::I32ShrS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // Negative number (sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -16 })
            .inst(InstructionKind::I32Const { value: 4 })
            .inst(InstructionKind::I32ShrS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]);

        // Shift by 32 (should wrap to 0)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000000u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 32 })
            .inst(InstructionKind::I32ShrS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x80000000u32 as i32)]);
    }

    #[test]
    fn test_i32_shr_s_sign_extension() {
        // Test that all high bits are set when shifting negative numbers
        // -1 >> 1 should be -1 (all bits set)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32ShrS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]);

        // -1 >> 31 should be -1 (sign bit propagated)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Const { value: 31 })
            .inst(InstructionKind::I32ShrS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]);

        // 0x80000000 >> 1 should be 0xC0000000 (sign extended)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000000u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32ShrS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0xC0000000u32 as i32)]);

        // 0x80000000 >> 31 should be -1 (all bits set from sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000000u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 31 })
            .inst(InstructionKind::I32ShrS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]);

        // i32::MIN >> 1 should maintain sign
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: i32::MIN })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32ShrS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0xC0000000u32 as i32)]);
    }

    #[test]
    fn test_i32_shr_u() {
        // Positive number
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 16 })
            .inst(InstructionKind::I32Const { value: 4 })
            .inst(InstructionKind::I32ShrU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // Negative number becomes positive (zero extension)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -16 })
            .inst(InstructionKind::I32Const { value: 4 })
            .inst(InstructionKind::I32ShrU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x0FFFFFFFu32 as i32)]);

        // 0x80000000 >> 1 should be 0x40000000 (zero fill)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000000u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32ShrU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x40000000)]);
    }

    #[test]
    fn test_i32_shr_u_zero_extension() {
        // Test that high bits are filled with zeros
        // -1 >>> 1 should be 0x7FFFFFFF (zero fill)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32ShrU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x7FFFFFFF)]);

        // -1 >>> 31 should be 1 (only LSB remains)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Const { value: 31 })
            .inst(InstructionKind::I32ShrU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // 0x80000000 >>> 31 should be 1 (only MSB moves to LSB)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000000u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 31 })
            .inst(InstructionKind::I32ShrU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // i32::MIN >>> 1 should be 0x40000000 (positive result)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: i32::MIN })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32ShrU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x40000000)]);

        // -2 >>> 1 should be 0x7FFFFFFF
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -2 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32ShrU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x7FFFFFFF)]);
    }

    #[test]
    fn test_i32_rotl() {
        // Basic rotation
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000001u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Rotl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x00000003)]);

        // Rotate by 0
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::I32Rotl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x12345678)]);

        // Rotate by 32 (full rotation)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32Const { value: 32 })
            .inst(InstructionKind::I32Rotl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x12345678)]);

        // Rotate by 4
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32Const { value: 4 })
            .inst(InstructionKind::I32Rotl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x23456781)]);
    }

    #[test]
    fn test_i32_rotl_large_counts() {
        // Rotate by 33 (wraps to 1)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000001u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 33 })
            .inst(InstructionKind::I32Rotl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x00000003)]);

        // Rotate by 100 (wraps to 4)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32Const { value: 100 })
            .inst(InstructionKind::I32Rotl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x23456781)]);

        // Rotate by -1 (wraps to 31, rotate right by 1)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000001u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Rotl)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0xC0000000u32 as i32)]);
    }

    #[test]
    fn test_i32_rotr() {
        // Basic rotation
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000001u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Rotr)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0xC0000000u32 as i32)]);

        // Rotate by 4
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x12345678 })
            .inst(InstructionKind::I32Const { value: 4 })
            .inst(InstructionKind::I32Rotr)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x81234567u32 as i32)]);
    }

    #[test]
    fn test_i32_rotr_edge_cases() {
        // Rotate -1 right by 1
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Rotr)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]);

        // Rotate 0x80000001 right by 31
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000001u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 31 })
            .inst(InstructionKind::I32Rotr)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x00000003)]);

        // Rotate by 65 (wraps to 1)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000001u32 as i32,
            })
            .inst(InstructionKind::I32Const { value: 65 })
            .inst(InstructionKind::I32Rotr)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0xC0000000u32 as i32)]);
    }

    // ============================================================================
    // i64 Bitwise Tests
    // ============================================================================

    #[test]
    fn test_i64_and() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0xFF00FF00FF00FF00u64 as i64,
            })
            .inst(InstructionKind::I64Const {
                value: 0x0F0F0F0F0F0F0F0Fu64 as i64,
            })
            .inst(InstructionKind::I64And)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x0F000F000F000F00u64 as i64)]);
    }

    #[test]
    fn test_i64_or() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0 })
            .inst(InstructionKind::I64Const {
                value: 0x123456789ABCDEF0u64 as i64,
            })
            .inst(InstructionKind::I64Or)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x123456789ABCDEF0u64 as i64)]);
    }

    #[test]
    fn test_i64_xor() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64Const {
                value: 0x123456789ABCDEF0u64 as i64,
            })
            .inst(InstructionKind::I64Xor)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0xEDCBA9876543210Fu64 as i64)]);
    }

    #[test]
    fn test_i64_bitwise_patterns() {
        // Clear specific bits with AND
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0xFFFFFFFFFFFFFFFFu64 as i64,
            })
            .inst(InstructionKind::I64Const {
                value: 0x00FFFFFF00FFFFFFu64 as i64,
            })
            .inst(InstructionKind::I64And)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x00FFFFFF00FFFFFFu64 as i64)]);

        // Set specific bits with OR
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0 })
            .inst(InstructionKind::I64Const {
                value: 0x8080808080808080u64 as i64,
            })
            .inst(InstructionKind::I64Or)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x8080808080808080u64 as i64)]);

        // Toggle pattern with XOR
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0xAAAAAAAAAAAAAAAAu64 as i64,
            })
            .inst(InstructionKind::I64Const {
                value: 0xFFFFFFFFFFFFFFFFu64 as i64,
            })
            .inst(InstructionKind::I64Xor)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x5555555555555555u64 as i64)]);

        // Complex mask operation
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x123456789ABCDEF0u64 as i64,
            })
            .inst(InstructionKind::I64Const {
                value: 0x0F0F0F0F0F0F0F0Fu64 as i64,
            })
            .inst(InstructionKind::I64And)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x020406080A0C0E00u64 as i64)]);
    }

    #[test]
    fn test_i64_shl() {
        // Basic shift
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Const { value: 8 })
            .inst(InstructionKind::I64Shl)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(256)]);

        // Shift by 64 (wraps to 0)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Const { value: 64 })
            .inst(InstructionKind::I64Shl)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(1)]);

        // Shift with overflow
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x8000000000000000u64 as i64,
            })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Shl)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0)]);
    }

    #[test]
    fn test_i64_shl_overflow() {
        // Large value shift
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x123456789ABCDEF0u64 as i64,
            })
            .inst(InstructionKind::I64Const { value: 4 })
            .inst(InstructionKind::I64Shl)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x23456789ABCDEF00u64 as i64)]);

        // Max negative shift
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: i64::MIN })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Shl)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0)]);
    }

    #[test]
    fn test_i64_shr_s() {
        // Positive shift
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 256 })
            .inst(InstructionKind::I64Const { value: 8 })
            .inst(InstructionKind::I64ShrS)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(1)]);

        // Negative with sign extension
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -256 })
            .inst(InstructionKind::I64Const { value: 8 })
            .inst(InstructionKind::I64ShrS)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]);
    }

    #[test]
    fn test_i64_shr_s_sign_extension() {
        // -1 >> 1 should be -1
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64ShrS)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]);

        // i64::MIN >> 1
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: i64::MIN })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64ShrS)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0xC000000000000000u64 as i64)]);

        // Sign bit propagation
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x8000000000000000u64 as i64,
            })
            .inst(InstructionKind::I64Const { value: 63 })
            .inst(InstructionKind::I64ShrS)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]);
    }

    #[test]
    fn test_i64_shr_u() {
        // Basic zero-fill shift
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 256 })
            .inst(InstructionKind::I64Const { value: 8 })
            .inst(InstructionKind::I64ShrU)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(1)]);

        // Negative becomes positive
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -256 })
            .inst(InstructionKind::I64Const { value: 8 })
            .inst(InstructionKind::I64ShrU)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x00FFFFFFFFFFFFFFu64 as i64)]);
    }

    #[test]
    fn test_i64_shr_u_zero_extension() {
        // -1 >>> 1
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64ShrU)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x7FFFFFFFFFFFFFFFu64 as i64)]);

        // i64::MIN >>> 1 (becomes positive)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: i64::MIN })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64ShrU)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x4000000000000000u64 as i64)]);

        // High bit shift to low
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x8000000000000000u64 as i64,
            })
            .inst(InstructionKind::I64Const { value: 63 })
            .inst(InstructionKind::I64ShrU)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(1)]);
    }

    #[test]
    fn test_i64_rotl() {
        // Basic rotation
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x8000000000000001u64 as i64,
            })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Rotl)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x0000000000000003)]);

        // Rotate by 64 (full rotation)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x123456789ABCDEF0u64 as i64,
            })
            .inst(InstructionKind::I64Const { value: 64 })
            .inst(InstructionKind::I64Rotl)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x123456789ABCDEF0u64 as i64)]);
    }

    #[test]
    fn test_i64_rotl_edge_cases() {
        // Rotate by 65 (wraps to 1)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x8000000000000001u64 as i64,
            })
            .inst(InstructionKind::I64Const { value: 65 })
            .inst(InstructionKind::I64Rotl)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x0000000000000003)]);

        // Rotate by -1 (wraps to 63, rotate right by 1)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x8000000000000001u64 as i64,
            })
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64Rotl)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0xC000000000000000u64 as i64)]);
    }

    #[test]
    fn test_i64_rotr() {
        // Basic rotation
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x8000000000000001u64 as i64,
            })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Rotr)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0xC000000000000000u64 as i64)]);

        // Complex rotation
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x123456789ABCDEF0u64 as i64,
            })
            .inst(InstructionKind::I64Const { value: 4 })
            .inst(InstructionKind::I64Rotr)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x0123456789ABCDEFu64 as i64)]);
    }

    #[test]
    fn test_i64_rotr_edge_cases() {
        // Rotate -1 right by 1
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Rotr)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]);

        // Rotate by 65 (wraps to 1)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x8000000000000001u64 as i64,
            })
            .inst(InstructionKind::I64Const { value: 65 })
            .inst(InstructionKind::I64Rotr)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0xC000000000000000u64 as i64)]);
    }
}
