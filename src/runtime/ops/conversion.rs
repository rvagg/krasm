//! Type conversion operations for WebAssembly
//!
//! This module provides implementations of type conversion operations
//! as specified in the WebAssembly specification section 4.4.1.6.
//!
//! Conversions include:
//! - Integer width conversions (wrap, extend)
//! - Float width conversions (promote, demote)
//! - Integer to float conversions
//! - Float to integer conversions (truncation)
//! - Saturating truncations (non-trapping)
//! - Reinterpretation (bit casting)

use super::*;

// Truncation range bounds (spec 4.4.1.6)
//
// Each pair defines [MIN, MAX) — the half-open range of truncated float values
// that fit in the target integer type. Values outside this range cause
// IntegerOverflow traps.

const I32_TRUNC_F32_S_MIN: f32 = -2147483648.0; // i32::MIN as f32
const I32_TRUNC_F32_S_MAX: f32 = 2147483648.0; // 2^31
const I32_TRUNC_F32_U_MIN: f32 = 0.0;
const I32_TRUNC_F32_U_MAX: f32 = 4294967296.0; // 2^32
const I32_TRUNC_F64_S_MIN: f64 = -2147483648.0; // i32::MIN as f64
const I32_TRUNC_F64_S_MAX: f64 = 2147483648.0; // 2^31
const I32_TRUNC_F64_U_MIN: f64 = 0.0;
const I32_TRUNC_F64_U_MAX: f64 = 4294967296.0; // 2^32
const I64_TRUNC_F32_S_MIN: f32 = -9223372036854775808.0; // i64::MIN as f32
const I64_TRUNC_F32_S_MAX: f32 = 9223372036854775808.0; // 2^63
const I64_TRUNC_F32_U_MIN: f32 = 0.0;
const I64_TRUNC_F32_U_MAX: f32 = 18446744073709551616.0; // 2^64
const I64_TRUNC_F64_S_MIN: f64 = -9223372036854775808.0; // i64::MIN as f64
const I64_TRUNC_F64_S_MAX: f64 = 9223372036854775808.0; // 2^63
const I64_TRUNC_F64_U_MIN: f64 = 0.0;
const I64_TRUNC_F64_U_MAX: f64 = 18446744073709551616.0; // 2^64

// ============================================================================
// Integer Width Conversions
// ============================================================================

/// i32.wrap_i64 - Truncate i64 to i32 (keep low 32 bits)
/// spec: 4.4.1.6
pub fn i32_wrap_i64(value: i64) -> i32 {
    value as i32
}

/// i64.extend_i32_s - Sign-extend i32 to i64
/// spec: 4.4.1.6
pub fn i64_extend_i32_s(value: i32) -> i64 {
    value as i64 // Rust's `as` does sign extension
}

/// i64.extend_i32_u - Zero-extend i32 to i64
/// spec: 4.4.1.6
pub fn i64_extend_i32_u(value: i32) -> i64 {
    (value as u32) as i64 // Cast to u32 first for zero-extension
}

// ============================================================================
// Sign Extension Operations
// ============================================================================

/// i32.extend8_s - Sign-extend 8-bit value to i32
/// spec: 4.4.1.6
///
/// Takes the lowest 8 bits of an i32 value and sign-extends them to 32 bits.
/// If bit 7 is set (negative in 8-bit), all upper bits are set to 1.
pub fn i32_extend8_s(value: i32) -> i32 {
    let byte = value as i8; // Truncate to 8 bits and interpret as signed
    byte as i32 // Sign-extend back to i32
}

/// i32.extend16_s - Sign-extend 16-bit value to i32
/// spec: 4.4.1.6
///
/// Takes the lowest 16 bits of an i32 value and sign-extends them to 32 bits.
/// If bit 15 is set (negative in 16-bit), all upper bits are set to 1.
pub fn i32_extend16_s(value: i32) -> i32 {
    let short = value as i16; // Truncate to 16 bits and interpret as signed
    short as i32 // Sign-extend back to i32
}

/// i64.extend8_s - Sign-extend 8-bit value to i64
/// spec: 4.4.1.6
///
/// Takes the lowest 8 bits of an i64 value and sign-extends them to 64 bits.
/// If bit 7 is set (negative in 8-bit), all upper bits are set to 1.
pub fn i64_extend8_s(value: i64) -> i64 {
    let byte = value as i8; // Truncate to 8 bits and interpret as signed
    byte as i64 // Sign-extend back to i64
}

/// i64.extend16_s - Sign-extend 16-bit value to i64
/// spec: 4.4.1.6
///
/// Takes the lowest 16 bits of an i64 value and sign-extends them to 64 bits.
/// If bit 15 is set (negative in 16-bit), all upper bits are set to 1.
pub fn i64_extend16_s(value: i64) -> i64 {
    let short = value as i16; // Truncate to 16 bits and interpret as signed
    short as i64 // Sign-extend back to i64
}

/// i64.extend32_s - Sign-extend 32-bit value to i64
/// spec: 4.4.1.6
///
/// Takes the lowest 32 bits of an i64 value and sign-extends them to 64 bits.
/// If bit 31 is set (negative in 32-bit), all upper bits are set to 1.
pub fn i64_extend32_s(value: i64) -> i64 {
    let int = value as i32; // Truncate to 32 bits and interpret as signed
    int as i64 // Sign-extend back to i64
}

// ============================================================================
// Reinterpretation / Bit Casting
// ============================================================================

/// i32.reinterpret_f32 - Reinterpret f32 bits as i32
/// spec: 4.4.1.6
pub fn i32_reinterpret_f32(value: f32) -> i32 {
    value.to_bits() as i32
}

/// i64.reinterpret_f64 - Reinterpret f64 bits as i64
/// spec: 4.4.1.6
pub fn i64_reinterpret_f64(value: f64) -> i64 {
    value.to_bits() as i64
}

/// f32.reinterpret_i32 - Reinterpret i32 bits as f32
/// spec: 4.4.1.6
pub fn f32_reinterpret_i32(value: i32) -> f32 {
    f32::from_bits(value as u32)
}

/// f64.reinterpret_i64 - Reinterpret i64 bits as f64
/// spec: 4.4.1.6
pub fn f64_reinterpret_i64(value: i64) -> f64 {
    f64::from_bits(value as u64)
}

// ============================================================================
// Float Width Conversions
// ============================================================================

/// f32.demote_f64 - Convert f64 to f32 (may lose precision)
/// spec: 4.4.1.6
pub fn f32_demote_f64(value: f64) -> f32 {
    value as f32
}

/// f64.promote_f32 - Convert f32 to f64 (exact conversion)
/// spec: 4.4.1.6
pub fn f64_promote_f32(value: f32) -> f64 {
    value as f64
}

// ============================================================================
// Integer to Float Conversions
// ============================================================================

/// f32.convert_i32_s - Convert signed i32 to f32
/// spec: 4.4.1.6
pub fn f32_convert_i32_s(value: i32) -> f32 {
    value as f32
}

/// f32.convert_i32_u - Convert unsigned i32 to f32
/// spec: 4.4.1.6
pub fn f32_convert_i32_u(value: i32) -> f32 {
    (value as u32) as f32
}

/// f32.convert_i64_s - Convert signed i64 to f32
/// spec: 4.4.1.6
pub fn f32_convert_i64_s(value: i64) -> f32 {
    value as f32
}

/// f32.convert_i64_u - Convert unsigned i64 to f32
/// spec: 4.4.1.6
pub fn f32_convert_i64_u(value: i64) -> f32 {
    (value as u64) as f32
}

/// f64.convert_i32_s - Convert signed i32 to f64
/// spec: 4.4.1.6
pub fn f64_convert_i32_s(value: i32) -> f64 {
    value as f64
}

/// f64.convert_i32_u - Convert unsigned i32 to f64
/// spec: 4.4.1.6
pub fn f64_convert_i32_u(value: i32) -> f64 {
    (value as u32) as f64
}

/// f64.convert_i64_s - Convert signed i64 to f64
/// spec: 4.4.1.6
pub fn f64_convert_i64_s(value: i64) -> f64 {
    value as f64
}

/// f64.convert_i64_u - Convert unsigned i64 to f64
/// spec: 4.4.1.6
pub fn f64_convert_i64_u(value: i64) -> f64 {
    (value as u64) as f64
}

// ============================================================================
// Float to Integer Conversions (Trapping)
// ============================================================================

/// i32.trunc_f32_s - Truncate f32 to signed i32
/// spec: 4.4.1.6
/// Traps on: NaN, infinity, or out-of-range values
pub fn i32_trunc_f32_s(value: f32) -> Result<i32, Trap> {
    // Check for NaN
    if value.is_nan() {
        return Err(Trap::InvalidConversionToInteger);
    }

    // Check for infinity
    if value.is_infinite() {
        return Err(Trap::IntegerOverflow);
    }

    // Truncate first, then check range
    let truncated = value.trunc();

    if !(I32_TRUNC_F32_S_MIN..I32_TRUNC_F32_S_MAX).contains(&truncated) {
        return Err(Trap::IntegerOverflow);
    }

    Ok(truncated as i32)
}

/// i32.trunc_f32_u - Truncate f32 to unsigned i32
/// spec: 4.4.1.6
pub fn i32_trunc_f32_u(value: f32) -> Result<i32, Trap> {
    if value.is_nan() {
        return Err(Trap::InvalidConversionToInteger);
    }

    if value.is_infinite() {
        return Err(Trap::IntegerOverflow);
    }

    // Truncate first to handle subnormals that become -0.0
    let truncated = value.trunc();

    // -0.0 is valid and converts to 0
    if truncated == 0.0 {
        return Ok(0);
    }

    if !(I32_TRUNC_F32_U_MIN..I32_TRUNC_F32_U_MAX).contains(&truncated) {
        return Err(Trap::IntegerOverflow);
    }

    Ok(truncated as u32 as i32)
}

/// i32.trunc_f64_s - Truncate f64 to signed i32
/// spec: 4.4.1.6
pub fn i32_trunc_f64_s(value: f64) -> Result<i32, Trap> {
    if value.is_nan() {
        return Err(Trap::InvalidConversionToInteger);
    }

    if value.is_infinite() {
        return Err(Trap::IntegerOverflow);
    }

    // Truncate first, then check range
    let truncated = value.trunc();

    if !(I32_TRUNC_F64_S_MIN..I32_TRUNC_F64_S_MAX).contains(&truncated) {
        return Err(Trap::IntegerOverflow);
    }

    Ok(truncated as i32)
}

/// i32.trunc_f64_u - Truncate f64 to unsigned i32
/// spec: 4.4.1.6
pub fn i32_trunc_f64_u(value: f64) -> Result<i32, Trap> {
    if value.is_nan() {
        return Err(Trap::InvalidConversionToInteger);
    }

    if value.is_infinite() {
        return Err(Trap::IntegerOverflow);
    }

    // Truncate first to handle subnormals that become -0.0
    let truncated = value.trunc();

    // -0.0 is valid and converts to 0
    if truncated == 0.0 {
        return Ok(0);
    }

    if !(I32_TRUNC_F64_U_MIN..I32_TRUNC_F64_U_MAX).contains(&truncated) {
        return Err(Trap::IntegerOverflow);
    }

    Ok(truncated as u32 as i32)
}

/// i64.trunc_f32_s - Truncate f32 to signed i64
/// spec: 4.4.1.6
pub fn i64_trunc_f32_s(value: f32) -> Result<i64, Trap> {
    if value.is_nan() {
        return Err(Trap::InvalidConversionToInteger);
    }

    if value.is_infinite() {
        return Err(Trap::IntegerOverflow);
    }

    // Truncate first, then check range
    let truncated = value.trunc();

    if !(I64_TRUNC_F32_S_MIN..I64_TRUNC_F32_S_MAX).contains(&truncated) {
        return Err(Trap::IntegerOverflow);
    }

    Ok(truncated as i64)
}

/// i64.trunc_f32_u - Truncate f32 to unsigned i64
/// spec: 4.4.1.6
pub fn i64_trunc_f32_u(value: f32) -> Result<i64, Trap> {
    if value.is_nan() {
        return Err(Trap::InvalidConversionToInteger);
    }

    if value.is_infinite() {
        return Err(Trap::IntegerOverflow);
    }

    // Truncate first to handle subnormals that become -0.0
    let truncated = value.trunc();

    // -0.0 is valid and converts to 0
    if truncated == 0.0 {
        return Ok(0);
    }

    if !(I64_TRUNC_F32_U_MIN..I64_TRUNC_F32_U_MAX).contains(&truncated) {
        return Err(Trap::IntegerOverflow);
    }

    Ok(truncated as u64 as i64)
}

/// i64.trunc_f64_s - Truncate f64 to signed i64
/// spec: 4.4.1.6
pub fn i64_trunc_f64_s(value: f64) -> Result<i64, Trap> {
    if value.is_nan() {
        return Err(Trap::InvalidConversionToInteger);
    }

    if value.is_infinite() {
        return Err(Trap::IntegerOverflow);
    }

    // Truncate first, then check range
    let truncated = value.trunc();

    if !(I64_TRUNC_F64_S_MIN..I64_TRUNC_F64_S_MAX).contains(&truncated) {
        return Err(Trap::IntegerOverflow);
    }

    Ok(truncated as i64)
}

/// i64.trunc_f64_u - Truncate f64 to unsigned i64
/// spec: 4.4.1.6
pub fn i64_trunc_f64_u(value: f64) -> Result<i64, Trap> {
    if value.is_nan() {
        return Err(Trap::InvalidConversionToInteger);
    }

    if value.is_infinite() {
        return Err(Trap::IntegerOverflow);
    }

    // Truncate first to handle subnormals that become -0.0
    let truncated = value.trunc();

    // -0.0 is valid and converts to 0
    if truncated == 0.0 {
        return Ok(0);
    }

    if !(I64_TRUNC_F64_U_MIN..I64_TRUNC_F64_U_MAX).contains(&truncated) {
        return Err(Trap::IntegerOverflow);
    }

    Ok(truncated as u64 as i64)
}

// ============================================================================
// Saturating Truncation (Non-trapping)
// ============================================================================

/// i32.trunc_sat_f32_s - Saturating truncation of f32 to signed i32
/// spec: Saturating truncation proposal
pub fn i32_trunc_sat_f32_s(value: f32) -> i32 {
    if value.is_nan() {
        0
    } else if value >= i32::MAX as f32 {
        i32::MAX
    } else if value <= i32::MIN as f32 {
        i32::MIN
    } else {
        value.trunc() as i32
    }
}

/// i32.trunc_sat_f32_u - Saturating truncation of f32 to unsigned i32
/// spec: Saturating truncation proposal
pub fn i32_trunc_sat_f32_u(value: f32) -> i32 {
    if value.is_nan() {
        0
    } else if value >= u32::MAX as f32 {
        u32::MAX as i32
    } else if value <= 0.0 {
        0
    } else {
        value.trunc() as u32 as i32
    }
}

/// i32.trunc_sat_f64_s - Saturating truncation of f64 to signed i32
/// spec: Saturating truncation proposal
pub fn i32_trunc_sat_f64_s(value: f64) -> i32 {
    if value.is_nan() {
        0
    } else if value >= i32::MAX as f64 {
        i32::MAX
    } else if value <= i32::MIN as f64 {
        i32::MIN
    } else {
        value.trunc() as i32
    }
}

/// i32.trunc_sat_f64_u - Saturating truncation of f64 to unsigned i32
/// spec: Saturating truncation proposal
pub fn i32_trunc_sat_f64_u(value: f64) -> i32 {
    if value.is_nan() {
        0
    } else if value >= u32::MAX as f64 {
        u32::MAX as i32
    } else if value <= 0.0 {
        0
    } else {
        value.trunc() as u32 as i32
    }
}

/// i64.trunc_sat_f32_s - Saturating truncation of f32 to signed i64
/// spec: Saturating truncation proposal
pub fn i64_trunc_sat_f32_s(value: f32) -> i64 {
    if value.is_nan() {
        0
    } else if value >= i64::MAX as f32 {
        i64::MAX
    } else if value <= i64::MIN as f32 {
        i64::MIN
    } else {
        value.trunc() as i64
    }
}

/// i64.trunc_sat_f32_u - Saturating truncation of f32 to unsigned i64
/// spec: Saturating truncation proposal
pub fn i64_trunc_sat_f32_u(value: f32) -> i64 {
    if value.is_nan() {
        0
    } else if value >= u64::MAX as f32 {
        u64::MAX as i64
    } else if value <= 0.0 {
        0
    } else {
        value.trunc() as u64 as i64
    }
}

/// i64.trunc_sat_f64_s - Saturating truncation of f64 to signed i64
/// spec: Saturating truncation proposal
pub fn i64_trunc_sat_f64_s(value: f64) -> i64 {
    if value.is_nan() {
        0
    } else if value >= i64::MAX as f64 {
        i64::MAX
    } else if value <= i64::MIN as f64 {
        i64::MIN
    } else {
        value.trunc() as i64
    }
}

/// i64.trunc_sat_f64_u - Saturating truncation of f64 to unsigned i64
/// spec: Saturating truncation proposal
pub fn i64_trunc_sat_f64_u(value: f64) -> i64 {
    if value.is_nan() {
        0
    } else if value >= u64::MAX as f64 {
        u64::MAX as i64
    } else if value <= 0.0 {
        0
    } else {
        value.trunc() as u64 as i64
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::instruction::InstructionKind;
    use crate::parser::module::ValueType;
    use crate::runtime::Value;
    use crate::runtime::test_utils::test::ExecutorTest;

    // ============================================================================
    // Integer Width Conversion Tests
    // ============================================================================

    #[test]
    fn test_i32_wrap_i64() {
        // Basic wrapping
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x123456789ABCDEF0u64 as i64,
            })
            .inst(InstructionKind::I32WrapI64)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x9ABCDEF0u32 as i32)]);

        // Wrap negative
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I32WrapI64)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]);

        // Wrap zero
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0 })
            .inst(InstructionKind::I32WrapI64)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn test_i64_extend_i32_s() {
        // Positive extension
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I64ExtendI32S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(42)]);

        // Negative extension (sign bit propagated)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I64ExtendI32S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]);

        // High bit set
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000000u32 as i32,
            })
            .inst(InstructionKind::I64ExtendI32S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0xFFFFFFFF80000000u64 as i64)]);
    }

    #[test]
    fn test_i64_extend_i32_u() {
        // Positive extension
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I64ExtendI32U)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(42)]);

        // High bit set (zero extended)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000000u32 as i32,
            })
            .inst(InstructionKind::I64ExtendI32U)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x80000000)]);

        // -1 becomes large positive
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I64ExtendI32U)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0xFFFFFFFF)]);
    }

    // ============================================================================
    // Reinterpret Tests
    // ============================================================================

    #[test]
    fn test_reinterpret_f32_i32() {
        // Round trip test: i32 -> f32 -> i32
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x3F800000 }) // 1.0 in f32
            .inst(InstructionKind::F32ReinterpretI32)
            .inst(InstructionKind::I32ReinterpretF32)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x3F800000)]);

        // Test f32 -> i32
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 1.0 })
            .inst(InstructionKind::I32ReinterpretF32)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x3F800000)]);
    }

    #[test]
    fn test_reinterpret_f64_i64() {
        // Specific bit pattern
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: 1.0 })
            .inst(InstructionKind::I64ReinterpretF64)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x3FF0000000000000u64 as i64)]);

        // Round trip
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x3FF0000000000000u64 as i64,
            })
            .inst(InstructionKind::F64ReinterpretI64)
            .inst(InstructionKind::I64ReinterpretF64)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x3FF0000000000000u64 as i64)]);
    }

    // ============================================================================
    // Float Width Tests
    // ============================================================================

    #[test]
    fn test_f32_demote_f64() {
        // Normal value - can't test exact value due to precision loss
        // so we test special values instead

        // Infinity preserved
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: f64::INFINITY })
            .inst(InstructionKind::F32DemoteF64)
            .returns(vec![ValueType::F32])
            .expect_stack(vec![Value::F32(f32::INFINITY)]);

        // Negative infinity
        ExecutorTest::new()
            .inst(InstructionKind::F64Const {
                value: f64::NEG_INFINITY,
            })
            .inst(InstructionKind::F32DemoteF64)
            .returns(vec![ValueType::F32])
            .expect_stack(vec![Value::F32(f32::NEG_INFINITY)]);

        // Zero
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: 0.0 })
            .inst(InstructionKind::F32DemoteF64)
            .returns(vec![ValueType::F32])
            .expect_stack(vec![Value::F32(0.0)]);
    }

    #[test]
    fn test_f64_promote_f32() {
        // Special values
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: f32::INFINITY })
            .inst(InstructionKind::F64PromoteF32)
            .returns(vec![ValueType::F64])
            .expect_stack(vec![Value::F64(f64::INFINITY)]);

        // Negative infinity
        ExecutorTest::new()
            .inst(InstructionKind::F32Const {
                value: f32::NEG_INFINITY,
            })
            .inst(InstructionKind::F64PromoteF32)
            .returns(vec![ValueType::F64])
            .expect_stack(vec![Value::F64(f64::NEG_INFINITY)]);

        // Zero
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 0.0 })
            .inst(InstructionKind::F64PromoteF32)
            .returns(vec![ValueType::F64])
            .expect_stack(vec![Value::F64(0.0)]);

        // Negative zero
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: -0.0 })
            .inst(InstructionKind::F64PromoteF32)
            .returns(vec![ValueType::F64])
            .expect_stack(vec![Value::F64(-0.0)]);
    }

    // ============================================================================
    // Boundary Tests for Float-to-Int Conversions
    // ============================================================================

    #[test]
    fn test_i32_trunc_f32_s_boundaries() {
        // Test the exact boundary value (2^31) - should trap
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 2147483648.0 })
            .inst(InstructionKind::I32TruncF32S)
            .expect_error("overflow");

        // Test i32::MAX when converted to f32 (rounds to 2^31) - should trap
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: i32::MAX as f32 })
            .inst(InstructionKind::I32TruncF32S)
            .expect_error("overflow");

        // Test a safe value just below
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 2147483520.0 })
            .inst(InstructionKind::I32TruncF32S)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(2147483520)]);

        // Test negative boundary
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: -2147483648.0 })
            .inst(InstructionKind::I32TruncF32S)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(i32::MIN)]);

        // Test value below negative boundary - should trap
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: -2147483904.0 }) // Next f32 below -2^31
            .inst(InstructionKind::I32TruncF32S)
            .expect_error("overflow");
    }

    #[test]
    fn test_i32_trunc_f32_u_boundaries() {
        // Test u32::MAX boundary - should trap
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 4294967296.0 }) // 2^32
            .inst(InstructionKind::I32TruncF32U)
            .expect_error("overflow");

        // Test a safe value near the max
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 4294967040.0 })
            .inst(InstructionKind::I32TruncF32U)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(4294967040u32 as i32)]);

        // Test negative value - should trap
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: -1.0 })
            .inst(InstructionKind::I32TruncF32U)
            .expect_error("overflow");

        // Test zero
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 0.0 })
            .inst(InstructionKind::I32TruncF32U)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    // ============================================================================
    // Float-to-Int Conversion Tests
    // ============================================================================

    #[test]
    fn test_f32_convert_i32() {
        // Signed conversion
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::F32ConvertI32S)
            .returns(vec![ValueType::F32])
            .expect_stack(vec![Value::F32(42.0)]);

        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -42 })
            .inst(InstructionKind::F32ConvertI32S)
            .returns(vec![ValueType::F32])
            .expect_stack(vec![Value::F32(-42.0)]);

        // Unsigned conversion
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: 0x80000000u32 as i32,
            })
            .inst(InstructionKind::F32ConvertI32U)
            .returns(vec![ValueType::F32])
            .expect_stack(vec![Value::F32(2147483648.0)]);
    }

    #[test]
    fn test_f64_convert_i64() {
        // Signed conversion
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 42 })
            .inst(InstructionKind::F64ConvertI64S)
            .returns(vec![ValueType::F64])
            .expect_stack(vec![Value::F64(42.0)]);

        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -42 })
            .inst(InstructionKind::F64ConvertI64S)
            .returns(vec![ValueType::F64])
            .expect_stack(vec![Value::F64(-42.0)]);

        // Unsigned conversion
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x8000000000000000u64 as i64,
            })
            .inst(InstructionKind::F64ConvertI64U)
            .returns(vec![ValueType::F64])
            .expect_stack(vec![Value::F64(9223372036854775808.0)]);
    }

    #[test]
    fn test_special_float_conversions() {
        // NaN handling in truncation - should trap
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: f32::NAN })
            .inst(InstructionKind::I32TruncF32S)
            .expect_error("invalid conversion to integer: NaN");

        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: f64::NAN })
            .inst(InstructionKind::I64TruncF64S)
            .expect_error("invalid conversion to integer: NaN");

        // Infinity handling - should trap
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: f32::INFINITY })
            .inst(InstructionKind::I32TruncF32S)
            .expect_error("integer overflow");

        ExecutorTest::new()
            .inst(InstructionKind::F64Const {
                value: f64::NEG_INFINITY,
            })
            .inst(InstructionKind::I64TruncF64S)
            .expect_error("integer overflow");
    }

    // ============================================================================
    // Sign Extension Tests
    // ============================================================================

    #[test]
    fn test_i32_extend8_s() {
        // Positive value (no sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x7F }) // 127
            .inst(InstructionKind::I32Extend8S)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x7F)]);

        // Negative value (sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x80 }) // 128 -> -128 in i8
            .inst(InstructionKind::I32Extend8S)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-128)]);

        // -1 in 8-bit
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0xFF }) // 255 -> -1 in i8
            .inst(InstructionKind::I32Extend8S)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]);

        // Value with upper bits set (should be ignored)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x123456FF })
            .inst(InstructionKind::I32Extend8S)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]); // Only lowest 8 bits matter
    }

    #[test]
    fn test_i32_extend16_s() {
        // Positive value (no sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x7FFF }) // 32767
            .inst(InstructionKind::I32Extend16S)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0x7FFF)]);

        // Negative value (sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x8000 }) // 32768 -> -32768 in i16
            .inst(InstructionKind::I32Extend16S)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-32768)]);

        // -1 in 16-bit
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0xFFFF }) // 65535 -> -1 in i16
            .inst(InstructionKind::I32Extend16S)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]);

        // Value with upper bits set (should be ignored)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0x1234FFFF })
            .inst(InstructionKind::I32Extend16S)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]); // Only lowest 16 bits matter
    }

    #[test]
    fn test_i64_extend8_s() {
        // Positive value (no sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0x7F }) // 127
            .inst(InstructionKind::I64Extend8S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x7F)]);

        // Negative value (sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0x80 }) // 128 -> -128 in i8
            .inst(InstructionKind::I64Extend8S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-128)]);

        // -1 in 8-bit
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0xFF }) // 255 -> -1 in i8
            .inst(InstructionKind::I64Extend8S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]);

        // Value with upper bits set (should be ignored)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x123456789ABCDEFF,
            })
            .inst(InstructionKind::I64Extend8S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]); // Only lowest 8 bits matter
    }

    #[test]
    fn test_i64_extend16_s() {
        // Positive value (no sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0x7FFF }) // 32767
            .inst(InstructionKind::I64Extend16S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x7FFF)]);

        // Negative value (sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0x8000 }) // 32768 -> -32768 in i16
            .inst(InstructionKind::I64Extend16S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-32768)]);

        // -1 in 16-bit
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0xFFFF }) // 65535 -> -1 in i16
            .inst(InstructionKind::I64Extend16S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]);

        // Value with upper bits set (should be ignored)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x123456789ABCFFFF,
            })
            .inst(InstructionKind::I64Extend16S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]); // Only lowest 16 bits matter
    }

    #[test]
    fn test_i64_extend32_s() {
        // Positive value (no sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0x7FFFFFFF }) // MAX i32
            .inst(InstructionKind::I64Extend32S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x7FFFFFFF)]);

        // Negative value (sign extension)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0x80000000 }) // MIN i32
            .inst(InstructionKind::I64Extend32S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-2147483648)]);

        // -1 in 32-bit
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0xFFFFFFFF }) // -1 in i32
            .inst(InstructionKind::I64Extend32S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]);

        // Value with upper bits set (should be ignored)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x12345678FFFFFFFF,
            })
            .inst(InstructionKind::I64Extend32S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]); // Only lowest 32 bits matter

        // Another example with upper bits
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0xABCDEF0012345678u64 as i64,
            })
            .inst(InstructionKind::I64Extend32S)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x12345678)]);
    }
}
