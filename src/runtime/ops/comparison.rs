//! Comparison operations for WebAssembly
//!
//! This module provides implementations of comparison operations
//! as specified in the WebAssembly specification section 4.4.1.4 (Test Instructions)
//! and 4.4.1.5 (Comparison Instructions).
//!
//! All comparison operations return an i32 value: 1 for true, 0 for false.

// ============================================================================
// Integer Comparison Operations (i32)
// ============================================================================

/// i32.eqz - Test if i32 is zero
/// spec: 4.4.1.4
pub fn i32_eqz(value: i32) -> i32 {
    if value == 0 { 1 } else { 0 }
}

/// i32.eq - Test if two i32 values are equal
/// spec: 4.4.1.5
pub fn i32_eq(a: i32, b: i32) -> i32 {
    if a == b { 1 } else { 0 }
}

/// i32.ne - Test if two i32 values are not equal
/// spec: 4.4.1.5
pub fn i32_ne(a: i32, b: i32) -> i32 {
    if a != b { 1 } else { 0 }
}

/// i32.lt_s - Test if a < b (signed)
/// spec: 4.4.1.5
pub fn i32_lt_s(a: i32, b: i32) -> i32 {
    if a < b { 1 } else { 0 }
}

/// i32.lt_u - Test if a < b (unsigned)
/// spec: 4.4.1.5
pub fn i32_lt_u(a: i32, b: i32) -> i32 {
    if (a as u32) < (b as u32) { 1 } else { 0 }
}

/// i32.gt_s - Test if a > b (signed)
/// spec: 4.4.1.5
pub fn i32_gt_s(a: i32, b: i32) -> i32 {
    if a > b { 1 } else { 0 }
}

/// i32.gt_u - Test if a > b (unsigned)
/// spec: 4.4.1.5
pub fn i32_gt_u(a: i32, b: i32) -> i32 {
    if (a as u32) > (b as u32) { 1 } else { 0 }
}

/// i32.le_s - Test if a <= b (signed)
/// spec: 4.4.1.5
pub fn i32_le_s(a: i32, b: i32) -> i32 {
    if a <= b { 1 } else { 0 }
}

/// i32.le_u - Test if a <= b (unsigned)
/// spec: 4.4.1.5
pub fn i32_le_u(a: i32, b: i32) -> i32 {
    if (a as u32) <= (b as u32) { 1 } else { 0 }
}

/// i32.ge_s - Test if a >= b (signed)
/// spec: 4.4.1.5
pub fn i32_ge_s(a: i32, b: i32) -> i32 {
    if a >= b { 1 } else { 0 }
}

/// i32.ge_u - Test if a >= b (unsigned)
/// spec: 4.4.1.5
pub fn i32_ge_u(a: i32, b: i32) -> i32 {
    if (a as u32) >= (b as u32) { 1 } else { 0 }
}

// ============================================================================
// Integer Comparison Operations (i64)
// ============================================================================

/// i64.eqz - Test if i64 is zero
/// spec: 4.4.1.4
pub fn i64_eqz(value: i64) -> i32 {
    if value == 0 { 1 } else { 0 }
}

/// i64.eq - Test if two i64 values are equal
/// spec: 4.4.1.5
pub fn i64_eq(a: i64, b: i64) -> i32 {
    if a == b { 1 } else { 0 }
}

/// i64.ne - Test if two i64 values are not equal
/// spec: 4.4.1.5
pub fn i64_ne(a: i64, b: i64) -> i32 {
    if a != b { 1 } else { 0 }
}

/// i64.lt_s - Test if a < b (signed)
/// spec: 4.4.1.5
pub fn i64_lt_s(a: i64, b: i64) -> i32 {
    if a < b { 1 } else { 0 }
}

/// i64.lt_u - Test if a < b (unsigned)
/// spec: 4.4.1.5
pub fn i64_lt_u(a: i64, b: i64) -> i32 {
    if (a as u64) < (b as u64) { 1 } else { 0 }
}

/// i64.gt_s - Test if a > b (signed)
/// spec: 4.4.1.5
pub fn i64_gt_s(a: i64, b: i64) -> i32 {
    if a > b { 1 } else { 0 }
}

/// i64.gt_u - Test if a > b (unsigned)
/// spec: 4.4.1.5
pub fn i64_gt_u(a: i64, b: i64) -> i32 {
    if (a as u64) > (b as u64) { 1 } else { 0 }
}

/// i64.le_s - Test if a <= b (signed)
/// spec: 4.4.1.5
pub fn i64_le_s(a: i64, b: i64) -> i32 {
    if a <= b { 1 } else { 0 }
}

/// i64.le_u - Test if a <= b (unsigned)
/// spec: 4.4.1.5
pub fn i64_le_u(a: i64, b: i64) -> i32 {
    if (a as u64) <= (b as u64) { 1 } else { 0 }
}

/// i64.ge_s - Test if a >= b (signed)
/// spec: 4.4.1.5
pub fn i64_ge_s(a: i64, b: i64) -> i32 {
    if a >= b { 1 } else { 0 }
}

/// i64.ge_u - Test if a >= b (unsigned)
/// spec: 4.4.1.5
pub fn i64_ge_u(a: i64, b: i64) -> i32 {
    if (a as u64) >= (b as u64) { 1 } else { 0 }
}

// ============================================================================
// Floating-Point Comparison Operations (f32)
// ============================================================================

/// f32.eq - Test if two f32 values are equal
/// spec: 4.4.1.5
/// Note: NaN != NaN, -0.0 == +0.0
pub fn f32_eq(a: f32, b: f32) -> i32 {
    if a == b { 1 } else { 0 }
}

/// f32.ne - Test if two f32 values are not equal
/// spec: 4.4.1.5
/// Note: NaN != NaN returns true
pub fn f32_ne(a: f32, b: f32) -> i32 {
    if a != b { 1 } else { 0 }
}

/// f32.lt - Test if a < b
/// spec: 4.4.1.5
/// Note: NaN comparisons return false
pub fn f32_lt(a: f32, b: f32) -> i32 {
    if a < b { 1 } else { 0 }
}

/// f32.gt - Test if a > b
/// spec: 4.4.1.5
pub fn f32_gt(a: f32, b: f32) -> i32 {
    if a > b { 1 } else { 0 }
}

/// f32.le - Test if a <= b
/// spec: 4.4.1.5
pub fn f32_le(a: f32, b: f32) -> i32 {
    if a <= b { 1 } else { 0 }
}

/// f32.ge - Test if a >= b
/// spec: 4.4.1.5
pub fn f32_ge(a: f32, b: f32) -> i32 {
    if a >= b { 1 } else { 0 }
}

// ============================================================================
// Floating-Point Comparison Operations (f64)
// ============================================================================

/// f64.eq - Test if two f64 values are equal
/// spec: 4.4.1.5
pub fn f64_eq(a: f64, b: f64) -> i32 {
    if a == b { 1 } else { 0 }
}

/// f64.ne - Test if two f64 values are not equal
/// spec: 4.4.1.5
pub fn f64_ne(a: f64, b: f64) -> i32 {
    if a != b { 1 } else { 0 }
}

/// f64.lt - Test if a < b
/// spec: 4.4.1.5
pub fn f64_lt(a: f64, b: f64) -> i32 {
    if a < b { 1 } else { 0 }
}

/// f64.gt - Test if a > b
/// spec: 4.4.1.5
pub fn f64_gt(a: f64, b: f64) -> i32 {
    if a > b { 1 } else { 0 }
}

/// f64.le - Test if a <= b
/// spec: 4.4.1.5
pub fn f64_le(a: f64, b: f64) -> i32 {
    if a <= b { 1 } else { 0 }
}

/// f64.ge - Test if a >= b
/// spec: 4.4.1.5
pub fn f64_ge(a: f64, b: f64) -> i32 {
    if a >= b { 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use crate::parser::instruction::InstructionKind;
    use crate::parser::module::ValueType;
    use crate::runtime::Value;
    use crate::runtime::test_utils::test::ExecutorTest;

    // ============================================================================
    // i32 Test Operations
    // ============================================================================

    #[test]
    fn test_i32_eqz() {
        // Test zero
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::I32Eqz)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // Test non-zero positive
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Eqz)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);

        // Test non-zero negative
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Eqz)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    // ============================================================================
    // i32 Comparison Operations
    // ============================================================================

    #[test]
    fn test_i32_eq() {
        // Equal values
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Eq)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // Different values
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 43 })
            .inst(InstructionKind::I32Eq)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);

        // Negative values equal
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Eq)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }

    #[test]
    fn test_i32_ne() {
        // Different values
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 43 })
            .inst(InstructionKind::I32Ne)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // Equal values
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Ne)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn test_i32_lt_s() {
        // 1 < 2 (true)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32LtS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // 2 < 1 (false)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32LtS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);

        // -2 < -1 (true, signed)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -2 })
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32LtS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }

    #[test]
    fn test_i32_lt_u() {
        // -1 as unsigned is greater than 1
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32LtU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);

        // 1 < 2 (true)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32LtU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }

    #[test]
    fn test_i32_gt_s() {
        // 2 > 1 (true)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32GtS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // 1 > 2 (false)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32GtS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn test_i32_le_s() {
        // 1 <= 2 (true)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32LeS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // 2 <= 2 (true)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32LeS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // 3 <= 2 (false)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 3 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32LeS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn test_i32_ge_s() {
        // 2 >= 1 (true)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32GeS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // More tests for unsigned comparisons
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32GeU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32GtU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32LeU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }

    // ============================================================================
    // i64 Comparison Operations
    // ============================================================================

    #[test]
    fn test_i64_eqz() {
        // Test zero
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 0 })
            .inst(InstructionKind::I64Eqz)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // Test non-zero
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: 0x123456789ABCDEF0u64 as i64,
            })
            .inst(InstructionKind::I64Eqz)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn test_i64_comparisons() {
        // i64.eq
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 42 })
            .inst(InstructionKind::I64Const { value: 42 })
            .inst(InstructionKind::I64Eq)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // i64.ne
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 42 })
            .inst(InstructionKind::I64Const { value: 43 })
            .inst(InstructionKind::I64Ne)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // i64.lt_s
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64LtS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // i64.lt_u
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64LtU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);

        // i64.gt_s
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64GtS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // i64.gt_u
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64GtU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // i64.le_s
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64LeS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // i64.le_u
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Const { value: 2 })
            .inst(InstructionKind::I64LeU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // i64.ge_s
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64GeS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // i64.ge_u
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64GeU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }

    // ============================================================================
    // f32 Comparison Operations
    // ============================================================================

    #[test]
    fn test_f32_eq() {
        // Equal values
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 42.0 })
            .inst(InstructionKind::F32Const { value: 42.0 })
            .inst(InstructionKind::F32Eq)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // Different values
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 42.0 })
            .inst(InstructionKind::F32Const { value: 43.0 })
            .inst(InstructionKind::F32Eq)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);

        // NaN never equals anything, even itself
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: f32::NAN })
            .inst(InstructionKind::F32Const { value: f32::NAN })
            .inst(InstructionKind::F32Eq)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn test_f32_ne() {
        // Different values
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 42.0 })
            .inst(InstructionKind::F32Const { value: 43.0 })
            .inst(InstructionKind::F32Ne)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // NaN is not equal to itself
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: f32::NAN })
            .inst(InstructionKind::F32Const { value: f32::NAN })
            .inst(InstructionKind::F32Ne)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }

    #[test]
    fn test_f32_lt() {
        // 1.0 < 2.0
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 1.0 })
            .inst(InstructionKind::F32Const { value: 2.0 })
            .inst(InstructionKind::F32Lt)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // NaN comparisons always false
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: f32::NAN })
            .inst(InstructionKind::F32Const { value: 1.0 })
            .inst(InstructionKind::F32Lt)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);

        // Other comparisons
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 2.0 })
            .inst(InstructionKind::F32Const { value: 1.0 })
            .inst(InstructionKind::F32Gt)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 1.0 })
            .inst(InstructionKind::F32Const { value: 2.0 })
            .inst(InstructionKind::F32Le)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 2.0 })
            .inst(InstructionKind::F32Const { value: 1.0 })
            .inst(InstructionKind::F32Ge)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }

    #[test]
    fn test_f32_infinity() {
        // Infinity is greater than any finite number
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: f32::INFINITY })
            .inst(InstructionKind::F32Const { value: f32::MAX })
            .inst(InstructionKind::F32Gt)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // -Infinity is less than any finite number
        ExecutorTest::new()
            .inst(InstructionKind::F32Const {
                value: f32::NEG_INFINITY,
            })
            .inst(InstructionKind::F32Const { value: f32::MIN })
            .inst(InstructionKind::F32Lt)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // Infinity equals itself
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: f32::INFINITY })
            .inst(InstructionKind::F32Const { value: f32::INFINITY })
            .inst(InstructionKind::F32Eq)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }

    // ============================================================================
    // f64 Comparison Operations
    // ============================================================================

    #[test]
    fn test_f64_comparisons() {
        // f64.eq
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: 42.0 })
            .inst(InstructionKind::F64Const { value: 42.0 })
            .inst(InstructionKind::F64Eq)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // f64.ne with NaN
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: f64::NAN })
            .inst(InstructionKind::F64Const { value: f64::NAN })
            .inst(InstructionKind::F64Ne)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // f64.lt
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: -1.0 })
            .inst(InstructionKind::F64Const { value: 1.0 })
            .inst(InstructionKind::F64Lt)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // f64.gt
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: 1.0 })
            .inst(InstructionKind::F64Const { value: -1.0 })
            .inst(InstructionKind::F64Gt)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // f64.le
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: 1.0 })
            .inst(InstructionKind::F64Const { value: 1.0 })
            .inst(InstructionKind::F64Le)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // f64.ge
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: 1.0 })
            .inst(InstructionKind::F64Const { value: 1.0 })
            .inst(InstructionKind::F64Ge)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        // Infinity comparisons
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: f64::INFINITY })
            .inst(InstructionKind::F64Const { value: 1000.0 })
            .inst(InstructionKind::F64Gt)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);

        ExecutorTest::new()
            .inst(InstructionKind::F64Const {
                value: f64::NEG_INFINITY,
            })
            .inst(InstructionKind::F64Const { value: -1000.0 })
            .inst(InstructionKind::F64Lt)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }
}
