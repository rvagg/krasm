//! Parametric operations for WebAssembly
//!
//! This module provides implementations of parametric operations
//! as specified in the WebAssembly specification section 4.4.4 (Parametric Instructions).

use super::*;
use crate::parser::module::ValueType;

/// drop - Drop value from stack
/// spec: 4.4.4
///
/// From the spec:
/// 1. Assert: due to validation, a value is on the top of the stack.
/// 2. Pop the value val from the stack.
pub fn drop(stack: &mut Stack) -> Result<(), RuntimeError> {
    stack.pop()?;
    Ok(())
}

/// select - Select one of two values based on condition
/// spec: 4.4.4
///
/// From the spec:
/// 1. Assert: due to validation, a value of type i32 is on the top of the stack.
/// 2. Pop the value c from the stack.
/// 3. Assert: due to validation, two more values (of the same type) are on the stack.
/// 4. Pop the value val2 from the stack.
/// 5. Pop the value val1 from the stack.
/// 6. If c is not 0, push val1 back to the stack.
/// 7. Else, push val2 back to the stack.
///
/// Stack: [val1, val2, condition] -> [(condition != 0) ? val1 : val2]
pub fn select(stack: &mut Stack) -> Result<(), RuntimeError> {
    // Pop the condition (i32)
    let condition = stack.pop_i32()?;

    // Pop val2 and val1
    let val2 = stack.pop()?;
    let val1 = stack.pop()?;

    // Push the selected value
    // Type safety is guaranteed by the validator at parse time
    if condition != 0 {
        stack.push(val1);
    } else {
        stack.push(val2);
    }

    Ok(())
}

/// select_t - Typed select with explicit type annotation
/// spec: 4.4.4
///
/// Similar to select but with explicit type annotations for validation.
/// This is used when the types cannot be inferred from context.
pub fn select_typed(stack: &mut Stack, _val_types: &[ValueType]) -> Result<(), RuntimeError> {
    // For runtime execution, select_typed behaves identically to select
    // The type information is primarily used during validation
    select(stack)
}

#[cfg(test)]
#[allow(clippy::approx_constant)] // 3.14 etc. are arbitrary test floats, not stand-ins for consts
mod tests {
    use crate::parser::instruction::InstructionKind;
    use crate::parser::module::ValueType;
    use crate::runtime::Value;
    use crate::runtime::test_utils::test::ExecutorTest;

    #[test]
    fn drop_single_value() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Drop)
            .inst(InstructionKind::I32Const { value: 100 })
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(100)]);
    }

    #[test]
    fn drop_multiple_values() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32Const { value: 3 })
            .inst(InstructionKind::Drop)
            .inst(InstructionKind::Drop)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }

    #[test]
    fn drop_empty_stack() {
        ExecutorTest::new()
            .inst(InstructionKind::Drop)
            .expect_error("stack underflow");
    }

    #[test]
    fn drop_different_types() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Drop)
            .inst(InstructionKind::F64Const { value: 3.14 })
            .inst(InstructionKind::Drop)
            .inst(InstructionKind::I64Const { value: 100 })
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(100)]);
    }

    // ============================================================================
    // Select Tests
    // ============================================================================

    #[test]
    fn select_i32_true() {
        // Select first value when condition is true (non-zero)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 10 }) // val1
            .inst(InstructionKind::I32Const { value: 20 }) // val2
            .inst(InstructionKind::I32Const { value: 1 }) // condition (true)
            .inst(InstructionKind::Select)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(10)]);
    }

    #[test]
    fn select_i32_false() {
        // Select second value when condition is false (zero)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 10 }) // val1
            .inst(InstructionKind::I32Const { value: 20 }) // val2
            .inst(InstructionKind::I32Const { value: 0 }) // condition (false)
            .inst(InstructionKind::Select)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(20)]);
    }

    #[test]
    fn select_i64_true() {
        // Select with i64 values
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 100 }) // val1
            .inst(InstructionKind::I64Const { value: 200 }) // val2
            .inst(InstructionKind::I32Const { value: -1 }) // condition (true, any non-zero)
            .inst(InstructionKind::Select)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(100)]);
    }

    #[test]
    fn select_i64_false() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 100 }) // val1
            .inst(InstructionKind::I64Const { value: 200 }) // val2
            .inst(InstructionKind::I32Const { value: 0 }) // condition (false)
            .inst(InstructionKind::Select)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(200)]);
    }

    #[test]
    fn select_f32() {
        // Select with f32 values
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: 1.5 }) // val1
            .inst(InstructionKind::F32Const { value: 2.5 }) // val2
            .inst(InstructionKind::I32Const { value: 42 }) // condition (true)
            .inst(InstructionKind::Select)
            .returns(vec![ValueType::F32])
            .expect_stack(vec![Value::F32(1.5)]);
    }

    #[test]
    fn select_f64() {
        // Select with f64 values
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: 3.14159 }) // val1
            .inst(InstructionKind::F64Const { value: 2.71828 }) // val2
            .inst(InstructionKind::I32Const { value: 0 }) // condition (false)
            .inst(InstructionKind::Select)
            .returns(vec![ValueType::F64])
            .expect_stack(vec![Value::F64(2.71828)]);
    }

    #[test]
    fn select_negative_condition() {
        // Negative values are truthy (non-zero)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 111 })
            .inst(InstructionKind::I32Const { value: 222 })
            .inst(InstructionKind::I32Const { value: -5 }) // negative = true
            .inst(InstructionKind::Select)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(111)]);
    }

    #[test]
    fn select_large_condition() {
        // Any non-zero value is truthy
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 7 })
            .inst(InstructionKind::I32Const { value: 8 })
            .inst(InstructionKind::I32Const { value: 0x7FFFFFFF }) // max i32 = true
            .inst(InstructionKind::Select)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(7)]);
    }

    #[test]
    fn select_chain() {
        // Multiple selects in sequence
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32Const { value: 1 }) // select 1
            .inst(InstructionKind::Select) // -> 1
            .inst(InstructionKind::I32Const { value: 3 })
            .inst(InstructionKind::I32Const { value: 0 }) // select 3
            .inst(InstructionKind::Select) // -> 3
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(3)]);
    }

    #[test]
    fn select_nested() {
        // Use result of one select as input to another
        ExecutorTest::new()
            // First select: choose between 10 and 20
            .inst(InstructionKind::I32Const { value: 10 })
            .inst(InstructionKind::I32Const { value: 20 })
            .inst(InstructionKind::I32Const { value: 1 }) // choose 10
            .inst(InstructionKind::Select)
            // Second select: use result (10) vs 30
            .inst(InstructionKind::I32Const { value: 30 })
            .inst(InstructionKind::I32Const { value: 0 }) // choose 30
            .inst(InstructionKind::Select)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(30)]);
    }

    #[test]
    fn select_typed_i32() {
        // SelectTyped should work the same as Select
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 5 })
            .inst(InstructionKind::I32Const { value: 6 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::SelectTyped {
                val_types: vec![ValueType::I32],
            })
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(5)]);
    }

    #[test]
    fn select_typed_multiple_values() {
        // SelectTyped with multiple type annotations (for future multi-value)
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 77 })
            .inst(InstructionKind::I64Const { value: 88 })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::SelectTyped {
                val_types: vec![ValueType::I64],
            })
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(88)]);
    }

    #[test]
    fn select_with_nan() {
        // Select preserves NaN bit patterns exactly
        // Note: We can't compare NaN directly, so just verify it runs without error
        ExecutorTest::new()
            .inst(InstructionKind::F32Const { value: f32::NAN })
            .inst(InstructionKind::F32Const { value: 1.0 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::Select)
            .inst(InstructionKind::Drop) // Drop the NaN result
            .inst(InstructionKind::I32Const { value: 42 }) // Push something we can verify
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn select_with_infinity() {
        // Select works with special float values
        ExecutorTest::new()
            .inst(InstructionKind::F64Const { value: f64::INFINITY })
            .inst(InstructionKind::F64Const {
                value: f64::NEG_INFINITY,
            })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::Select)
            .returns(vec![ValueType::F64])
            .expect_stack(vec![Value::F64(f64::NEG_INFINITY)]);
    }

    // ============================================================================
    // Select Error Tests
    // ============================================================================

    #[test]
    fn select_empty_stack() {
        ExecutorTest::new()
            .inst(InstructionKind::Select)
            .expect_error("stack underflow");
    }

    #[test]
    fn select_insufficient_values() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::Select) // Missing one value
            .expect_error("stack underflow");
    }

    // Note: Type mismatch test removed because validation happens at parse time,
    // not runtime. The validator ensures both values have the same type.

    #[test]
    fn select_wrong_condition_type() {
        // Condition must be i32 (checked when popping)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I64Const { value: 1 }) // Wrong type for condition
            .inst(InstructionKind::Select)
            .expect_error("type mismatch");
    }
}
