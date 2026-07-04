//! Numeric operations for WebAssembly
//!
//! This module implements numeric instructions as defined in the WebAssembly specification:
//! - Section 4.4.1: Numeric Instructions
//! - Section 4.4.1.1: Constants (t.const)
//! - Section 4.4.1.2: Unary operations (t.unop)
//! - Section 4.4.1.3: Binary operations (t.binop)
//! - Section 4.4.1.4: Test operations (t.testop)
//! - Section 4.4.1.5: Comparison operations (t.relop)

use super::{RuntimeError, Stack, Value};

// ============================================================================
// Numeric Constants (Section 4.4.1.1)
// ============================================================================

/// i32.const
/// From the spec (4.4.1.1):
/// Push the value i32.const c to the stack.
pub fn i32_const(stack: &mut Stack, value: i32) -> Result<(), RuntimeError> {
    stack.push(Value::I32(value));
    Ok(())
}

/// i64.const
/// From the spec (4.4.1.1):
/// Push the value i64.const c to the stack.
pub fn i64_const(stack: &mut Stack, value: i64) -> Result<(), RuntimeError> {
    stack.push(Value::I64(value));
    Ok(())
}

/// f32.const
/// From the spec (4.4.1.1):
/// Push the value f32.const c to the stack.
pub fn f32_const(stack: &mut Stack, value: f32) -> Result<(), RuntimeError> {
    stack.push(Value::F32(value));
    Ok(())
}

/// f64.const
/// From the spec (4.4.1.1):
/// Push the value f64.const c to the stack.
pub fn f64_const(stack: &mut Stack, value: f64) -> Result<(), RuntimeError> {
    stack.push(Value::F64(value));
    Ok(())
}

// ============================================================================
// Integer Binary Operations (Section 4.4.1.3)
// ============================================================================

/// i32.add
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i32, trap
/// 4. Compute c1 + c2 modulo 2^32
/// 5. Push result to stack
pub fn i32_add(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i32()?;
    let c1 = stack.pop_i32()?;
    stack.push(Value::I32(c1.wrapping_add(c2)));
    Ok(())
}

/// i32.sub
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i32, trap
/// 4. Compute c1 - c2 modulo 2^32
/// 5. Push result to stack
pub fn i32_sub(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i32()?;
    let c1 = stack.pop_i32()?;
    stack.push(Value::I32(c1.wrapping_sub(c2)));
    Ok(())
}

/// i32.mul
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i32, trap
/// 4. Compute c1 * c2 modulo 2^32
/// 5. Push result to stack
pub fn i32_mul(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i32()?;
    let c1 = stack.pop_i32()?;
    stack.push(Value::I32(c1.wrapping_mul(c2)));
    Ok(())
}

/// i32.div_s
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i32, trap
/// 4. If c2 is 0, trap
/// 5. If c1 is INT32_MIN and c2 is -1, trap (result 2^31 doesn't fit in i32)
/// 6. Compute signed division, truncating toward zero
/// 7. Push result to stack
pub fn i32_div_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i32()?;
    let c1 = stack.pop_i32()?;
    if c2 == 0 {
        return Err(RuntimeError::DivisionByZero);
    }
    // Check for overflow: i32::MIN / -1
    if c1 == i32::MIN && c2 == -1 {
        return Err(RuntimeError::IntegerOverflow);
    }
    stack.push(Value::I32(c1 / c2));
    Ok(())
}

/// i32.div_u
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i32, trap
/// 4. If c2 is 0, trap
/// 5. Compute unsigned division
/// 6. Push result to stack
pub fn i32_div_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i32()?;
    let c1 = stack.pop_i32()?;
    if c2 == 0 {
        return Err(RuntimeError::DivisionByZero);
    }
    // Interpret as unsigned for division
    let u1 = c1 as u32;
    let u2 = c2 as u32;
    stack.push(Value::I32((u1 / u2) as i32));
    Ok(())
}

/// i32.rem_s
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i32, trap
/// 4. If c2 is 0, trap
/// 5. Compute signed remainder
/// 6. Push result to stack
///
/// Note: any integer % ±1 = 0 mathematically
pub fn i32_rem_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i32()?;
    let c1 = stack.pop_i32()?;
    if c2 == 0 {
        return Err(RuntimeError::DivisionByZero);
    }
    // Mathematical fact: any integer % ±1 = 0
    // Special handling for i32::MIN % -1 to avoid overflow
    if c2 == 1 || c2 == -1 {
        stack.push(Value::I32(0));
    } else {
        stack.push(Value::I32(c1 % c2));
    }
    Ok(())
}

/// i32.rem_u
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i32, trap
/// 4. If c2 is 0, trap
/// 5. Compute unsigned remainder
/// 6. Push result to stack
pub fn i32_rem_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i32()?;
    let c1 = stack.pop_i32()?;
    if c2 == 0 {
        return Err(RuntimeError::DivisionByZero);
    }
    // Optimise: any unsigned % 1 = 0
    if c2 == 1 {
        stack.push(Value::I32(0));
    } else {
        // Interpret as unsigned for remainder
        let u1 = c1 as u32;
        let u2 = c2 as u32;
        stack.push(Value::I32((u1 % u2) as i32));
    }
    Ok(())
}

// ============================================================================
// Integer Unary Operations (Section 4.4.1.2)
// ============================================================================

/// i32.clz - Count leading zeros
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute clz(c1) = count of leading zero bits
/// 3. Push result to stack
pub fn i32_clz(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_i32()?;
    stack.push(Value::I32(value.leading_zeros() as i32));
    Ok(())
}

/// i32.ctz - Count trailing zeros
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute ctz(c1) = count of trailing zero bits
/// 3. Push result to stack
pub fn i32_ctz(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_i32()?;
    stack.push(Value::I32(value.trailing_zeros() as i32));
    Ok(())
}

/// i32.popcnt - Population count
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute popcnt(c1) = count of non-zero bits
/// 3. Push result to stack
pub fn i32_popcnt(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_i32()?;
    stack.push(Value::I32(value.count_ones() as i32));
    Ok(())
}

// ============================================================================
// i64 Binary Operations (Section 4.4.1.3)
// ============================================================================

/// i64.add
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i64, trap
/// 4. Compute c1 + c2 modulo 2^64
/// 5. Push result to stack
pub fn i64_add(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i64()?;
    let c1 = stack.pop_i64()?;
    stack.push(Value::I64(c1.wrapping_add(c2)));
    Ok(())
}

/// i64.sub
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i64, trap
/// 4. Compute c1 - c2 modulo 2^64
/// 5. Push result to stack
pub fn i64_sub(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i64()?;
    let c1 = stack.pop_i64()?;
    stack.push(Value::I64(c1.wrapping_sub(c2)));
    Ok(())
}

/// i64.mul
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i64, trap
/// 4. Compute c1 * c2 modulo 2^64
/// 5. Push result to stack
pub fn i64_mul(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i64()?;
    let c1 = stack.pop_i64()?;
    stack.push(Value::I64(c1.wrapping_mul(c2)));
    Ok(())
}

/// i64.div_s
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i64, trap
/// 4. If c2 is 0, trap
/// 5. If c1 is INT64_MIN and c2 is -1, trap (result 2^63 doesn't fit in i64)
/// 6. Compute signed division, truncating toward zero
/// 7. Push result to stack
pub fn i64_div_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i64()?;
    let c1 = stack.pop_i64()?;
    if c2 == 0 {
        return Err(RuntimeError::DivisionByZero);
    }
    // Check for overflow: i64::MIN / -1
    if c1 == i64::MIN && c2 == -1 {
        return Err(RuntimeError::IntegerOverflow);
    }
    stack.push(Value::I64(c1 / c2));
    Ok(())
}

/// i64.div_u
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i64, trap
/// 4. If c2 is 0, trap
/// 5. Compute unsigned division
/// 6. Push result to stack
pub fn i64_div_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i64()?;
    let c1 = stack.pop_i64()?;
    if c2 == 0 {
        return Err(RuntimeError::DivisionByZero);
    }
    // Interpret as unsigned for division
    let u1 = c1 as u64;
    let u2 = c2 as u64;
    stack.push(Value::I64((u1 / u2) as i64));
    Ok(())
}

/// i64.rem_s
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i64, trap
/// 4. If c2 is 0, trap
/// 5. Compute signed remainder
/// 6. Push result to stack
///
/// Note: any integer % ±1 = 0 mathematically
pub fn i64_rem_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i64()?;
    let c1 = stack.pop_i64()?;
    if c2 == 0 {
        return Err(RuntimeError::DivisionByZero);
    }
    // Mathematical fact: any integer % ±1 = 0
    // Special handling for i64::MIN % -1 to avoid overflow
    if c2 == 1 || c2 == -1 {
        stack.push(Value::I64(0));
    } else {
        stack.push(Value::I64(c1 % c2));
    }
    Ok(())
}

/// i64.rem_u
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. If either value is not i64, trap
/// 4. If c2 is 0, trap
/// 5. Compute unsigned remainder
/// 6. Push result to stack
pub fn i64_rem_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_i64()?;
    let c1 = stack.pop_i64()?;
    if c2 == 0 {
        return Err(RuntimeError::DivisionByZero);
    }
    // Optimise: any unsigned % 1 = 0
    if c2 == 1 {
        stack.push(Value::I64(0));
    } else {
        // Interpret as unsigned for remainder
        let u1 = c1 as u64;
        let u2 = c2 as u64;
        stack.push(Value::I64((u1 % u2) as i64));
    }
    Ok(())
}

// ============================================================================
// Floating-Point Unary Operations (Section 4.4.1.2)
// ============================================================================

/// f32.abs - Absolute value
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute abs(c1) = absolute value of c1
/// 3. Push result to stack
///
/// Note: Preserves NaN payload and sign handling per IEEE 754
pub fn f32_abs(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f32()?;
    stack.push(Value::F32(value.abs()));
    Ok(())
}

/// f32.neg - Negation
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute neg(c1) = negation of c1
/// 3. Push result to stack
///
/// Note: Flips sign bit, including for NaN and infinity
pub fn f32_neg(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f32()?;
    stack.push(Value::F32(f32::from_bits(value.to_bits() ^ 0x80000000)));
    Ok(())
}

/// f32.sqrt - Square root
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute sqrt(c1) per IEEE 754-2019
/// 3. Push result to stack
///
/// Note: Returns NaN for negative inputs
pub fn f32_sqrt(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f32()?;
    stack.push(Value::F32(value.sqrt()));
    Ok(())
}

/// f32.ceil - Ceiling
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute ceil(c1) = smallest integer >= c1
/// 3. Push result to stack
///
/// Note: Preserves NaN, ±∞, and ±0
pub fn f32_ceil(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f32()?;
    stack.push(Value::F32(value.ceil()));
    Ok(())
}

/// f32.floor - Floor
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute floor(c1) = largest integer <= c1
/// 3. Push result to stack
///
/// Note: Preserves NaN, ±∞, and ±0
pub fn f32_floor(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f32()?;
    stack.push(Value::F32(value.floor()));
    Ok(())
}

/// f32.trunc - Truncate
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute trunc(c1) = round toward zero
/// 3. Push result to stack
///
/// Note: Discards fractional part, preserves sign
pub fn f32_trunc(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f32()?;
    stack.push(Value::F32(value.trunc()));
    Ok(())
}

/// f32.nearest - Round to nearest even
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute nearest(c1) = round to nearest integer
/// 3. If tie, round to even (banker's rounding)
/// 4. Push result to stack
///
/// Note: Rust's round_ties_even() implements IEEE 754 roundTiesToEven
pub fn f32_nearest(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f32()?;
    stack.push(Value::F32(value.round_ties_even()));
    Ok(())
}

/// f64.abs - Absolute value
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute abs(c1) = absolute value of c1
/// 3. Push result to stack
///
/// Note: Preserves NaN payload and sign handling per IEEE 754
pub fn f64_abs(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f64()?;
    stack.push(Value::F64(value.abs()));
    Ok(())
}

/// f64.neg - Negation
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute neg(c1) = negation of c1
/// 3. Push result to stack
///
/// Note: Flips sign bit, including for NaN and infinity
pub fn f64_neg(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f64()?;
    stack.push(Value::F64(f64::from_bits(value.to_bits() ^ 0x8000000000000000)));
    Ok(())
}

/// f64.sqrt - Square root
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute sqrt(c1) per IEEE 754-2019
/// 3. Push result to stack
///
/// Note: Returns NaN for negative inputs
pub fn f64_sqrt(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f64()?;
    stack.push(Value::F64(value.sqrt()));
    Ok(())
}

/// f64.ceil - Ceiling
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute ceil(c1) = smallest integer >= c1
/// 3. Push result to stack
///
/// Note: Preserves NaN, ±∞, and ±0
pub fn f64_ceil(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f64()?;
    stack.push(Value::F64(value.ceil()));
    Ok(())
}

/// f64.floor - Floor
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute floor(c1) = largest integer <= c1
/// 3. Push result to stack
///
/// Note: Preserves NaN, ±∞, and ±0
pub fn f64_floor(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f64()?;
    stack.push(Value::F64(value.floor()));
    Ok(())
}

/// f64.trunc - Truncate
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute trunc(c1) = round toward zero
/// 3. Push result to stack
///
/// Note: Discards fractional part, preserves sign
pub fn f64_trunc(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f64()?;
    stack.push(Value::F64(value.trunc()));
    Ok(())
}

/// f64.nearest - Round to nearest even
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute nearest(c1) = round to nearest integer
/// 3. If tie, round to even (banker's rounding)
/// 4. Push result to stack
///
/// Note: Rust's round_ties_even() implements IEEE 754 roundTiesToEven
pub fn f64_nearest(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_f64()?;
    stack.push(Value::F64(value.round_ties_even()));
    Ok(())
}

// ============================================================================
// Floating-Point Binary Operations (Section 4.4.1.3)
// ============================================================================

/// f32.add
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. Compute fadd(c1, c2) per IEEE 754-2019
/// 4. Push result to stack
///
/// Note: NaN propagation is non-deterministic per spec
pub fn f32_add(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f32()?;
    let c1 = stack.pop_f32()?;
    stack.push(Value::F32(c1 + c2));
    Ok(())
}

/// f32.sub
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. Compute fsub(c1, c2) per IEEE 754-2019
/// 4. Push result to stack
pub fn f32_sub(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f32()?;
    let c1 = stack.pop_f32()?;
    stack.push(Value::F32(c1 - c2));
    Ok(())
}

/// f32.mul
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. Compute fmul(c1, c2) per IEEE 754-2019
/// 4. Push result to stack
pub fn f32_mul(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f32()?;
    let c1 = stack.pop_f32()?;
    stack.push(Value::F32(c1 * c2));
    Ok(())
}

/// f32.div
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. Compute fdiv(c1, c2) per IEEE 754-2019
/// 4. Push result to stack
///
/// Note: Division by zero returns ±∞ per IEEE 754
pub fn f32_div(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f32()?;
    let c1 = stack.pop_f32()?;
    stack.push(Value::F32(c1 / c2));
    Ok(())
}

/// f32.min
/// From the spec (4.4.1.3 - t.binop):
/// Returns the minimum of two values
/// Special cases: -0.0 < +0.0, NaN propagation
pub fn f32_min(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f32()?;
    let c1 = stack.pop_f32()?;
    // WebAssembly spec requires special NaN and -0.0 handling
    let result = if c1.is_nan() || c2.is_nan() {
        f32::NAN
    } else if c1 == 0.0 && c2 == 0.0 && c1.is_sign_negative() != c2.is_sign_negative() {
        // -0.0 is less than +0.0
        if c1.is_sign_negative() { c1 } else { c2 }
    } else {
        c1.min(c2)
    };
    stack.push(Value::F32(result));
    Ok(())
}

/// f32.max
/// From the spec (4.4.1.3 - t.binop):
/// Returns the maximum of two values
/// Special cases: +0.0 > -0.0, NaN propagation
pub fn f32_max(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f32()?;
    let c1 = stack.pop_f32()?;
    // WebAssembly spec requires special NaN and -0.0 handling
    let result = if c1.is_nan() || c2.is_nan() {
        f32::NAN
    } else if c1 == 0.0 && c2 == 0.0 && c1.is_sign_negative() != c2.is_sign_negative() {
        // +0.0 is greater than -0.0
        if c1.is_sign_negative() { c2 } else { c1 }
    } else {
        c1.max(c2)
    };
    stack.push(Value::F32(result));
    Ok(())
}

/// f32.copysign
/// From the spec (4.4.1.3 - t.binop):
/// Returns c1 with the sign of c2
pub fn f32_copysign(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f32()?;
    let c1 = stack.pop_f32()?;
    stack.push(Value::F32(c1.copysign(c2)));
    Ok(())
}

/// f64.add
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. Compute fadd(c1, c2) per IEEE 754-2019
/// 4. Push result to stack
///
/// Note: NaN propagation is non-deterministic per spec
pub fn f64_add(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f64()?;
    let c1 = stack.pop_f64()?;
    stack.push(Value::F64(c1 + c2));
    Ok(())
}

/// f64.sub
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. Compute fsub(c1, c2) per IEEE 754-2019
/// 4. Push result to stack
pub fn f64_sub(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f64()?;
    let c1 = stack.pop_f64()?;
    stack.push(Value::F64(c1 - c2));
    Ok(())
}

/// f64.mul
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. Compute fmul(c1, c2) per IEEE 754-2019
/// 4. Push result to stack
pub fn f64_mul(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f64()?;
    let c1 = stack.pop_f64()?;
    stack.push(Value::F64(c1 * c2));
    Ok(())
}

/// f64.div
/// From the spec (4.4.1.3 - t.binop):
/// 1. Pop value c2 from stack
/// 2. Pop value c1 from stack
/// 3. Compute fdiv(c1, c2) per IEEE 754-2019
/// 4. Push result to stack
///
/// Note: Division by zero returns ±∞ per IEEE 754
pub fn f64_div(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f64()?;
    let c1 = stack.pop_f64()?;
    stack.push(Value::F64(c1 / c2));
    Ok(())
}

/// f64.min
/// From the spec (4.4.1.3 - t.binop):
/// Returns the minimum of two values
/// Special cases: -0.0 < +0.0, NaN propagation
pub fn f64_min(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f64()?;
    let c1 = stack.pop_f64()?;
    // WebAssembly spec requires special NaN and -0.0 handling
    let result = if c1.is_nan() || c2.is_nan() {
        f64::NAN
    } else if c1 == 0.0 && c2 == 0.0 && c1.is_sign_negative() != c2.is_sign_negative() {
        // -0.0 is less than +0.0
        if c1.is_sign_negative() { c1 } else { c2 }
    } else {
        c1.min(c2)
    };
    stack.push(Value::F64(result));
    Ok(())
}

/// f64.max
/// From the spec (4.4.1.3 - t.binop):
/// Returns the maximum of two values
/// Special cases: +0.0 > -0.0, NaN propagation
pub fn f64_max(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f64()?;
    let c1 = stack.pop_f64()?;
    // WebAssembly spec requires special NaN and -0.0 handling
    let result = if c1.is_nan() || c2.is_nan() {
        f64::NAN
    } else if c1 == 0.0 && c2 == 0.0 && c1.is_sign_negative() != c2.is_sign_negative() {
        // +0.0 is greater than -0.0
        if c1.is_sign_negative() { c2 } else { c1 }
    } else {
        c1.max(c2)
    };
    stack.push(Value::F64(result));
    Ok(())
}

/// f64.copysign
/// From the spec (4.4.1.3 - t.binop):
/// Returns c1 with the sign of c2
pub fn f64_copysign(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c2 = stack.pop_f64()?;
    let c1 = stack.pop_f64()?;
    stack.push(Value::F64(c1.copysign(c2)));
    Ok(())
}

// ============================================================================
// i64 Unary Operations (Section 4.4.1.2)
// ============================================================================

/// i64.clz - Count leading zeros
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute clz(c1) = count of leading zero bits
/// 3. Push result to stack
pub fn i64_clz(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_i64()?;
    stack.push(Value::I64(value.leading_zeros() as i64));
    Ok(())
}

/// i64.ctz - Count trailing zeros
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute ctz(c1) = count of trailing zero bits
/// 3. Push result to stack
pub fn i64_ctz(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_i64()?;
    stack.push(Value::I64(value.trailing_zeros() as i64));
    Ok(())
}

/// i64.popcnt - Population count
/// From the spec (4.4.1.2 - t.unop):
/// 1. Pop value c1 from stack
/// 2. Compute popcnt(c1) = count of non-zero bits
/// 3. Push result to stack
pub fn i64_popcnt(stack: &mut Stack) -> Result<(), RuntimeError> {
    let value = stack.pop_i64()?;
    stack.push(Value::I64(value.count_ones() as i64));
    Ok(())
}

#[cfg(test)]
#[allow(clippy::approx_constant)] // 3.14 etc. are arbitrary test floats, not stand-ins for consts
mod tests {
    use crate::parser::instruction::InstructionKind;
    use crate::parser::module::ValueType;
    use crate::runtime::Value;
    use crate::runtime::test_utils::test::ExecutorTest;

    mod constants {
        use super::*;

        #[test]
        fn i32_const() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 42 })
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(42)]);

            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: -1 })
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(-1)]);

            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: i32::MAX })
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(i32::MAX)]);
        }

        #[test]
        fn i64_const() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: 42 })
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(42)]);

            ExecutorTest::new()
                .inst(InstructionKind::I64Const {
                    value: 0x123456789ABCDEF0u64 as i64,
                })
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(0x123456789ABCDEF0u64 as i64)]);
        }

        #[test]
        fn f32_const() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 3.14 })
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(3.14)]);

            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: f32::INFINITY })
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(f32::INFINITY)]);
        }

        #[test]
        fn f64_const() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 3.14159 })
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(3.14159)]);

            ExecutorTest::new()
                .inst(InstructionKind::F64Const {
                    value: f64::NEG_INFINITY,
                })
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(f64::NEG_INFINITY)]);
        }
    }

    mod i32_binary_ops {
        use super::*;

        #[test]
        fn i32_add_basic() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 5 })
                .inst(InstructionKind::I32Const { value: 3 })
                .inst(InstructionKind::I32Add)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(8)]);
        }

        #[test]
        fn i32_add_overflow() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: i32::MAX })
                .inst(InstructionKind::I32Const { value: 1 })
                .inst(InstructionKind::I32Add)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(i32::MIN)]);
        }

        #[test]
        fn i32_add_negative() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: -10 })
                .inst(InstructionKind::I32Const { value: 5 })
                .inst(InstructionKind::I32Add)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(-5)]);
        }

        #[test]
        fn i32_sub_basic() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 10 })
                .inst(InstructionKind::I32Const { value: 3 })
                .inst(InstructionKind::I32Sub)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(7)]);
        }

        #[test]
        fn i32_sub_underflow() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: i32::MIN })
                .inst(InstructionKind::I32Const { value: 1 })
                .inst(InstructionKind::I32Sub)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(i32::MAX)]);
        }

        #[test]
        fn i32_div_s_by_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 42 })
                .inst(InstructionKind::I32Const { value: 0 })
                .inst(InstructionKind::I32DivS)
                .expect_error("integer divide by zero");
        }

        #[test]
        fn i32_div_s_overflow() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: i32::MIN })
                .inst(InstructionKind::I32Const { value: -1 })
                .inst(InstructionKind::I32DivS)
                .expect_error("integer overflow");
        }

        #[test]
        fn i32_rem_s_by_negative_one() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: -100 })
                .inst(InstructionKind::I32Const { value: -1 })
                .inst(InstructionKind::I32RemS)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(0)]);
        }

        #[test]
        fn i32_rem_s_special_case() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: i32::MIN })
                .inst(InstructionKind::I32Const { value: -1 })
                .inst(InstructionKind::I32RemS)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(0)]);
        }

        #[test]
        fn i32_mul_basic() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 6 })
                .inst(InstructionKind::I32Const { value: 7 })
                .inst(InstructionKind::I32Mul)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(42)]);
        }

        #[test]
        fn i32_mul_overflow() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const {
                    value: i32::MAX / 2 + 1,
                })
                .inst(InstructionKind::I32Const { value: 2 })
                .inst(InstructionKind::I32Mul)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(i32::MIN)]);
        }

        #[test]
        fn i32_mul_by_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: i32::MAX })
                .inst(InstructionKind::I32Const { value: 0 })
                .inst(InstructionKind::I32Mul)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(0)]);
        }

        #[test]
        fn i32_mul_by_negative_one() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 42 })
                .inst(InstructionKind::I32Const { value: -1 })
                .inst(InstructionKind::I32Mul)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(-42)]);
        }

        #[test]
        fn i32_div_s_basic() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 42 })
                .inst(InstructionKind::I32Const { value: 6 })
                .inst(InstructionKind::I32DivS)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(7)]);
        }

        #[test]
        fn i32_div_s_negative() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: -42 })
                .inst(InstructionKind::I32Const { value: 6 })
                .inst(InstructionKind::I32DivS)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(-7)]);
        }

        #[test]
        fn i32_div_s_truncation() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: -7 })
                .inst(InstructionKind::I32Const { value: 3 })
                .inst(InstructionKind::I32DivS)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(-2)]);
        }

        #[test]
        fn i32_div_u_basic() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 42 })
                .inst(InstructionKind::I32Const { value: 6 })
                .inst(InstructionKind::I32DivU)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(7)]);
        }

        #[test]
        fn i32_div_u_large() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: -1 })
                .inst(InstructionKind::I32Const { value: 2 })
                .inst(InstructionKind::I32DivU)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32((u32::MAX / 2) as i32)]);
        }

        #[test]
        fn i32_div_u_by_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 42 })
                .inst(InstructionKind::I32Const { value: 0 })
                .inst(InstructionKind::I32DivU)
                .expect_error("integer divide by zero");
        }

        #[test]
        fn i32_rem_s_basic() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 43 })
                .inst(InstructionKind::I32Const { value: 6 })
                .inst(InstructionKind::I32RemS)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(1)]);
        }

        #[test]
        fn i32_rem_s_negative() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: -43 })
                .inst(InstructionKind::I32Const { value: 6 })
                .inst(InstructionKind::I32RemS)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(-1)]);
        }

        #[test]
        fn i32_rem_s_by_one() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 42 })
                .inst(InstructionKind::I32Const { value: 1 })
                .inst(InstructionKind::I32RemS)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(0)]);
        }

        #[test]
        fn i32_rem_s_by_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 42 })
                .inst(InstructionKind::I32Const { value: 0 })
                .inst(InstructionKind::I32RemS)
                .expect_error("integer divide by zero");
        }

        #[test]
        fn i32_rem_u_basic() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 43 })
                .inst(InstructionKind::I32Const { value: 6 })
                .inst(InstructionKind::I32RemU)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(1)]);
        }

        #[test]
        fn i32_rem_u_large() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: -1 })
                .inst(InstructionKind::I32Const { value: 10 })
                .inst(InstructionKind::I32RemU)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32((u32::MAX % 10) as i32)]);
        }

        #[test]
        fn i32_rem_u_by_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 42 })
                .inst(InstructionKind::I32Const { value: 0 })
                .inst(InstructionKind::I32RemU)
                .expect_error("integer divide by zero");
        }

        #[test]
        fn i32_add_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 42 })
                .inst(InstructionKind::I32Const { value: 0 })
                .inst(InstructionKind::I32Add)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(42)]);
        }

        #[test]
        fn i32_sub_same() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 42 })
                .inst(InstructionKind::I32Const { value: 42 })
                .inst(InstructionKind::I32Sub)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(0)]);
        }
    }

    mod i32_unary_ops {
        use super::*;

        #[test]
        fn i32_clz() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 0x0F000000 })
                .inst(InstructionKind::I32Clz)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(4)]);

            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 0 })
                .inst(InstructionKind::I32Clz)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(32)]);

            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: -1 })
                .inst(InstructionKind::I32Clz)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(0)]);
        }

        #[test]
        fn i32_ctz() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 0x00008000 })
                .inst(InstructionKind::I32Ctz)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(15)]);

            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 0 })
                .inst(InstructionKind::I32Ctz)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(32)]);
        }

        #[test]
        fn i32_popcnt() {
            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 0x55555555 })
                .inst(InstructionKind::I32Popcnt)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(16)]);

            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: 0 })
                .inst(InstructionKind::I32Popcnt)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(0)]);

            ExecutorTest::new()
                .inst(InstructionKind::I32Const { value: -1 })
                .inst(InstructionKind::I32Popcnt)
                .returns(vec![ValueType::I32])
                .expect_stack(vec![Value::I32(32)]);
        }
    }

    mod i64_binary_ops {
        use super::*;

        #[test]
        fn i64_add_overflow() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: i64::MAX })
                .inst(InstructionKind::I64Const { value: 1 })
                .inst(InstructionKind::I64Add)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(i64::MIN)]);
        }

        #[test]
        fn i64_sub_underflow() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: i64::MIN })
                .inst(InstructionKind::I64Const { value: 1 })
                .inst(InstructionKind::I64Sub)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(i64::MAX)]);
        }

        #[test]
        fn i64_mul_basic() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: 1000000 })
                .inst(InstructionKind::I64Const { value: 1000000 })
                .inst(InstructionKind::I64Mul)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(1000000000000)]);
        }

        #[test]
        fn i64_mul_overflow() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const {
                    value: i64::MAX / 2 + 1,
                })
                .inst(InstructionKind::I64Const { value: 2 })
                .inst(InstructionKind::I64Mul)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(i64::MIN)]);
        }

        #[test]
        fn i64_mul_by_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: i64::MAX })
                .inst(InstructionKind::I64Const { value: 0 })
                .inst(InstructionKind::I64Mul)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(0)]);
        }

        #[test]
        fn i64_div_s_overflow() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: i64::MIN })
                .inst(InstructionKind::I64Const { value: -1 })
                .inst(InstructionKind::I64DivS)
                .expect_error("integer overflow");
        }

        #[test]
        fn i64_div_s_by_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: 42 })
                .inst(InstructionKind::I64Const { value: 0 })
                .inst(InstructionKind::I64DivS)
                .expect_error("integer divide by zero");
        }

        #[test]
        fn i64_div_s_negative() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: -42 })
                .inst(InstructionKind::I64Const { value: 6 })
                .inst(InstructionKind::I64DivS)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(-7)]);
        }

        #[test]
        fn i64_div_s_truncation() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: -7 })
                .inst(InstructionKind::I64Const { value: 3 })
                .inst(InstructionKind::I64DivS)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(-2)]);
        }

        #[test]
        fn i64_div_u_large() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: -1 })
                .inst(InstructionKind::I64Const { value: 2 })
                .inst(InstructionKind::I64DivU)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64((u64::MAX / 2) as i64)]);
        }

        #[test]
        fn i64_div_u_by_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: 42 })
                .inst(InstructionKind::I64Const { value: 0 })
                .inst(InstructionKind::I64DivU)
                .expect_error("integer divide by zero");
        }

        #[test]
        fn i64_rem_s_special_case() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: i64::MIN })
                .inst(InstructionKind::I64Const { value: -1 })
                .inst(InstructionKind::I64RemS)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(0)]);
        }

        #[test]
        fn i64_rem_s_negative() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: -43 })
                .inst(InstructionKind::I64Const { value: 6 })
                .inst(InstructionKind::I64RemS)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(-1)]);
        }

        #[test]
        fn i64_rem_s_by_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: 42 })
                .inst(InstructionKind::I64Const { value: 0 })
                .inst(InstructionKind::I64RemS)
                .expect_error("integer divide by zero");
        }

        #[test]
        fn i64_rem_u_basic() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: 100 })
                .inst(InstructionKind::I64Const { value: 7 })
                .inst(InstructionKind::I64RemU)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(2)]);
        }

        #[test]
        fn i64_rem_u_large() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: -1 })
                .inst(InstructionKind::I64Const { value: 10 })
                .inst(InstructionKind::I64RemU)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64((u64::MAX % 10) as i64)]);
        }

        #[test]
        fn i64_rem_u_by_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: 42 })
                .inst(InstructionKind::I64Const { value: 0 })
                .inst(InstructionKind::I64RemU)
                .expect_error("integer divide by zero");
        }
    }

    mod f32_unary_ops {
        use super::*;

        #[test]
        fn f32_abs() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: -3.14 })
                .inst(InstructionKind::F32Abs)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(3.14)]);

            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 3.14 })
                .inst(InstructionKind::F32Abs)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(3.14)]);
        }

        #[test]
        fn f32_neg() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 3.14 })
                .inst(InstructionKind::F32Neg)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(-3.14)]);

            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: -3.14 })
                .inst(InstructionKind::F32Neg)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(3.14)]);
        }

        #[test]
        fn f32_sqrt() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 4.0 })
                .inst(InstructionKind::F32Sqrt)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(2.0)]);
        }

        #[test]
        fn f32_ceil() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 3.3 })
                .inst(InstructionKind::F32Ceil)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(4.0)]);

            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: -3.3 })
                .inst(InstructionKind::F32Ceil)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(-3.0)]);
        }

        #[test]
        fn f32_floor() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 3.7 })
                .inst(InstructionKind::F32Floor)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(3.0)]);

            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: -3.7 })
                .inst(InstructionKind::F32Floor)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(-4.0)]);
        }

        #[test]
        fn f32_trunc() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 3.7 })
                .inst(InstructionKind::F32Trunc)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(3.0)]);

            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: -3.7 })
                .inst(InstructionKind::F32Trunc)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(-3.0)]);
        }

        #[test]
        fn f32_nearest() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 3.5 })
                .inst(InstructionKind::F32Nearest)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(4.0)]); // Round to even

            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 2.5 })
                .inst(InstructionKind::F32Nearest)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(2.0)]); // Round to even

            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 3.3 })
                .inst(InstructionKind::F32Nearest)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(3.0)]);
        }
    }

    mod f64_unary_ops {
        use super::*;

        #[test]
        fn f64_abs() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: -3.14159 })
                .inst(InstructionKind::F64Abs)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(3.14159)]);
        }

        #[test]
        fn f64_neg() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 3.14159 })
                .inst(InstructionKind::F64Neg)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(-3.14159)]);
        }

        #[test]
        fn f64_sqrt() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 9.0 })
                .inst(InstructionKind::F64Sqrt)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(3.0)]);
        }

        #[test]
        fn f64_ceil() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 3.3 })
                .inst(InstructionKind::F64Ceil)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(4.0)]);
        }

        #[test]
        fn f64_floor() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 3.7 })
                .inst(InstructionKind::F64Floor)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(3.0)]);
        }

        #[test]
        fn f64_trunc() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: -3.7 })
                .inst(InstructionKind::F64Trunc)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(-3.0)]);
        }

        #[test]
        fn f64_nearest() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 4.5 })
                .inst(InstructionKind::F64Nearest)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(4.0)]); // Round to even
        }
    }

    mod f32_binary_ops {
        use super::*;

        #[test]
        fn f32_add_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 1.5 })
                .inst(InstructionKind::F32Const { value: 2.5 })
                .inst(InstructionKind::F32Add)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(4.0)]);
        }

        #[test]
        fn f32_add_inf_inf() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: f32::INFINITY })
                .inst(InstructionKind::F32Const { value: f32::INFINITY })
                .inst(InstructionKind::F32Add)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(f32::INFINITY)]);
        }

        #[test]
        fn f32_sub_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 5.0 })
                .inst(InstructionKind::F32Const { value: 2.0 })
                .inst(InstructionKind::F32Sub)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(3.0)]);
        }

        #[test]
        fn f32_mul_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 3.0 })
                .inst(InstructionKind::F32Const { value: 4.0 })
                .inst(InstructionKind::F32Mul)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(12.0)]);
        }

        #[test]
        fn f32_div_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 12.0 })
                .inst(InstructionKind::F32Const { value: 3.0 })
                .inst(InstructionKind::F32Div)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(4.0)]);
        }

        #[test]
        fn f32_div_by_zero() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 1.0 })
                .inst(InstructionKind::F32Const { value: 0.0 })
                .inst(InstructionKind::F32Div)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(f32::INFINITY)]);
        }

        #[test]
        fn f32_min_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 3.0 })
                .inst(InstructionKind::F32Const { value: 5.0 })
                .inst(InstructionKind::F32Min)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(3.0)]);
        }

        #[test]
        fn f32_max_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 3.0 })
                .inst(InstructionKind::F32Const { value: 5.0 })
                .inst(InstructionKind::F32Max)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(5.0)]);
        }

        #[test]
        fn f32_copysign_positive() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: -5.0 })
                .inst(InstructionKind::F32Const { value: 1.0 })
                .inst(InstructionKind::F32Copysign)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(5.0)]);
        }

        #[test]
        fn f32_copysign_negative() {
            ExecutorTest::new()
                .inst(InstructionKind::F32Const { value: 5.0 })
                .inst(InstructionKind::F32Const { value: -1.0 })
                .inst(InstructionKind::F32Copysign)
                .returns(vec![ValueType::F32])
                .expect_stack(vec![Value::F32(-5.0)]);
        }
    }

    mod f64_binary_ops {
        use super::*;

        #[test]
        fn f64_add_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 1.5 })
                .inst(InstructionKind::F64Const { value: 2.5 })
                .inst(InstructionKind::F64Add)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(4.0)]);
        }

        #[test]
        fn f64_sub_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 5.0 })
                .inst(InstructionKind::F64Const { value: 2.0 })
                .inst(InstructionKind::F64Sub)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(3.0)]);
        }

        #[test]
        fn f64_mul_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 3.0 })
                .inst(InstructionKind::F64Const { value: 4.0 })
                .inst(InstructionKind::F64Mul)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(12.0)]);
        }

        #[test]
        fn f64_div_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 12.0 })
                .inst(InstructionKind::F64Const { value: 3.0 })
                .inst(InstructionKind::F64Div)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(4.0)]);
        }

        #[test]
        fn f64_min_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 3.0 })
                .inst(InstructionKind::F64Const { value: 5.0 })
                .inst(InstructionKind::F64Min)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(3.0)]);
        }

        #[test]
        fn f64_max_normal() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: 3.0 })
                .inst(InstructionKind::F64Const { value: 5.0 })
                .inst(InstructionKind::F64Max)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(5.0)]);
        }

        #[test]
        fn f64_copysign_inf() {
            ExecutorTest::new()
                .inst(InstructionKind::F64Const { value: f64::INFINITY })
                .inst(InstructionKind::F64Const { value: -1.0 })
                .inst(InstructionKind::F64Copysign)
                .returns(vec![ValueType::F64])
                .expect_stack(vec![Value::F64(f64::NEG_INFINITY)]);
        }
    }

    mod i64_unary_ops {
        use super::*;

        #[test]
        fn i64_clz() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const {
                    value: 0x0F00000000000000,
                })
                .inst(InstructionKind::I64Clz)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(4)]);

            ExecutorTest::new()
                .inst(InstructionKind::I64Const { value: 0 })
                .inst(InstructionKind::I64Clz)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(64)]);
        }

        #[test]
        fn i64_ctz() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const {
                    value: 0x8000000000000000u64 as i64,
                })
                .inst(InstructionKind::I64Ctz)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(63)]);
        }

        #[test]
        fn i64_popcnt() {
            ExecutorTest::new()
                .inst(InstructionKind::I64Const {
                    value: 0x5555555555555555,
                })
                .inst(InstructionKind::I64Popcnt)
                .returns(vec![ValueType::I64])
                .expect_stack(vec![Value::I64(32)]);
        }
    }

    // Floating-point Binary Operation Tests

    // ============================================================================
    // Additional integer binary operation tests moved from executor.rs
    // ============================================================================

    #[test]
    fn i32_add_basic() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 5 })
            .inst(InstructionKind::I32Const { value: 3 })
            .inst(InstructionKind::I32Add)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(8)]);
    }

    #[test]
    fn i32_add_overflow() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: i32::MAX })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Add)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(i32::MIN)]); // Wraps around
    }

    #[test]
    fn i32_add_negative() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -10 })
            .inst(InstructionKind::I32Const { value: 5 })
            .inst(InstructionKind::I32Add)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-5)]);
    }

    #[test]
    fn i32_sub_basic() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 10 })
            .inst(InstructionKind::I32Const { value: 3 })
            .inst(InstructionKind::I32Sub)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(7)]);
    }

    #[test]
    fn i32_sub_underflow() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: i32::MIN })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Sub)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(i32::MAX)]); // Wraps around
    }

    #[test]
    fn i32_mul_basic() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 6 })
            .inst(InstructionKind::I32Const { value: 7 })
            .inst(InstructionKind::I32Mul)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn i32_mul_overflow() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const {
                value: i32::MAX / 2 + 1,
            })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32Mul)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(i32::MIN)]); // Wraps
    }

    #[test]
    fn i32_div_s_basic() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 6 })
            .inst(InstructionKind::I32DivS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(7)]);
    }

    #[test]
    fn i32_div_s_negative() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -42 })
            .inst(InstructionKind::I32Const { value: 6 })
            .inst(InstructionKind::I32DivS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-7)]);
    }

    #[test]
    fn i32_div_s_by_zero() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::I32DivS)
            .expect_error("integer divide by zero");
    }

    #[test]
    fn i32_div_s_overflow() {
        // i32::MIN / -1 causes overflow
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: i32::MIN })
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32DivS)
            .expect_error("integer overflow");
    }

    #[test]
    fn i32_div_u_basic() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 6 })
            .inst(InstructionKind::I32DivU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(7)]);
    }

    #[test]
    fn i32_div_u_large() {
        // -1 as u32 is u32::MAX
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::I32DivU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32((u32::MAX / 2) as i32)]);
    }

    #[test]
    fn i32_rem_s_basic() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 43 })
            .inst(InstructionKind::I32Const { value: 6 })
            .inst(InstructionKind::I32RemS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }

    #[test]
    fn i32_rem_s_negative() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -43 })
            .inst(InstructionKind::I32Const { value: 6 })
            .inst(InstructionKind::I32RemS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]);
    }

    #[test]
    fn i32_rem_s_special_case() {
        // i32::MIN % -1 = 0 (avoids overflow)
        // This is because ANY integer % ±1 = 0 mathematically
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: i32::MIN })
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32RemS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn i32_rem_s_by_negative_one() {
        // Test that ANY value % -1 = 0
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -100 })
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32RemS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn i32_rem_s_by_one() {
        // Test that ANY value % 1 = 0
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32RemS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn i32_rem_u_basic() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 43 })
            .inst(InstructionKind::I32Const { value: 6 })
            .inst(InstructionKind::I32RemU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(1)]);
    }

    #[test]
    fn i32_rem_u_by_zero() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::I32RemU)
            .expect_error("integer divide by zero");
    }

    #[test]
    fn i32_div_u_by_zero() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::I32DivU)
            .expect_error("integer divide by zero");
    }

    #[test]
    fn i32_mul_by_zero() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: i32::MAX })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::I32Mul)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn i32_mul_by_negative_one() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: -1 })
            .inst(InstructionKind::I32Mul)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-42)]);
    }

    #[test]
    fn i32_div_s_truncation() {
        // Test truncation toward zero for negative results
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -7 })
            .inst(InstructionKind::I32Const { value: 3 })
            .inst(InstructionKind::I32DivS)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-2)]); // -7/3 = -2.33.. truncates to -2
    }

    #[test]
    fn i32_rem_s_by_zero() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::I32RemS)
            .expect_error("integer divide by zero");
    }

    #[test]
    fn i32_rem_u_large() {
        // Test with large unsigned values
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: -1 }) // u32::MAX
            .inst(InstructionKind::I32Const { value: 10 })
            .inst(InstructionKind::I32RemU)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32((u32::MAX % 10) as i32)]);
    }

    #[test]
    fn i32_add_zero() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::I32Add)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn i32_sub_same() {
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Sub)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(0)]);
    }

    #[test]
    fn i64_add_overflow() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: i64::MAX })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Add)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(i64::MIN)]);
    }

    #[test]
    fn i64_sub_underflow() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: i64::MIN })
            .inst(InstructionKind::I64Const { value: 1 })
            .inst(InstructionKind::I64Sub)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(i64::MAX)]);
    }

    #[test]
    fn i64_mul_basic() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 1000000 })
            .inst(InstructionKind::I64Const { value: 1000000 })
            .inst(InstructionKind::I64Mul)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(1000000000000)]);
    }

    #[test]
    fn i64_div_s_overflow() {
        // i64::MIN / -1 causes overflow
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: i64::MIN })
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64DivS)
            .expect_error("integer overflow");
    }

    #[test]
    fn i64_div_u_large() {
        // -1 as u64 is u64::MAX
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64Const { value: 2 })
            .inst(InstructionKind::I64DivU)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64((u64::MAX / 2) as i64)]);
    }

    #[test]
    fn i64_rem_s_special_case() {
        // i64::MIN % -1 = 0 per spec
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: i64::MIN })
            .inst(InstructionKind::I64Const { value: -1 })
            .inst(InstructionKind::I64RemS)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0)]);
    }

    #[test]
    fn i64_rem_u_basic() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 100 })
            .inst(InstructionKind::I64Const { value: 7 })
            .inst(InstructionKind::I64RemU)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(2)]);
    }

    #[test]
    fn i64_div_s_by_zero() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 42 })
            .inst(InstructionKind::I64Const { value: 0 })
            .inst(InstructionKind::I64DivS)
            .expect_error("integer divide by zero");
    }

    #[test]
    fn i64_div_u_by_zero() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 42 })
            .inst(InstructionKind::I64Const { value: 0 })
            .inst(InstructionKind::I64DivU)
            .expect_error("integer divide by zero");
    }

    #[test]
    fn i64_rem_s_by_zero() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 42 })
            .inst(InstructionKind::I64Const { value: 0 })
            .inst(InstructionKind::I64RemS)
            .expect_error("integer divide by zero");
    }

    #[test]
    fn i64_rem_u_by_zero() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: 42 })
            .inst(InstructionKind::I64Const { value: 0 })
            .inst(InstructionKind::I64RemU)
            .expect_error("integer divide by zero");
    }

    #[test]
    fn i64_div_s_truncation() {
        // Test truncation toward zero for negative results
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -7 })
            .inst(InstructionKind::I64Const { value: 3 })
            .inst(InstructionKind::I64DivS)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-2)]);
    }

    #[test]
    fn i64_mul_overflow() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const {
                value: i64::MAX / 2 + 1,
            })
            .inst(InstructionKind::I64Const { value: 2 })
            .inst(InstructionKind::I64Mul)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(i64::MIN)]);
    }

    #[test]
    fn i64_mul_by_zero() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: i64::MAX })
            .inst(InstructionKind::I64Const { value: 0 })
            .inst(InstructionKind::I64Mul)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0)]);
    }

    #[test]
    fn i64_rem_u_large() {
        // Test with large unsigned values
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -1 }) // u64::MAX
            .inst(InstructionKind::I64Const { value: 10 })
            .inst(InstructionKind::I64RemU)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64((u64::MAX % 10) as i64)]);
    }

    #[test]
    fn i64_div_s_negative() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -42 })
            .inst(InstructionKind::I64Const { value: 6 })
            .inst(InstructionKind::I64DivS)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-7)]);
    }

    #[test]
    fn i64_rem_s_negative() {
        ExecutorTest::new()
            .inst(InstructionKind::I64Const { value: -43 })
            .inst(InstructionKind::I64Const { value: 6 })
            .inst(InstructionKind::I64RemS)
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]);
    }
}
