//! SIMD (v128) operations for WebAssembly.
//!
//! Implements all 236 instructions under the `0xFD` SIMD prefix
//! (spec section 4.4.3). The v128 value is stored as `[u8; 16]` in
//! little-endian byte order; lane interpretation is determined by the
//! instruction:
//!
//! | Interpretation | Lanes | Rust type per lane |
//! |----------------|-------|--------------------|
//! | i8x16 / u8x16 | 16 | `i8` / `u8` |
//! | i16x8 / u16x8 | 8 | `i16` / `u16` |
//! | i32x4 / u32x4 | 4 | `i32` / `u32` |
//! | i64x2 / u64x2 | 2 | `i64` / `u64` |
//! | f32x4 | 4 | `f32` |
//! | f64x2 | 2 | `f64` |
//!
//! # NaN canonicalisation
//!
//! Arithmetic floating-point operations (`add`, `sub`, `mul`, `div`, `sqrt`,
//! `ceil`, `floor`, `trunc`, `nearest`, `min`, `max`) canonicalise NaN results
//! by setting the quiet bit. Non-arithmetic operations (`abs`, `neg`, `pmin`,
//! `pmax`, `copysign`) preserve the NaN payload unchanged.

use super::*;

// ============================================================================
// Lane extraction/insertion helpers
//
// These convert between the raw `[u8; 16]` byte representation and typed
// lane arrays. All conversions use little-endian byte order.
// ============================================================================

/// Interpret a v128 as four `f32` lanes.
fn get_f32_lanes(v: [u8; 16]) -> [f32; 4] {
    [
        f32::from_le_bytes([v[0], v[1], v[2], v[3]]),
        f32::from_le_bytes([v[4], v[5], v[6], v[7]]),
        f32::from_le_bytes([v[8], v[9], v[10], v[11]]),
        f32::from_le_bytes([v[12], v[13], v[14], v[15]]),
    ]
}

/// Pack four `f32` lanes into a v128.
fn set_f32_lanes(lanes: [f32; 4]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for (i, &f) in lanes.iter().enumerate() {
        r[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    r
}

/// Interpret a v128 as two `f64` lanes.
fn get_f64_lanes(v: [u8; 16]) -> [f64; 2] {
    [
        f64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]),
        f64::from_le_bytes([v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]]),
    ]
}

/// Pack two `f64` lanes into a v128.
fn set_f64_lanes(lanes: [f64; 2]) -> [u8; 16] {
    let mut r = [0u8; 16];
    r[0..8].copy_from_slice(&lanes[0].to_le_bytes());
    r[8..16].copy_from_slice(&lanes[1].to_le_bytes());
    r
}

// ============================================================================
// Integer lane extraction/insertion helpers
//
// Signed and unsigned variants for each lane width. The `get_*` functions
// reinterpret bytes as typed lanes; the `set_*` functions do the reverse.
// ============================================================================

/// Interpret a v128 as sixteen signed `i8` lanes.
fn get_i8x16_lanes(v: [u8; 16]) -> [i8; 16] {
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = v[i] as i8;
    }
    r
}

fn get_u8x16_lanes(v: [u8; 16]) -> [u8; 16] {
    v
}

fn set_i8x16_lanes(lanes: [i8; 16]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = lanes[i] as u8;
    }
    r
}

fn set_u8x16_lanes(lanes: [u8; 16]) -> [u8; 16] {
    lanes
}

/// Interpret a v128 as eight signed `i16` lanes.
fn get_i16x8_lanes(v: [u8; 16]) -> [i16; 8] {
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = i16::from_le_bytes([v[i * 2], v[i * 2 + 1]]);
    }
    r
}

fn get_u16x8_lanes(v: [u8; 16]) -> [u16; 8] {
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = u16::from_le_bytes([v[i * 2], v[i * 2 + 1]]);
    }
    r
}

fn set_i16x8_lanes(lanes: [i16; 8]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for (i, &val) in lanes.iter().enumerate() {
        r[i * 2..i * 2 + 2].copy_from_slice(&val.to_le_bytes());
    }
    r
}

fn set_u16x8_lanes(lanes: [u16; 8]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for (i, &val) in lanes.iter().enumerate() {
        r[i * 2..i * 2 + 2].copy_from_slice(&val.to_le_bytes());
    }
    r
}

/// Interpret a v128 as four signed `i32` lanes.
fn get_i32x4_lanes(v: [u8; 16]) -> [i32; 4] {
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = i32::from_le_bytes([v[i * 4], v[i * 4 + 1], v[i * 4 + 2], v[i * 4 + 3]]);
    }
    r
}

fn get_u32x4_lanes(v: [u8; 16]) -> [u32; 4] {
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = u32::from_le_bytes([v[i * 4], v[i * 4 + 1], v[i * 4 + 2], v[i * 4 + 3]]);
    }
    r
}

fn set_i32x4_lanes(lanes: [i32; 4]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for (i, &val) in lanes.iter().enumerate() {
        r[i * 4..i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }
    r
}

fn set_u32x4_lanes(lanes: [u32; 4]) -> [u8; 16] {
    let mut r = [0u8; 16];
    for (i, &val) in lanes.iter().enumerate() {
        r[i * 4..i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }
    r
}

/// Interpret a v128 as two signed `i64` lanes.
fn get_i64x2_lanes(v: [u8; 16]) -> [i64; 2] {
    [
        i64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]),
        i64::from_le_bytes([v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]]),
    ]
}

fn get_u64x2_lanes(v: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]),
        u64::from_le_bytes([v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]]),
    ]
}

fn set_i64x2_lanes(lanes: [i64; 2]) -> [u8; 16] {
    let mut r = [0u8; 16];
    r[0..8].copy_from_slice(&lanes[0].to_le_bytes());
    r[8..16].copy_from_slice(&lanes[1].to_le_bytes());
    r
}

fn set_u64x2_lanes(lanes: [u64; 2]) -> [u8; 16] {
    let mut r = [0u8; 16];
    r[0..8].copy_from_slice(&lanes[0].to_le_bytes());
    r[8..16].copy_from_slice(&lanes[1].to_le_bytes());
    r
}

/// Compute effective address from i32 base + memarg offset, with overflow checking
fn effective_address(stack: &mut Stack, memarg: &MemArg) -> Result<u32, RuntimeError> {
    let addr = stack.pop_i32()?;
    let ea = (addr as u64)
        .checked_add(memarg.offset as u64)
        .ok_or_else(|| RuntimeError::MemoryError("out of bounds memory access".to_string()))?;
    if ea > u32::MAX as u64 {
        return Err(RuntimeError::MemoryError("out of bounds memory access".to_string()));
    }
    Ok(ea as u32)
}

/// v128.load — load 16 bytes from memory.
/// \[i32\] -> \[v128\]
pub fn v128_load(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let bytes = memory.read_bytes(ea, 16)?;
    let mut value = [0u8; 16];
    value.copy_from_slice(&bytes);
    stack.push(Value::V128(value));
    Ok(())
}

/// v128.store — store 16 bytes to memory
/// [i32, v128] → []
pub fn v128_store(stack: &mut Stack, memory: &mut Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let value = stack.pop_v128()?;
    let ea = effective_address(stack, memarg)?;
    memory.write_bytes(ea, &value)?;
    Ok(())
}

// ============================================================================
// f32x4 operations
// ============================================================================

/// Apply a unary f32 operation lane-wise, canonicalising NaN results.
fn f32x4_unary(stack: &mut Stack, op: fn(f32) -> f32) -> Result<(), RuntimeError> {
    let a = get_f32_lanes(stack.pop_v128()?);
    let r = [
        canonicalise_f32_nan(op(a[0])),
        canonicalise_f32_nan(op(a[1])),
        canonicalise_f32_nan(op(a[2])),
        canonicalise_f32_nan(op(a[3])),
    ];
    stack.push(Value::V128(set_f32_lanes(r)));
    Ok(())
}

/// Apply a binary f32 operation lane-wise, canonicalising NaN results.
fn f32x4_binary(stack: &mut Stack, op: fn(f32, f32) -> f32) -> Result<(), RuntimeError> {
    let b = get_f32_lanes(stack.pop_v128()?);
    let a = get_f32_lanes(stack.pop_v128()?);
    let r = [
        canonicalise_f32_nan(op(a[0], b[0])),
        canonicalise_f32_nan(op(a[1], b[1])),
        canonicalise_f32_nan(op(a[2], b[2])),
        canonicalise_f32_nan(op(a[3], b[3])),
    ];
    stack.push(Value::V128(set_f32_lanes(r)));
    Ok(())
}

/// Apply a binary f32 operation without NaN canonicalisation (for pmin/pmax)
fn f32x4_bitwise_binary(stack: &mut Stack, op: fn(f32, f32) -> f32) -> Result<(), RuntimeError> {
    let b = get_f32_lanes(stack.pop_v128()?);
    let a = get_f32_lanes(stack.pop_v128()?);
    stack.push(Value::V128(set_f32_lanes([
        op(a[0], b[0]),
        op(a[1], b[1]),
        op(a[2], b[2]),
        op(a[3], b[3]),
    ])));
    Ok(())
}

/// Apply a comparison f32 operation lane-wise; all-ones (0xFFFFFFFF) for true, all-zeros for false
fn f32x4_compare(stack: &mut Stack, op: fn(f32, f32) -> bool) -> Result<(), RuntimeError> {
    let b = get_f32_lanes(stack.pop_v128()?);
    let a = get_f32_lanes(stack.pop_v128()?);
    let mut r = [0u8; 16];
    for i in 0..4 {
        let mask: u32 = if op(a[i], b[i]) { 0xFFFFFFFF } else { 0 };
        r[i * 4..i * 4 + 4].copy_from_slice(&mask.to_le_bytes());
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// IEEE 754-2019 minimum: propagate NaN, -0 < +0
fn ieee_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 && a.is_sign_negative() != b.is_sign_negative() {
        if a.is_sign_negative() { a } else { b }
    } else {
        a.min(b)
    }
}

/// IEEE 754-2019 maximum: propagate NaN, +0 > -0
fn ieee_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 && a.is_sign_negative() != b.is_sign_negative() {
        if a.is_sign_negative() { b } else { a }
    } else {
        a.max(b)
    }
}

/// abs/neg operate on the sign bit only — they do NOT canonicalise NaN
pub fn f32x4_abs(stack: &mut Stack) -> Result<(), RuntimeError> {
    let mut v = stack.pop_v128()?;
    for i in 0..4 {
        v[i * 4 + 3] &= 0x7F;
    }
    stack.push(Value::V128(v));
    Ok(())
}

/// Flip the sign bit of each f32 lane — does NOT canonicalise NaN.
pub fn f32x4_neg(stack: &mut Stack) -> Result<(), RuntimeError> {
    let mut v = stack.pop_v128()?;
    for i in 0..4 {
        v[i * 4 + 3] ^= 0x80;
    }
    stack.push(Value::V128(v));
    Ok(())
}

pub fn f32x4_sqrt(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_unary(stack, f32::sqrt)
}

pub fn f32x4_ceil(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_unary(stack, f32::ceil)
}

pub fn f32x4_floor(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_unary(stack, f32::floor)
}

pub fn f32x4_trunc(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_unary(stack, f32::trunc)
}

pub fn f32x4_nearest(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_unary(stack, wasm_nearest_f32)
}

pub fn f32x4_add(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_binary(stack, std::ops::Add::add)
}

pub fn f32x4_sub(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_binary(stack, std::ops::Sub::sub)
}

pub fn f32x4_mul(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_binary(stack, std::ops::Mul::mul)
}

pub fn f32x4_div(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_binary(stack, std::ops::Div::div)
}

pub fn f32x4_min(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_binary(stack, ieee_min_f32)
}

pub fn f32x4_max(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_binary(stack, ieee_max_f32)
}

/// pmin: if b < a then b, else a — bitwise, passes NaN through unchanged
pub fn f32x4_pmin(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_bitwise_binary(stack, |a, b| if b < a { b } else { a })
}

/// pmax: if a < b then b, else a — bitwise, passes NaN through unchanged
pub fn f32x4_pmax(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_bitwise_binary(stack, |a, b| if a < b { b } else { a })
}

pub fn f32x4_eq(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_compare(stack, |a, b| a == b)
}

pub fn f32x4_ne(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_compare(stack, |a, b| a != b)
}

pub fn f32x4_lt(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_compare(stack, |a, b| a < b)
}

pub fn f32x4_gt(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_compare(stack, |a, b| a > b)
}

pub fn f32x4_le(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_compare(stack, |a, b| a <= b)
}

pub fn f32x4_ge(stack: &mut Stack) -> Result<(), RuntimeError> {
    f32x4_compare(stack, |a, b| a >= b)
}

// ============================================================================
// f64x2 operations
// ============================================================================

/// Apply a unary f64 operation lane-wise, canonicalising NaN results.
fn f64x2_unary(stack: &mut Stack, op: fn(f64) -> f64) -> Result<(), RuntimeError> {
    let a = get_f64_lanes(stack.pop_v128()?);
    let r = [canonicalise_f64_nan(op(a[0])), canonicalise_f64_nan(op(a[1]))];
    stack.push(Value::V128(set_f64_lanes(r)));
    Ok(())
}

/// Apply a binary f64 operation lane-wise, canonicalising NaN results.
fn f64x2_binary(stack: &mut Stack, op: fn(f64, f64) -> f64) -> Result<(), RuntimeError> {
    let b = get_f64_lanes(stack.pop_v128()?);
    let a = get_f64_lanes(stack.pop_v128()?);
    let r = [
        canonicalise_f64_nan(op(a[0], b[0])),
        canonicalise_f64_nan(op(a[1], b[1])),
    ];
    stack.push(Value::V128(set_f64_lanes(r)));
    Ok(())
}

/// Apply a binary f64 operation without NaN canonicalisation (for pmin/pmax).
fn f64x2_bitwise_binary(stack: &mut Stack, op: fn(f64, f64) -> f64) -> Result<(), RuntimeError> {
    let b = get_f64_lanes(stack.pop_v128()?);
    let a = get_f64_lanes(stack.pop_v128()?);
    stack.push(Value::V128(set_f64_lanes([op(a[0], b[0]), op(a[1], b[1])])));
    Ok(())
}

/// Apply a comparison f64 operation lane-wise; all-ones for true, all-zeros for false.
fn f64x2_compare(stack: &mut Stack, op: fn(f64, f64) -> bool) -> Result<(), RuntimeError> {
    let b = get_f64_lanes(stack.pop_v128()?);
    let a = get_f64_lanes(stack.pop_v128()?);
    let mut r = [0u8; 16];
    for i in 0..2 {
        let mask: u64 = if op(a[i], b[i]) { 0xFFFFFFFFFFFFFFFF } else { 0 };
        r[i * 8..i * 8 + 8].copy_from_slice(&mask.to_le_bytes());
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// IEEE 754-2019 minimum for f64: propagate NaN, -0 < +0.
fn ieee_min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 && a.is_sign_negative() != b.is_sign_negative() {
        if a.is_sign_negative() { a } else { b }
    } else {
        a.min(b)
    }
}

/// IEEE 754-2019 maximum for f64: propagate NaN, +0 > -0.
fn ieee_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 && a.is_sign_negative() != b.is_sign_negative() {
        if a.is_sign_negative() { b } else { a }
    } else {
        a.max(b)
    }
}

/// Clear the sign bit of each f64 lane — does NOT canonicalise NaN.
pub fn f64x2_abs(stack: &mut Stack) -> Result<(), RuntimeError> {
    let mut v = stack.pop_v128()?;
    v[7] &= 0x7F;
    v[15] &= 0x7F;
    stack.push(Value::V128(v));
    Ok(())
}

/// Flip the sign bit of each f64 lane — does NOT canonicalise NaN.
pub fn f64x2_neg(stack: &mut Stack) -> Result<(), RuntimeError> {
    let mut v = stack.pop_v128()?;
    v[7] ^= 0x80;
    v[15] ^= 0x80;
    stack.push(Value::V128(v));
    Ok(())
}

pub fn f64x2_sqrt(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_unary(stack, f64::sqrt)
}

pub fn f64x2_ceil(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_unary(stack, f64::ceil)
}

pub fn f64x2_floor(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_unary(stack, f64::floor)
}

pub fn f64x2_trunc(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_unary(stack, f64::trunc)
}

pub fn f64x2_nearest(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_unary(stack, wasm_nearest_f64)
}

pub fn f64x2_add(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_binary(stack, std::ops::Add::add)
}

pub fn f64x2_sub(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_binary(stack, std::ops::Sub::sub)
}

pub fn f64x2_mul(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_binary(stack, std::ops::Mul::mul)
}

pub fn f64x2_div(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_binary(stack, std::ops::Div::div)
}

pub fn f64x2_min(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_binary(stack, ieee_min_f64)
}

pub fn f64x2_max(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_binary(stack, ieee_max_f64)
}

pub fn f64x2_pmin(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_bitwise_binary(stack, |a, b| if b < a { b } else { a })
}

pub fn f64x2_pmax(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_bitwise_binary(stack, |a, b| if a < b { b } else { a })
}

pub fn f64x2_eq(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_compare(stack, |a, b| a == b)
}

pub fn f64x2_ne(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_compare(stack, |a, b| a != b)
}

pub fn f64x2_lt(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_compare(stack, |a, b| a < b)
}

pub fn f64x2_gt(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_compare(stack, |a, b| a > b)
}

pub fn f64x2_le(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_compare(stack, |a, b| a <= b)
}

pub fn f64x2_ge(stack: &mut Stack) -> Result<(), RuntimeError> {
    f64x2_compare(stack, |a, b| a >= b)
}

// ============================================================================
// v128 bitwise operations
// ============================================================================

/// v128.not — bitwise NOT of all 128 bits
pub fn v128_not(stack: &mut Stack) -> Result<(), RuntimeError> {
    let mut v = stack.pop_v128()?;
    for b in &mut v {
        *b = !*b;
    }
    stack.push(Value::V128(v));
    Ok(())
}

/// v128.and — bitwise AND of two v128 values
pub fn v128_and(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i] & b[i];
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// v128.andnot — bitwise AND of a with NOT b
pub fn v128_andnot(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i] & !b[i];
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// v128.or — bitwise OR of two v128 values
pub fn v128_or(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i] | b[i];
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// v128.xor — bitwise XOR of two v128 values
pub fn v128_xor(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i] ^ b[i];
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// v128.bitselect — per-bit selection: (v1 & c) | (v2 & !c).
/// \[v128, v128, v128\] -> \[v128\]
pub fn v128_bitselect(stack: &mut Stack) -> Result<(), RuntimeError> {
    let c = stack.pop_v128()?;
    let v2 = stack.pop_v128()?;
    let v1 = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = (v1[i] & c[i]) | (v2[i] & !c[i]);
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// v128.any_true — returns 1 if any bit is non-zero, 0 otherwise
pub fn v128_any_true(stack: &mut Stack) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let result = if v.iter().any(|&b| b != 0) { 1i32 } else { 0i32 };
    stack.push(Value::I32(result));
    Ok(())
}

// ============================================================================
// Splat operations
// ============================================================================

/// i8x16.splat — broadcast i32 (truncated to i8) across all 16 lanes
pub fn i8x16_splat(stack: &mut Stack) -> Result<(), RuntimeError> {
    let val = stack.pop_i32()? as u8;
    stack.push(Value::V128([val; 16]));
    Ok(())
}

/// i16x8.splat — broadcast i32 (truncated to i16) across all 8 lanes
pub fn i16x8_splat(stack: &mut Stack) -> Result<(), RuntimeError> {
    let val = stack.pop_i32()? as u16;
    let mut r = [0u8; 16];
    let bytes = val.to_le_bytes();
    for i in 0..8 {
        r[i * 2] = bytes[0];
        r[i * 2 + 1] = bytes[1];
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// i32x4.splat — broadcast i32 across all 4 lanes
pub fn i32x4_splat(stack: &mut Stack) -> Result<(), RuntimeError> {
    let val = stack.pop_i32()? as u32;
    let mut r = [0u8; 16];
    let bytes = val.to_le_bytes();
    for i in 0..4 {
        r[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// i64x2.splat — broadcast i64 across both lanes
pub fn i64x2_splat(stack: &mut Stack) -> Result<(), RuntimeError> {
    let val = stack.pop_i64()? as u64;
    let mut r = [0u8; 16];
    let bytes = val.to_le_bytes();
    r[0..8].copy_from_slice(&bytes);
    r[8..16].copy_from_slice(&bytes);
    stack.push(Value::V128(r));
    Ok(())
}

/// f32x4.splat — broadcast f32 across all 4 lanes
pub fn f32x4_splat(stack: &mut Stack) -> Result<(), RuntimeError> {
    let val = stack.pop_f32()?;
    stack.push(Value::V128(set_f32_lanes([val, val, val, val])));
    Ok(())
}

/// f64x2.splat — broadcast f64 across both lanes
pub fn f64x2_splat(stack: &mut Stack) -> Result<(), RuntimeError> {
    let val = stack.pop_f64()?;
    stack.push(Value::V128(set_f64_lanes([val, val])));
    Ok(())
}

// ============================================================================
// Extract lane operations
// ============================================================================

/// i8x16.extract_lane_s — sign-extend i8 lane to i32
pub fn i8x16_extract_lane_s(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let val = v[lane as usize] as i8 as i32;
    stack.push(Value::I32(val));
    Ok(())
}

/// i8x16.extract_lane_u — zero-extend i8 lane to i32
pub fn i8x16_extract_lane_u(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let val = v[lane as usize] as i32;
    stack.push(Value::I32(val));
    Ok(())
}

/// i16x8.extract_lane_s — sign-extend i16 lane to i32
pub fn i16x8_extract_lane_s(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let lanes = get_i16x8_lanes(v);
    stack.push(Value::I32(lanes[lane as usize] as i32));
    Ok(())
}

/// i16x8.extract_lane_u — zero-extend i16 lane to i32
pub fn i16x8_extract_lane_u(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let lanes = get_u16x8_lanes(v);
    stack.push(Value::I32(lanes[lane as usize] as i32));
    Ok(())
}

/// i32x4.extract_lane
pub fn i32x4_extract_lane(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let lanes = get_i32x4_lanes(v);
    stack.push(Value::I32(lanes[lane as usize]));
    Ok(())
}

/// i64x2.extract_lane
pub fn i64x2_extract_lane(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let lanes = get_i64x2_lanes(v);
    stack.push(Value::I64(lanes[lane as usize]));
    Ok(())
}

/// f32x4.extract_lane
pub fn f32x4_extract_lane(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let lanes = get_f32_lanes(v);
    stack.push(Value::F32(lanes[lane as usize]));
    Ok(())
}

/// f64x2.extract_lane
pub fn f64x2_extract_lane(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let lanes = get_f64_lanes(v);
    stack.push(Value::F64(lanes[lane as usize]));
    Ok(())
}

// ============================================================================
// Replace lane operations
// ============================================================================

/// i8x16.replace_lane — replace a single i8 lane (from i32, truncated)
pub fn i8x16_replace_lane(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let val = stack.pop_i32()? as u8;
    let mut v = stack.pop_v128()?;
    v[lane as usize] = val;
    stack.push(Value::V128(v));
    Ok(())
}

/// i16x8.replace_lane — replace a single i16 lane (from i32, truncated)
pub fn i16x8_replace_lane(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let val = stack.pop_i32()? as u16;
    let mut v = stack.pop_v128()?;
    let offset = lane as usize * 2;
    v[offset..offset + 2].copy_from_slice(&val.to_le_bytes());
    stack.push(Value::V128(v));
    Ok(())
}

/// i32x4.replace_lane
pub fn i32x4_replace_lane(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let val = stack.pop_i32()?;
    let mut v = stack.pop_v128()?;
    let offset = lane as usize * 4;
    v[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    stack.push(Value::V128(v));
    Ok(())
}

/// i64x2.replace_lane
pub fn i64x2_replace_lane(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let val = stack.pop_i64()?;
    let mut v = stack.pop_v128()?;
    let offset = lane as usize * 8;
    v[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
    stack.push(Value::V128(v));
    Ok(())
}

/// f32x4.replace_lane
pub fn f32x4_replace_lane(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let val = stack.pop_f32()?;
    let mut v = stack.pop_v128()?;
    let offset = lane as usize * 4;
    v[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    stack.push(Value::V128(v));
    Ok(())
}

/// f64x2.replace_lane
pub fn f64x2_replace_lane(stack: &mut Stack, lane: u8) -> Result<(), RuntimeError> {
    let val = stack.pop_f64()?;
    let mut v = stack.pop_v128()?;
    let offset = lane as usize * 8;
    v[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
    stack.push(Value::V128(v));
    Ok(())
}

// ============================================================================
// i8x16 comparisons
// ============================================================================

/// Signed i8x16 lane-wise comparison; 0xFF for true, 0x00 for false.
fn i8x16_compare(stack: &mut Stack, op: fn(i8, i8) -> bool) -> Result<(), RuntimeError> {
    let b = get_i8x16_lanes(stack.pop_v128()?);
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = if op(a[i], b[i]) { 0xFF } else { 0x00 };
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// Unsigned u8x16 lane-wise comparison; 0xFF for true, 0x00 for false.
fn u8x16_compare(stack: &mut Stack, op: fn(u8, u8) -> bool) -> Result<(), RuntimeError> {
    let b = get_u8x16_lanes(stack.pop_v128()?);
    let a = get_u8x16_lanes(stack.pop_v128()?);
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = if op(a[i], b[i]) { 0xFF } else { 0x00 };
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i8x16_eq(stack: &mut Stack) -> Result<(), RuntimeError> {
    i8x16_compare(stack, |a, b| a == b)
}
pub fn i8x16_ne(stack: &mut Stack) -> Result<(), RuntimeError> {
    i8x16_compare(stack, |a, b| a != b)
}
pub fn i8x16_lt_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i8x16_compare(stack, |a, b| a < b)
}
pub fn i8x16_lt_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u8x16_compare(stack, |a, b| a < b)
}
pub fn i8x16_gt_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i8x16_compare(stack, |a, b| a > b)
}
pub fn i8x16_gt_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u8x16_compare(stack, |a, b| a > b)
}
pub fn i8x16_le_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i8x16_compare(stack, |a, b| a <= b)
}
pub fn i8x16_le_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u8x16_compare(stack, |a, b| a <= b)
}
pub fn i8x16_ge_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i8x16_compare(stack, |a, b| a >= b)
}
pub fn i8x16_ge_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u8x16_compare(stack, |a, b| a >= b)
}

// ============================================================================
// i16x8 comparisons
// ============================================================================

/// Signed i16x8 lane-wise comparison; all-ones for true, all-zeros for false.
fn i16x8_compare(stack: &mut Stack, op: fn(i16, i16) -> bool) -> Result<(), RuntimeError> {
    let b = get_i16x8_lanes(stack.pop_v128()?);
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0u8; 16];
    for i in 0..8 {
        let mask: u16 = if op(a[i], b[i]) { 0xFFFF } else { 0 };
        r[i * 2..i * 2 + 2].copy_from_slice(&mask.to_le_bytes());
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// Unsigned u16x8 lane-wise comparison; all-ones for true, all-zeros for false.
fn u16x8_compare(stack: &mut Stack, op: fn(u16, u16) -> bool) -> Result<(), RuntimeError> {
    let b = get_u16x8_lanes(stack.pop_v128()?);
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u8; 16];
    for i in 0..8 {
        let mask: u16 = if op(a[i], b[i]) { 0xFFFF } else { 0 };
        r[i * 2..i * 2 + 2].copy_from_slice(&mask.to_le_bytes());
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i16x8_eq(stack: &mut Stack) -> Result<(), RuntimeError> {
    i16x8_compare(stack, |a, b| a == b)
}
pub fn i16x8_ne(stack: &mut Stack) -> Result<(), RuntimeError> {
    i16x8_compare(stack, |a, b| a != b)
}
pub fn i16x8_lt_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i16x8_compare(stack, |a, b| a < b)
}
pub fn i16x8_lt_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u16x8_compare(stack, |a, b| a < b)
}
pub fn i16x8_gt_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i16x8_compare(stack, |a, b| a > b)
}
pub fn i16x8_gt_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u16x8_compare(stack, |a, b| a > b)
}
pub fn i16x8_le_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i16x8_compare(stack, |a, b| a <= b)
}
pub fn i16x8_le_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u16x8_compare(stack, |a, b| a <= b)
}
pub fn i16x8_ge_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i16x8_compare(stack, |a, b| a >= b)
}
pub fn i16x8_ge_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u16x8_compare(stack, |a, b| a >= b)
}

// ============================================================================
// i32x4 comparisons
// ============================================================================

/// Signed i32x4 lane-wise comparison; all-ones for true, all-zeros for false.
fn i32x4_compare(stack: &mut Stack, op: fn(i32, i32) -> bool) -> Result<(), RuntimeError> {
    let b = get_i32x4_lanes(stack.pop_v128()?);
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let mut r = [0u8; 16];
    for i in 0..4 {
        let mask: u32 = if op(a[i], b[i]) { 0xFFFFFFFF } else { 0 };
        r[i * 4..i * 4 + 4].copy_from_slice(&mask.to_le_bytes());
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// Unsigned u32x4 lane-wise comparison; all-ones for true, all-zeros for false.
fn u32x4_compare(stack: &mut Stack, op: fn(u32, u32) -> bool) -> Result<(), RuntimeError> {
    let b = get_u32x4_lanes(stack.pop_v128()?);
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let mut r = [0u8; 16];
    for i in 0..4 {
        let mask: u32 = if op(a[i], b[i]) { 0xFFFFFFFF } else { 0 };
        r[i * 4..i * 4 + 4].copy_from_slice(&mask.to_le_bytes());
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i32x4_eq(stack: &mut Stack) -> Result<(), RuntimeError> {
    i32x4_compare(stack, |a, b| a == b)
}
pub fn i32x4_ne(stack: &mut Stack) -> Result<(), RuntimeError> {
    i32x4_compare(stack, |a, b| a != b)
}
pub fn i32x4_lt_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i32x4_compare(stack, |a, b| a < b)
}
pub fn i32x4_lt_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u32x4_compare(stack, |a, b| a < b)
}
pub fn i32x4_gt_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i32x4_compare(stack, |a, b| a > b)
}
pub fn i32x4_gt_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u32x4_compare(stack, |a, b| a > b)
}
pub fn i32x4_le_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i32x4_compare(stack, |a, b| a <= b)
}
pub fn i32x4_le_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u32x4_compare(stack, |a, b| a <= b)
}
pub fn i32x4_ge_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i32x4_compare(stack, |a, b| a >= b)
}
pub fn i32x4_ge_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    u32x4_compare(stack, |a, b| a >= b)
}

// ============================================================================
// i64x2 comparisons
// ============================================================================

/// Signed i64x2 lane-wise comparison; all-ones for true, all-zeros for false.
fn i64x2_compare(stack: &mut Stack, op: fn(i64, i64) -> bool) -> Result<(), RuntimeError> {
    let b = get_i64x2_lanes(stack.pop_v128()?);
    let a = get_i64x2_lanes(stack.pop_v128()?);
    let mut r = [0u8; 16];
    for i in 0..2 {
        let mask: u64 = if op(a[i], b[i]) { 0xFFFFFFFFFFFFFFFF } else { 0 };
        r[i * 8..i * 8 + 8].copy_from_slice(&mask.to_le_bytes());
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i64x2_eq(stack: &mut Stack) -> Result<(), RuntimeError> {
    i64x2_compare(stack, |a, b| a == b)
}
pub fn i64x2_ne(stack: &mut Stack) -> Result<(), RuntimeError> {
    i64x2_compare(stack, |a, b| a != b)
}
pub fn i64x2_lt_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i64x2_compare(stack, |a, b| a < b)
}
pub fn i64x2_gt_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i64x2_compare(stack, |a, b| a > b)
}
pub fn i64x2_le_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i64x2_compare(stack, |a, b| a <= b)
}
pub fn i64x2_ge_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    i64x2_compare(stack, |a, b| a >= b)
}

// ============================================================================
// i8x16 operations
// ============================================================================

pub fn i8x16_abs(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = a[i].wrapping_abs();
    }
    stack.push(Value::V128(set_i8x16_lanes(r)));
    Ok(())
}

pub fn i8x16_neg(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = (0i8).wrapping_sub(a[i]);
    }
    stack.push(Value::V128(set_i8x16_lanes(r)));
    Ok(())
}

pub fn i8x16_popcnt(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i].count_ones() as u8;
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i8x16_all_true(stack: &mut Stack) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let result = if v.iter().all(|&b| b != 0) { 1i32 } else { 0i32 };
    stack.push(Value::I32(result));
    Ok(())
}

pub fn i8x16_bitmask(stack: &mut Stack) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let mut result = 0u32;
    for (i, &byte) in v.iter().enumerate() {
        if byte & 0x80 != 0 {
            result |= 1 << i;
        }
    }
    stack.push(Value::I32(result as i32));
    Ok(())
}

pub fn i8x16_shl(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = (stack.pop_i32()? as u32 % 8) as u8;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i] << shift;
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i8x16_shr_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = (stack.pop_i32()? as u32 % 8) as u8;
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = a[i] >> shift;
    }
    stack.push(Value::V128(set_i8x16_lanes(r)));
    Ok(())
}

pub fn i8x16_shr_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = (stack.pop_i32()? as u32 % 8) as u8;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i] >> shift;
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i8x16_add(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i].wrapping_add(b[i]);
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i8x16_add_sat_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i8x16_lanes(stack.pop_v128()?);
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = a[i].saturating_add(b[i]);
    }
    stack.push(Value::V128(set_i8x16_lanes(r)));
    Ok(())
}

pub fn i8x16_add_sat_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i].saturating_add(b[i]);
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i8x16_sub(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i].wrapping_sub(b[i]);
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i8x16_sub_sat_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i8x16_lanes(stack.pop_v128()?);
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = a[i].saturating_sub(b[i]);
    }
    stack.push(Value::V128(set_i8x16_lanes(r)));
    Ok(())
}

pub fn i8x16_sub_sat_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i].saturating_sub(b[i]);
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i8x16_min_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i8x16_lanes(stack.pop_v128()?);
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = a[i].min(b[i]);
    }
    stack.push(Value::V128(set_i8x16_lanes(r)));
    Ok(())
}

pub fn i8x16_min_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i].min(b[i]);
    }
    stack.push(Value::V128(r));
    Ok(())
}

pub fn i8x16_max_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i8x16_lanes(stack.pop_v128()?);
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = a[i].max(b[i]);
    }
    stack.push(Value::V128(set_i8x16_lanes(r)));
    Ok(())
}

pub fn i8x16_max_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i].max(b[i]);
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// i8x16.avgr_u — unsigned rounding average: (a + b + 1) / 2
pub fn i8x16_avgr_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = (a[i] as u16 + b[i] as u16).div_ceil(2) as u8;
    }
    stack.push(Value::V128(r));
    Ok(())
}

// ============================================================================
// i16x8 operations
// ============================================================================

pub fn i16x8_abs(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = a[i].wrapping_abs();
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_neg(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = (0i16).wrapping_sub(a[i]);
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_all_true(stack: &mut Stack) -> Result<(), RuntimeError> {
    let lanes = get_u16x8_lanes(stack.pop_v128()?);
    let result = if lanes.iter().all(|&v| v != 0) { 1i32 } else { 0i32 };
    stack.push(Value::I32(result));
    Ok(())
}

pub fn i16x8_bitmask(stack: &mut Stack) -> Result<(), RuntimeError> {
    let lanes = get_i16x8_lanes(stack.pop_v128()?);
    let mut result = 0u32;
    for (i, &lane) in lanes.iter().enumerate() {
        if lane < 0 {
            result |= 1 << i;
        }
    }
    stack.push(Value::I32(result as i32));
    Ok(())
}

pub fn i16x8_shl(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = (stack.pop_i32()? as u32 % 16) as u16;
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = a[i] << shift;
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_shr_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = (stack.pop_i32()? as u32 % 16) as u16;
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = a[i] >> shift;
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_shr_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = (stack.pop_i32()? as u32 % 16) as u16;
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = a[i] >> shift;
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_add(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u16x8_lanes(stack.pop_v128()?);
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = a[i].wrapping_add(b[i]);
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_add_sat_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i16x8_lanes(stack.pop_v128()?);
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = a[i].saturating_add(b[i]);
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_add_sat_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u16x8_lanes(stack.pop_v128()?);
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = a[i].saturating_add(b[i]);
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_sub(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u16x8_lanes(stack.pop_v128()?);
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = a[i].wrapping_sub(b[i]);
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_sub_sat_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i16x8_lanes(stack.pop_v128()?);
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = a[i].saturating_sub(b[i]);
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_sub_sat_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u16x8_lanes(stack.pop_v128()?);
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = a[i].saturating_sub(b[i]);
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_mul(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u16x8_lanes(stack.pop_v128()?);
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = a[i].wrapping_mul(b[i]);
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_min_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i16x8_lanes(stack.pop_v128()?);
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = a[i].min(b[i]);
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_min_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u16x8_lanes(stack.pop_v128()?);
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = a[i].min(b[i]);
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_max_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i16x8_lanes(stack.pop_v128()?);
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = a[i].max(b[i]);
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

pub fn i16x8_max_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u16x8_lanes(stack.pop_v128()?);
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = a[i].max(b[i]);
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

/// i16x8.avgr_u — unsigned rounding average: (a + b + 1) / 2
pub fn i16x8_avgr_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u16x8_lanes(stack.pop_v128()?);
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = (a[i] as u32 + b[i] as u32).div_ceil(2) as u16;
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

/// i16x8.q15mulr_sat_s — Q15 fixed-point multiply with saturation
/// result = saturate_i16((a * b + 0x4000) >> 15)
pub fn i16x8_q15mulr_sat_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i16x8_lanes(stack.pop_v128()?);
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        let product = (a[i] as i32) * (b[i] as i32);
        let shifted = (product + 0x4000) >> 15;
        r[i] = shifted.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

// ============================================================================
// i32x4 operations
// ============================================================================

pub fn i32x4_abs(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = a[i].wrapping_abs();
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

pub fn i32x4_neg(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = (0i32).wrapping_sub(a[i]);
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

pub fn i32x4_all_true(stack: &mut Stack) -> Result<(), RuntimeError> {
    let lanes = get_u32x4_lanes(stack.pop_v128()?);
    let result = if lanes.iter().all(|&v| v != 0) { 1i32 } else { 0i32 };
    stack.push(Value::I32(result));
    Ok(())
}

pub fn i32x4_bitmask(stack: &mut Stack) -> Result<(), RuntimeError> {
    let lanes = get_i32x4_lanes(stack.pop_v128()?);
    let mut result = 0u32;
    for (i, &lane) in lanes.iter().enumerate() {
        if lane < 0 {
            result |= 1 << i;
        }
    }
    stack.push(Value::I32(result as i32));
    Ok(())
}

pub fn i32x4_shl(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = stack.pop_i32()? as u32 % 32;
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = a[i] << shift;
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

pub fn i32x4_shr_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = stack.pop_i32()? as u32 % 32;
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = a[i] >> shift;
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

pub fn i32x4_shr_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = stack.pop_i32()? as u32 % 32;
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = a[i] >> shift;
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

pub fn i32x4_add(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u32x4_lanes(stack.pop_v128()?);
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = a[i].wrapping_add(b[i]);
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

pub fn i32x4_sub(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u32x4_lanes(stack.pop_v128()?);
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = a[i].wrapping_sub(b[i]);
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

pub fn i32x4_mul(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u32x4_lanes(stack.pop_v128()?);
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = a[i].wrapping_mul(b[i]);
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

pub fn i32x4_min_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i32x4_lanes(stack.pop_v128()?);
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = a[i].min(b[i]);
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

pub fn i32x4_min_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u32x4_lanes(stack.pop_v128()?);
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = a[i].min(b[i]);
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

pub fn i32x4_max_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i32x4_lanes(stack.pop_v128()?);
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = a[i].max(b[i]);
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

pub fn i32x4_max_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u32x4_lanes(stack.pop_v128()?);
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = a[i].max(b[i]);
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

// ============================================================================
// i64x2 operations
// ============================================================================

pub fn i64x2_abs(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i64x2_lanes(stack.pop_v128()?);
    let r = [a[0].wrapping_abs(), a[1].wrapping_abs()];
    stack.push(Value::V128(set_i64x2_lanes(r)));
    Ok(())
}

pub fn i64x2_neg(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i64x2_lanes(stack.pop_v128()?);
    let r = [(0i64).wrapping_sub(a[0]), (0i64).wrapping_sub(a[1])];
    stack.push(Value::V128(set_i64x2_lanes(r)));
    Ok(())
}

pub fn i64x2_all_true(stack: &mut Stack) -> Result<(), RuntimeError> {
    let lanes = get_u64x2_lanes(stack.pop_v128()?);
    let result = if lanes.iter().all(|&v| v != 0) { 1i32 } else { 0i32 };
    stack.push(Value::I32(result));
    Ok(())
}

pub fn i64x2_bitmask(stack: &mut Stack) -> Result<(), RuntimeError> {
    let lanes = get_i64x2_lanes(stack.pop_v128()?);
    let mut result = 0u32;
    for (i, &lane) in lanes.iter().enumerate() {
        if lane < 0 {
            result |= 1 << i;
        }
    }
    stack.push(Value::I32(result as i32));
    Ok(())
}

pub fn i64x2_shl(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = stack.pop_i32()? as u64 % 64;
    let a = get_u64x2_lanes(stack.pop_v128()?);
    let r = [a[0] << shift, a[1] << shift];
    stack.push(Value::V128(set_u64x2_lanes(r)));
    Ok(())
}

pub fn i64x2_shr_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = stack.pop_i32()? as u64 % 64;
    let a = get_i64x2_lanes(stack.pop_v128()?);
    let r = [a[0] >> shift, a[1] >> shift];
    stack.push(Value::V128(set_i64x2_lanes(r)));
    Ok(())
}

pub fn i64x2_shr_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let shift = stack.pop_i32()? as u64 % 64;
    let a = get_u64x2_lanes(stack.pop_v128()?);
    let r = [a[0] >> shift, a[1] >> shift];
    stack.push(Value::V128(set_u64x2_lanes(r)));
    Ok(())
}

pub fn i64x2_add(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u64x2_lanes(stack.pop_v128()?);
    let a = get_u64x2_lanes(stack.pop_v128()?);
    let r = [a[0].wrapping_add(b[0]), a[1].wrapping_add(b[1])];
    stack.push(Value::V128(set_u64x2_lanes(r)));
    Ok(())
}

pub fn i64x2_sub(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u64x2_lanes(stack.pop_v128()?);
    let a = get_u64x2_lanes(stack.pop_v128()?);
    let r = [a[0].wrapping_sub(b[0]), a[1].wrapping_sub(b[1])];
    stack.push(Value::V128(set_u64x2_lanes(r)));
    Ok(())
}

pub fn i64x2_mul(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u64x2_lanes(stack.pop_v128()?);
    let a = get_u64x2_lanes(stack.pop_v128()?);
    let r = [a[0].wrapping_mul(b[0]), a[1].wrapping_mul(b[1])];
    stack.push(Value::V128(set_u64x2_lanes(r)));
    Ok(())
}

// ============================================================================
// Shuffle and swizzle
// ============================================================================

/// i8x16.shuffle — select lanes from two v128 values using 16-byte immediate indices.
/// Each index in `lanes[i]` selects from the concatenation \[a, b\] (0..15 from a, 16..31 from b)
pub fn i8x16_shuffle(stack: &mut Stack, lanes: &[u8; 16]) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut concat = [0u8; 32];
    concat[0..16].copy_from_slice(&a);
    concat[16..32].copy_from_slice(&b);
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = concat[lanes[i] as usize];
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// i8x16.swizzle — dynamic lane selection from a single v128
/// Each index byte in b selects a lane from a; out-of-range indices yield 0
pub fn i8x16_swizzle(stack: &mut Stack) -> Result<(), RuntimeError> {
    let idx = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u8; 16];
    for i in 0..16 {
        let j = idx[i];
        r[i] = if j < 16 { a[j as usize] } else { 0 };
    }
    stack.push(Value::V128(r));
    Ok(())
}

// ============================================================================
// Narrow operations
// ============================================================================

/// i8x16.narrow_i16x8_s — narrow two i16x8 to one i8x16 with signed saturation
pub fn i8x16_narrow_i16x8_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i16x8_lanes(stack.pop_v128()?);
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i8; 16];
    for i in 0..8 {
        r[i] = a[i].clamp(i8::MIN as i16, i8::MAX as i16) as i8;
    }
    for i in 0..8 {
        r[i + 8] = b[i].clamp(i8::MIN as i16, i8::MAX as i16) as i8;
    }
    stack.push(Value::V128(set_i8x16_lanes(r)));
    Ok(())
}

/// i8x16.narrow_i16x8_u — narrow two i16x8 to one i8x16 with unsigned saturation
pub fn i8x16_narrow_i16x8_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i16x8_lanes(stack.pop_v128()?);
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0u8; 16];
    for i in 0..8 {
        r[i] = a[i].clamp(0, u8::MAX as i16) as u8;
    }
    for i in 0..8 {
        r[i + 8] = b[i].clamp(0, u8::MAX as i16) as u8;
    }
    stack.push(Value::V128(set_u8x16_lanes(r)));
    Ok(())
}

/// i16x8.narrow_i32x4_s — narrow two i32x4 to one i16x8 with signed saturation
pub fn i16x8_narrow_i32x4_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i32x4_lanes(stack.pop_v128()?);
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..4 {
        r[i] = a[i].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    for i in 0..4 {
        r[i + 4] = b[i].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

/// i16x8.narrow_i32x4_u — narrow two i32x4 to one i16x8 with unsigned saturation
pub fn i16x8_narrow_i32x4_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i32x4_lanes(stack.pop_v128()?);
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let mut r = [0u16; 8];
    for i in 0..4 {
        r[i] = a[i].clamp(0, u16::MAX as i32) as u16;
    }
    for i in 0..4 {
        r[i + 4] = b[i].clamp(0, u16::MAX as i32) as u16;
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

// ============================================================================
// Integer extension operations
// ============================================================================

/// i16x8.extend_low_i8x16_s — sign-extend the low 8 i8 lanes to i16
pub fn i16x8_extend_low_i8x16_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = a[i] as i16;
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

/// i16x8.extend_high_i8x16_s — sign-extend the high 8 i8 lanes to i16
pub fn i16x8_extend_high_i8x16_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = a[i + 8] as i16;
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

/// i16x8.extend_low_i8x16_u — zero-extend the low 8 i8 lanes to i16
pub fn i16x8_extend_low_i8x16_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = stack.pop_v128()?;
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = a[i] as u16;
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

/// i16x8.extend_high_i8x16_u — zero-extend the high 8 i8 lanes to i16
pub fn i16x8_extend_high_i8x16_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = stack.pop_v128()?;
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = a[i + 8] as u16;
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

/// i32x4.extend_low_i16x8_s — sign-extend the low 4 i16 lanes to i32
pub fn i32x4_extend_low_i16x8_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = a[i] as i32;
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

/// i32x4.extend_high_i16x8_s — sign-extend the high 4 i16 lanes to i32
pub fn i32x4_extend_high_i16x8_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = a[i + 4] as i32;
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

/// i32x4.extend_low_i16x8_u — zero-extend the low 4 i16 lanes to i32
pub fn i32x4_extend_low_i16x8_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = a[i] as u32;
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

/// i32x4.extend_high_i16x8_u — zero-extend the high 4 i16 lanes to i32
pub fn i32x4_extend_high_i16x8_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = a[i + 4] as u32;
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

/// i64x2.extend_low_i32x4_s — sign-extend the low 2 i32 lanes to i64
pub fn i64x2_extend_low_i32x4_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let r = [a[0] as i64, a[1] as i64];
    stack.push(Value::V128(set_i64x2_lanes(r)));
    Ok(())
}

/// i64x2.extend_high_i32x4_s — sign-extend the high 2 i32 lanes to i64
pub fn i64x2_extend_high_i32x4_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let r = [a[2] as i64, a[3] as i64];
    stack.push(Value::V128(set_i64x2_lanes(r)));
    Ok(())
}

/// i64x2.extend_low_i32x4_u — zero-extend the low 2 i32 lanes to i64
pub fn i64x2_extend_low_i32x4_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let r = [a[0] as u64, a[1] as u64];
    stack.push(Value::V128(set_u64x2_lanes(r)));
    Ok(())
}

/// i64x2.extend_high_i32x4_u — zero-extend the high 2 i32 lanes to i64
pub fn i64x2_extend_high_i32x4_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let r = [a[2] as u64, a[3] as u64];
    stack.push(Value::V128(set_u64x2_lanes(r)));
    Ok(())
}

// ============================================================================
// Extended multiply operations
// ============================================================================

/// i16x8.extmul_low_i8x16_s — widening signed multiply of low i8 lanes
pub fn i16x8_extmul_low_i8x16_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i8x16_lanes(stack.pop_v128()?);
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = (a[i] as i16) * (b[i] as i16);
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

/// i16x8.extmul_high_i8x16_s — widening signed multiply of high i8 lanes
pub fn i16x8_extmul_high_i8x16_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i8x16_lanes(stack.pop_v128()?);
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = (a[i + 8] as i16) * (b[i + 8] as i16);
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

/// i16x8.extmul_low_i8x16_u — widening unsigned multiply of low i8 lanes
pub fn i16x8_extmul_low_i8x16_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = (a[i] as u16) * (b[i] as u16);
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

/// i16x8.extmul_high_i8x16_u — widening unsigned multiply of high i8 lanes
pub fn i16x8_extmul_high_i8x16_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = stack.pop_v128()?;
    let a = stack.pop_v128()?;
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = (a[i + 8] as u16) * (b[i + 8] as u16);
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

/// i32x4.extmul_low_i16x8_s — widening signed multiply of low i16 lanes
pub fn i32x4_extmul_low_i16x8_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i16x8_lanes(stack.pop_v128()?);
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = (a[i] as i32) * (b[i] as i32);
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

/// i32x4.extmul_high_i16x8_s — widening signed multiply of high i16 lanes
pub fn i32x4_extmul_high_i16x8_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i16x8_lanes(stack.pop_v128()?);
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = (a[i + 4] as i32) * (b[i + 4] as i32);
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

/// i32x4.extmul_low_i16x8_u — widening unsigned multiply of low i16 lanes
pub fn i32x4_extmul_low_i16x8_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u16x8_lanes(stack.pop_v128()?);
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = (a[i] as u32) * (b[i] as u32);
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

/// i32x4.extmul_high_i16x8_u — widening unsigned multiply of high i16 lanes
pub fn i32x4_extmul_high_i16x8_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u16x8_lanes(stack.pop_v128()?);
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = (a[i + 4] as u32) * (b[i + 4] as u32);
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

/// i64x2.extmul_low_i32x4_s — widening signed multiply of low i32 lanes
pub fn i64x2_extmul_low_i32x4_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i32x4_lanes(stack.pop_v128()?);
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let r = [(a[0] as i64) * (b[0] as i64), (a[1] as i64) * (b[1] as i64)];
    stack.push(Value::V128(set_i64x2_lanes(r)));
    Ok(())
}

/// i64x2.extmul_high_i32x4_s — widening signed multiply of high i32 lanes
pub fn i64x2_extmul_high_i32x4_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i32x4_lanes(stack.pop_v128()?);
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let r = [(a[2] as i64) * (b[2] as i64), (a[3] as i64) * (b[3] as i64)];
    stack.push(Value::V128(set_i64x2_lanes(r)));
    Ok(())
}

/// i64x2.extmul_low_i32x4_u — widening unsigned multiply of low i32 lanes
pub fn i64x2_extmul_low_i32x4_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u32x4_lanes(stack.pop_v128()?);
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let r = [(a[0] as u64) * (b[0] as u64), (a[1] as u64) * (b[1] as u64)];
    stack.push(Value::V128(set_u64x2_lanes(r)));
    Ok(())
}

/// i64x2.extmul_high_i32x4_u — widening unsigned multiply of high i32 lanes
pub fn i64x2_extmul_high_i32x4_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_u32x4_lanes(stack.pop_v128()?);
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let r = [(a[2] as u64) * (b[2] as u64), (a[3] as u64) * (b[3] as u64)];
    stack.push(Value::V128(set_u64x2_lanes(r)));
    Ok(())
}

// ============================================================================
// Pairwise add operations
// ============================================================================

/// i16x8.extadd_pairwise_i8x16_s — pairwise signed widening add of i8 lanes
pub fn i16x8_extadd_pairwise_i8x16_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i8x16_lanes(stack.pop_v128()?);
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = (a[i * 2] as i16) + (a[i * 2 + 1] as i16);
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

/// i16x8.extadd_pairwise_i8x16_u — pairwise unsigned widening add of i8 lanes
pub fn i16x8_extadd_pairwise_i8x16_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = stack.pop_v128()?;
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = (a[i * 2] as u16) + (a[i * 2 + 1] as u16);
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

/// i32x4.extadd_pairwise_i16x8_s — pairwise signed widening add of i16 lanes
pub fn i32x4_extadd_pairwise_i16x8_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = (a[i * 2] as i32) + (a[i * 2 + 1] as i32);
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

/// i32x4.extadd_pairwise_i16x8_u — pairwise unsigned widening add of i16 lanes
pub fn i32x4_extadd_pairwise_i16x8_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_u16x8_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = (a[i * 2] as u32) + (a[i * 2 + 1] as u32);
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

// ============================================================================
// Dot product
// ============================================================================

/// i32x4.dot_i16x8_s — signed dot product of i16 pairs producing i32 lanes.
/// `r[i] = a[i*2]*b[i*2] + a[i*2+1]*b[i*2+1]`  (all as i32)
pub fn i32x4_dot_i16x8_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let b = get_i16x8_lanes(stack.pop_v128()?);
    let a = get_i16x8_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        let lo = (a[i * 2] as i32) * (b[i * 2] as i32);
        let hi = (a[i * 2 + 1] as i32) * (b[i * 2 + 1] as i32);
        r[i] = lo.wrapping_add(hi);
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

// ============================================================================
// Truncation / conversion operations
// ============================================================================

/// i32x4.trunc_sat_f32x4_s — saturating truncation of f32 to signed i32
pub fn i32x4_trunc_sat_f32x4_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_f32_lanes(stack.pop_v128()?);
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = trunc_sat_f32_to_i32_s(a[i]);
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

/// i32x4.trunc_sat_f32x4_u — saturating truncation of f32 to unsigned i32
pub fn i32x4_trunc_sat_f32x4_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_f32_lanes(stack.pop_v128()?);
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = trunc_sat_f32_to_u32(a[i]);
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

/// f32x4.convert_i32x4_s — convert signed i32 lanes to f32
pub fn f32x4_convert_i32x4_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let r = [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32];
    stack.push(Value::V128(set_f32_lanes(r)));
    Ok(())
}

/// f32x4.convert_i32x4_u — convert unsigned i32 lanes to f32
pub fn f32x4_convert_i32x4_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let r = [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32];
    stack.push(Value::V128(set_f32_lanes(r)));
    Ok(())
}

/// i32x4.trunc_sat_f64x2_s_zero — truncate two f64 to signed i32, zero-extend to 4 lanes
pub fn i32x4_trunc_sat_f64x2_s_zero(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_f64_lanes(stack.pop_v128()?);
    let r = [trunc_sat_f64_to_i32_s(a[0]), trunc_sat_f64_to_i32_s(a[1]), 0i32, 0i32];
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

/// i32x4.trunc_sat_f64x2_u_zero — truncate two f64 to unsigned i32, zero-extend to 4 lanes
pub fn i32x4_trunc_sat_f64x2_u_zero(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_f64_lanes(stack.pop_v128()?);
    let r = [trunc_sat_f64_to_u32(a[0]), trunc_sat_f64_to_u32(a[1]), 0u32, 0u32];
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

/// f64x2.convert_low_i32x4_s — convert low two signed i32 lanes to f64
pub fn f64x2_convert_low_i32x4_s(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_i32x4_lanes(stack.pop_v128()?);
    let r = [a[0] as f64, a[1] as f64];
    stack.push(Value::V128(set_f64_lanes(r)));
    Ok(())
}

/// f64x2.convert_low_i32x4_u — convert low two unsigned i32 lanes to f64
pub fn f64x2_convert_low_i32x4_u(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_u32x4_lanes(stack.pop_v128()?);
    let r = [a[0] as f64, a[1] as f64];
    stack.push(Value::V128(set_f64_lanes(r)));
    Ok(())
}

/// f32x4.demote_f64x2_zero — demote two f64 to f32, zero-fill high lanes
pub fn f32x4_demote_f64x2_zero(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_f64_lanes(stack.pop_v128()?);
    let r = [
        canonicalise_f32_nan(a[0] as f32),
        canonicalise_f32_nan(a[1] as f32),
        0.0f32,
        0.0f32,
    ];
    stack.push(Value::V128(set_f32_lanes(r)));
    Ok(())
}

/// f64x2.promote_low_f32x4 — promote low two f32 to f64
pub fn f64x2_promote_low_f32x4(stack: &mut Stack) -> Result<(), RuntimeError> {
    let a = get_f32_lanes(stack.pop_v128()?);
    let r = [canonicalise_f64_nan(a[0] as f64), canonicalise_f64_nan(a[1] as f64)];
    stack.push(Value::V128(set_f64_lanes(r)));
    Ok(())
}

// ============================================================================
// Memory splat/zero/extend load operations
// ============================================================================

/// v128.load8_splat — load one i8 and broadcast to all 16 lanes
pub fn v128_load8_splat(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let val = memory.read_u8(ea)?;
    stack.push(Value::V128([val; 16]));
    Ok(())
}

/// v128.load16_splat — load one i16 and broadcast to all 8 lanes
pub fn v128_load16_splat(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let val = memory.read_u16(ea)?;
    let bytes = val.to_le_bytes();
    let mut r = [0u8; 16];
    for i in 0..8 {
        r[i * 2] = bytes[0];
        r[i * 2 + 1] = bytes[1];
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// v128.load32_splat — load one i32 and broadcast to all 4 lanes
pub fn v128_load32_splat(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let val = memory.read_u32(ea)?;
    let bytes = val.to_le_bytes();
    let mut r = [0u8; 16];
    for i in 0..4 {
        r[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
    stack.push(Value::V128(r));
    Ok(())
}

/// v128.load64_splat — load one i64 and broadcast to both lanes
pub fn v128_load64_splat(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let val = memory.read_u64(ea)?;
    let bytes = val.to_le_bytes();
    let mut r = [0u8; 16];
    r[0..8].copy_from_slice(&bytes);
    r[8..16].copy_from_slice(&bytes);
    stack.push(Value::V128(r));
    Ok(())
}

/// v128.load32_zero — load 32 bits into low lane, zero the rest
pub fn v128_load32_zero(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let val = memory.read_u32(ea)?;
    let mut r = [0u8; 16];
    r[0..4].copy_from_slice(&val.to_le_bytes());
    stack.push(Value::V128(r));
    Ok(())
}

/// v128.load64_zero — load 64 bits into low lane, zero the rest
pub fn v128_load64_zero(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let val = memory.read_u64(ea)?;
    let mut r = [0u8; 16];
    r[0..8].copy_from_slice(&val.to_le_bytes());
    stack.push(Value::V128(r));
    Ok(())
}

/// v128.load8x8_s — load 8 i8, sign-extend each to i16
pub fn v128_load8x8_s(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let bytes = memory.read_bytes(ea, 8)?;
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = bytes[i] as i8 as i16;
    }
    stack.push(Value::V128(set_i16x8_lanes(r)));
    Ok(())
}

/// v128.load8x8_u — load 8 i8, zero-extend each to i16
pub fn v128_load8x8_u(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let bytes = memory.read_bytes(ea, 8)?;
    let mut r = [0u16; 8];
    for i in 0..8 {
        r[i] = bytes[i] as u16;
    }
    stack.push(Value::V128(set_u16x8_lanes(r)));
    Ok(())
}

/// v128.load16x4_s — load 4 i16, sign-extend each to i32
pub fn v128_load16x4_s(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let bytes = memory.read_bytes(ea, 8)?;
    let mut r = [0i32; 4];
    for i in 0..4 {
        let val = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        r[i] = val as i32;
    }
    stack.push(Value::V128(set_i32x4_lanes(r)));
    Ok(())
}

/// v128.load16x4_u — load 4 i16, zero-extend each to i32
pub fn v128_load16x4_u(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let bytes = memory.read_bytes(ea, 8)?;
    let mut r = [0u32; 4];
    for i in 0..4 {
        let val = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        r[i] = val as u32;
    }
    stack.push(Value::V128(set_u32x4_lanes(r)));
    Ok(())
}

/// v128.load32x2_s — load 2 i32, sign-extend each to i64
pub fn v128_load32x2_s(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let bytes = memory.read_bytes(ea, 8)?;
    let lo = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let hi = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let r = [lo as i64, hi as i64];
    stack.push(Value::V128(set_i64x2_lanes(r)));
    Ok(())
}

/// v128.load32x2_u — load 2 i32, zero-extend each to i64
pub fn v128_load32x2_u(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let ea = effective_address(stack, memarg)?;
    let bytes = memory.read_bytes(ea, 8)?;
    let lo = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let hi = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let r = [lo as u64, hi as u64];
    stack.push(Value::V128(set_u64x2_lanes(r)));
    Ok(())
}

// ============================================================================
// Load/store lane operations
// ============================================================================

/// v128.load8_lane — load a single byte from memory into the specified lane
pub fn v128_load8_lane(stack: &mut Stack, memory: &Memory, memarg: &MemArg, lane: u8) -> Result<(), RuntimeError> {
    let mut v = stack.pop_v128()?;
    let ea = effective_address(stack, memarg)?;
    v[lane as usize] = memory.read_u8(ea)?;
    stack.push(Value::V128(v));
    Ok(())
}

/// v128.load16_lane — load 2 bytes from memory into the specified i16 lane
pub fn v128_load16_lane(stack: &mut Stack, memory: &Memory, memarg: &MemArg, lane: u8) -> Result<(), RuntimeError> {
    let mut v = stack.pop_v128()?;
    let ea = effective_address(stack, memarg)?;
    let val = memory.read_u16(ea)?;
    let offset = lane as usize * 2;
    v[offset..offset + 2].copy_from_slice(&val.to_le_bytes());
    stack.push(Value::V128(v));
    Ok(())
}

/// v128.load32_lane — load 4 bytes from memory into the specified i32 lane
pub fn v128_load32_lane(stack: &mut Stack, memory: &Memory, memarg: &MemArg, lane: u8) -> Result<(), RuntimeError> {
    let mut v = stack.pop_v128()?;
    let ea = effective_address(stack, memarg)?;
    let val = memory.read_u32(ea)?;
    let offset = lane as usize * 4;
    v[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    stack.push(Value::V128(v));
    Ok(())
}

/// v128.load64_lane — load 8 bytes from memory into the specified i64 lane
pub fn v128_load64_lane(stack: &mut Stack, memory: &Memory, memarg: &MemArg, lane: u8) -> Result<(), RuntimeError> {
    let mut v = stack.pop_v128()?;
    let ea = effective_address(stack, memarg)?;
    let val = memory.read_u64(ea)?;
    let offset = lane as usize * 8;
    v[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
    stack.push(Value::V128(v));
    Ok(())
}

/// v128.store8_lane — store the specified byte lane to memory
pub fn v128_store8_lane(stack: &mut Stack, memory: &mut Memory, memarg: &MemArg, lane: u8) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let ea = effective_address(stack, memarg)?;
    memory.write_u8(ea, v[lane as usize])?;
    Ok(())
}

/// v128.store16_lane — store the specified i16 lane to memory
pub fn v128_store16_lane(
    stack: &mut Stack,
    memory: &mut Memory,
    memarg: &MemArg,
    lane: u8,
) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let ea = effective_address(stack, memarg)?;
    let offset = lane as usize * 2;
    let val = u16::from_le_bytes([v[offset], v[offset + 1]]);
    memory.write_u16(ea, val)?;
    Ok(())
}

/// v128.store32_lane — store the specified i32 lane to memory
pub fn v128_store32_lane(
    stack: &mut Stack,
    memory: &mut Memory,
    memarg: &MemArg,
    lane: u8,
) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let ea = effective_address(stack, memarg)?;
    let offset = lane as usize * 4;
    let val = u32::from_le_bytes([v[offset], v[offset + 1], v[offset + 2], v[offset + 3]]);
    memory.write_u32(ea, val)?;
    Ok(())
}

/// v128.store64_lane — store the specified i64 lane to memory
pub fn v128_store64_lane(
    stack: &mut Stack,
    memory: &mut Memory,
    memarg: &MemArg,
    lane: u8,
) -> Result<(), RuntimeError> {
    let v = stack.pop_v128()?;
    let ea = effective_address(stack, memarg)?;
    let offset = lane as usize * 8;
    let val = u64::from_le_bytes([
        v[offset],
        v[offset + 1],
        v[offset + 2],
        v[offset + 3],
        v[offset + 4],
        v[offset + 5],
        v[offset + 6],
        v[offset + 7],
    ]);
    memory.write_u64(ea, val)?;
    Ok(())
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Saturating truncation of f32 to signed i32
fn trunc_sat_f32_to_i32_s(v: f32) -> i32 {
    if v.is_nan() {
        0
    } else if v >= i32::MAX as f32 {
        i32::MAX
    } else if v <= i32::MIN as f32 {
        i32::MIN
    } else {
        v as i32
    }
}

/// Saturating truncation of f32 to unsigned i32
fn trunc_sat_f32_to_u32(v: f32) -> u32 {
    if v.is_nan() {
        0
    } else if v >= u32::MAX as f32 {
        u32::MAX
    } else if v <= 0.0 {
        0
    } else {
        v as u32
    }
}

/// Saturating truncation of f64 to signed i32
fn trunc_sat_f64_to_i32_s(v: f64) -> i32 {
    if v.is_nan() {
        0
    } else if v >= i32::MAX as f64 {
        i32::MAX
    } else if v <= i32::MIN as f64 {
        i32::MIN
    } else {
        v as i32
    }
}

/// Saturating truncation of f64 to unsigned i32
fn trunc_sat_f64_to_u32(v: f64) -> u32 {
    if v.is_nan() {
        0
    } else if v >= u32::MAX as f64 {
        u32::MAX
    } else if v <= 0.0 {
        0
    } else {
        v as u32
    }
}

/// WebAssembly `nearest` rounding: round to nearest, ties to even.
fn wasm_nearest_f32(v: f32) -> f32 {
    v.round_ties_even()
}

/// WebAssembly `nearest` rounding: round to nearest, ties to even.
fn wasm_nearest_f64(v: f64) -> f64 {
    v.round_ties_even()
}

/// Ensure an f32 NaN result has the quiet bit set (signaling → quiet conversion).
/// Per the WebAssembly spec, floating-point operations never produce signaling NaNs.
#[inline]
fn canonicalise_f32_nan(v: f32) -> f32 {
    if v.is_nan() {
        let bits = v.to_bits();
        f32::from_bits(bits | 0x00400000)
    } else {
        v
    }
}

/// Ensure an f64 NaN result has the quiet bit set (signaling -> quiet conversion).
/// Per the WebAssembly spec, floating-point operations never produce signaling NaNs.
#[inline]
fn canonicalise_f64_nan(v: f64) -> f64 {
    if v.is_nan() {
        let bits = v.to_bits();
        f64::from_bits(bits | 0x0008000000000000)
    } else {
        v
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)] // 3.14 etc. are arbitrary test floats, not stand-ins for consts
mod tests {
    use super::*;
    use crate::parser::instruction::MemArg;
    use crate::runtime::memory::Memory;

    fn make_stack(values: Vec<Value>) -> Stack {
        let mut stack = Stack::new();
        for v in values {
            stack.push(v);
        }
        stack
    }

    fn v128_from_f32s(a: f32, b: f32, c: f32, d: f32) -> [u8; 16] {
        set_f32_lanes([a, b, c, d])
    }

    fn v128_from_f64s(a: f64, b: f64) -> [u8; 16] {
        set_f64_lanes([a, b])
    }

    // ---- v128 load/store ----

    #[test]
    fn test_v128_load() {
        let mut mem = Memory::new(1, None).unwrap();
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        mem.write_bytes(100, &data).unwrap();

        let mut stack = make_stack(vec![Value::I32(100)]);
        let memarg = MemArg { align: 0, offset: 0 };
        v128_load(&mut stack, &mem, &memarg).unwrap();
        assert_eq!(stack.pop_v128().unwrap(), data);
    }

    #[test]
    fn test_v128_load_with_offset() {
        let mut mem = Memory::new(1, None).unwrap();
        let data = [0xAA; 16];
        mem.write_bytes(200, &data).unwrap();

        let mut stack = make_stack(vec![Value::I32(100)]);
        let memarg = MemArg { align: 0, offset: 100 };
        v128_load(&mut stack, &mem, &memarg).unwrap();
        assert_eq!(stack.pop_v128().unwrap(), data);
    }

    #[test]
    fn test_v128_store() {
        let mut mem = Memory::new(1, None).unwrap();
        let data = [10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];

        let mut stack = make_stack(vec![Value::I32(50), Value::V128(data)]);
        let memarg = MemArg { align: 0, offset: 0 };
        v128_store(&mut stack, &mut mem, &memarg).unwrap();
        assert_eq!(mem.read_bytes(50, 16).unwrap(), data);
    }

    #[test]
    fn test_v128_load_oob() {
        let mem = Memory::new(1, None).unwrap();
        let mut stack = make_stack(vec![Value::I32(65530)]);
        let memarg = MemArg { align: 0, offset: 0 };
        assert!(v128_load(&mut stack, &mem, &memarg).is_err());
    }

    // ---- f32x4 arithmetic ----

    #[test]
    fn test_f32x4_add() {
        let a = v128_from_f32s(1.0, 2.0, 3.0, 4.0);
        let b = v128_from_f32s(10.0, 20.0, 30.0, 40.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f32x4_add(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        assert_eq!(result, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_f32x4_sub() {
        let a = v128_from_f32s(10.0, 20.0, 30.0, 40.0);
        let b = v128_from_f32s(1.0, 2.0, 3.0, 4.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f32x4_sub(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        assert_eq!(result, [9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn test_f32x4_mul() {
        let a = v128_from_f32s(2.0, 3.0, 4.0, 5.0);
        let b = v128_from_f32s(10.0, 10.0, 10.0, 10.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f32x4_mul(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        assert_eq!(result, [20.0, 30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_f32x4_div() {
        let a = v128_from_f32s(10.0, 20.0, 30.0, 40.0);
        let b = v128_from_f32s(2.0, 4.0, 5.0, 8.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f32x4_div(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        assert_eq!(result, [5.0, 5.0, 6.0, 5.0]);
    }

    // ---- f32x4 unary ----

    #[test]
    fn test_f32x4_abs_preserves_nan_payload() {
        let snan = f32::from_bits(0xFFA00000); // negative signaling NaN
        let a = v128_from_f32s(snan, -3.0, 0.0, -0.0);
        let mut stack = make_stack(vec![Value::V128(a)]);
        f32x4_abs(&mut stack).unwrap();
        let result = stack.pop_v128().unwrap();
        let lanes = get_f32_lanes(result);
        // abs of sNaN should clear sign bit, keep payload (not canonicalise)
        assert_eq!(lanes[0].to_bits(), 0x7FA00000);
        assert_eq!(lanes[1], 3.0);
        assert_eq!(lanes[2], 0.0);
        assert!(lanes[3].is_sign_positive() && lanes[3] == 0.0);
    }

    #[test]
    fn test_f32x4_neg_preserves_nan_payload() {
        let qnan = f32::from_bits(0x7FC00000); // positive canonical NaN
        let a = v128_from_f32s(qnan, 5.0, -5.0, 0.0);
        let mut stack = make_stack(vec![Value::V128(a)]);
        f32x4_neg(&mut stack).unwrap();
        let result = stack.pop_v128().unwrap();
        let lanes = get_f32_lanes(result);
        assert_eq!(lanes[0].to_bits(), 0xFFC00000); // sign flipped, payload preserved
        assert_eq!(lanes[1], -5.0);
        assert_eq!(lanes[2], 5.0);
        assert!(lanes[3].is_sign_negative()); // -0.0
    }

    #[test]
    fn test_f32x4_sqrt() {
        let a = v128_from_f32s(4.0, 9.0, 16.0, 25.0);
        let mut stack = make_stack(vec![Value::V128(a)]);
        f32x4_sqrt(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        assert_eq!(result, [2.0, 3.0, 4.0, 5.0]);
    }

    // ---- f32x4 NaN canonicalisation ----

    #[test]
    fn test_f32x4_add_canonicalises_nan() {
        // Adding a signaling NaN should produce a quiet NaN
        let snan = f32::from_bits(0x7F800001); // signaling NaN
        let a = v128_from_f32s(snan, 0.0, 0.0, 0.0);
        let b = v128_from_f32s(1.0, 0.0, 0.0, 0.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f32x4_add(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        assert!(result[0].is_nan());
        // Quiet bit must be set
        assert_ne!(result[0].to_bits() & 0x00400000, 0);
    }

    // ---- f32x4 comparison ----

    #[test]
    fn test_f32x4_eq() {
        let a = v128_from_f32s(1.0, 2.0, 3.0, f32::NAN);
        let b = v128_from_f32s(1.0, 3.0, 3.0, f32::NAN);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f32x4_eq(&mut stack).unwrap();
        let result = stack.pop_v128().unwrap();
        let lanes: [u32; 4] = [
            u32::from_le_bytes(result[0..4].try_into().unwrap()),
            u32::from_le_bytes(result[4..8].try_into().unwrap()),
            u32::from_le_bytes(result[8..12].try_into().unwrap()),
            u32::from_le_bytes(result[12..16].try_into().unwrap()),
        ];
        assert_eq!(lanes[0], 0xFFFFFFFF); // 1.0 == 1.0
        assert_eq!(lanes[1], 0x00000000); // 2.0 != 3.0
        assert_eq!(lanes[2], 0xFFFFFFFF); // 3.0 == 3.0
        assert_eq!(lanes[3], 0x00000000); // NaN != NaN
    }

    #[test]
    fn test_f32x4_lt() {
        let a = v128_from_f32s(1.0, 5.0, 3.0, f32::NAN);
        let b = v128_from_f32s(2.0, 3.0, 3.0, 1.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f32x4_lt(&mut stack).unwrap();
        let result = stack.pop_v128().unwrap();
        let lanes: [u32; 4] = [
            u32::from_le_bytes(result[0..4].try_into().unwrap()),
            u32::from_le_bytes(result[4..8].try_into().unwrap()),
            u32::from_le_bytes(result[8..12].try_into().unwrap()),
            u32::from_le_bytes(result[12..16].try_into().unwrap()),
        ];
        assert_eq!(lanes[0], 0xFFFFFFFF); // 1.0 < 2.0
        assert_eq!(lanes[1], 0x00000000); // 5.0 not < 3.0
        assert_eq!(lanes[2], 0x00000000); // 3.0 not < 3.0
        assert_eq!(lanes[3], 0x00000000); // NaN not < anything
    }

    // ---- f32x4 IEEE min/max ----

    #[test]
    fn test_f32x4_min_zero_signs() {
        let a = v128_from_f32s(-0.0, 0.0, -0.0, 0.0);
        let b = v128_from_f32s(0.0, -0.0, -0.0, 0.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f32x4_min(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        assert!(result[0].is_sign_negative()); // min(-0, +0) = -0
        assert!(result[1].is_sign_negative()); // min(+0, -0) = -0
    }

    #[test]
    fn test_f32x4_min_nan_propagation() {
        let a = v128_from_f32s(f32::NAN, 1.0, f32::NAN, 5.0);
        let b = v128_from_f32s(1.0, f32::NAN, f32::NAN, 3.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f32x4_min(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        assert!(result[0].is_nan()); // NaN propagated
        assert!(result[1].is_nan()); // NaN propagated
        assert!(result[2].is_nan()); // NaN propagated
        assert_eq!(result[3], 3.0); // min(5, 3) = 3
    }

    // ---- f32x4 pmin/pmax (bitwise, no NaN canonicalisation) ----

    #[test]
    fn test_f32x4_pmin_passes_through_nan() {
        let snan = f32::from_bits(0x7F800001); // signaling NaN
        let a = v128_from_f32s(snan, 1.0, 0.0, 0.0);
        let b = v128_from_f32s(1.0, snan, 0.0, 0.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f32x4_pmin(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        // pmin(sNaN, 1.0): b < a is false (NaN comparison), so result = a = sNaN
        assert_eq!(result[0].to_bits(), 0x7F800001); // sNaN preserved, NOT canonicalised
        // pmin(1.0, sNaN): b < a is false, so result = a = 1.0
        assert_eq!(result[1], 1.0);
    }

    #[test]
    fn test_f32x4_pmax_basic() {
        let a = v128_from_f32s(1.0, 5.0, -1.0, 0.0);
        let b = v128_from_f32s(2.0, 3.0, -2.0, 0.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f32x4_pmax(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        assert_eq!(result[0], 2.0); // a < b, so b
        assert_eq!(result[1], 5.0); // a not < b, so a
        assert_eq!(result[2], -1.0); // a not < b, so a
    }

    // ---- f32x4 rounding ----

    #[test]
    fn test_f32x4_ceil() {
        let a = v128_from_f32s(1.1, -1.1, 2.0, -0.0);
        let mut stack = make_stack(vec![Value::V128(a)]);
        f32x4_ceil(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], -1.0);
        assert_eq!(result[2], 2.0);
        assert!(result[3] == 0.0 && result[3].is_sign_negative());
    }

    #[test]
    fn test_f32x4_nearest_ties_to_even() {
        let a = v128_from_f32s(2.5, 3.5, -2.5, -3.5);
        let mut stack = make_stack(vec![Value::V128(a)]);
        f32x4_nearest(&mut stack).unwrap();
        let result = get_f32_lanes(stack.pop_v128().unwrap());
        assert_eq!(result[0], 2.0); // ties to even
        assert_eq!(result[1], 4.0); // ties to even
        assert_eq!(result[2], -2.0);
        assert_eq!(result[3], -4.0);
    }

    // ---- f64x2 arithmetic ----

    #[test]
    fn test_f64x2_add() {
        let a = v128_from_f64s(1.5, 2.5);
        let b = v128_from_f64s(10.0, 20.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f64x2_add(&mut stack).unwrap();
        let result = get_f64_lanes(stack.pop_v128().unwrap());
        assert_eq!(result, [11.5, 22.5]);
    }

    #[test]
    fn test_f64x2_neg_preserves_nan_payload() {
        let snan = f64::from_bits(0x7FF0000000000001);
        let a = v128_from_f64s(snan, -5.0);
        let mut stack = make_stack(vec![Value::V128(a)]);
        f64x2_neg(&mut stack).unwrap();
        let result = get_f64_lanes(stack.pop_v128().unwrap());
        // Sign bit flipped, payload preserved
        assert_eq!(result[0].to_bits(), 0xFFF0000000000001);
        assert_eq!(result[1], 5.0);
    }

    #[test]
    fn test_f64x2_eq() {
        let a = v128_from_f64s(1.0, f64::NAN);
        let b = v128_from_f64s(1.0, f64::NAN);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f64x2_eq(&mut stack).unwrap();
        let result = stack.pop_v128().unwrap();
        let lane0 = u64::from_le_bytes(result[0..8].try_into().unwrap());
        let lane1 = u64::from_le_bytes(result[8..16].try_into().unwrap());
        assert_eq!(lane0, 0xFFFFFFFFFFFFFFFF); // 1.0 == 1.0
        assert_eq!(lane1, 0x0000000000000000); // NaN != NaN
    }

    #[test]
    fn test_f64x2_pmin_passes_through_nan() {
        let snan = f64::from_bits(0x7FF4000000000000); // signaling NaN
        let a = v128_from_f64s(snan, 1.0);
        let b = v128_from_f64s(1.0, 2.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f64x2_pmin(&mut stack).unwrap();
        let result = get_f64_lanes(stack.pop_v128().unwrap());
        // pmin(sNaN, 1.0): b < a is false, so result = a = sNaN preserved
        assert_eq!(result[0].to_bits(), 0x7FF4000000000000);
        assert_eq!(result[1], 1.0); // pmin(1.0, 2.0) = 1.0
    }

    #[test]
    fn test_f64x2_min_nan_propagation() {
        let a = v128_from_f64s(f64::NAN, 5.0);
        let b = v128_from_f64s(1.0, 3.0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        f64x2_min(&mut stack).unwrap();
        let result = get_f64_lanes(stack.pop_v128().unwrap());
        assert!(result[0].is_nan());
        assert_eq!(result[1], 3.0);
    }

    // ---- lane helper round-trip ----

    #[test]
    fn test_f32_lane_roundtrip() {
        let lanes = [1.0f32, -2.5, f32::INFINITY, f32::NEG_INFINITY];
        let bytes = set_f32_lanes(lanes);
        let back = get_f32_lanes(bytes);
        assert_eq!(back, lanes);
    }

    #[test]
    fn test_f64_lane_roundtrip() {
        let lanes = [std::f64::consts::PI, -std::f64::consts::E];
        let bytes = set_f64_lanes(lanes);
        let back = get_f64_lanes(bytes);
        assert_eq!(back, lanes);
    }

    // ---- integer lane helpers ----

    fn v128_from_i32s(a: i32, b: i32, c: i32, d: i32) -> [u8; 16] {
        set_i32x4_lanes([a, b, c, d])
    }

    fn v128_from_u32s(a: u32, b: u32, c: u32, d: u32) -> [u8; 16] {
        set_u32x4_lanes([a, b, c, d])
    }

    fn v128_from_i64s(a: i64, b: i64) -> [u8; 16] {
        set_i64x2_lanes([a, b])
    }

    fn v128_from_i16s(vals: [i16; 8]) -> [u8; 16] {
        set_i16x8_lanes(vals)
    }

    fn v128_from_i8s(vals: [i8; 16]) -> [u8; 16] {
        set_i8x16_lanes(vals)
    }

    fn v128_from_u8s(vals: [u8; 16]) -> [u8; 16] {
        set_u8x16_lanes(vals)
    }

    // ==== Phase 4-15 tests ====

    // ---- 1. i32x4 arithmetic ----

    #[test]
    fn test_i32x4_add() {
        let a = v128_from_i32s(1, 2, 3, 4);
        let b = v128_from_i32s(10, 20, 30, 40);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_add(&mut stack).unwrap();
        assert_eq!(get_i32x4_lanes(stack.pop_v128().unwrap()), [11, 22, 33, 44]);
    }

    #[test]
    fn test_i32x4_add_wrapping() {
        let a = v128_from_i32s(i32::MAX, i32::MIN, 0, -1);
        let b = v128_from_i32s(1, -1, 0, 1);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_add(&mut stack).unwrap();
        let r = get_i32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], i32::MIN); // wraps
        assert_eq!(r[1], i32::MAX); // wraps
        assert_eq!(r[2], 0);
        assert_eq!(r[3], 0);
    }

    #[test]
    fn test_i32x4_sub() {
        let a = v128_from_i32s(10, 20, 30, 40);
        let b = v128_from_i32s(1, 2, 3, 4);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_sub(&mut stack).unwrap();
        assert_eq!(get_i32x4_lanes(stack.pop_v128().unwrap()), [9, 18, 27, 36]);
    }

    #[test]
    fn test_i32x4_mul() {
        let a = v128_from_i32s(2, -3, 4, 0);
        let b = v128_from_i32s(5, 7, -8, 100);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_mul(&mut stack).unwrap();
        assert_eq!(get_i32x4_lanes(stack.pop_v128().unwrap()), [10, -21, -32, 0]);
    }

    #[test]
    fn test_i32x4_neg() {
        let a = v128_from_i32s(1, -1, 0, i32::MIN);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i32x4_neg(&mut stack).unwrap();
        let r = get_i32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], -1);
        assert_eq!(r[1], 1);
        assert_eq!(r[2], 0);
        assert_eq!(r[3], i32::MIN); // wrapping neg of MIN is MIN
    }

    #[test]
    fn test_i32x4_abs() {
        let a = v128_from_i32s(-5, 5, 0, i32::MIN);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i32x4_abs(&mut stack).unwrap();
        let r = get_i32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 5);
        assert_eq!(r[1], 5);
        assert_eq!(r[2], 0);
        assert_eq!(r[3], i32::MIN); // wrapping abs of MIN is MIN
    }

    // ---- 2. i64x2 arithmetic ----

    #[test]
    fn test_i64x2_add() {
        let a = v128_from_i64s(100, -200);
        let b = v128_from_i64s(50, 300);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i64x2_add(&mut stack).unwrap();
        assert_eq!(get_i64x2_lanes(stack.pop_v128().unwrap()), [150, 100]);
    }

    #[test]
    fn test_i64x2_sub() {
        let a = v128_from_i64s(100, 200);
        let b = v128_from_i64s(50, 300);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i64x2_sub(&mut stack).unwrap();
        assert_eq!(get_i64x2_lanes(stack.pop_v128().unwrap()), [50, -100]);
    }

    #[test]
    fn test_i64x2_mul() {
        let a = v128_from_i64s(7, -3);
        let b = v128_from_i64s(6, 4);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i64x2_mul(&mut stack).unwrap();
        assert_eq!(get_i64x2_lanes(stack.pop_v128().unwrap()), [42, -12]);
    }

    #[test]
    fn test_i64x2_neg() {
        let a = v128_from_i64s(1, i64::MIN);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i64x2_neg(&mut stack).unwrap();
        let r = get_i64x2_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], -1);
        assert_eq!(r[1], i64::MIN); // wrapping
    }

    #[test]
    fn test_i64x2_abs() {
        let a = v128_from_i64s(-42, 42);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i64x2_abs(&mut stack).unwrap();
        assert_eq!(get_i64x2_lanes(stack.pop_v128().unwrap()), [42, 42]);
    }

    // ---- 3. i8x16 arithmetic ----

    #[test]
    fn test_i8x16_add() {
        let a = v128_from_i8s([1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8]);
        let b = v128_from_i8s([10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i8x16_add(&mut stack).unwrap();
        let r = get_i8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 11);
        assert_eq!(r[8], 0);
    }

    #[test]
    fn test_i8x16_add_wrapping_overflow() {
        let a = v128_from_i8s([127, -128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let b = v128_from_i8s([1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i8x16_add(&mut stack).unwrap();
        let r = get_i8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], -128); // 127 + 1 wraps to -128
        assert_eq!(r[1], 127); // -128 + (-1) wraps to 127
    }

    #[test]
    fn test_i8x16_sub() {
        let a = v128_from_i8s([10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let b = v128_from_i8s([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i8x16_sub(&mut stack).unwrap();
        assert_eq!(get_i8x16_lanes(stack.pop_v128().unwrap())[0], 7);
    }

    #[test]
    fn test_i8x16_neg() {
        let a = v128_from_i8s([1, -1, 0, 127, -128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i8x16_neg(&mut stack).unwrap();
        let r = get_i8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], -1);
        assert_eq!(r[1], 1);
        assert_eq!(r[2], 0);
        assert_eq!(r[3], -127);
        assert_eq!(r[4], -128); // wrapping neg of -128
    }

    #[test]
    fn test_i8x16_abs() {
        let a = v128_from_i8s([-5, 5, 0, -128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i8x16_abs(&mut stack).unwrap();
        let r = get_i8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 5);
        assert_eq!(r[1], 5);
        assert_eq!(r[2], 0);
        assert_eq!(r[3], -128); // wrapping abs of -128
    }

    // ---- 4. i16x8 arithmetic ----

    #[test]
    fn test_i16x8_add() {
        let a = v128_from_i16s([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = v128_from_i16s([10, 20, 30, 40, 50, 60, 70, 80]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i16x8_add(&mut stack).unwrap();
        assert_eq!(
            get_i16x8_lanes(stack.pop_v128().unwrap()),
            [11, 22, 33, 44, 55, 66, 77, 88]
        );
    }

    #[test]
    fn test_i16x8_sub() {
        let a = v128_from_i16s([100, 200, 0, 0, 0, 0, 0, 0]);
        let b = v128_from_i16s([50, 300, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i16x8_sub(&mut stack).unwrap();
        let r = get_i16x8_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 50);
        assert_eq!(r[1], -100);
    }

    #[test]
    fn test_i16x8_mul() {
        let a = v128_from_i16s([3, -4, 0, 0, 0, 0, 0, 0]);
        let b = v128_from_i16s([7, 5, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i16x8_mul(&mut stack).unwrap();
        let r = get_i16x8_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 21);
        assert_eq!(r[1], -20);
    }

    #[test]
    fn test_i16x8_neg() {
        let a = v128_from_i16s([1, -1, i16::MIN, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i16x8_neg(&mut stack).unwrap();
        let r = get_i16x8_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], -1);
        assert_eq!(r[1], 1);
        assert_eq!(r[2], i16::MIN); // wrapping
    }

    #[test]
    fn test_i16x8_abs() {
        let a = v128_from_i16s([-10, 10, 0, i16::MIN, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i16x8_abs(&mut stack).unwrap();
        let r = get_i16x8_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 10);
        assert_eq!(r[1], 10);
        assert_eq!(r[2], 0);
        assert_eq!(r[3], i16::MIN); // wrapping abs of MIN
    }

    // ---- 5. Saturating arithmetic ----

    #[test]
    fn test_i8x16_add_sat_s_clamp() {
        let a = v128_from_i8s([127, -128, 100, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let b = v128_from_i8s([1, -1, 100, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i8x16_add_sat_s(&mut stack).unwrap();
        let r = get_i8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 127); // saturates at max
        assert_eq!(r[1], -128); // saturates at min
        assert_eq!(r[2], 127); // 200 clamped
        assert_eq!(r[3], -128); // -200 clamped
    }

    #[test]
    fn test_i8x16_add_sat_u_clamp() {
        let a = v128_from_u8s([255, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let b = v128_from_u8s([1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i8x16_add_sat_u(&mut stack).unwrap();
        let r = get_u8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 255); // saturates at 255
        assert_eq!(r[1], 255); // 300 clamped to 255
    }

    #[test]
    fn test_i16x8_sub_sat_s() {
        let a = v128_from_i16s([-32000, 32000, 0, 0, 0, 0, 0, 0]);
        let b = v128_from_i16s([1000, -1000, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i16x8_sub_sat_s(&mut stack).unwrap();
        let r = get_i16x8_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], -32768); // saturates
        assert_eq!(r[1], 32767); // saturates
    }

    // ---- 6. Splat ----

    #[test]
    fn test_i32x4_splat() {
        let mut stack = make_stack(vec![Value::I32(42)]);
        i32x4_splat(&mut stack).unwrap();
        assert_eq!(get_i32x4_lanes(stack.pop_v128().unwrap()), [42, 42, 42, 42]);
    }

    #[test]
    fn test_i8x16_splat() {
        let mut stack = make_stack(vec![Value::I32(0xFF)]);
        i8x16_splat(&mut stack).unwrap();
        let r = get_i8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r, [-1i8; 16]); // 0xFF truncated to i8 = -1
    }

    #[test]
    fn test_f32x4_splat() {
        let mut stack = make_stack(vec![Value::F32(3.14)]);
        f32x4_splat(&mut stack).unwrap();
        let r = get_f32_lanes(stack.pop_v128().unwrap());
        assert_eq!(r, [3.14, 3.14, 3.14, 3.14]);
    }

    #[test]
    fn test_i64x2_splat() {
        let mut stack = make_stack(vec![Value::I64(999)]);
        i64x2_splat(&mut stack).unwrap();
        assert_eq!(get_i64x2_lanes(stack.pop_v128().unwrap()), [999, 999]);
    }

    // ---- 7. Extract lane ----

    #[test]
    fn test_i32x4_extract_lane() {
        let a = v128_from_i32s(10, 20, 30, 40);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i32x4_extract_lane(&mut stack, 2).unwrap();
        assert_eq!(stack.pop_i32().unwrap(), 30);
    }

    #[test]
    fn test_i8x16_extract_lane_s_sign_extension() {
        let a = v128_from_u8s([0xFF, 0x7F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i8x16_extract_lane_s(&mut stack, 0).unwrap();
        assert_eq!(stack.pop_i32().unwrap(), -1); // 0xFF sign-extended

        let mut stack = make_stack(vec![Value::V128(a)]);
        i8x16_extract_lane_s(&mut stack, 1).unwrap();
        assert_eq!(stack.pop_i32().unwrap(), 127); // 0x7F stays positive
    }

    #[test]
    fn test_i8x16_extract_lane_u() {
        let a = v128_from_u8s([0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i8x16_extract_lane_u(&mut stack, 0).unwrap();
        assert_eq!(stack.pop_i32().unwrap(), 255); // zero-extended
    }

    // ---- 8. Replace lane ----

    #[test]
    fn test_i32x4_replace_lane() {
        let a = v128_from_i32s(1, 2, 3, 4);
        let mut stack = make_stack(vec![Value::V128(a), Value::I32(99)]);
        i32x4_replace_lane(&mut stack, 2).unwrap();
        assert_eq!(get_i32x4_lanes(stack.pop_v128().unwrap()), [1, 2, 99, 4]);
    }

    // ---- 9. Integer comparisons ----

    #[test]
    fn test_i32x4_eq() {
        let a = v128_from_i32s(1, 2, 3, 4);
        let b = v128_from_i32s(1, 99, 3, 0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_eq(&mut stack).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0xFFFFFFFF); // equal
        assert_eq!(r[1], 0x00000000); // not equal
        assert_eq!(r[2], 0xFFFFFFFF); // equal
        assert_eq!(r[3], 0x00000000); // not equal
    }

    #[test]
    fn test_i32x4_lt_s() {
        let a = v128_from_i32s(-1, 5, i32::MIN, 0);
        let b = v128_from_i32s(0, 5, i32::MAX, 0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_lt_s(&mut stack).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0xFFFFFFFF); // -1 < 0 (signed)
        assert_eq!(r[1], 0x00000000); // 5 not < 5
        assert_eq!(r[2], 0xFFFFFFFF); // MIN < MAX
        assert_eq!(r[3], 0x00000000); // 0 not < 0
    }

    #[test]
    fn test_i32x4_lt_u_signed_vs_unsigned() {
        // As unsigned, -1 (0xFFFFFFFF) is MAX, so it's > 0
        let a = v128_from_i32s(-1, 0, 1, -2);
        let b = v128_from_i32s(0, 1, 0, -1);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_lt_u(&mut stack).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0x00000000); // 0xFFFFFFFF not < 0 (unsigned)
        assert_eq!(r[1], 0xFFFFFFFF); // 0 < 1
        assert_eq!(r[2], 0x00000000); // 1 not < 0
        assert_eq!(r[3], 0xFFFFFFFF); // 0xFFFFFFFE < 0xFFFFFFFF
    }

    #[test]
    fn test_i8x16_eq() {
        let a = v128_from_i8s([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let b = v128_from_i8s([1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i8x16_eq(&mut stack).unwrap();
        let r = get_i8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], -1); // equal -> 0xFF
        assert_eq!(r[1], 0); // not equal
        assert_eq!(r[2], -1); // equal
        assert_eq!(r[3], 0); // not equal
    }

    // ---- 10. Bitwise ----

    #[test]
    fn test_v128_and() {
        let a = v128_from_u32s(0xFF00FF00, 0x0F0F0F0F, 0, 0xFFFFFFFF);
        let b = v128_from_u32s(0xFFFF0000, 0xF0F0F0F0, 0xFFFFFFFF, 0x12345678);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        v128_and(&mut stack).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0xFF000000);
        assert_eq!(r[1], 0x00000000);
        assert_eq!(r[2], 0x00000000);
        assert_eq!(r[3], 0x12345678);
    }

    #[test]
    fn test_v128_or() {
        let a = v128_from_u32s(0xFF000000, 0, 0, 0);
        let b = v128_from_u32s(0x00FF0000, 0, 0, 0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        v128_or(&mut stack).unwrap();
        assert_eq!(get_u32x4_lanes(stack.pop_v128().unwrap())[0], 0xFFFF0000);
    }

    #[test]
    fn test_v128_xor() {
        let a = v128_from_u32s(0xFFFF0000, 0, 0, 0);
        let b = v128_from_u32s(0xFF00FF00, 0, 0, 0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        v128_xor(&mut stack).unwrap();
        assert_eq!(get_u32x4_lanes(stack.pop_v128().unwrap())[0], 0x00FFFF00);
    }

    #[test]
    fn test_v128_not() {
        let a = v128_from_u32s(0x00000000, 0xFFFFFFFF, 0x0F0F0F0F, 0);
        let mut stack = make_stack(vec![Value::V128(a)]);
        v128_not(&mut stack).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0xFFFFFFFF);
        assert_eq!(r[1], 0x00000000);
        assert_eq!(r[2], 0xF0F0F0F0);
    }

    #[test]
    fn test_v128_andnot() {
        // andnot(a, b) = a & ~b
        let a = v128_from_u32s(0xFF00FF00, 0, 0, 0);
        let b = v128_from_u32s(0xFFFF0000, 0, 0, 0);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        v128_andnot(&mut stack).unwrap();
        assert_eq!(get_u32x4_lanes(stack.pop_v128().unwrap())[0], 0x0000FF00);
    }

    #[test]
    fn test_v128_bitselect() {
        // bitselect(v1, v2, mask): mask=1 picks v1, mask=0 picks v2
        let v1 = v128_from_u32s(0xAAAAAAAA, 0, 0, 0);
        let v2 = v128_from_u32s(0x55555555, 0, 0, 0);
        let mask = v128_from_u32s(0xFF00FF00, 0, 0, 0);
        let mut stack = make_stack(vec![Value::V128(v1), Value::V128(v2), Value::V128(mask)]);
        v128_bitselect(&mut stack).unwrap();
        assert_eq!(get_u32x4_lanes(stack.pop_v128().unwrap())[0], 0xAA55AA55);
    }

    #[test]
    fn test_v128_any_true() {
        let zero = [0u8; 16];
        let mut stack = make_stack(vec![Value::V128(zero)]);
        v128_any_true(&mut stack).unwrap();
        assert_eq!(stack.pop_i32().unwrap(), 0);

        let mut nonzero = [0u8; 16];
        nonzero[7] = 1;
        let mut stack = make_stack(vec![Value::V128(nonzero)]);
        v128_any_true(&mut stack).unwrap();
        assert_eq!(stack.pop_i32().unwrap(), 1);
    }

    // ---- 11. All_true and bitmask ----

    #[test]
    fn test_i8x16_all_true() {
        let a = v128_from_i8s([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i8x16_all_true(&mut stack).unwrap();
        assert_eq!(stack.pop_i32().unwrap(), 1);

        let b = v128_from_i8s([1, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mut stack = make_stack(vec![Value::V128(b)]);
        i8x16_all_true(&mut stack).unwrap();
        assert_eq!(stack.pop_i32().unwrap(), 0);
    }

    #[test]
    fn test_i32x4_all_true() {
        let a = v128_from_i32s(1, 2, 3, 4);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i32x4_all_true(&mut stack).unwrap();
        assert_eq!(stack.pop_i32().unwrap(), 1);

        let b = v128_from_i32s(1, 0, 3, 4);
        let mut stack = make_stack(vec![Value::V128(b)]);
        i32x4_all_true(&mut stack).unwrap();
        assert_eq!(stack.pop_i32().unwrap(), 0);
    }

    #[test]
    fn test_i8x16_bitmask() {
        // MSB of each byte lane forms the bitmask
        let mut bytes = [0u8; 16];
        bytes[0] = 0x80; // bit 0 set
        bytes[3] = 0xFF; // bit 3 set
        bytes[15] = 0x80; // bit 15 set
        let mut stack = make_stack(vec![Value::V128(bytes)]);
        i8x16_bitmask(&mut stack).unwrap();
        let r = stack.pop_i32().unwrap();
        assert_eq!(r, (1 << 0) | (1 << 3) | (1 << 15));
    }

    #[test]
    fn test_i32x4_bitmask() {
        let a = v128_from_i32s(-1, 0, -1, 0);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i32x4_bitmask(&mut stack).unwrap();
        let r = stack.pop_i32().unwrap();
        assert_eq!(r, 0b0101); // lanes 0 and 2 have MSB set
    }

    // ---- 12. Bit shifts ----

    #[test]
    fn test_i32x4_shl() {
        let a = v128_from_i32s(1, 2, 3, 0x80000000u32 as i32);
        let mut stack = make_stack(vec![Value::V128(a), Value::I32(4)]);
        i32x4_shl(&mut stack).unwrap();
        assert_eq!(get_i32x4_lanes(stack.pop_v128().unwrap()), [16, 32, 48, 0]);
    }

    #[test]
    fn test_i32x4_shr_s() {
        let a = v128_from_i32s(-16, 16, -1, 0);
        let mut stack = make_stack(vec![Value::V128(a), Value::I32(2)]);
        i32x4_shr_s(&mut stack).unwrap();
        let r = get_i32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], -4); // arithmetic shift preserves sign
        assert_eq!(r[1], 4);
        assert_eq!(r[2], -1); // -1 >> n stays -1
    }

    #[test]
    fn test_i32x4_shr_u() {
        let a = v128_from_i32s(-1, 0, 256, 0);
        let mut stack = make_stack(vec![Value::V128(a), Value::I32(8)]);
        i32x4_shr_u(&mut stack).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0x00FFFFFF); // logical shift, zero-fill
        assert_eq!(r[2], 1);
    }

    #[test]
    fn test_i8x16_shl_mask() {
        // Shift amount masked to 3 bits (mod 8)
        let a = v128_from_u8s([1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::I32(9)]); // 9 & 7 = 1
        i8x16_shl(&mut stack).unwrap();
        let r = get_u8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 2); // 1 << 1
        assert_eq!(r[1], 4); // 2 << 1
    }

    // ---- 13. Min/max/avgr/popcnt ----

    #[test]
    fn test_i32x4_min_s() {
        let a = v128_from_i32s(-1, 5, i32::MIN, 0);
        let b = v128_from_i32s(0, 3, 0, i32::MAX);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_min_s(&mut stack).unwrap();
        assert_eq!(get_i32x4_lanes(stack.pop_v128().unwrap()), [-1, 3, i32::MIN, 0]);
    }

    #[test]
    fn test_i32x4_max_u() {
        let a = v128_from_u32s(0, 5, 0xFFFFFFFF, 100);
        let b = v128_from_u32s(1, 3, 0, 200);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_max_u(&mut stack).unwrap();
        assert_eq!(get_u32x4_lanes(stack.pop_v128().unwrap()), [1, 5, 0xFFFFFFFF, 200]);
    }

    #[test]
    fn test_i8x16_avgr_u() {
        let a = v128_from_u8s([0, 10, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let b = v128_from_u8s([1, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i8x16_avgr_u(&mut stack).unwrap();
        let r = get_u8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 1); // (0 + 1 + 1) / 2 = 1
        assert_eq!(r[1], 11); // (10 + 11 + 1) / 2 = 11
        assert_eq!(r[2], 128); // (255 + 0 + 1) / 2 = 128
        assert_eq!(r[3], 1); // (1 + 0 + 1) / 2 = 1
    }

    #[test]
    fn test_i8x16_popcnt() {
        let a = v128_from_u8s([0, 1, 3, 7, 0xFF, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i8x16_popcnt(&mut stack).unwrap();
        let r = get_u8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0);
        assert_eq!(r[1], 1);
        assert_eq!(r[2], 2);
        assert_eq!(r[3], 3);
        assert_eq!(r[4], 8);
        assert_eq!(r[5], 1);
    }

    // ---- 14. Shuffle and swizzle ----

    #[test]
    fn test_i8x16_shuffle() {
        let a = v128_from_u8s([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]);
        let b = v128_from_u8s([30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]);
        let lanes: [u8; 16] = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i8x16_shuffle(&mut stack, &lanes).unwrap();
        let r = get_u8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 10); // from a[0]
        assert_eq!(r[1], 30); // from b[0]
        assert_eq!(r[2], 11); // from a[1]
        assert_eq!(r[3], 31); // from b[1]
    }

    #[test]
    fn test_i8x16_swizzle() {
        let a = v128_from_u8s([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]);
        let idx = v128_from_u8s([0, 2, 4, 6, 15, 255, 16, 128, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(idx)]);
        i8x16_swizzle(&mut stack).unwrap();
        let r = get_u8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 10); // a[0]
        assert_eq!(r[1], 30); // a[2]
        assert_eq!(r[2], 50); // a[4]
        assert_eq!(r[4], 160); // a[15]
        assert_eq!(r[5], 0); // index 255 out of range -> 0
        assert_eq!(r[6], 0); // index 16 out of range -> 0
        assert_eq!(r[7], 0); // index 128 out of range -> 0
    }

    // ---- 15. Narrow ----

    #[test]
    fn test_i8x16_narrow_i16x8_s() {
        let a = v128_from_i16s([0, 127, -128, 200, -200, 50, -50, 0]);
        let b = v128_from_i16s([1, -1, 1000, -1000, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i8x16_narrow_i16x8_s(&mut stack).unwrap();
        let r = get_i8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0);
        assert_eq!(r[1], 127);
        assert_eq!(r[2], -128);
        assert_eq!(r[3], 127); // 200 clamped to 127
        assert_eq!(r[4], -128); // -200 clamped to -128
        assert_eq!(r[8], 1); // from b
        assert_eq!(r[9], -1);
        assert_eq!(r[10], 127); // 1000 clamped
        assert_eq!(r[11], -128); // -1000 clamped
    }

    #[test]
    fn test_i8x16_narrow_i16x8_u() {
        let a = v128_from_i16s([0, 255, 256, -1, 128, 0, 0, 0]);
        let b = v128_from_i16s([100, 300, -5, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i8x16_narrow_i16x8_u(&mut stack).unwrap();
        let r = get_u8x16_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0);
        assert_eq!(r[1], 255);
        assert_eq!(r[2], 255); // 256 clamped to 255
        assert_eq!(r[3], 0); // -1 clamped to 0
        assert_eq!(r[4], 128);
        assert_eq!(r[8], 100);
        assert_eq!(r[9], 255); // 300 clamped
        assert_eq!(r[10], 0); // -5 clamped to 0
    }

    // ---- 16. Conversions ----

    #[test]
    fn test_i32x4_trunc_sat_f32x4_s() {
        let a = v128_from_f32s(1.9, -2.9, f32::NAN, f32::INFINITY);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i32x4_trunc_sat_f32x4_s(&mut stack).unwrap();
        let r = get_i32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 1);
        assert_eq!(r[1], -2);
        assert_eq!(r[2], 0); // NaN -> 0
        assert_eq!(r[3], i32::MAX); // +inf -> MAX
    }

    #[test]
    fn test_i32x4_trunc_sat_f32x4_s_neg_inf() {
        let a = v128_from_f32s(f32::NEG_INFINITY, 0.0, 0.0, 0.0);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i32x4_trunc_sat_f32x4_s(&mut stack).unwrap();
        assert_eq!(get_i32x4_lanes(stack.pop_v128().unwrap())[0], i32::MIN);
    }

    #[test]
    fn test_i32x4_trunc_sat_f32x4_u() {
        let a = v128_from_f32s(1.9, -1.0, f32::NAN, 4294967040.0);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i32x4_trunc_sat_f32x4_u(&mut stack).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 1);
        assert_eq!(r[1], 0); // negative -> 0
        assert_eq!(r[2], 0); // NaN -> 0
        assert_eq!(r[3], 4294967040); // large but representable
    }

    #[test]
    fn test_f32x4_convert_i32x4_s() {
        let a = v128_from_i32s(0, 1, -1, i32::MAX);
        let mut stack = make_stack(vec![Value::V128(a)]);
        f32x4_convert_i32x4_s(&mut stack).unwrap();
        let r = get_f32_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0.0);
        assert_eq!(r[1], 1.0);
        assert_eq!(r[2], -1.0);
        assert_eq!(r[3], i32::MAX as f32);
    }

    #[test]
    fn test_f64x2_promote_low_f32x4() {
        let a = v128_from_f32s(1.5, -2.5, 99.0, 100.0);
        let mut stack = make_stack(vec![Value::V128(a)]);
        f64x2_promote_low_f32x4(&mut stack).unwrap();
        let r = get_f64_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 1.5); // exact promotion
        assert_eq!(r[1], -2.5);
    }

    #[test]
    fn test_f32x4_demote_f64x2_zero() {
        let a = v128_from_f64s(1.5, -2.5);
        let mut stack = make_stack(vec![Value::V128(a)]);
        f32x4_demote_f64x2_zero(&mut stack).unwrap();
        let r = get_f32_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 1.5);
        assert_eq!(r[1], -2.5);
        assert_eq!(r[2], 0.0); // zero-filled
        assert_eq!(r[3], 0.0);
    }

    // ---- 17. Integer extend ----

    #[test]
    fn test_i16x8_extend_low_i8x16_s() {
        let a = v128_from_i8s([-1, 127, -128, 0, 1, 2, 3, 4, 99, 99, 99, 99, 99, 99, 99, 99]);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i16x8_extend_low_i8x16_s(&mut stack).unwrap();
        let r = get_i16x8_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], -1); // sign-extended
        assert_eq!(r[1], 127);
        assert_eq!(r[2], -128); // sign-extended
        assert_eq!(r[3], 0);
    }

    #[test]
    fn test_i32x4_extend_high_i16x8_u() {
        let a = v128_from_i16s([0, 0, 0, 0, 1, 0xFFFF_u16 as i16, 0x8000_u16 as i16, 100]);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i32x4_extend_high_i16x8_u(&mut stack).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 1);
        assert_eq!(r[1], 65535); // zero-extended
        assert_eq!(r[2], 32768); // zero-extended
        assert_eq!(r[3], 100);
    }

    // ---- 18. ExtMul ----

    #[test]
    fn test_i16x8_extmul_low_i8x16_s() {
        let a = v128_from_i8s([-1, 2, -128, 127, 0, 0, 0, 0, 99, 99, 99, 99, 99, 99, 99, 99]);
        let b = v128_from_i8s([2, 3, 2, 2, 0, 0, 0, 0, 99, 99, 99, 99, 99, 99, 99, 99]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i16x8_extmul_low_i8x16_s(&mut stack).unwrap();
        let r = get_i16x8_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], -2); // -1 * 2
        assert_eq!(r[1], 6); // 2 * 3
        assert_eq!(r[2], -256); // -128 * 2
        assert_eq!(r[3], 254); // 127 * 2
    }

    #[test]
    fn test_i32x4_extmul_high_i16x8_u() {
        let a = v128_from_i16s([0, 0, 0, 0, 100, 0xFFFF_u16 as i16, 0, 1]);
        let b = v128_from_i16s([0, 0, 0, 0, 200, 2, 0, 1]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_extmul_high_i16x8_u(&mut stack).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 20000); // 100 * 200
        assert_eq!(r[1], 131070); // 65535 * 2
        assert_eq!(r[2], 0);
        assert_eq!(r[3], 1);
    }

    // ---- 19. Pairwise add ----

    #[test]
    fn test_i16x8_extadd_pairwise_i8x16_s() {
        let a = v128_from_i8s([1, 2, -1, -2, 127, 127, -128, -128, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a)]);
        i16x8_extadd_pairwise_i8x16_s(&mut stack).unwrap();
        let r = get_i16x8_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 3); // 1 + 2
        assert_eq!(r[1], -3); // -1 + -2
        assert_eq!(r[2], 254); // 127 + 127
        assert_eq!(r[3], -256); // -128 + -128
    }

    // ---- 20. Dot product ----

    #[test]
    fn test_i32x4_dot_i16x8_s() {
        let a = v128_from_i16s([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = v128_from_i16s([10, 20, 30, 40, 50, 60, 70, 80]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_dot_i16x8_s(&mut stack).unwrap();
        let r = get_i32x4_lanes(stack.pop_v128().unwrap());
        // dot products: [1*10+2*20, 3*30+4*40, 5*50+6*60, 7*70+8*80]
        assert_eq!(r, [50, 250, 610, 1130]);
    }

    #[test]
    fn test_i32x4_dot_i16x8_s_negative() {
        let a = v128_from_i16s([-1, -1, i16::MIN, i16::MAX, 0, 0, 0, 0]);
        let b = v128_from_i16s([1, 1, 1, 1, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i32x4_dot_i16x8_s(&mut stack).unwrap();
        let r = get_i32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], -2); // -1*1 + -1*1
        assert_eq!(r[1], -1); // -32768 + 32767
    }

    // ---- 21. Q15 ----

    #[test]
    fn test_i16x8_q15mulr_sat_s() {
        // Q15 format: 0x7FFF represents ~1.0, 0x4000 represents 0.5
        let a = v128_from_i16s([0x4000, 0x7FFF, -0x4000, 0, 0, 0, 0, 0]);
        let b = v128_from_i16s([0x4000, 0x7FFF, 0x7FFF, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i16x8_q15mulr_sat_s(&mut stack).unwrap();
        let r = get_i16x8_lanes(stack.pop_v128().unwrap());
        // (0x4000 * 0x4000 + 0x4000) >> 15 = (0x10000000 + 0x4000) >> 15 = 0x2001
        assert_eq!(r[0], 0x2000);
        // (0x7FFF * 0x7FFF + 0x4000) >> 15 = near max
        assert_eq!(r[1], 0x7FFE);
    }

    #[test]
    fn test_i16x8_q15mulr_sat_s_saturation() {
        // -32768 * -32768 overflows i16 range after the Q15 shift
        let a = v128_from_i16s([i16::MIN, 0, 0, 0, 0, 0, 0, 0]);
        let b = v128_from_i16s([i16::MIN, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::V128(a), Value::V128(b)]);
        i16x8_q15mulr_sat_s(&mut stack).unwrap();
        let r = get_i16x8_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], i16::MAX); // saturates to 32767
    }

    // ---- 22. Memory load splat ----

    #[test]
    fn test_v128_load8_splat() {
        let mut mem = Memory::new(1, None).unwrap();
        mem.write_u8(50, 0xAB).unwrap();

        let mut stack = make_stack(vec![Value::I32(50)]);
        let memarg = MemArg { align: 0, offset: 0 };
        v128_load8_splat(&mut stack, &mem, &memarg).unwrap();
        let r = stack.pop_v128().unwrap();
        assert_eq!(r, [0xAB; 16]);
    }

    #[test]
    fn test_v128_load32_splat() {
        let mut mem = Memory::new(1, None).unwrap();
        mem.write_u32(100, 0xDEADBEEF).unwrap();

        let mut stack = make_stack(vec![Value::I32(100)]);
        let memarg = MemArg { align: 0, offset: 0 };
        v128_load32_splat(&mut stack, &mem, &memarg).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r, [0xDEADBEEF; 4]);
    }

    // ---- 23. Memory load zero ----

    #[test]
    fn test_v128_load32_zero() {
        let mut mem = Memory::new(1, None).unwrap();
        mem.write_u32(200, 0x12345678).unwrap();

        let mut stack = make_stack(vec![Value::I32(200)]);
        let memarg = MemArg { align: 0, offset: 0 };
        v128_load32_zero(&mut stack, &mem, &memarg).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0x12345678);
        assert_eq!(r[1], 0);
        assert_eq!(r[2], 0);
        assert_eq!(r[3], 0);
    }

    #[test]
    fn test_v128_load64_zero() {
        let mut mem = Memory::new(1, None).unwrap();
        mem.write_u64(300, 0xCAFEBABE_DEADBEEF).unwrap();

        let mut stack = make_stack(vec![Value::I32(300)]);
        let memarg = MemArg { align: 0, offset: 0 };
        v128_load64_zero(&mut stack, &mem, &memarg).unwrap();
        let r = get_u64x2_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0xCAFEBABE_DEADBEEF);
        assert_eq!(r[1], 0);
    }

    // ---- 24. Memory load extend ----

    #[test]
    fn test_v128_load8x8_s() {
        let mut mem = Memory::new(1, None).unwrap();
        mem.write_bytes(0, &[0x01, 0x80, 0xFF, 0x7F, 0, 0, 0, 0]).unwrap();

        let mut stack = make_stack(vec![Value::I32(0)]);
        let memarg = MemArg { align: 0, offset: 0 };
        v128_load8x8_s(&mut stack, &mem, &memarg).unwrap();
        let r = get_i16x8_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 1); // sign-extended
        assert_eq!(r[1], -128); // 0x80 sign-extended
        assert_eq!(r[2], -1); // 0xFF sign-extended
        assert_eq!(r[3], 127); // 0x7F sign-extended
    }

    #[test]
    fn test_v128_load8x8_u() {
        let mut mem = Memory::new(1, None).unwrap();
        mem.write_bytes(0, &[0x01, 0x80, 0xFF, 0x7F, 0, 0, 0, 0]).unwrap();

        let mut stack = make_stack(vec![Value::I32(0)]);
        let memarg = MemArg { align: 0, offset: 0 };
        v128_load8x8_u(&mut stack, &mem, &memarg).unwrap();
        let r = get_u16x8_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 1);
        assert_eq!(r[1], 128); // zero-extended
        assert_eq!(r[2], 255); // zero-extended
        assert_eq!(r[3], 127);
    }

    // ---- 25. Memory load/store lane ----

    #[test]
    fn test_v128_load32_lane() {
        let mut mem = Memory::new(1, None).unwrap();
        mem.write_u32(400, 0x11223344).unwrap();

        let existing = v128_from_u32s(0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD);
        let mut stack = make_stack(vec![Value::I32(400), Value::V128(existing)]);
        let memarg = MemArg { align: 0, offset: 0 };
        v128_load32_lane(&mut stack, &mem, &memarg, 1).unwrap();
        let r = get_u32x4_lanes(stack.pop_v128().unwrap());
        assert_eq!(r[0], 0xAAAAAAAA);
        assert_eq!(r[1], 0x11223344); // replaced from memory
        assert_eq!(r[2], 0xCCCCCCCC);
        assert_eq!(r[3], 0xDDDDDDDD);
    }

    #[test]
    fn test_v128_store8_lane() {
        let mut mem = Memory::new(1, None).unwrap();
        let v = v128_from_u8s([0xAA, 0xBB, 0xCC, 0xDD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut stack = make_stack(vec![Value::I32(500), Value::V128(v)]);
        let memarg = MemArg { align: 0, offset: 0 };
        v128_store8_lane(&mut stack, &mut mem, &memarg, 2).unwrap();
        assert_eq!(mem.read_u8(500).unwrap(), 0xCC);
    }
}
