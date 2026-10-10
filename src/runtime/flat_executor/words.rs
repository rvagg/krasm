//! Values as stack words.
//!
//! The flat executor's stack is untagged: it holds raw 64-bit words, and the
//! bytecode, typed by the compiler, knows what every word means. Numbers are
//! stored as their bits in the low bytes of one word. A `v128` takes two
//! words, low half first. A reference takes one word: 0 for null, or its
//! address or index plus one. Every type's zero is therefore all-zero words,
//! so fresh locals are zeroed in one pass.
//!
//! Values convert to and from [`Value`] only where they cross into tagged
//! storage: arguments and results, globals, tables and host calls.

use crate::parser::module::ValueType;
use crate::runtime::Value;
use crate::runtime::store::FuncAddr;

pub(super) type Word = u64;

/// A Rust type that operations take or return, stored in `WIDTH` words.
pub(super) trait WordValue: Sized {
    const WIDTH: usize;
    /// # Safety
    /// `p` must point to `WIDTH` readable words.
    unsafe fn read(p: *const Word) -> Self;
    /// # Safety
    /// `p` must point to `WIDTH` writable words.
    unsafe fn write(self, p: *mut Word);
}

macro_rules! scalar {
    ($ty:ty, |$v:ident| $to:expr, |$s:ident| $from:expr) => {
        impl WordValue for $ty {
            const WIDTH: usize = 1;

            #[inline(always)]
            unsafe fn read(p: *const Word) -> Self {
                let $s = unsafe { *p };
                $from
            }

            #[inline(always)]
            unsafe fn write(self, p: *mut Word) {
                let $v = self;
                unsafe { *p = $to }
            }
        }
    };
}

scalar!(i32, |v| v as u32 as Word, |s| s as u32 as i32);
scalar!(i64, |v| v as Word, |s| s as i64);
scalar!(f32, |v| v.to_bits() as Word, |s| f32::from_bits(s as u32));
scalar!(f64, |v| v.to_bits(), |s| f64::from_bits(s));

impl WordValue for [u8; 16] {
    const WIDTH: usize = 2;

    #[inline(always)]
    unsafe fn read(p: *const Word) -> Self {
        let (lo, hi) = unsafe { (*p, *p.add(1)) };
        (lo as u128 | (hi as u128) << 64).to_le_bytes()
    }

    #[inline(always)]
    unsafe fn write(self, p: *mut Word) {
        let bits = u128::from_le_bytes(self);
        unsafe {
            *p = bits as Word;
            *p.add(1) = (bits >> 64) as Word;
        }
    }
}

/// What an operation leaves on the stack: one value, or nothing.
pub(super) trait WordResults {
    /// # Safety
    /// `*sp` must have room for the result.
    unsafe fn push_to(self, sp: &mut *mut Word);
}

impl WordResults for () {
    #[inline(always)]
    unsafe fn push_to(self, _sp: &mut *mut Word) {}
}

impl<T: WordValue> WordResults for T {
    #[inline(always)]
    unsafe fn push_to(self, sp: &mut *mut Word) {
        unsafe {
            self.write(*sp);
            *sp = sp.add(T::WIDTH);
        }
    }
}

/// Pop a `T` from the word stack.
///
/// # Safety
/// The top `T::WIDTH` words must hold a `T`.
#[inline(always)]
pub(super) unsafe fn pop<T: WordValue>(sp: &mut *mut Word) -> T {
    unsafe {
        *sp = sp.sub(T::WIDTH);
        T::read(*sp)
    }
}

/// Push an operation's result, if it has one.
///
/// # Safety
/// The stack must have room for the result.
#[inline(always)]
pub(super) unsafe fn push<R: WordResults>(sp: &mut *mut Word, result: R) {
    unsafe { result.push_to(sp) }
}

/// Push a tagged value.
///
/// # Safety
/// The stack must have room for the value.
pub(super) unsafe fn push_value(sp: &mut *mut Word, value: Value) {
    unsafe {
        match value {
            Value::I32(v) => push(sp, v),
            Value::I64(v) => push(sp, v),
            Value::F32(v) => push(sp, v),
            Value::F64(v) => push(sp, v),
            Value::V128(v) => push(sp, v),
            Value::FuncRef(r) => push(sp, r.map_or(0, |FuncAddr(addr)| addr as i64 + 1)),
            Value::ExternRef(r) => push(sp, r.map_or(0, |index| index as i64 + 1)),
        }
    }
}

/// Pop a value of type `value_type` as a tagged value.
///
/// # Safety
/// The top words must hold a value of `value_type`.
pub(super) unsafe fn pop_value(sp: &mut *mut Word, value_type: ValueType) -> Value {
    unsafe {
        match value_type {
            ValueType::I32 => Value::I32(pop(sp)),
            ValueType::I64 => Value::I64(pop(sp)),
            ValueType::F32 => Value::F32(pop(sp)),
            ValueType::F64 => Value::F64(pop(sp)),
            ValueType::V128 => Value::V128(pop(sp)),
            ValueType::FuncRef => Value::FuncRef(reference(pop(sp)).map(FuncAddr)),
            ValueType::ExternRef => Value::ExternRef(reference(pop(sp)).map(|index| index as u32)),
        }
    }
}

/// Decode a reference word: `None` for null, else the address or index.
fn reference(word: i64) -> Option<usize> {
    (word as u64).checked_sub(1).map(|index| index as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::bytecode::value_words;

    #[test]
    fn values_round_trip_through_words() {
        let values = [
            (Value::I32(-7), ValueType::I32),
            (Value::I64(i64::MIN), ValueType::I64),
            (Value::F32(-0.0), ValueType::F32),
            (Value::F64(f64::NAN), ValueType::F64),
            (Value::V128(std::array::from_fn(|i| i as u8)), ValueType::V128),
            (Value::FuncRef(None), ValueType::FuncRef),
            (Value::FuncRef(Some(FuncAddr(0))), ValueType::FuncRef),
            (Value::ExternRef(Some(u32::MAX)), ValueType::ExternRef),
        ];
        let mut words = [0; 16];
        let mut sp = words.as_mut_ptr();
        for (value, _) in values {
            unsafe { push_value(&mut sp, value) };
        }
        assert_eq!(unsafe { sp.offset_from(words.as_ptr()) }, 9);
        for (value, value_type) in values.into_iter().rev() {
            let back = unsafe { pop_value(&mut sp, value_type) };
            // NaN never equals itself; compare bits.
            assert_eq!(format!("{back:?}"), format!("{value:?}"));
        }
    }

    #[test]
    fn zero_words_are_every_type_zero() {
        for value_type in [ValueType::I32, ValueType::F64, ValueType::V128, ValueType::FuncRef] {
            let mut words = [0; 2];
            let mut sp = unsafe { words.as_mut_ptr().add(value_words(value_type) as usize) };
            let value = unsafe { pop_value(&mut sp, value_type) };
            assert_eq!(value, crate::runtime::imports::default_value_for_type(value_type));
        }
    }
}
