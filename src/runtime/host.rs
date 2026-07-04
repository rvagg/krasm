//! Typed host function support
//!
//! Maps Rust types to WebAssembly types and converts typed closures into host
//! functions. The function type signature is inferred from the closure.
//!
//! # Usage
//!
//! ```ignore
//! // Simple function (no caller access)
//! let addr = store.wrap(|a: i32, b: i32| -> i32 { a + b });
//!
//! // With caller access (memory + user data)
//! let addr = store.wrap_with_caller(|caller: &mut Caller<MyState>, ptr: i32| -> i32 {
//!     let data = caller.data_mut();
//!     data.call_count += 1;
//!     0
//! });
//! ```

use super::store::{Caller, HostFunc};
use super::{RuntimeError, Value};
use crate::parser::module::{FunctionType, ValueType};

/// Maps a Rust type to a WebAssembly value type with bidirectional conversion.
pub trait WasmType: Sized + 'static {
    const VALUE_TYPE: ValueType;
    fn from_value(val: Value) -> Result<Self, RuntimeError>;
    fn into_value(self) -> Value;
}

impl WasmType for i32 {
    const VALUE_TYPE: ValueType = ValueType::I32;
    fn from_value(val: Value) -> Result<Self, RuntimeError> {
        match val {
            Value::I32(v) => Ok(v),
            other => Err(RuntimeError::TypeMismatch {
                expected: "i32".into(),
                actual: format!("{}", other.typ()),
            }),
        }
    }
    fn into_value(self) -> Value {
        Value::I32(self)
    }
}

impl WasmType for i64 {
    const VALUE_TYPE: ValueType = ValueType::I64;
    fn from_value(val: Value) -> Result<Self, RuntimeError> {
        match val {
            Value::I64(v) => Ok(v),
            other => Err(RuntimeError::TypeMismatch {
                expected: "i64".into(),
                actual: format!("{}", other.typ()),
            }),
        }
    }
    fn into_value(self) -> Value {
        Value::I64(self)
    }
}

impl WasmType for f32 {
    const VALUE_TYPE: ValueType = ValueType::F32;
    fn from_value(val: Value) -> Result<Self, RuntimeError> {
        match val {
            Value::F32(v) => Ok(v),
            other => Err(RuntimeError::TypeMismatch {
                expected: "f32".into(),
                actual: format!("{}", other.typ()),
            }),
        }
    }
    fn into_value(self) -> Value {
        Value::F32(self)
    }
}

impl WasmType for f64 {
    const VALUE_TYPE: ValueType = ValueType::F64;
    fn from_value(val: Value) -> Result<Self, RuntimeError> {
        match val {
            Value::F64(v) => Ok(v),
            other => Err(RuntimeError::TypeMismatch {
                expected: "f64".into(),
                actual: format!("{}", other.typ()),
            }),
        }
    }
    fn into_value(self) -> Value {
        Value::F64(self)
    }
}

/// Represents valid return types for typed host functions.
///
/// Implemented for `()` (no return), single `WasmType` values, and
/// `Result<R, RuntimeError>` for fallible host functions.
pub trait WasmResult {
    fn return_types() -> Vec<ValueType>;
    fn into_values(self) -> Result<Vec<Value>, RuntimeError>;
}

impl WasmResult for () {
    fn return_types() -> Vec<ValueType> {
        vec![]
    }
    fn into_values(self) -> Result<Vec<Value>, RuntimeError> {
        Ok(vec![])
    }
}

impl<T: WasmType> WasmResult for T {
    fn return_types() -> Vec<ValueType> {
        vec![T::VALUE_TYPE]
    }
    fn into_values(self) -> Result<Vec<Value>, RuntimeError> {
        Ok(vec![self.into_value()])
    }
}

impl<R: WasmResult> WasmResult for Result<R, RuntimeError> {
    fn return_types() -> Vec<ValueType> {
        R::return_types()
    }
    fn into_values(self) -> Result<Vec<Value>, RuntimeError> {
        self?.into_values()
    }
}

/// Converts a typed closure (without caller access) into a host function.
pub trait IntoHostFunc<T, Params, Returns> {
    fn into_host_func(self) -> (HostFunc<T>, FunctionType);
}

/// Converts a typed closure (with `&mut Caller<T>` access) into a host function.
pub trait IntoHostFuncWithCaller<T, Params, Returns> {
    fn into_host_func(self) -> (HostFunc<T>, FunctionType);
}

macro_rules! impl_host_func {
    ($($A:ident),*) => {
        #[allow(non_snake_case, unused_variables, unused_mut)]
        impl<UserData, F, R $(, $A)*> IntoHostFunc<UserData, ($($A,)*), R> for F
        where
            F: Fn($($A,)*) -> R + 'static,
            $($A: WasmType,)*
            R: WasmResult,
        {
            fn into_host_func(self) -> (HostFunc<UserData>, FunctionType) {
                let func_type = FunctionType {
                    parameters: vec![$($A::VALUE_TYPE,)*],
                    return_types: R::return_types(),
                };
                let func: HostFunc<UserData> = Box::new(move |_caller, args| {
                    let mut _iter = args.into_iter();
                    $(let $A = $A::from_value(_iter.next().unwrap())?;)*
                    self($($A,)*).into_values()
                });
                (func, func_type)
            }
        }

        #[allow(non_snake_case, unused_variables, unused_mut)]
        impl<UserData, F, R $(, $A)*> IntoHostFuncWithCaller<UserData, ($($A,)*), R> for F
        where
            F: Fn(&mut Caller<'_, UserData>, $($A,)*) -> R + 'static,
            $($A: WasmType,)*
            R: WasmResult,
        {
            fn into_host_func(self) -> (HostFunc<UserData>, FunctionType) {
                let func_type = FunctionType {
                    parameters: vec![$($A::VALUE_TYPE,)*],
                    return_types: R::return_types(),
                };
                let func: HostFunc<UserData> = Box::new(move |caller, args| {
                    let mut _iter = args.into_iter();
                    $(let $A = $A::from_value(_iter.next().unwrap())?;)*
                    self(caller, $($A,)*).into_values()
                });
                (func, func_type)
            }
        }
    };
}

impl_host_func!();
impl_host_func!(A0);
impl_host_func!(A0, A1);
impl_host_func!(A0, A1, A2);
impl_host_func!(A0, A1, A2, A3);
impl_host_func!(A0, A1, A2, A3, A4);
impl_host_func!(A0, A1, A2, A3, A4, A5);
impl_host_func!(A0, A1, A2, A3, A4, A5, A6);
impl_host_func!(A0, A1, A2, A3, A4, A5, A6, A7);
impl_host_func!(A0, A1, A2, A3, A4, A5, A6, A7, A8);

#[cfg(test)]
#[allow(clippy::approx_constant)] // 3.14 etc. are arbitrary test floats, not stand-ins for consts
mod tests {
    use super::*;

    #[test]
    fn wasm_type_i32_round_trip() {
        let val = 42i32.into_value();
        assert_eq!(i32::from_value(val).unwrap(), 42);
    }

    #[test]
    fn wasm_type_i64_round_trip() {
        let val = 123i64.into_value();
        assert_eq!(i64::from_value(val).unwrap(), 123);
    }

    #[test]
    fn wasm_type_f32_round_trip() {
        let val = 3.14f32.into_value();
        assert_eq!(f32::from_value(val).unwrap(), 3.14);
    }

    #[test]
    fn wasm_type_f64_round_trip() {
        let val = 2.718f64.into_value();
        assert_eq!(f64::from_value(val).unwrap(), 2.718);
    }

    #[test]
    fn wasm_type_mismatch_error() {
        let val = Value::I32(1);
        let err = i64::from_value(val).unwrap_err();
        assert!(err.to_string().contains("type mismatch"));
    }

    #[test]
    fn wasm_result_unit() {
        assert_eq!(<() as WasmResult>::return_types(), vec![]);
        assert_eq!(().into_values().unwrap(), vec![]);
    }

    #[test]
    fn wasm_result_single() {
        assert_eq!(<i32 as WasmResult>::return_types(), vec![ValueType::I32]);
        assert_eq!(42i32.into_values().unwrap(), vec![Value::I32(42)]);
    }

    #[test]
    fn wasm_result_fallible() {
        let ok: Result<i32, RuntimeError> = Ok(7);
        assert_eq!(ok.into_values().unwrap(), vec![Value::I32(7)]);

        let err: Result<i32, RuntimeError> = Err(RuntimeError::Trap("test".into()));
        assert!(err.into_values().is_err());
    }

    #[test]
    fn into_host_func_no_args() {
        let (func, ftype) = (|| -> i32 { 42 }).into_host_func();
        assert_eq!(ftype.parameters, vec![]);
        assert_eq!(ftype.return_types, vec![ValueType::I32]);

        let mut data = ();
        let mut caller = Caller::for_test(None, &mut data);
        let result = func(&mut caller, vec![]).unwrap();
        assert_eq!(result, vec![Value::I32(42)]);
    }

    #[test]
    fn into_host_func_two_args() {
        let (func, ftype) = (|a: i32, b: i32| -> i32 { a + b }).into_host_func();
        assert_eq!(ftype.parameters, vec![ValueType::I32, ValueType::I32]);
        assert_eq!(ftype.return_types, vec![ValueType::I32]);

        let mut data = ();
        let mut caller = Caller::for_test(None, &mut data);
        let result = func(&mut caller, vec![Value::I32(3), Value::I32(4)]).unwrap();
        assert_eq!(result, vec![Value::I32(7)]);
    }

    #[test]
    fn into_host_func_void_return() {
        let (func, ftype) = (|_a: i32| {}).into_host_func();
        assert_eq!(ftype.parameters, vec![ValueType::I32]);
        assert_eq!(ftype.return_types, vec![]);

        let mut data = ();
        let mut caller = Caller::for_test(None, &mut data);
        let result = func(&mut caller, vec![Value::I32(1)]).unwrap();
        assert_eq!(result, vec![]);
    }

    #[test]
    fn into_host_func_with_caller() {
        let (func, ftype): (HostFunc<u32>, _) =
            IntoHostFuncWithCaller::into_host_func(|caller: &mut Caller<'_, u32>, x: i32| -> i32 {
                *caller.data_mut() += x as u32;
                *caller.data() as i32
            });
        assert_eq!(ftype.parameters, vec![ValueType::I32]);
        assert_eq!(ftype.return_types, vec![ValueType::I32]);

        let mut data: u32 = 10;
        let mut caller = Caller::for_test(None, &mut data);
        let result = func(&mut caller, vec![Value::I32(5)]).unwrap();
        assert_eq!(result, vec![Value::I32(15)]);
    }

    #[test]
    fn into_host_func_fallible() {
        let (func, ftype) = (|x: i32| -> Result<i32, RuntimeError> {
            if x == 0 {
                Err(RuntimeError::Trap("zero".into()))
            } else {
                Ok(x * 2)
            }
        })
        .into_host_func();
        assert_eq!(ftype.return_types, vec![ValueType::I32]);

        let mut data = ();
        let mut caller = Caller::for_test(None, &mut data);
        assert_eq!(func(&mut caller, vec![Value::I32(3)]).unwrap(), vec![Value::I32(6)]);
        assert!(func(&mut caller, vec![Value::I32(0)]).is_err());
    }
}
