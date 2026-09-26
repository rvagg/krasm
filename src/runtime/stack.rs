//! WebAssembly value stack implementation

use super::{RuntimeError, Value};
use crate::parser::module::ValueType;

/// The WebAssembly value stack
#[derive(Debug, Default)]
pub struct Stack {
    values: Vec<Value>,
}

macro_rules! typed_pop {
    ($name:ident, $variant:ident, $ty:ty) => {
        pub fn $name(&mut self) -> Result<$ty, RuntimeError> {
            match self.pop()? {
                Value::$variant(value) => Ok(value),
                value => Err(type_mismatch(ValueType::$variant, value.typ())),
            }
        }
    };
}

impl Stack {
    pub fn new() -> Self {
        Stack { values: Vec::new() }
    }

    pub fn push(&mut self, value: Value) {
        self.values.push(value);
    }

    pub fn pop(&mut self) -> Result<Value, RuntimeError> {
        // Construct the error only on underflow; eager errors can retain drop glue.
        match self.values.pop() {
            Some(value) => Ok(value),
            None => Err(RuntimeError::StackUnderflow),
        }
    }

    /// Pop a value and check its type, returning `TypeMismatch` on failure.
    pub fn pop_typed(&mut self, expected_type: ValueType) -> Result<Value, RuntimeError> {
        let value = self.pop()?;
        if value.typ() != expected_type {
            return Err(type_mismatch(expected_type, value.typ()));
        }
        Ok(value)
    }

    typed_pop!(pop_i32, I32, i32);
    typed_pop!(pop_i64, I64, i64);
    typed_pop!(pop_f32, F32, f32);
    typed_pop!(pop_f64, F64, f64);
    typed_pop!(pop_v128, V128, [u8; 16]);

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Peek at the top value without popping.
    pub fn peek(&self) -> Option<&Value> {
        self.values.last()
    }
}

#[cold]
fn type_mismatch(expected: ValueType, actual: ValueType) -> RuntimeError {
    RuntimeError::TypeMismatch {
        expected: format!("{expected:?}"),
        actual: format!("{actual:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let mut stack = Stack::new();

        stack.push(Value::I32(42));
        stack.push(Value::I64(100));

        assert_eq!(stack.len(), 2);
        assert_eq!(stack.pop().unwrap(), Value::I64(100));
        assert_eq!(stack.pop().unwrap(), Value::I32(42));
        assert!(stack.pop().is_err());
    }

    #[test]
    fn test_pop_typed() {
        let mut stack = Stack::new();
        // Correct type
        stack.push(Value::I32(42));
        assert_eq!(stack.pop_typed(ValueType::I32).unwrap(), Value::I32(42));

        // Wrong type
        stack.push(Value::I32(42));
        assert!(stack.pop_typed(ValueType::I64).is_err());
    }

    #[test]
    fn test_typed_pop_methods() {
        let mut stack = Stack::new();

        stack.push(Value::I32(42));
        assert_eq!(stack.pop_i32().unwrap(), 42);

        stack.push(Value::I64(100));
        assert_eq!(stack.pop_i64().unwrap(), 100);

        stack.push(Value::F32(1.5));
        assert_eq!(stack.pop_f32().unwrap(), 1.5);

        stack.push(Value::F64(2.5));
        assert_eq!(stack.pop_f64().unwrap(), 2.5);
    }

    #[test]
    fn test_typed_pop_mismatches_consume_only_top() {
        fn check<T: std::fmt::Debug>(pop: fn(&mut Stack) -> Result<T, RuntimeError>, expected_type: ValueType) {
            for value in [
                Value::I32(1),
                Value::I64(1),
                Value::F32(1.0),
                Value::F64(1.0),
                Value::V128([1; 16]),
                Value::FuncRef(None),
                Value::ExternRef(None),
            ] {
                if value.typ() == expected_type {
                    continue;
                }
                let mut stack = Stack::new();
                stack.push(Value::I32(91));
                stack.push(value);
                match pop(&mut stack).unwrap_err() {
                    RuntimeError::TypeMismatch { expected, actual } => {
                        assert_eq!(expected, format!("{expected_type:?}"));
                        assert_eq!(actual, format!("{:?}", value.typ()));
                    }
                    other => panic!("expected type mismatch, got {other:?}"),
                }
                assert_eq!(stack.pop().unwrap(), Value::I32(91));
                assert!(matches!(pop(&mut stack), Err(RuntimeError::StackUnderflow)));
            }
        }

        check(Stack::pop_i32, ValueType::I32);
        check(Stack::pop_i64, ValueType::I64);
        check(Stack::pop_f32, ValueType::F32);
        check(Stack::pop_f64, ValueType::F64);
        check(Stack::pop_v128, ValueType::V128);
    }

    #[test]
    fn test_peek() {
        let mut stack = Stack::new();
        assert!(stack.peek().is_none());

        stack.push(Value::I32(42));
        assert_eq!(stack.peek(), Some(&Value::I32(42)));
        assert_eq!(stack.len(), 1); // peek doesn't remove
    }

    #[test]
    fn test_clear() {
        let mut stack = Stack::new();
        stack.push(Value::I32(42));
        assert_eq!(stack.len(), 1);

        stack.clear();
        assert_eq!(stack.len(), 0);
    }
}
