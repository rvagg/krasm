//! WebAssembly value stack implementation

use super::{RuntimeError, Value};
use crate::parser::module::ValueType;

/// The WebAssembly value stack
#[derive(Debug, Default)]
pub struct Stack {
    values: Vec<Value>,
}

impl Stack {
    pub fn new() -> Self {
        Stack { values: Vec::new() }
    }

    pub fn push(&mut self, value: Value) {
        self.values.push(value);
    }

    pub fn pop(&mut self) -> Result<Value, RuntimeError> {
        self.values.pop().ok_or(RuntimeError::StackUnderflow)
    }

    /// Pop a value and check its type, returning `TypeMismatch` on failure.
    pub fn pop_typed(&mut self, expected_type: ValueType) -> Result<Value, RuntimeError> {
        let value = self.pop()?;
        if value.typ() != expected_type {
            return Err(RuntimeError::TypeMismatch {
                expected: format!("{expected_type:?}"),
                actual: format!("{:?}", value.typ()),
            });
        }
        Ok(value)
    }

    pub fn pop_i32(&mut self) -> Result<i32, RuntimeError> {
        self.pop_typed(ValueType::I32)?
            .as_i32()
            .ok_or(RuntimeError::TypeMismatch {
                expected: "i32".to_string(),
                actual: "non-i32".to_string(),
            })
    }

    pub fn pop_i64(&mut self) -> Result<i64, RuntimeError> {
        self.pop_typed(ValueType::I64)?
            .as_i64()
            .ok_or(RuntimeError::TypeMismatch {
                expected: "i64".to_string(),
                actual: "non-i64".to_string(),
            })
    }

    pub fn pop_f32(&mut self) -> Result<f32, RuntimeError> {
        self.pop_typed(ValueType::F32)?
            .as_f32()
            .ok_or(RuntimeError::TypeMismatch {
                expected: "f32".to_string(),
                actual: "non-f32".to_string(),
            })
    }

    pub fn pop_f64(&mut self) -> Result<f64, RuntimeError> {
        self.pop_typed(ValueType::F64)?
            .as_f64()
            .ok_or(RuntimeError::TypeMismatch {
                expected: "f64".to_string(),
                actual: "non-f64".to_string(),
            })
    }

    pub fn pop_v128(&mut self) -> Result<[u8; 16], RuntimeError> {
        self.pop_typed(ValueType::V128)?
            .as_v128()
            .ok_or(RuntimeError::TypeMismatch {
                expected: "v128".to_string(),
                actual: "non-v128".to_string(),
            })
    }

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
