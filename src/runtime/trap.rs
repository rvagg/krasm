//! Traps raised by instructions.
//!
//! A trap aborts execution (spec 4.4). Operations return [`Trap`] rather than
//! [`RuntimeError`]: it is one byte, so a `Result<i32, Trap>` comes back in
//! registers, and the engine converts it only when a trap actually happens.

use super::RuntimeError;

/// Why an instruction trapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trap {
    IntegerDivideByZero,
    IntegerOverflow,
}

impl From<Trap> for RuntimeError {
    #[cold]
    fn from(trap: Trap) -> Self {
        match trap {
            Trap::IntegerDivideByZero => RuntimeError::DivisionByZero,
            Trap::IntegerOverflow => RuntimeError::IntegerOverflow,
        }
    }
}
