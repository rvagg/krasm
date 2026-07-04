//! Memory operations for WebAssembly
//!
//! This module provides implementations of memory load and store operations
//! as specified in the WebAssembly specification section 4.4.7 (Memory Instructions).

use super::*;

// ============================================================================
// Helper Functions
// ============================================================================

/// Copy data from a byte slice into memory
/// This is used by both memory.init and data section initialization
pub fn copy_to_memory(memory: &mut Memory, dest_addr: u32, data: &[u8]) -> Result<(), RuntimeError> {
    // Write data to memory with bounds checking
    for (i, &byte) in data.iter().enumerate() {
        memory.write_u8(dest_addr + i as u32, byte)?;
    }
    Ok(())
}

/// Compute the effective address for a memory load/store: base + memarg.offset.
///
/// Returns an error if the sum overflows or exceeds the 32-bit address space.
#[inline]
fn effective_address(base: i32, memarg: &MemArg) -> Result<u32, RuntimeError> {
    let ea = (base as u64)
        .checked_add(memarg.offset as u64)
        .ok_or_else(|| RuntimeError::MemoryError("out of bounds memory access".to_string()))?;
    if ea > u32::MAX as u64 {
        return Err(RuntimeError::MemoryError("out of bounds memory access".to_string()));
    }
    Ok(ea as u32)
}

/// Check that addr + length falls within memory bounds.
///
/// WebAssembly bulk memory operations require this even when length is zero.
#[inline]
fn check_memory_bounds(memory: &Memory, addr: u32, length: u32) -> Result<(), RuntimeError> {
    let mem_size = memory.size() as u64 * 65536;
    if addr as u64 + length as u64 > mem_size {
        return Err(RuntimeError::MemoryError("out of bounds memory access".to_string()));
    }
    Ok(())
}

// ============================================================================
// Memory Load Operations (spec section 4.4.7.1)
// ============================================================================

/// i32.load - Load 32-bit integer from memory
/// spec: 4.4.7.1
/// [i32] → [i32]
pub fn i32_load(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_i32(ea)?;
    stack.push(Value::I32(value));
    Ok(())
}

/// i32.load8_s - Load 8-bit signed integer and sign-extend to 32-bit
/// spec: 4.4.7.1
/// [i32] → [i32]
pub fn i32_load8_s(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_i8(ea)?;
    stack.push(Value::I32(value as i32));
    Ok(())
}

/// i32.load8_u - Load 8-bit unsigned integer and zero-extend to 32-bit
/// spec: 4.4.7.1
/// [i32] → [i32]
pub fn i32_load8_u(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_u8(ea)?;
    stack.push(Value::I32(value as i32));
    Ok(())
}

/// i32.load16_s - Load 16-bit signed integer and sign-extend to 32-bit
/// spec: 4.4.7.1
/// [i32] → [i32]
pub fn i32_load16_s(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_i16(ea)?;
    stack.push(Value::I32(value as i32));
    Ok(())
}

/// i32.load16_u - Load 16-bit unsigned integer and zero-extend to 32-bit
/// spec: 4.4.7.1
/// [i32] → [i32]
pub fn i32_load16_u(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_u16(ea)?;
    stack.push(Value::I32(value as i32));
    Ok(())
}

/// i64.load - Load 64-bit integer from memory
/// spec: 4.4.7.1
/// [i32] → [i64]
pub fn i64_load(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_i64(ea)?;
    stack.push(Value::I64(value));
    Ok(())
}

/// i64.load8_s - Load 8-bit signed integer and sign-extend to 64-bit
/// spec: 4.4.7.1
/// [i32] → [i64]
pub fn i64_load8_s(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_i8(ea)?;
    stack.push(Value::I64(value as i64));
    Ok(())
}

/// i64.load8_u - Load 8-bit unsigned integer and zero-extend to 64-bit
/// spec: 4.4.7.1
/// [i32] → [i64]
pub fn i64_load8_u(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_u8(ea)?;
    stack.push(Value::I64(value as i64));
    Ok(())
}

/// i64.load16_s - Load 16-bit signed integer and sign-extend to 64-bit
/// spec: 4.4.7.1
/// [i32] → [i64]
pub fn i64_load16_s(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_i16(ea)?;
    stack.push(Value::I64(value as i64));
    Ok(())
}

/// i64.load16_u - Load 16-bit unsigned integer and zero-extend to 64-bit
/// spec: 4.4.7.1
/// [i32] → [i64]
pub fn i64_load16_u(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_u16(ea)?;
    stack.push(Value::I64(value as i64));
    Ok(())
}

/// i64.load32_s - Load 32-bit signed integer and sign-extend to 64-bit
/// spec: 4.4.7.1
/// [i32] → [i64]
pub fn i64_load32_s(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_i32(ea)?;
    stack.push(Value::I64(value as i64));
    Ok(())
}

/// i64.load32_u - Load 32-bit unsigned integer and zero-extend to 64-bit
/// spec: 4.4.7.1
/// [i32] → [i64]
pub fn i64_load32_u(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_u32(ea)?;
    stack.push(Value::I64(value as i64));
    Ok(())
}

/// f32.load - Load 32-bit float from memory
/// spec: 4.4.7.1
/// [i32] → [f32]
pub fn f32_load(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_f32(ea)?;
    stack.push(Value::F32(value));
    Ok(())
}

/// f64.load - Load 64-bit float from memory
/// spec: 4.4.7.1
/// [i32] → [f64]
pub fn f64_load(stack: &mut Stack, memory: &Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    let value = memory.read_f64(ea)?;
    stack.push(Value::F64(value));
    Ok(())
}

// ============================================================================
// Memory Store Operations (spec section 4.4.7.6)
// ============================================================================

/// i32.store - Store 32-bit integer to memory
/// spec: 4.4.7.6
/// [i32, i32] → []
pub fn i32_store(stack: &mut Stack, memory: &mut Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let value = stack.pop_i32()?;
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    memory.write_i32(ea, value)?;
    Ok(())
}

/// i32.store8 - Store 32-bit integer as 8-bit value
/// spec: 4.4.7.6
/// [i32, i32] → []
pub fn i32_store8(stack: &mut Stack, memory: &mut Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let value = stack.pop_i32()?;
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    memory.write_u8(ea, value as u8)?;
    Ok(())
}

/// i32.store16 - Store 32-bit integer as 16-bit value
/// spec: 4.4.7.6
/// [i32, i32] → []
pub fn i32_store16(stack: &mut Stack, memory: &mut Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let value = stack.pop_i32()?;
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    memory.write_u16(ea, value as u16)?;
    Ok(())
}

/// i64.store - Store 64-bit integer to memory
/// spec: 4.4.7.6
/// [i32, i64] → []
pub fn i64_store(stack: &mut Stack, memory: &mut Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let value = stack.pop_i64()?;
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    memory.write_i64(ea, value)?;
    Ok(())
}

/// i64.store8 - Store 64-bit integer as 8-bit value
/// spec: 4.4.7.6
/// [i32, i64] → []
pub fn i64_store8(stack: &mut Stack, memory: &mut Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let value = stack.pop_i64()?;
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    memory.write_u8(ea, value as u8)?;
    Ok(())
}

/// i64.store16 - Store 64-bit integer as 16-bit value
/// spec: 4.4.7.6
/// [i32, i64] → []
pub fn i64_store16(stack: &mut Stack, memory: &mut Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let value = stack.pop_i64()?;
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    memory.write_u16(ea, value as u16)?;
    Ok(())
}

/// i64.store32 - Store 64-bit integer as 32-bit value
/// spec: 4.4.7.6
/// [i32, i64] → []
pub fn i64_store32(stack: &mut Stack, memory: &mut Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let value = stack.pop_i64()?;
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    memory.write_u32(ea, value as u32)?;
    Ok(())
}

/// f32.store - Store 32-bit float to memory
/// spec: 4.4.7.6
/// [i32, f32] → []
pub fn f32_store(stack: &mut Stack, memory: &mut Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let value = stack.pop_f32()?;
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    memory.write_f32(ea, value)?;
    Ok(())
}

/// f64.store - Store 64-bit float to memory
/// spec: 4.4.7.6
/// [i32, f64] → []
pub fn f64_store(stack: &mut Stack, memory: &mut Memory, memarg: &MemArg) -> Result<(), RuntimeError> {
    let value = stack.pop_f64()?;
    let addr = stack.pop_i32()?;

    let ea = effective_address(addr, memarg)?;

    memory.write_f64(ea, value)?;
    Ok(())
}

// ============================================================================
// Memory Size Operations (spec section 4.4.7.8, 4.4.7.9)
// ============================================================================

/// memory.size - Get memory size in pages
/// spec: 4.4.7.8
/// [] → [i32]
pub fn memory_size(stack: &mut Stack, memory: &Memory) -> Result<(), RuntimeError> {
    let size_in_pages = memory.size();
    stack.push(Value::I32(size_in_pages as i32));
    Ok(())
}

/// memory.grow - Grow memory by delta pages
/// spec: 4.4.7.9
/// [i32] → [i32]
///
/// Returns the previous size in pages, or -1 if the operation fails
pub fn memory_grow(stack: &mut Stack, memory: &mut Memory) -> Result<(), RuntimeError> {
    let delta = stack.pop_i32()?;

    if delta < 0 {
        stack.push(Value::I32(-1));
        return Ok(());
    }

    // The grow function expects pages, not bytes
    let result = memory.grow(delta as u32);
    stack.push(Value::I32(result));

    Ok(())
}

// Bulk Memory Operations
// These are part of the bulk memory operations proposal which is now standard

/// memory.init - Initialise memory from a data segment
/// spec: 4.4.7.12
/// [i32 i32 i32] → []
///
/// Stack: [dest_addr, src_offset, length]
/// Copies length bytes from data segment at src_offset to memory at dest_addr
pub fn memory_init(
    stack: &mut Stack,
    memory: &mut Memory,
    data_idx: u32,
    data_segments: &[crate::parser::module::Data],
    dropped_data: &std::collections::HashSet<u32>,
) -> Result<(), RuntimeError> {
    let length = stack.pop_i32()? as u32;
    let src_offset = stack.pop_i32()? as u32;
    let dest_addr = stack.pop_i32()? as u32;

    // A dropped segment acts as an empty segment
    let effective_len = if dropped_data.contains(&data_idx) {
        0
    } else {
        let data_segment = data_segments
            .get(data_idx as usize)
            .ok_or_else(|| RuntimeError::MemoryError(format!("invalid data segment index: {}", data_idx)))?;
        data_segment.init.len()
    };

    // Check bounds in data segment
    let src_end = (src_offset as usize)
        .checked_add(length as usize)
        .ok_or_else(|| RuntimeError::MemoryError("source range overflow in memory.init".to_string()))?;

    if src_end > effective_len {
        return Err(RuntimeError::MemoryError("out of bounds memory access".to_string()));
    }

    check_memory_bounds(memory, dest_addr, length)?;

    let data_segment = &data_segments[data_idx as usize];
    let src_data = &data_segment.init[src_offset as usize..src_end];
    copy_to_memory(memory, dest_addr, src_data)?;

    Ok(())
}

/// memory.copy - Copy memory within the same memory
/// spec: 4.4.7.11
/// [i32 i32 i32] → []
///
/// Stack: [dest_addr, src_addr, length]
/// Copies length bytes from src_addr to dest_addr
pub fn memory_copy(stack: &mut Stack, memory: &mut Memory) -> Result<(), RuntimeError> {
    let length = stack.pop_i32()? as u32;
    let src_addr = stack.pop_i32()? as u32;
    let dest_addr = stack.pop_i32()? as u32;

    check_memory_bounds(memory, src_addr, length)?;
    check_memory_bounds(memory, dest_addr, length)?;

    if length == 0 {
        return Ok(());
    }

    // Buffer the source bytes before writing — src and dest may overlap within
    // the same memory, so an in-place copy could corrupt unread source bytes.
    let mut bytes = Vec::with_capacity(length as usize);
    for i in 0..length {
        bytes.push(memory.read_u8(src_addr + i)?);
    }

    // Write bytes to destination
    for (i, byte) in bytes.iter().enumerate() {
        memory.write_u8(dest_addr + i as u32, *byte)?;
    }

    Ok(())
}

/// memory.fill - Fill memory with a byte value
/// spec: 4.4.7.10
/// [i32 i32 i32] → []
///
/// Stack: [dest_addr, value, length]
/// Fills length bytes at dest_addr with value
pub fn memory_fill(stack: &mut Stack, memory: &mut Memory) -> Result<(), RuntimeError> {
    let length = stack.pop_i32()? as u32;
    let value = stack.pop_i32()?;
    let dest_addr = stack.pop_i32()? as u32;

    check_memory_bounds(memory, dest_addr, length)?;

    // Value is treated as a byte (truncated to u8)
    let byte_value = (value & 0xFF) as u8;

    // Fill the memory
    for i in 0..length {
        memory.write_u8(dest_addr + i, byte_value)?;
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::approx_constant)] // 3.14 etc. are arbitrary test floats, not stand-ins for consts
mod tests {
    use crate::parser::instruction::{Instruction, InstructionKind, MemArg};
    use crate::parser::module::{Limits, Memory as MemoryDef, Module, ValueType};
    use crate::parser::structure_builder::StructureBuilder;
    use crate::runtime::executor::Executor;
    use crate::runtime::memory::PAGE_SIZE;
    use crate::runtime::test_utils::test::{ExecutorTest, make_instruction};
    use crate::runtime::{ExecutionOutcome, RuntimeError, Value};
    // Allow shorthand names for instruction kinds in tests
    use InstructionKind::*;
    use std::sync::Arc;

    /// Helper to unwrap ExecutionOutcome::Complete for tests
    fn unwrap_complete(outcome: ExecutionOutcome) -> Vec<Value> {
        match outcome {
            ExecutionOutcome::Complete(results) => results,
            ExecutionOutcome::NeedsExternalCall(_) => {
                panic!("Memory test unexpectedly needs external call");
            }
        }
    }

    // Helper function for tests that need manual module setup
    fn execute_memory_test(
        module: Module,
        instructions: Vec<Instruction>,
        args: Vec<Value>,
        return_types: &[ValueType],
    ) -> Result<Vec<Value>, RuntimeError> {
        let structured_func = StructureBuilder::build_function(&instructions, 0, return_types.to_vec())
            .expect("Structure building should succeed");
        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");
        let outcome = executor.execute_function(&structured_func, args, return_types, &mut resources)?;
        match outcome {
            crate::runtime::ExecutionOutcome::Complete(results) => Ok(results),
            crate::runtime::ExecutionOutcome::NeedsExternalCall(_) => {
                panic!("Memory test unexpectedly needs external call");
            }
        }
    }

    #[test]
    fn multiple_memories_error() {
        // WebAssembly 1.0 only supports one memory
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: None },
        });
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: None },
        });

        let result = Executor::new(Arc::new(module));
        assert!(result.is_err());
        let error_msg = result.err().unwrap().to_string();
        assert!(error_msg.contains("multiple memories not supported"));
    }

    #[test]
    fn memory_size_no_memory() {
        // memory.size with no memory should error
        ExecutorTest::new()
            .inst(InstructionKind::MemorySize)
            .expect_error("no memory instance available");
    }

    #[test]
    fn memory_grow_no_memory() {
        // memory.grow with no memory should error
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::MemoryGrow)
            .expect_error("no memory instance available");
    }

    #[test]
    fn memory_size_initial() {
        // Create module with 1 page of memory
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: None },
        });

        let result = execute_memory_test(
            module,
            vec![
                make_instruction(InstructionKind::MemorySize),
                make_instruction(InstructionKind::End),
            ],
            vec![],
            &[ValueType::I32],
        )
        .unwrap();

        assert_eq!(result, vec![Value::I32(1)]);
    }

    #[test]
    fn memory_grow_success() {
        // Create module with 1 page of memory, max 10
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: Some(10) },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");
        let instructions = vec![
            make_instruction(InstructionKind::I32Const { value: 2 }),
            make_instruction(InstructionKind::MemoryGrow),
            make_instruction(InstructionKind::MemorySize),
            make_instruction(InstructionKind::End),
        ];
        let func = StructureBuilder::build_function(&instructions, 0, vec![ValueType::I32, ValueType::I32])
            .expect("Structure building should succeed");
        let result = unwrap_complete(
            executor
                .execute_function(&func, vec![], &[ValueType::I32, ValueType::I32], &mut resources)
                .unwrap(),
        );

        // Should return old size (1) and new size (3)
        assert_eq!(result, vec![Value::I32(1), Value::I32(3)]);
    }

    #[test]
    fn memory_grow_failure_max() {
        // Create module with 1 page of memory, max 2
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: Some(2) },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");
        let instructions = vec![
            make_instruction(InstructionKind::I32Const { value: 5 }),
            make_instruction(InstructionKind::MemoryGrow),
            make_instruction(InstructionKind::End),
        ];
        let func = StructureBuilder::build_function(&instructions, 0, vec![ValueType::I32])
            .expect("Structure building should succeed");
        let result = unwrap_complete(
            executor
                .execute_function(&func, vec![], &[ValueType::I32], &mut resources)
                .unwrap(),
        );

        // Should return -1 (failure)
        assert_eq!(result, vec![Value::I32(-1)]);
    }

    #[test]
    fn memory_grow_negative() {
        // Create module with 1 page of memory
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: None },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");
        let instructions = vec![
            make_instruction(InstructionKind::I32Const { value: -1 }),
            make_instruction(InstructionKind::MemoryGrow),
            make_instruction(InstructionKind::End),
        ];
        let func = StructureBuilder::build_function(&instructions, 0, vec![ValueType::I32])
            .expect("Structure building should succeed");
        let result = unwrap_complete(
            executor
                .execute_function(&func, vec![], &[ValueType::I32], &mut resources)
                .unwrap(),
        );

        // Should return -1 (failure)
        assert_eq!(result, vec![Value::I32(-1)]);
    }

    #[test]
    fn memory_grow_zero() {
        // Create module with 2 pages of memory
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 2, max: None },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");
        let instructions = vec![
            make_instruction(InstructionKind::I32Const { value: 0 }),
            make_instruction(InstructionKind::MemoryGrow),
            make_instruction(InstructionKind::End),
        ];
        let func = StructureBuilder::build_function(&instructions, 0, vec![ValueType::I32])
            .expect("Structure building should succeed");
        let result = unwrap_complete(
            executor
                .execute_function(&func, vec![], &[ValueType::I32], &mut resources)
                .unwrap(),
        );

        // Should return current size (2)
        assert_eq!(result, vec![Value::I32(2)]);
    }

    #[test]
    fn memory_operations_sequence() {
        // Test a sequence of memory operations
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: Some(5) },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");
        let instructions = vec![
            make_instruction(InstructionKind::MemorySize),
            make_instruction(InstructionKind::I32Const { value: 1 }),
            make_instruction(InstructionKind::MemoryGrow),
            make_instruction(InstructionKind::I32Const { value: 2 }),
            make_instruction(InstructionKind::MemoryGrow),
            make_instruction(InstructionKind::MemorySize),
            make_instruction(InstructionKind::I32Const { value: 2 }),
            make_instruction(InstructionKind::MemoryGrow),
            make_instruction(InstructionKind::End),
        ];
        let func = StructureBuilder::build_function(
            &instructions,
            0,
            vec![
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
            ],
        )
        .expect("Structure building should succeed");
        let result = unwrap_complete(
            executor
                .execute_function(
                    &func,
                    vec![],
                    &[
                        ValueType::I32,
                        ValueType::I32,
                        ValueType::I32,
                        ValueType::I32,
                        ValueType::I32,
                    ],
                    &mut resources,
                )
                .unwrap(),
        );

        // Results: initial size (1), grow result (1), grow result (2),
        // final size (4), failed grow (-1)
        assert_eq!(
            result,
            vec![
                Value::I32(1),  // Initial size
                Value::I32(1),  // Old size after first grow
                Value::I32(2),  // Old size after second grow
                Value::I32(4),  // Final size
                Value::I32(-1), // Failed grow
            ]
        );
    }

    #[test]
    fn i32_load_basic() {
        // Create module with 1 page of memory
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: None },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");
        let instructions = vec![
            make_instruction(InstructionKind::I32Const { value: 100 }), // address,
            make_instruction(InstructionKind::I32Const { value: 42 }),  // value,
            make_instruction(InstructionKind::I32Store {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::I32Const { value: 100 }),
            make_instruction(InstructionKind::I32Load {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::End),
        ];
        let func = StructureBuilder::build_function(&instructions, 0, vec![ValueType::I32])
            .expect("Structure building should succeed");
        let result = unwrap_complete(
            executor
                .execute_function(&func, vec![], &[ValueType::I32], &mut resources)
                .unwrap(),
        );

        assert_eq!(result, vec![Value::I32(42)]);
    }

    #[test]
    fn i32_load_with_offset() {
        // Create module with 1 page of memory
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: None },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");
        let instructions = vec![
            make_instruction(InstructionKind::I32Const { value: 100 }),
            make_instruction(InstructionKind::I32Const { value: 0x12345678 }),
            make_instruction(InstructionKind::I32Store {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::I32Const { value: 90 }),
            make_instruction(InstructionKind::I32Load {
                memarg: MemArg { offset: 10, align: 2 },
            }),
            make_instruction(InstructionKind::End),
        ];
        let func = StructureBuilder::build_function(&instructions, 0, vec![ValueType::I32])
            .expect("Structure building should succeed");
        let result = unwrap_complete(
            executor
                .execute_function(&func, vec![], &[ValueType::I32], &mut resources)
                .unwrap(),
        );

        assert_eq!(result, vec![Value::I32(0x12345678)]);
    }

    #[test]
    fn i32_store_multiple() {
        // Test storing multiple values
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: None },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");
        let instructions = vec![
            make_instruction(InstructionKind::I32Const { value: 0 }),
            make_instruction(InstructionKind::I32Const { value: 100 }),
            make_instruction(InstructionKind::I32Store {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::I32Const { value: 4 }),
            make_instruction(InstructionKind::I32Const { value: 200 }),
            make_instruction(InstructionKind::I32Store {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::I32Const { value: 8 }),
            make_instruction(InstructionKind::I32Const { value: 300 }),
            make_instruction(InstructionKind::I32Store {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::I32Const { value: 0 }),
            make_instruction(InstructionKind::I32Load {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::I32Const { value: 4 }),
            make_instruction(InstructionKind::I32Load {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::I32Const { value: 8 }),
            make_instruction(InstructionKind::I32Load {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::End),
        ];
        let func =
            StructureBuilder::build_function(&instructions, 0, vec![ValueType::I32, ValueType::I32, ValueType::I32])
                .expect("Structure building should succeed");
        let result = unwrap_complete(
            executor
                .execute_function(
                    &func,
                    vec![],
                    &[ValueType::I32, ValueType::I32, ValueType::I32],
                    &mut resources,
                )
                .unwrap(),
        );

        assert_eq!(result, vec![Value::I32(100), Value::I32(200), Value::I32(300)]);
    }

    #[test]
    fn i32_load_bounds_check() {
        // Test bounds checking
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: None },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");

        // Try to load from last valid address (PAGE_SIZE - 4)
        let instructions = vec![
            make_instruction(InstructionKind::I32Const {
                value: (PAGE_SIZE - 4) as i32,
            }),
            make_instruction(InstructionKind::I32Load {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::End),
        ];
        let func = StructureBuilder::build_function(&instructions, 0, vec![ValueType::I32])
            .expect("Structure building should succeed");
        let result = unwrap_complete(
            executor
                .execute_function(&func, vec![], &[ValueType::I32], &mut resources)
                .unwrap(),
        );
        assert_eq!(result, vec![Value::I32(0)]); // Memory is zero-initialised

        // Try to load from out of bounds address
        let instructions = vec![
            make_instruction(InstructionKind::I32Const {
                value: (PAGE_SIZE - 3) as i32,
            }),
            make_instruction(InstructionKind::I32Load {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::End),
        ];
        let func =
            StructureBuilder::build_function(&instructions, 0, Vec::new()).expect("Structure building should succeed");
        let result = executor.execute_function(&func, vec![], &[], &mut resources);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn i32_store_bounds_check() {
        // Test bounds checking for store
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: None },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");

        // Try to store at last valid address
        let instructions = vec![
            make_instruction(InstructionKind::I32Const {
                value: (PAGE_SIZE - 4) as i32,
            }),
            make_instruction(InstructionKind::I32Const { value: 42 }),
            make_instruction(InstructionKind::I32Store {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::I32Const { value: 1 }), // Return success indicator,
            make_instruction(InstructionKind::End),
        ];
        let func = StructureBuilder::build_function(&instructions, 0, vec![ValueType::I32])
            .expect("Structure building should succeed");
        let result = unwrap_complete(
            executor
                .execute_function(&func, vec![], &[ValueType::I32], &mut resources)
                .unwrap(),
        );
        assert_eq!(result, vec![Value::I32(1)]);

        // Try to store at out of bounds address
        let instructions = vec![
            make_instruction(InstructionKind::I32Const {
                value: (PAGE_SIZE - 3) as i32,
            }),
            make_instruction(InstructionKind::I32Const { value: 42 }),
            make_instruction(InstructionKind::I32Store {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::End),
        ];
        let func =
            StructureBuilder::build_function(&instructions, 0, Vec::new()).expect("Structure building should succeed");
        let result = executor.execute_function(&func, vec![], &[], &mut resources);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn i32_load_offset_overflow() {
        // Test address calculation overflow
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: None },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");
        let instructions = vec![
            make_instruction(InstructionKind::I32Const { value: i32::MAX }),
            make_instruction(InstructionKind::I32Load {
                memarg: MemArg { offset: 10, align: 2 },
            }),
            make_instruction(InstructionKind::End),
        ];
        let func = StructureBuilder::build_function(&instructions, 0, vec![ValueType::I32])
            .expect("Structure building should succeed");
        let result = executor.execute_function(&func, vec![], &[ValueType::I32], &mut resources);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("overflow") || error_msg.contains("bounds"),
            "Expected overflow or bounds error, got: {}",
            error_msg
        );
    }

    #[test]
    fn i32_load_no_memory() {
        // Test load with no memory
        let module = Module::new("test");
        let result = execute_memory_test(
            module,
            vec![
                make_instruction(InstructionKind::I32Const { value: 0 }),
                make_instruction(InstructionKind::I32Load {
                    memarg: MemArg { offset: 0, align: 2 },
                }),
                make_instruction(InstructionKind::End),
            ],
            vec![],
            &[ValueType::I32],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no memory instance available"));
    }

    #[test]
    fn i32_store_no_memory() {
        // Test store with no memory
        let module = Module::new("test");
        let result = execute_memory_test(
            module,
            vec![
                make_instruction(InstructionKind::I32Const { value: 0 }),
                make_instruction(InstructionKind::I32Const { value: 42 }),
                make_instruction(InstructionKind::I32Store {
                    memarg: MemArg { offset: 0, align: 2 },
                }),
                make_instruction(InstructionKind::End),
            ],
            vec![],
            &[],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no memory instance available"));
    }

    #[test]
    fn i32_memory_persistence() {
        // Test that memory persists across multiple function calls with same executor
        let mut module = Module::new("test");
        module.memory.memory.push(MemoryDef {
            limits: Limits { min: 1, max: None },
        });

        let (mut executor, mut resources) = Executor::new(Arc::new(module)).expect("Executor creation should succeed");

        // First call: store a value
        let store_instructions = vec![
            make_instruction(InstructionKind::I32Const { value: 100 }),
            make_instruction(InstructionKind::I32Const { value: 42 }),
            make_instruction(InstructionKind::I32Store {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::End),
        ];
        let store_func = StructureBuilder::build_function(&store_instructions, 0, vec![])
            .expect("Structure building should succeed");

        executor
            .execute_function(&store_func, vec![], &[], &mut resources)
            .unwrap();

        // Second call: load the value back
        let load_instructions = vec![
            make_instruction(InstructionKind::I32Const { value: 100 }),
            make_instruction(InstructionKind::I32Load {
                memarg: MemArg { offset: 0, align: 2 },
            }),
            make_instruction(InstructionKind::End),
        ];
        let load_func = StructureBuilder::build_function(&load_instructions, 0, vec![ValueType::I32])
            .expect("Structure building should succeed");

        let result = unwrap_complete(
            executor
                .execute_function(&load_func, vec![], &[ValueType::I32], &mut resources)
                .unwrap(),
        );

        assert_eq!(result, vec![Value::I32(42)]);
    }

    #[test]
    fn i64_load_store() {
        ExecutorTest::new()
            .with_memory()
            .inst(I32Const { value: 8 }) // address
            .inst(I64Const {
                value: 0x123456789ABCDEF0,
            }) // value
            .inst(I64Store {
                memarg: MemArg { offset: 0, align: 3 },
            })
            .inst(I32Const { value: 8 }) // address
            .inst(I64Load {
                memarg: MemArg { offset: 0, align: 3 },
            })
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(0x123456789ABCDEF0)]);
    }

    #[test]
    fn f32_load_store() {
        ExecutorTest::new()
            .with_memory()
            .inst(I32Const { value: 16 }) // address
            .inst(F32Const { value: 3.14159 }) // value
            .inst(F32Store {
                memarg: MemArg { offset: 0, align: 2 },
            })
            .inst(I32Const { value: 16 }) // address
            .inst(F32Load {
                memarg: MemArg { offset: 0, align: 2 },
            })
            .returns(vec![ValueType::F32])
            .expect_stack(vec![Value::F32(3.14159)]);
    }

    #[test]
    fn f64_load_store() {
        ExecutorTest::new()
            .with_memory()
            .inst(I32Const { value: 24 }) // address
            .inst(F64Const {
                value: 2.718281828459045,
            }) // value
            .inst(F64Store {
                memarg: MemArg { offset: 0, align: 3 },
            })
            .inst(I32Const { value: 24 }) // address
            .inst(F64Load {
                memarg: MemArg { offset: 0, align: 3 },
            })
            .returns(vec![ValueType::F64])
            .expect_stack(vec![Value::F64(2.718281828459045)]);
    }

    #[test]
    fn i64_load_with_offset() {
        ExecutorTest::new()
            .with_memory()
            .inst(I32Const { value: 0 }) // address
            .inst(I64Const { value: i64::MAX }) // value
            .inst(I64Store {
                memarg: MemArg { offset: 100, align: 3 },
            })
            .inst(I32Const { value: 0 }) // address
            .inst(I64Load {
                memarg: MemArg { offset: 100, align: 3 },
            })
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(i64::MAX)]);
    }

    #[test]
    fn f32_store_multiple() {
        ExecutorTest::new()
            .with_memory()
            // Store multiple f32 values
            .inst(I32Const { value: 0 })
            .inst(F32Const { value: 1.0 })
            .inst(F32Store {
                memarg: MemArg { offset: 0, align: 2 },
            })
            .inst(I32Const { value: 4 })
            .inst(F32Const { value: 2.0 })
            .inst(F32Store {
                memarg: MemArg { offset: 0, align: 2 },
            })
            .inst(I32Const { value: 8 })
            .inst(F32Const { value: 3.0 })
            .inst(F32Store {
                memarg: MemArg { offset: 0, align: 2 },
            })
            // Load them back
            .inst(I32Const { value: 0 })
            .inst(F32Load {
                memarg: MemArg { offset: 0, align: 2 },
            })
            .inst(I32Const { value: 4 })
            .inst(F32Load {
                memarg: MemArg { offset: 0, align: 2 },
            })
            .inst(I32Const { value: 8 })
            .inst(F32Load {
                memarg: MemArg { offset: 0, align: 2 },
            })
            .returns(vec![ValueType::F32, ValueType::F32, ValueType::F32])
            .expect_stack(vec![Value::F32(1.0), Value::F32(2.0), Value::F32(3.0)]);
    }

    #[test]
    fn f64_bounds_check() {
        ExecutorTest::new()
            .with_memory()
            .inst(I32Const { value: 65536 - 4 }) // Near end of single page
            .inst(F64Const { value: 1.0 })
            .inst(F64Store {
                memarg: MemArg { offset: 0, align: 3 },
            })
            .expect_error("out of bounds");
    }

    #[test]
    fn i64_no_memory() {
        ExecutorTest::new()
            .inst(I32Const { value: 0 })
            .inst(I64Load {
                memarg: MemArg { offset: 0, align: 3 },
            })
            .expect_error("no memory instance");
    }

    #[test]
    fn i32_load8_u() {
        ExecutorTest::new()
            .with_memory()
            .inst(I32Const { value: 0 })
            .inst(I32Const { value: 0xFF }) // 255
            .inst(I32Store8 {
                memarg: MemArg { offset: 0, align: 0 },
            })
            .inst(I32Const { value: 0 })
            .inst(I32Load8U {
                memarg: MemArg { offset: 0, align: 0 },
            })
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(255)]);
    }

    #[test]
    fn i32_load8_s() {
        ExecutorTest::new()
            .with_memory()
            .inst(I32Const { value: 0 })
            .inst(I32Const { value: 0xFF }) // -1 when sign-extended
            .inst(I32Store8 {
                memarg: MemArg { offset: 0, align: 0 },
            })
            .inst(I32Const { value: 0 })
            .inst(I32Load8S {
                memarg: MemArg { offset: 0, align: 0 },
            })
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]);
    }

    #[test]
    fn i32_load16_u() {
        ExecutorTest::new()
            .with_memory()
            .inst(I32Const { value: 0 })
            .inst(I32Const { value: 0xFFFF }) // 65535
            .inst(I32Store16 {
                memarg: MemArg { offset: 0, align: 1 },
            })
            .inst(I32Const { value: 0 })
            .inst(I32Load16U {
                memarg: MemArg { offset: 0, align: 1 },
            })
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(65535)]);
    }

    #[test]
    fn i32_load16_s() {
        ExecutorTest::new()
            .with_memory()
            .inst(I32Const { value: 0 })
            .inst(I32Const { value: 0xFFFF }) // -1 when sign-extended
            .inst(I32Store16 {
                memarg: MemArg { offset: 0, align: 1 },
            })
            .inst(I32Const { value: 0 })
            .inst(I32Load16S {
                memarg: MemArg { offset: 0, align: 1 },
            })
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(-1)]);
    }

    #[test]
    fn i64_load8_u() {
        ExecutorTest::new()
            .with_memory()
            .inst(I32Const { value: 0 })
            .inst(I64Const { value: 0xFF }) // 255
            .inst(I64Store8 {
                memarg: MemArg { offset: 0, align: 0 },
            })
            .inst(I32Const { value: 0 })
            .inst(I64Load8U {
                memarg: MemArg { offset: 0, align: 0 },
            })
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(255)]);
    }

    #[test]
    fn i64_load32_s() {
        ExecutorTest::new()
            .with_memory()
            .inst(I32Const { value: 0 })
            .inst(I64Const { value: 0xFFFFFFFF }) // -1 when sign-extended from 32-bit
            .inst(I64Store32 {
                memarg: MemArg { offset: 0, align: 2 },
            })
            .inst(I32Const { value: 0 })
            .inst(I64Load32S {
                memarg: MemArg { offset: 0, align: 2 },
            })
            .returns(vec![ValueType::I64])
            .expect_stack(vec![Value::I64(-1)]);
    }
}
