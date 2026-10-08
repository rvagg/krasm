//! Control flow operations for WebAssembly
//!
//! This module provides implementations of control flow operations
//! as specified in the WebAssembly specification section 4.4.8 (Control Instructions).

use super::*;
use crate::parser::structured::BlockEnd;
use crate::runtime::control::LabelStack;

/// Perform the actual branch operation to a target label
/// This is the common implementation used by br, br_if, and br_table
///
/// Steps:
/// 1. Get the target label to determine arity and stack height
/// 2. Pop arity values from the stack (these will be the results/parameters)
/// 3. Restore stack to the label's entry height (removes intermediate values)
/// 4. Push the result values back
/// 5. Return Branch(label_idx) to signal the branch
fn perform_branch(stack: &mut Stack, label_stack: &LabelStack, label_idx: u32) -> Result<BlockEnd, RuntimeError> {
    // Get the target label to determine arity and stack height
    let label = label_stack
        .get(label_idx)
        .ok_or(RuntimeError::InvalidLabel(label_idx))?;

    // Get the arity based on label type
    // For loops: we need to preserve parameters
    // For blocks/ifs: we need to preserve return values
    let arity = label.arity();

    // Pop arity values from stack (these will be the results/parameters)
    let mut values = Vec::with_capacity(arity);
    for _ in 0..arity {
        values.push(stack.pop()?);
    }
    values.reverse(); // Restore original order

    // Restore stack to the label's entry height
    // This removes any intermediate values that were on the stack
    while stack.len() > label.stack_height {
        stack.pop()?;
    }

    // Push the result values back
    for value in values {
        stack.push(value);
    }

    Ok(BlockEnd::Branch(label_idx))
}

/// br l - Unconditional branch
/// spec: 4.4.8
///
/// From the spec:
/// 1. Assert: due to validation, the stack contains at least l + 1 labels.
/// 2. Let L be the l-th label appearing on the stack, starting from the top and counting from zero.
/// 3. Let n be the arity of L.
/// 4. Assert: due to validation, there are at least n values on the top of the stack.
/// 5. Pop the values val^n from the stack.
/// 6. Repeat l + 1 times: pop a label from the stack.
/// 7. Push the values val^n to the stack.
/// 8. Jump to the continuation of L.
pub fn br(stack: &mut Stack, label_stack: &LabelStack, label_idx: u32) -> Result<BlockEnd, RuntimeError> {
    perform_branch(stack, label_stack, label_idx)
}

/// br_if l - Conditional branch
/// spec: 4.4.8
///
/// From the spec:
/// 1. Assert: due to validation, a value of type i32 is on the top of the stack.
/// 2. Pop the value c from the stack.
/// 3. If c is non-zero, then execute the instruction br l.
/// 4. Else, do nothing.
pub fn br_if(stack: &mut Stack, label_stack: &LabelStack, label_idx: u32) -> Result<BlockEnd, RuntimeError> {
    let condition = stack.pop_i32()?;
    if condition != 0 {
        perform_branch(stack, label_stack, label_idx)
    } else {
        Ok(BlockEnd::Normal)
    }
}

/// br_table l* lN - Indirect branch via table
/// spec: 4.4.8
///
/// From the spec:
/// A br_table performs an indirect branch through an operand indexing into
/// a list of labels.
/// 1. Pop i32 index from stack
/// 2. If `index` < len(labels), branch to labels\[index\]
/// 3. Else branch to default
pub fn br_table(
    stack: &mut Stack,
    label_stack: &LabelStack,
    labels: &[u32],
    default: u32,
) -> Result<BlockEnd, RuntimeError> {
    // Pop index from stack
    let index = stack.pop_i32()?;

    // Choose the target label based on index
    let target = if index >= 0 && (index as usize) < labels.len() {
        labels[index as usize]
    } else {
        default
    };

    // Now branch to the target using the common implementation
    perform_branch(stack, label_stack, target)
}

/// return - Return from function
/// spec: 4.4.8
///
/// From the spec:
/// The return instruction is a shortcut for an unconditional branch
/// to the outermost block, which implicitly is the body of the current function.
///
/// The branch unwinds the stack to the function label like any other
/// branch, then BlockEnd::Return propagates up through all nested blocks to
/// exit the function. The function label is the outermost in `label_stack`.
pub fn return_op(stack: &mut Stack, label_stack: &LabelStack) -> Result<BlockEnd, RuntimeError> {
    let function_label = label_stack.len().checked_sub(1).ok_or(RuntimeError::InvalidLabel(0))?;
    perform_branch(stack, label_stack, function_label as u32)?;
    Ok(BlockEnd::Return)
}

/// unreachable - Trap immediately
/// spec: 4.4.8
///
/// From the spec:
/// The unreachable instruction causes an immediate trap.
/// It is typically used to indicate unreachable code.
pub fn unreachable() -> Result<BlockEnd, RuntimeError> {
    Err(RuntimeError::Trap("unreachable instruction executed".to_string()))
}

#[cfg(test)]
mod tests {
    use crate::parser::instruction::{BlockType, InstructionKind};
    use crate::parser::module::ValueType;
    use crate::runtime::Value;
    use crate::runtime::test_utils::test::ExecutorTest;

    // ============================================================================
    // Block and Control Flow Tests
    // ============================================================================

    #[test]
    fn block_empty() {
        // Empty block
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 42 })
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn block_with_value() {
        // Block that produces a value
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn nested_blocks() {
        // Nested blocks
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::End)
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn br_simple() {
        // Branch out of a block
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Br { label_idx: 0 })
            .inst(InstructionKind::I32Const { value: 99 }) // Should be skipped
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn br_with_value() {
        // Branch with a value
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Br { label_idx: 0 })
            .inst(InstructionKind::Drop) // Should be skipped
            .inst(InstructionKind::I32Const { value: 99 }) // Should be skipped
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn br_nested() {
        // Branch from inner to outer block
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::Br { label_idx: 1 }) // Branch to outer block
            .inst(InstructionKind::I32Const { value: 99 }) // Should be skipped
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 88 }) // Should be skipped
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn br_if_true() {
        // Conditional branch taken
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 1 }) // True condition
            .inst(InstructionKind::BrIf { label_idx: 0 })
            .inst(InstructionKind::Drop) // Should be skipped
            .inst(InstructionKind::I32Const { value: 99 }) // Should be skipped
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn br_if_false() {
        // Conditional branch not taken
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 0 }) // False condition
            .inst(InstructionKind::BrIf { label_idx: 0 })
            .inst(InstructionKind::Drop) // Should execute
            .inst(InstructionKind::I32Const { value: 99 }) // Should execute
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(99)]);
    }

    #[test]
    fn br_if_with_value() {
        // Conditional branch with block value
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: -1 }) // True condition (non-zero)
            .inst(InstructionKind::BrIf { label_idx: 0 })
            .inst(InstructionKind::Drop)
            .inst(InstructionKind::I32Const { value: 99 })
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn loop_simple() {
        // Simple loop that exits immediately
        ExecutorTest::new()
            .inst(InstructionKind::Loop {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn loop_with_counter() {
        // Loop with a counter (simplified - normally would use locals)
        ExecutorTest::new()
            .args(vec![Value::I32(3)]) // Counter in local 0
            .inst(InstructionKind::Loop {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::LocalGet { local_idx: 0 })
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 0 }) // Simulate i32.sub result of 0
            .inst(InstructionKind::LocalSet { local_idx: 0 })
            .inst(InstructionKind::LocalGet { local_idx: 0 })
            .inst(InstructionKind::BrIf { label_idx: 0 }) // Continue loop if non-zero
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 42 })
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn if_true_no_else() {
        // If with true condition, no else branch
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 }) // True condition
            .inst(InstructionKind::If {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 99 })
            .returns(vec![ValueType::I32, ValueType::I32])
            .expect_stack(vec![Value::I32(42), Value::I32(99)]);
    }

    #[test]
    fn if_false_no_else() {
        // If with false condition, no else branch
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0 }) // False condition
            .inst(InstructionKind::If {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 }) // Should be skipped
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 99 })
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(99)]);
    }

    #[test]
    fn if_true_with_else() {
        // If with true condition and else branch
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 }) // True condition
            .inst(InstructionKind::If {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Else)
            .inst(InstructionKind::I32Const { value: 88 }) // Should be skipped
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn if_false_with_else() {
        // If with false condition and else branch
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0 }) // False condition
            .inst(InstructionKind::If {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 }) // Should be skipped
            .inst(InstructionKind::Else)
            .inst(InstructionKind::I32Const { value: 88 })
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(88)]);
    }

    #[test]
    fn if_with_value() {
        // If that produces a value
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::If {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Else)
            .inst(InstructionKind::I32Const { value: 88 })
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn nested_if() {
        // Nested if statements
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::If {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::If {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 11 }) // Should be skipped
            .inst(InstructionKind::Else)
            .inst(InstructionKind::I32Const { value: 22 })
            .inst(InstructionKind::End)
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(22)]);
    }

    #[test]
    fn if_br() {
        // Branch out of if
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::If {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Br { label_idx: 0 })
            .inst(InstructionKind::Drop)
            .inst(InstructionKind::I32Const { value: 99 })
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn return_simple() {
        // Simple return with value
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Return)
            .inst(InstructionKind::I32Const { value: 99 }) // Should not be executed
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn return_no_value() {
        // Return with no value
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Drop)
            .inst(InstructionKind::Return)
            .inst(InstructionKind::I32Const { value: 99 }) // Should not be executed
            .returns(vec![])
            .expect_stack(vec![]);
    }

    #[test]
    fn return_multiple_values() {
        // Return with multiple values
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::I32Const { value: 2 })
            .inst(InstructionKind::Return)
            .inst(InstructionKind::I32Const { value: 99 }) // Should not be executed
            .returns(vec![ValueType::I32, ValueType::I32])
            .expect_stack(vec![Value::I32(1), Value::I32(2)]);
    }

    #[test]
    fn return_from_block() {
        // Return from inside a block
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Return)
            .inst(InstructionKind::I32Const { value: 88 }) // Should not be executed
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 99 }) // Should not be executed
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn return_from_nested_blocks() {
        // Return from deeply nested blocks
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Return)
            .inst(InstructionKind::End)
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 99 }) // Should not be executed
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn return_from_if() {
        // Return from inside if branch
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::If {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Return)
            .inst(InstructionKind::Else)
            .inst(InstructionKind::I32Const { value: 88 })
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 99 }) // Should not be executed
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn return_from_loop() {
        // Return from inside a loop
        ExecutorTest::new()
            .inst(InstructionKind::Loop {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Return)
            .inst(InstructionKind::Br { label_idx: 0 }) // Should not be executed
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn br_table_index_0() {
        // Branch to first label in table (innermost block)
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 0 }) // index 0
            .inst(InstructionKind::BrTable {
                labels: vec![0, 1],
                default: 1,
            })
            .inst(InstructionKind::I32Const { value: 99 }) // Should be skipped
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 88 })
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32, ValueType::I32])
            .expect_stack(vec![Value::I32(42), Value::I32(88)]);
    }

    #[test]
    fn br_table_index_1() {
        // Branch to second label in table (outer block)
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 1 }) // index 1 -> label 1
            .inst(InstructionKind::BrTable {
                labels: vec![0, 1],
                default: 0,
            })
            .inst(InstructionKind::I32Const { value: 99 }) // Should be skipped
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 88 }) // Should be skipped
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn br_table_default() {
        // Index out of bounds, use default
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 5 }) // index out of bounds
            .inst(InstructionKind::BrTable {
                labels: vec![0, 1],
                default: 1,
            })
            .inst(InstructionKind::I32Const { value: 99 }) // Should be skipped
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 88 }) // Should be skipped
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn br_table_negative_index() {
        // Negative index uses default
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: -1 }) // negative index
            .inst(InstructionKind::BrTable {
                labels: vec![0],
                default: 0,
            })
            .inst(InstructionKind::I32Const { value: 99 }) // Should be skipped
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn br_table_single_label() {
        // Table with only one label plus default
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::BrTable {
                labels: vec![0],
                default: 0,
            })
            .inst(InstructionKind::Drop)
            .inst(InstructionKind::I32Const { value: 99 })
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    #[test]
    fn br_table_simple() {
        // Simple br_table in a single block
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::BrTable {
                labels: vec![0],
                default: 0,
            })
            .inst(InstructionKind::I32Const { value: 99 }) // Should be skipped
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 88 })
            .returns(vec![ValueType::I32, ValueType::I32])
            .expect_stack(vec![Value::I32(42), Value::I32(88)]);
    }

    #[test]
    fn br_table_three_way_branch() {
        // Test 3-way branch with clear signal which path was taken
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                // Label 2 (outermost)
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::Block {
                // Label 1 (middle)
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::Block {
                // Label 0 (innermost)
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 1 }) // index 1 -> middle block
            .inst(InstructionKind::BrTable {
                labels: vec![0, 1, 2],
                default: 2,
            })
            .inst(InstructionKind::I32Const { value: 100 }) // innermost - skipped
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 200 }) // middle - skipped
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 300 }) // outermost - executed
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(300)]);
    }

    #[test]
    fn br_table_with_stack_values() {
        // Test that branch properly handles stack values according to label arity
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 100 })
            .inst(InstructionKind::I32Const { value: 200 })
            .inst(InstructionKind::I32Const { value: 0 }) // index 0 -> inner block (Empty)
            .inst(InstructionKind::BrTable {
                labels: vec![0, 1],
                default: 1,
            })
            .inst(InstructionKind::Drop) // Should be skipped
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 300 }) // Should execute after inner branch
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(300)]); // The 300 value pushed after the branch
    }

    // ============================================================================
    // BrTable Error Tests
    // ============================================================================

    #[test]
    fn br_table_empty_stack_error() {
        // br_table should fail if there's no index on stack
        ExecutorTest::new()
            .inst(InstructionKind::BrTable {
                labels: vec![0],
                default: 0,
            })
            .expect_error("stack underflow");
    }

    #[test]
    fn br_table_invalid_label_depth() {
        // br_table with label depth greater than available labels should fail
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::BrTable {
                labels: vec![5], // Label 5 doesn't exist (no blocks)
                default: 0,
            })
            .expect_error("invalid label");
    }

    #[test]
    fn br_table_invalid_default_label() {
        // br_table with invalid default label
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 99 }) // Out of bounds index
            .inst(InstructionKind::BrTable {
                labels: vec![0],
                default: 5, // Invalid default label
            })
            .expect_error("invalid label");
    }

    #[test]
    fn br_table_mixed_valid_invalid_labels() {
        // Some labels valid, some invalid - should catch during execution
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 1 }) // index 1 -> invalid label 5
            .inst(InstructionKind::BrTable {
                labels: vec![0, 5], // label 5 doesn't exist
                default: 0,
            })
            .expect_error("invalid label");
    }

    #[test]
    fn br_table_very_large_index() {
        // Test with very large index (should use default)
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Value(ValueType::I32),
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::I32Const { value: 1000000 }) // Very large index
            .inst(InstructionKind::BrTable {
                labels: vec![0],
                default: 0,
            })
            .inst(InstructionKind::I32Const { value: 99 }) // Should be skipped
            .inst(InstructionKind::End)
            .returns(vec![ValueType::I32])
            .expect_stack(vec![Value::I32(42)]);
    }

    // ============================================================================
    // Unreachable Tests
    // ============================================================================

    #[test]
    fn unreachable_immediate() {
        // Unreachable should trap immediately
        ExecutorTest::new()
            .inst(InstructionKind::Unreachable)
            .inst(InstructionKind::I32Const { value: 42 }) // Should not be reached
            .expect_error("unreachable");
    }

    #[test]
    fn unreachable_after_value() {
        // Unreachable after pushing a value
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Unreachable)
            .inst(InstructionKind::I32Const { value: 99 }) // Should not be reached
            .expect_error("unreachable");
    }

    #[test]
    fn unreachable_in_block() {
        // Unreachable inside a block
        ExecutorTest::new()
            .inst(InstructionKind::Block {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::Unreachable)
            .inst(InstructionKind::End)
            .inst(InstructionKind::I32Const { value: 42 }) // Should not be reached
            .expect_error("unreachable");
    }

    #[test]
    fn unreachable_in_if_branch() {
        // Unreachable in if branch (taken)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 1 })
            .inst(InstructionKind::If {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::Unreachable)
            .inst(InstructionKind::Else)
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::End)
            .expect_error("unreachable");
    }

    #[test]
    fn unreachable_in_else_branch() {
        // Unreachable in else branch (taken)
        ExecutorTest::new()
            .inst(InstructionKind::I32Const { value: 0 })
            .inst(InstructionKind::If {
                block_type: BlockType::Empty,
            })
            .inst(InstructionKind::I32Const { value: 42 })
            .inst(InstructionKind::Else)
            .inst(InstructionKind::Unreachable)
            .inst(InstructionKind::End)
            .expect_error("unreachable");
    }
}
