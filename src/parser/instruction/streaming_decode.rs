//! Streaming instruction decoder that can build structures incrementally

use super::{DecodeError, Instruction, InstructionIterator, ParseType};
use crate::parser::module::{Locals, Module};
use crate::parser::structure_builder::StructureBuilder;
use crate::parser::structured::StructuredFunction;
use crate::parser::validate::{CodeValidator, Validator};

/// Trait for processing instructions as they're decoded
#[allow(dead_code)]
pub trait InstructionProcessor {
    /// Process a single instruction
    fn process(&mut self, instruction: Instruction) -> Result<(), DecodeError>;

    /// Called when decoding is complete
    fn finalise(self) -> Result<(), DecodeError>;
}

/// Implementation that collects instructions into a Vec (current behaviour)
pub struct VecCollector {
    instructions: Vec<Instruction>,
}

impl VecCollector {
    pub fn new() -> Self {
        VecCollector {
            instructions: Vec::new(),
        }
    }

    pub fn into_instructions(self) -> Vec<Instruction> {
        self.instructions
    }
}

impl Default for VecCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl InstructionProcessor for VecCollector {
    fn process(&mut self, instruction: Instruction) -> Result<(), DecodeError> {
        self.instructions.push(instruction);
        Ok(())
    }

    fn finalise(self) -> Result<(), DecodeError> {
        Ok(())
    }
}

/// Implementation that builds structured representation on the fly
pub struct StreamingStructureProcessor {
    builder: StructureBuilder,
}

impl StreamingStructureProcessor {
    pub fn new(local_count: usize, return_types: Vec<super::super::module::ValueType>) -> Self {
        StreamingStructureProcessor {
            builder: StructureBuilder::new(local_count, return_types),
        }
    }

    pub fn into_function(self) -> Result<StructuredFunction, DecodeError> {
        self.builder.finalise().map_err(DecodeError::Io)
    }
}

impl InstructionProcessor for StreamingStructureProcessor {
    fn process(&mut self, instruction: Instruction) -> Result<(), DecodeError> {
        self.builder.process_instruction(&instruction).map_err(DecodeError::Io)
    }

    fn finalise(self) -> Result<(), DecodeError> {
        // Finalisation happens in into_function
        Ok(())
    }
}

/// Generic decode function that works with any processor
#[allow(dead_code)]
pub fn decode_with_processor<T: Validator, P: InstructionProcessor>(
    validator: &mut T,
    processor: &mut P,
    parse_type: ParseType,
    reader: &mut super::super::reader::Reader,
) -> Result<(), DecodeError> {
    let instruction_iter = InstructionIterator::new(reader, parse_type);

    for result in instruction_iter {
        let instruction = result?;

        validator.validate(&instruction)?;
        processor.process(instruction)?;

        if validator.ended() {
            validator.finalise()?;
            return Ok(());
        }
    }

    validator.finalise()?;
    Ok(())
}

/// Decode a function body into structured representation
pub fn decode_function(
    reader: &mut super::super::reader::Reader,
    module: &Module,
    locals: &Locals,
    function_index: u32,
) -> Result<StructuredFunction, DecodeError> {
    let ftype = module
        .get_function_type(function_index as usize)
        .ok_or(super::super::validate::ValidationError::UnknownFunctionType)?;

    let ctx = module.validation_context();
    let mut validator = CodeValidator::new(module, ctx, locals, ftype, function_index);

    // local_count covers parameters plus declared locals: the code section's
    // locals vector never includes parameters, but StructuredFunction's
    // local_count does (matching the WAT parser).
    let local_count = ftype.parameters.len() + locals.len() as usize;
    let mut processor = StreamingStructureProcessor::new(local_count, ftype.return_types.clone());

    decode_with_processor(&mut validator, &mut processor, ParseType::ReadAll, reader)?;

    processor.into_function()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::instruction::InstructionKind;
    use crate::parser::module::{FunctionType, Module};
    use crate::parser::reader::Reader;

    fn make_test_module() -> Module {
        let mut module = Module::new("test");
        // Add a function type
        module.types.types.push(FunctionType {
            parameters: vec![],
            return_types: vec![],
        });
        // Add a function that references the type
        module
            .functions
            .functions
            .push(crate::parser::module::Function { ftype_index: 0 });
        // Add empty code for the function
        module.code.code.push(crate::parser::module::FunctionBody {
            locals: crate::parser::module::Locals::new(vec![]),
            body: crate::parser::structured::StructuredFunction {
                body: vec![],
                local_count: 0,
                return_types: vec![],
                end_instruction: None,
            },
            position: crate::parser::module::SectionPosition { start: 0, end: 0 },
        });
        module
    }

    #[test]
    fn test_streaming_decode() {
        let module = make_test_module();

        // Create simple bytecode: just "end"
        let bytecode = vec![0x0b]; // end opcode

        let mut reader = Reader::new(bytecode);
        let locals = crate::parser::module::Locals::new(vec![]);

        // Decode to structured function directly
        let result = decode_function(&mut reader, &module, &locals, 0);

        if let Err(e) = &result {
            eprintln!("Decode error: {:?}", e);
        }
        assert!(result.is_ok());
        let func = result.unwrap();

        // Should have an empty body
        assert_eq!(func.body.len(), 0);
    }

    #[test]
    fn test_processor_trait() {
        // Test that the VecCollector works
        let mut collector = VecCollector::new();

        let inst1 = Instruction {
            kind: InstructionKind::I32Const { value: 42 },
            position: crate::parser::instruction::ByteRange { offset: 0, length: 1 },
            original_bytes: vec![],
        };

        collector.process(inst1.clone()).unwrap();

        let inst2 = Instruction {
            kind: InstructionKind::End,
            position: crate::parser::instruction::ByteRange { offset: 1, length: 1 },
            original_bytes: vec![],
        };

        collector.process(inst2.clone()).unwrap();

        let instructions = collector.into_instructions();
        assert_eq!(instructions.len(), 2);
        assert_eq!(instructions[0].kind, InstructionKind::I32Const { value: 42 });
        assert_eq!(instructions[1].kind, InstructionKind::End);
    }
}
