//! Cumulative flat-bytecode dispatch profiles, enabled by `instruction-profile`.
//!
//! Counts are dispatch attempts, not timings. A trapping operation is counted;
//! the operations of a run rejected by the instruction budget are not.
//! Floating constants serialise as their unsigned IEEE bit patterns,
//! preserving NaN payloads and signed zero. Each op also records whether it
//! [exits](Op::exits) a run, its [weight](Op::weight) and the pcs of its
//! [targets](Op::targets), so tools need not know which opcodes branch. Bump
//! `op_format_version` when the serialised Op contract changes.

use super::bytecode::{CompiledFunction, Op};
use crate::parser::module::{ExportIndex, Module, Positional};
use serde::{Serialize, Serializer};

/// CLI capture envelope; fixture hashes are recorded by the capture harness.
#[derive(Serialize)]
pub struct RunProfile<'a> {
    pub module: &'a str,
    pub instance_id: usize,
    pub outcome: ProfileOutcome,
    pub profile: InstructionProfile<'a>,
}

#[derive(Serialize)]
pub struct ProfileOutcome {
    pub exit_code: i32,
    pub trap: Option<String>,
}

#[derive(Serialize)]
pub struct InstructionProfile<'a> {
    pub schema_version: u32,
    pub op_format_version: u32,
    pub count_semantics: &'static str,
    pub functions: Vec<FunctionProfile<'a>>,
}

#[derive(Serialize)]
pub struct FunctionProfile<'a> {
    /// Module-level index, including the imported-function prefix.
    pub function_index: usize,
    pub exports: Vec<&'a str>,
    pub is_start: bool,
    #[serde(serialize_with = "ops_with_properties")]
    pub ops: Vec<&'a Op>,
    /// Indexed by PC in `ops`; cold instructions retain zero counts.
    pub counts: &'a [u64],
}

impl<'a> InstructionProfile<'a> {
    pub(crate) fn new(module: &'a Module, funcs: &'a [CompiledFunction], counts: &'a [Vec<u64>]) -> Self {
        let imported = module.imports.function_count();
        let functions = funcs
            .iter()
            .zip(counts)
            .enumerate()
            .map(|(index, (func, counts))| {
                let function_index = index + imported;
                FunctionProfile {
                    function_index,
                    exports: module
                        .exports
                        .exports
                        .iter()
                        .filter_map(|export| match export.index {
                            ExportIndex::Function(idx) if idx as usize == function_index => Some(export.name.as_str()),
                            _ => None,
                        })
                        .collect(),
                    is_start: module.start.has_position() && module.start.start as usize == function_index,
                    ops: func.ops().collect(),
                    counts,
                }
            })
            .collect();
        Self {
            schema_version: 1,
            op_format_version: 2,
            count_semantics: "dispatch-attempts",
            functions,
        }
    }
}

/// An op as serialised: its opcode and immediates, then its properties.
#[derive(Serialize)]
struct ProfiledOp<'a> {
    #[serde(flatten)]
    op: &'a Op,
    exits: bool,
    weight: u32,
    targets: Vec<u32>,
}

fn ops_with_properties<S: Serializer>(ops: &[&Op], serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(ops.iter().map(|op| ProfiledOp {
        op,
        exits: op.exits(),
        weight: op.weight(),
        targets: op.targets().map(|target| target.pc).collect(),
    }))
}

pub(crate) fn f32_bits<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_u32(value.to_bits())
}

pub(crate) fn f64_bits<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_u64(value.to_bits())
}
