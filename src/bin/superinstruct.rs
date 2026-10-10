//! Superinstruction analysis tool
//!
//! Analyses WASM modules to find common instruction sequences that could be
//! fused into superinstructions for improved interpreter performance.
//!
//! Usage: cargo run --bin superinstruct -- module.wasm [--min-count N] [--ngram N]

use InstructionKind::*;
use krasm::parser;
use krasm::parser::instruction::InstructionKind;
use krasm::parser::structured::StructuredInstruction;
use std::collections::HashMap;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    fn print_help(prog: &str) {
        eprintln!("Usage: {} <module.wasm> [--min-count N] [--ngram N]", prog);
        eprintln!();
        eprintln!("Analyses instruction sequences to find superinstruction candidates.");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --min-count N   Only show sequences with at least N occurrences (default: 5)");
        eprintln!("  --ngram N       Analyse N-gram sequences (2, 3, or 4; default: all)");
        eprintln!("  --detailed      Show operand patterns for top sequences");
        eprintln!("  -h, --help      Show this help message");
    }

    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        print_help(&args[0]);
        std::process::exit(if args.len() < 2 { 1 } else { 0 });
    }

    let file = &args[1];
    let mut min_count = 5;
    let mut ngram_filter: Option<usize> = None;
    let mut detailed = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--min-count" => {
                i += 1;
                if i < args.len() {
                    min_count = args[i].parse().expect("--min-count requires a number");
                }
            }
            "--ngram" => {
                i += 1;
                if i < args.len() {
                    ngram_filter = Some(args[i].parse().expect("--ngram requires 2, 3, or 4"));
                }
            }
            "--detailed" => {
                detailed = true;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Parse the module
    let bytes = match fs::read(file) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error reading {}: {}", file, e);
            std::process::exit(1);
        }
    };

    let module_registry = HashMap::new();
    let module = match parser::parse(&module_registry, file, &mut parser::reader::Reader::new(bytes)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error parsing {}: {}", file, e);
            std::process::exit(1);
        }
    };

    // Collect all instruction mnemonics from all functions
    let mut all_mnemonics: Vec<Vec<String>> = Vec::new();
    let mut total_instructions = 0usize;

    for func_body in &module.code.code {
        let flattened = func_body.body.flatten();
        let mnemonics: Vec<String> = flattened.iter().map(|inst| inst.kind.mnemonic().to_string()).collect();
        total_instructions += mnemonics.len();
        all_mnemonics.push(mnemonics);
    }

    println!("=== Superinstruction Analysis: {} ===", file);
    println!();
    println!("Functions: {}", all_mnemonics.len());
    println!("Total instructions: {}", total_instructions);
    println!();

    // Analyse N-grams
    let ngrams_to_analyse: Vec<usize> = match ngram_filter {
        Some(n) => vec![n],
        None => vec![2, 3, 4],
    };

    for n in ngrams_to_analyse {
        let counts = count_ngrams(&all_mnemonics, n);
        print_ngram_analysis(&counts, n, min_count, total_instructions);
    }

    // Detailed analysis with operand patterns
    if detailed {
        println!("=== Detailed Operand Analysis ===");
        println!();

        for func_body in &module.code.code {
            analyse_with_operands(&func_body.body);
        }
    }
}

/// Count N-grams across all functions
fn count_ngrams(all_mnemonics: &[Vec<String>], n: usize) -> HashMap<Vec<String>, usize> {
    let mut counts: HashMap<Vec<String>, usize> = HashMap::new();

    for mnemonics in all_mnemonics {
        if mnemonics.len() < n {
            continue;
        }
        for window in mnemonics.windows(n) {
            *counts.entry(window.to_vec()).or_insert(0) += 1;
        }
    }

    counts
}

/// Print analysis for a specific N-gram size
fn print_ngram_analysis(counts: &HashMap<Vec<String>, usize>, n: usize, min_count: usize, total_instructions: usize) {
    let mut sorted: Vec<_> = counts.iter().filter(|(_, count)| **count >= min_count).collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    let total_ngrams: usize = counts.values().sum();

    println!("=== {}-gram Sequences (min count: {}) ===", n, min_count);
    println!();

    if sorted.is_empty() {
        println!("No sequences found with {} or more occurrences.", min_count);
        println!();
        return;
    }

    // Header
    println!("{:>6} {:>6}  Sequence", "Count", "%");
    println!("{}", "-".repeat(60));

    // Top sequences (limit to 30)
    for (sequence, count) in sorted.iter().take(30) {
        let percentage = (**count as f64 / total_ngrams as f64) * 100.0;
        let seq_str = sequence.join(" → ");
        println!("{:>6} {:>5.1}%  {}", count, percentage, seq_str);
    }

    // Summary statistics
    let unique_sequences = counts.len();
    let sequences_above_threshold = sorted.len();
    let instructions_in_top = sorted.iter().take(30).map(|(_, c)| **c * n).sum::<usize>();

    println!();
    println!("Summary:");
    println!("  Unique {}-grams: {}", n, unique_sequences);
    println!(
        "  Sequences with {}+ occurrences: {}",
        min_count, sequences_above_threshold
    );
    if sequences_above_threshold > 0 {
        let coverage = (instructions_in_top as f64 / total_instructions as f64) * 100.0;
        println!("  Top 30 sequences cover: {:.1}% of instructions", coverage);
    }
    println!();
}

/// Analyse instruction patterns with operand details
fn analyse_with_operands(function: &parser::structured::StructuredFunction) {
    let mut patterns: HashMap<String, usize> = HashMap::new();

    fn visit(inst: &StructuredInstruction, prev: &mut Option<String>, patterns: &mut HashMap<String, usize>) {
        let current = format_with_operand_pattern(inst);

        if let Some(p) = prev.as_ref() {
            let pattern = format!("{} → {}", p, current);
            *patterns.entry(pattern).or_insert(0) += 1;
        }

        *prev = Some(current);

        // Recurse into nested structures
        match inst {
            StructuredInstruction::Plain(_) => {}
            StructuredInstruction::Block { body, .. } | StructuredInstruction::Loop { body, .. } => {
                for child in body {
                    visit(child, prev, patterns);
                }
            }
            StructuredInstruction::If {
                then_branch,
                else_branch,
                ..
            } => {
                for child in then_branch {
                    visit(child, prev, patterns);
                }
                if let Some(else_body) = else_branch {
                    for child in else_body {
                        visit(child, prev, patterns);
                    }
                }
            }
        }
    }

    let mut prev = None;
    for inst in &function.body {
        visit(inst, &mut prev, &mut patterns);
    }

    // Print top patterns with operands
    let mut sorted: Vec<_> = patterns.iter().filter(|(_, c)| **c >= 3).collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    for (pattern, count) in sorted.iter().take(20) {
        println!("  {:>4}x  {}", count, pattern);
    }
}

/// Format an instruction with operand pattern info
fn format_with_operand_pattern(inst: &StructuredInstruction) -> String {
    match inst {
        StructuredInstruction::Plain(i) => {
            let mnem = i.kind.mnemonic();
            match &i.kind {
                // Show local/global indices as patterns
                LocalGet { local_idx } => format!("{}[{}]", mnem, local_idx),
                LocalSet { local_idx } => format!("{}[{}]", mnem, local_idx),
                LocalTee { local_idx } => format!("{}[{}]", mnem, local_idx),
                GlobalGet { global_idx } => format!("{}[{}]", mnem, global_idx),
                GlobalSet { global_idx } => format!("{}[{}]", mnem, global_idx),

                // Show small constants, categorise large ones
                I32Const { value } => {
                    if *value >= -128 && *value <= 127 {
                        format!("{}({})", mnem, value)
                    } else {
                        format!("{}(large)", mnem)
                    }
                }
                I64Const { value } => {
                    if *value >= -128 && *value <= 127 {
                        format!("{}({})", mnem, value)
                    } else {
                        format!("{}(large)", mnem)
                    }
                }

                // Show branch depth
                Br { label_idx } => format!("{}[{}]", mnem, label_idx),
                BrIf { label_idx } => format!("{}[{}]", mnem, label_idx),

                // Memory operations - show offset pattern
                I32Load { memarg }
                | I64Load { memarg }
                | I32Load8S { memarg }
                | I32Load8U { memarg }
                | I32Load16S { memarg }
                | I32Load16U { memarg } => {
                    if memarg.offset == 0 {
                        format!("{}[0]", mnem)
                    } else {
                        format!("{}[+]", mnem)
                    }
                }
                I32Store { memarg } | I64Store { memarg } | I32Store8 { memarg } | I32Store16 { memarg } => {
                    if memarg.offset == 0 {
                        format!("{}[0]", mnem)
                    } else {
                        format!("{}[+]", mnem)
                    }
                }

                // Call - show function index
                Call { func_idx } => format!("{}[{}]", mnem, func_idx),

                _ => mnem.to_string(),
            }
        }
        StructuredInstruction::Block { .. } => "block".to_string(),
        StructuredInstruction::Loop { .. } => "loop".to_string(),
        StructuredInstruction::If { .. } => "if".to_string(),
    }
}
