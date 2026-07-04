//! CLI entry point for krasm.
//!
//! Provides three subcommands:
//! - `run` -- execute a WASI module (`.wasm` or `.wat`)
//! - `dump` -- display module structure (header, details, disassembly)
//! - `compile` -- compile WAT text format to `.wasm` binary

use clap::{Parser, Subcommand, ValueEnum};
use krasm::parser;
use krasm::wasi::{WasiContext, create_wasi_instance};
use krasm::{EngineKind, Module, Store};
use std::collections::HashMap;
use std::fs;
use std::io::{stderr, stdin, stdout};
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "krasm")]
#[command(about = "WebAssembly runtime and toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute a WASI module (.wasm or .wat)
    Run {
        /// Path to the WebAssembly module
        file: String,

        /// Preopen a directory for filesystem access
        #[arg(long = "dir", value_name = "PATH")]
        dirs: Vec<String>,

        /// Interpreter engine (flat is experimental with partial coverage)
        #[arg(long, value_enum, default_value_t = EngineArg::Structured)]
        engine: EngineArg,

        /// Arguments to pass to the module (after --)
        #[arg(last = true)]
        args: Vec<String>,
    },

    /// Dump module information (defaults to detailed view)
    Dump {
        /// Path to the WebAssembly module (.wasm or .wat)
        file: String,

        /// Show only module header (magic, version)
        #[arg(long)]
        header: bool,

        /// Show disassembled code
        #[arg(long, short = 'd')]
        disassemble: bool,
    },

    /// Compile a WAT file to WebAssembly binary
    Compile {
        /// Path to the WAT source file
        file: String,

        /// Output path (defaults to input with .wasm extension)
        #[arg(short, long)]
        output: Option<String>,
    },
}

/// Load and parse a WebAssembly module from a `.wasm` or `.wat` file.
fn load_module(file: &str) -> Result<Module, String> {
    if file.ends_with(".wat") {
        let source = fs::read_to_string(file).map_err(|e| format!("Error reading {}: {}", file, e))?;
        krasm::wat::parse(&source).map_err(|e| format!("Error parsing {}: {}", file, e))
    } else {
        let bytes = fs::read(file).map_err(|e| format!("Error reading {}: {}", file, e))?;
        let module_registry = HashMap::new();
        parser::parse(&module_registry, file, &mut parser::reader::Reader::new(bytes))
            .map_err(|e| format!("Error parsing {}: {}", file, e))
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            file,
            dirs,
            engine,
            args,
        } => run_wasi_module(&file, dirs, engine, args),
        Commands::Dump {
            file,
            header,
            disassemble,
        } => dump_module(&file, header, disassemble),
        Commands::Compile { file, output } => compile_module(&file, output),
    }
}

fn dump_module(file: &str, header: bool, disassemble: bool) -> ExitCode {
    let module = match load_module(file) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::FAILURE;
        }
    };

    let format = if header {
        parser::module::ParsedUnitFormat::Header
    } else if disassemble {
        parser::module::ParsedUnitFormat::Disassemble
    } else {
        parser::module::ParsedUnitFormat::Details
    };

    println!("{}", module.to_string(format));
    ExitCode::SUCCESS
}

fn compile_module(file: &str, output: Option<String>) -> ExitCode {
    let source = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {}", file, e);
            return ExitCode::FAILURE;
        }
    };

    let module = match krasm::wat::parse(&source) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error parsing {}: {}", file, e);
            return ExitCode::FAILURE;
        }
    };

    let bytes = match krasm::encoder::encode(&module) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error encoding {}: {}", file, e);
            return ExitCode::FAILURE;
        }
    };

    let output_path = output.unwrap_or_else(|| {
        let p = Path::new(file);
        p.with_extension("wasm").to_string_lossy().into_owned()
    });

    if let Err(e) = fs::write(&output_path, &bytes) {
        eprintln!("Error writing {}: {}", output_path, e);
        return ExitCode::FAILURE;
    }

    eprintln!("{}: {} bytes", output_path, bytes.len());
    ExitCode::SUCCESS
}

/// CLI-facing engine choice, mapped onto [`EngineKind`].
#[derive(Clone, Copy, ValueEnum)]
enum EngineArg {
    Structured,
    Flat,
}

impl From<EngineArg> for EngineKind {
    fn from(arg: EngineArg) -> Self {
        match arg {
            EngineArg::Structured => EngineKind::Structured,
            EngineArg::Flat => EngineKind::Flat,
        }
    }
}

fn run_wasi_module(file: &str, dirs: Vec<String>, engine: EngineArg, module_args: Vec<String>) -> ExitCode {
    let module = match load_module(file) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::FAILURE;
        }
    };

    // Build args array: program name + provided args
    let mut args = vec![file.to_string()];
    args.extend(module_args);

    let mut builder = WasiContext::builder()
        .args(args)
        .stdin(Box::new(stdin()))
        .stdout(Box::new(stdout()))
        .stderr(Box::new(stderr()));

    for dir in &dirs {
        let path = std::path::Path::new(dir);
        if !path.is_dir() {
            eprintln!("Error: --dir path is not a directory: {}", dir);
            return ExitCode::FAILURE;
        }
        builder = builder.preopen_dir(dir, dir);
    }

    let ctx = Arc::new(builder.build());

    let mut store = Store::new();
    store.set_engine(engine.into());
    let module = Arc::new(module);
    let instance_id = match create_wasi_instance(&mut store, module, ctx.clone(), true) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Error instantiating module: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // WASI modules must export _start as the entry point
    let has_start = store
        .get_instance(instance_id)
        .map(|i| i.get_function_addr("_start").is_ok())
        .unwrap_or(false);

    if !has_start {
        eprintln!("Error: module does not export _start (not a WASI command module)");
        return ExitCode::FAILURE;
    }

    let result = store.invoke_export(instance_id, "_start", vec![], None);

    // proc_exit() raises a trap to halt execution, so check for an exit
    // code in both the Ok and Err arms.
    match result {
        Ok(_) => {
            if let Some(code) = ctx.exit_code() {
                ExitCode::from(code as u8)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            if let Some(code) = ctx.exit_code() {
                ExitCode::from(code as u8)
            } else {
                eprintln!("Error: {}", e);
                ExitCode::FAILURE
            }
        }
    }
}
