//! CommP SIMD CLI: compute a Filecoin piece commitment from stdin.

use std::io::{self, Read, Write};

fn main() {
    let mut hasher = commp_simd::CommPHasher::new();
    let mut input = io::stdin().lock();
    let mut buffer = [0u8; 8192];

    loop {
        match input.read(&mut buffer) {
            Ok(0) => break,
            Ok(len) => {
                if let Err(error) = hasher.write(&buffer[..len]) {
                    eprintln!("Error hashing stdin: {error}");
                    std::process::exit(1);
                }
            }
            Err(error) => {
                eprintln!("Error reading stdin: {error}");
                std::process::exit(1);
            }
        }
    }

    let root = hasher.root();
    let mut output = [0u8; 65];
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (index, byte) in root.iter().enumerate() {
        output[index * 2] = HEX[(byte >> 4) as usize];
        output[index * 2 + 1] = HEX[(byte & 0x0f) as usize];
    }
    output[64] = b'\n';

    if let Err(error) = io::stdout().lock().write_all(&output) {
        eprintln!("Error writing stdout: {error}");
        std::process::exit(1);
    }
}
