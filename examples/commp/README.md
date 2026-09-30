# CommP - Filecoin Piece Commitment

A WASI example implementing Filecoin's CommP (Piece Commitment) algorithm. This is a real-world cryptographic hash used in Filecoin storage deals.

## Usage

```bash
# Build for WASI
cargo build --target wasm32-wasip1 --release

# Run with krasm
cat file.bin | krasm run target/wasm32-wasip1/release/commp.wasm

# Output: 64-character hex CommP root
ea94b28b4c72336a925aa555376cbca087b9aae7cf16bc69eb19e913106f6f0c
```

## Algorithm Overview

CommP is a SHA256-based binary merkle tree with two quirks required for Filecoin's proof system:

1. **FR32 expansion**: Input is expanded from 127 to 128 bytes per block by inserting 2 zero bits every 254 bits. This ensures all 32-byte chunks are valid BLS12-381 field elements.

2. **254-bit truncation**: All SHA256 hashes have their top 2 bits zeroed, keeping values within the field.

```
Input bytes (streaming)
       ↓
FR32 expand (127 → 128 bytes)
       ↓
Split into 32-byte leaves (no leaf hashing - chunks ARE leaves)
       ↓
Binary merkle tree with SHA256-trunc254 internal nodes
       ↓
32-byte CommP root
```

## Why This Example

- **Real-world algorithm**: Used in production Filecoin since 2020
- **SHA256-heavy**: Good stress test for interpreter performance
- **Streaming input**: Processes stdin in 8KB chunks
- **Pure Rust**: No external dependencies beyond `sha2` crate
- **SIMD-enabled**: Compiled with `+simd128` for SIMD-accelerated SHA256
- **Compact WASM**: ~67KB release binary

## Test Vectors

From the [reference implementation](https://github.com/filecoin-project/go-fil-commp-hashhash):

| Input | CommP |
|-------|-------|
| 127 × 0x42 | `ea94b28b4c72336a925aa555376cbca087b9aae7cf16bc69eb19e913106f6f0c` |
| 254 × 0x42 | `3f3019433e31133007948d56fe896fdbb42b6ecfe430e22728b49ca9355af30b` |
| 508 × 0x42 | `004f6f290bdcc62e84ed8f2c88a3fa713709a5382f70d79ae473c0cdcca7d131` |

## Building

```bash
# Native (for testing)
cargo test --release

# WASI target (with SIMD)
rustup target add wasm32-wasip1
RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-wasip1 --release
```

## SIMD companion

`examples/commp-simd/` is an additive WASI adaptation of
[Hugo Dias's CommP core](https://github.com/hugomrdias/commp/tree/a37a5b2bb1f272d0862e3d59accd318c84ea00e6/rs/commp),
pinned to revision `a37a5b2bb1f272d0862e3d59accd318c84ea00e6`
(MIT OR Apache-2.0). It preserves the four-message SIMD SHA-256, batched FR32
processing and O(log n) tree. The wasm-bindgen interface, JS multihash wrappers,
custom allocator and no_std panic handler are omitted. Rust's standard WASI
environment supplies I/O and allocation.

Both examples read stdin in 8 KB chunks and print the same raw 32-byte root as
64 lowercase hex characters. The SIMD example requires `+simd128` for Wasm;
its native build uses scalar hashing for independent verification. This is a
benchmark of the adapted WASI core, not Hugo's published JS package or its
wasm-bindgen binary. The existing `commp.wasm` remains unchanged.

From the repository root:

```bash
rustup target add wasm32-wasip1
cargo build --release --bin krasm
python3 examples/commp-simd/build.py
python3 scripts/profile.py bench --workload commp-simd --cpu 4
python3 scripts/profile.py sample --workload commp-simd --cpu 4
```

The builder uses a fresh target directory and the example's committed lockfile.
It checks reference vectors and the existing 500 KB profiling input in both
engines before writing `commp-simd.wasm` and `fixture.json`. The metadata pins
the upstream source hash, adapted sources, toolchain, flags, Wasm hash and
expected root. Rebuild after changing the example; the profiler rejects stale
sources or a mismatched binary.

`./check.sh` runs native example tests and builds a fresh SIMD Wasm for
both-engine vector checks. Its `--check` builder invocation does not overwrite
the frozen profiling fixture and does not require the 500 KB benchmark file.
