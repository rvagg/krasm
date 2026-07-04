#!/bin/bash
# Run all checks for KASM project

set -e  # Exit on error

FUZZ_DURATION=60  # Default fuzz duration in seconds
RUN_FUZZ=false
FUZZ_TARGET="parse_module"  # Default fuzz target

usage() {
    echo "Usage: $0 [-f [duration]] [-t target] [-h]"
    echo "  -f [duration]  Run fuzzing after checks (default: ${FUZZ_DURATION}s)"
    echo "  -t target      Fuzz target: parse_module, execute_module, generate_module (default: $FUZZ_TARGET)"
    echo "  -h             Show this help message"
    echo ""
    echo "Fuzz targets:"
    echo "  parse_module    - Fuzz the binary parser with random/malformed bytes"
    echo "  execute_module  - Fuzz parser + execution with typed arguments"
    echo "  generate_module - Structure-aware fuzzing with generated valid modules"
    exit 0
}

while getopts "f:t:h" opt; do
    case $opt in
        f)
            RUN_FUZZ=true
            if [[ -n "$OPTARG" && "$OPTARG" =~ ^[0-9]+$ ]]; then
                FUZZ_DURATION="$OPTARG"
            fi
            ;;
        t)
            FUZZ_TARGET="$OPTARG"
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

# Build AssemblyScript examples if needed (Make-like behavior)
AS_DIR="examples/assemblyscript"
AS_BUILD="$AS_DIR/build/release.wasm"
AS_SOURCES="$AS_DIR/assembly"

needs_rebuild() {
    # Rebuild if output doesn't exist
    if [[ ! -f "$AS_BUILD" ]]; then
        return 0
    fi

    # Rebuild if any source file is newer than output
    if [[ -d "$AS_SOURCES" ]]; then
        for src in "$AS_SOURCES"/*.ts; do
            if [[ -f "$src" && "$src" -nt "$AS_BUILD" ]]; then
                return 0
            fi
        done
    fi

    # No rebuild needed
    return 1
}

if [[ -f "$AS_DIR/package.json" ]]; then
    if needs_rebuild; then
        echo "=== Building AssemblyScript examples ==="
        (cd "$AS_DIR" && npm install --silent && npm run asbuild:all --silent)
    else
        echo "=== AssemblyScript examples up to date ==="
    fi
fi

echo "=== Running cargo check ==="
cargo check

echo -e "\n=== Running rustfmt check ==="
cargo fmt -- --check

echo -e "\n=== Running clippy ==="
cargo clippy --all-targets -- -D warnings

echo -e "\n=== Checking documentation ==="
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --quiet

echo -e "\n=== Running tests ==="
cargo test

# Integration tests: run AssemblyScript examples and verify output
if [[ -f "$AS_DIR/build/mktree.wasm" && -f "$AS_DIR/build/tree.wasm" && -f "$AS_DIR/build/stat.wasm" ]]; then
    echo -e "\n=== Running WASI integration examples ==="
    KASM="cargo run -q --bin krasm --"
    WORK=$(mktemp -d)
    trap 'rm -rf "$WORK"' EXIT

    # mktree: create a known directory structure
    MKTREE_OUT=$($KASM run --dir="$WORK" "$AS_DIR/build/mktree.wasm" 2>&1)
    echo "$MKTREE_OUT" | grep -q "^done$" || { echo "FAIL: mktree did not complete"; echo "$MKTREE_OUT"; exit 1; }
    # Verify all operations returned errno 0
    if echo "$MKTREE_OUT" | grep -v '^done$' | grep -qv ': 0$'; then
        echo "FAIL: mktree had non-zero errno"
        echo "$MKTREE_OUT"
        exit 1
    fi

    # tree: verify the structure matches expected output
    TREE_EXPECTED="sub
sub/deep
sub/deep/nested.txt
sub/empty-dir
sub/greeting.txt"
    TREE_OUT=$($KASM run --dir="$WORK" "$AS_DIR/build/tree.wasm" 2>&1)
    if [[ "$TREE_OUT" != "$TREE_EXPECTED" ]]; then
        echo "FAIL: tree output mismatch"
        echo "Expected:"
        echo "$TREE_EXPECTED"
        echo "Got:"
        echo "$TREE_OUT"
        exit 1
    fi

    # stat: verify file metadata
    STAT_OUT=$($KASM run --dir="$WORK" "$AS_DIR/build/stat.wasm" -- sub/greeting.txt 2>&1)
    echo "$STAT_OUT" | grep -q "type: regular" || { echo "FAIL: stat type"; echo "$STAT_OUT"; exit 1; }
    echo "$STAT_OUT" | grep -q "size: 14" || { echo "FAIL: stat size"; echo "$STAT_OUT"; exit 1; }
    echo "$STAT_OUT" | grep -q "fd_size: 14" || { echo "FAIL: stat fd_size"; echo "$STAT_OUT"; exit 1; }

    # cat: read a file created by mktree
    CAT_OUT=$($KASM run --dir="$WORK" "$AS_DIR/build/cat.wasm" -- sub/greeting.txt 2>&1)
    if [[ "$CAT_OUT" != "Hello, world!" ]]; then
        echo "FAIL: cat output mismatch"
        echo "Expected: Hello, world!"
        echo "Got: $CAT_OUT"
        exit 1
    fi

    # wc: count the file
    WC_OUT=$($KASM run --dir="$WORK" "$AS_DIR/build/wc.wasm" -- sub/greeting.txt 2>&1)
    echo "$WC_OUT" | grep -q "1.*2.*14" || { echo "FAIL: wc counts"; echo "$WC_OUT"; exit 1; }

    echo "  mktree + tree + stat + cat + wc: ok"
    rm -rf "$WORK"
    trap - EXIT
fi

echo -e "\n=== All checks passed! ==="

if $RUN_FUZZ; then
    echo -e "\n=== Running fuzzer ($FUZZ_TARGET) for ${FUZZ_DURATION}s ==="
    if ! command -v cargo-fuzz &> /dev/null; then
        echo "cargo-fuzz not installed. Install with: cargo install cargo-fuzz"
        exit 1
    fi
    cd fuzz

    # Build arguments based on target
    FUZZ_ARGS=("-max_total_time=$FUZZ_DURATION")

    # Use dictionary for byte-based targets
    if [[ "$FUZZ_TARGET" == "parse_module" || "$FUZZ_TARGET" == "execute_module" ]]; then
        if [[ -f "wasm.dict" ]]; then
            FUZZ_ARGS+=("-dict=wasm.dict")
        fi
    fi

    if cargo +nightly fuzz run "$FUZZ_TARGET" -- "${FUZZ_ARGS[@]}"; then
        echo -e "\n=== Fuzzing complete, no issues found ==="
    else
        echo -e "\n=== Fuzzing found issues (see above) ==="
        exit 1
    fi
fi
