#!/bin/bash
# Seed fuzzing corpus with .wasm files from the project and spec tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create corpus directories
mkdir -p "$SCRIPT_DIR/corpus/parse_module"
mkdir -p "$SCRIPT_DIR/corpus/execute_module"
mkdir -p "$SCRIPT_DIR/corpus/generate_module"

echo "Seeding corpus from .wasm files..."
wasm_count=0

# Find all .wasm files and copy them to corpus
while IFS= read -r -d '' wasm_file; do
    if [[ -f "$wasm_file" ]]; then
        hash=$(sha1sum "$wasm_file" | cut -d' ' -f1)
        dest="$SCRIPT_DIR/corpus/parse_module/$hash"
        if [[ ! -f "$dest" ]]; then
            cp "$wasm_file" "$dest"
            ((wasm_count++)) || true
        fi
    fi
done < <(find "$PROJECT_ROOT" -name "*.wasm" -type f -print0 2>/dev/null)

echo "  Added $wasm_count .wasm files"

# Extract wasm modules from spec test JSON files
echo "Extracting from spec test JSON files..."
json_count=0

for json_file in "$PROJECT_ROOT"/tests/spec/*.json; do
    if [[ -f "$json_file" ]]; then
        # Extract all .wasm base64 entries from the "bin" object
        # Use Python for reliable JSON + base64 handling
        python3 - "$json_file" "$SCRIPT_DIR/corpus/parse_module" <<'PYTHON'
import json
import sys
import base64
import hashlib
import os

json_path = sys.argv[1]
corpus_dir = sys.argv[2]

try:
    with open(json_path, 'r') as f:
        data = json.load(f)

    bin_section = data.get('bin', {})
    added = 0

    for filename, b64_content in bin_section.items():
        if filename.endswith('.wasm'):
            try:
                wasm_bytes = base64.b64decode(b64_content)
                # Use sha1 as filename (libFuzzer convention)
                hash_name = hashlib.sha1(wasm_bytes).hexdigest()
                dest_path = os.path.join(corpus_dir, hash_name)
                if not os.path.exists(dest_path):
                    with open(dest_path, 'wb') as out:
                        out.write(wasm_bytes)
                    added += 1
            except Exception:
                pass

    if added > 0:
        print(f"  {os.path.basename(json_path)}: {added} modules", file=sys.stderr)
except Exception as e:
    pass
PYTHON
        ((json_count++)) || true
    fi
done

echo "  Processed $json_count JSON files"

# execute_module reserves a 64-byte suffix for typed function arguments.
echo "Seeding execute_module with argument suffixes..."
python3 - "$SCRIPT_DIR/corpus/parse_module" "$SCRIPT_DIR/corpus/execute_module" <<'PYTHON'
import hashlib
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
for path in source.iterdir():
    if path.is_file():
        data = path.read_bytes() + bytes(64)
        target = destination / hashlib.sha1(data).hexdigest()
        if not target.exists():
            target.write_bytes(data)
PYTHON

# Create minimal seeds for generate_module (uses arbitrary, not raw wasm)
echo "Creating minimal seeds for generate_module..."

# These are seed bytes that arbitrary will parse into ModuleConfig structures
printf '\x00\x00\x00\x00\x00\x00\x00\x00' > "$SCRIPT_DIR/corpus/generate_module/seed_empty"
printf '\x01\x00\x02\x01\x00' > "$SCRIPT_DIR/corpus/generate_module/seed_simple"
printf '\x02\x01\x05\x01\x00\x00\x00\x00' > "$SCRIPT_DIR/corpus/generate_module/seed_with_memory"
printf '\x03\x02\x08\x01\x01\x41\x2a' > "$SCRIPT_DIR/corpus/generate_module/seed_with_const"

echo ""
echo "Done! Corpus sizes:"
echo "  parse_module:    $(ls -1 "$SCRIPT_DIR/corpus/parse_module" 2>/dev/null | wc -l) files"
echo "  execute_module:  $(ls -1 "$SCRIPT_DIR/corpus/execute_module" 2>/dev/null | wc -l) files"
echo "  generate_module: $(ls -1 "$SCRIPT_DIR/corpus/generate_module" 2>/dev/null | wc -l) files"
