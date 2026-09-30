#!/usr/bin/env python3
"""Build and verify the adapted SIMD CommP WASI workload."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REVISION = "a37a5b2bb1f272d0862e3d59accd318c84ea00e6"
UPSTREAM_SOURCE_SHA256 = "0b8d644353466c67cbc6e2185dff8b1de5997bd15230ba6fbe63b51e55ab52c2"
INPUT_SHA256 = "ec8a9c811f2abd8f233c256a019e8686506b73669439c084479ab4f60fd28979"
EXPECTED = "c1bb8f1985dbf4bf34d06c7190d10a916d228dccd668ba87a10cb1cf0cf3b523"
VECTORS = (
    (127, "ea94b28b4c72336a925aa555376cbca087b9aae7cf16bc69eb19e913106f6f0c"),
    (254, "3f3019433e31133007948d56fe896fdbb42b6ecfe430e22728b49ca9355af30b"),
    (508, "004f6f290bdcc62e84ed8f2c88a3fa713709a5382f70d79ae473c0cdcca7d131"),
    (32768, "cb042c314b73c00e5e1743099e0abb01937bd3e1b8f37d5b4b76f44c6ee9a30c"),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [HERE / name for name in ("Cargo.toml", "Cargo.lock", "build.py")]
    paths += sorted((HERE / "src").glob("*.rs"))
    return {str(path.relative_to(HERE)): sha256(path) for path in paths}


def verify(runner, wasm, data, expected):
    for engine in ("flat", "structured"):
        process = subprocess.run(
            [str(runner), "run", str(wasm), "--engine", engine],
            input=data, capture_output=True, check=True, timeout=120,
        )
        if process.stdout.strip() != expected.encode():
            raise RuntimeError(f"{engine} root mismatch for {len(data)} bytes: {process.stdout!r}")
    print(f"{len(data)} bytes: matching root in both engines", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=ROOT / "target/release/krasm")
    parser.add_argument("--check", action="store_true", help="verify a fresh build without updating the frozen fixture")
    args = parser.parse_args()
    runner = args.runner.resolve()
    if not runner.is_file():
        parser.error(f"missing {runner}; build krasm first or supply --runner")
    try:
        before = source_hashes()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        target = ROOT / "target/commp-simd" / stamp
        env = os.environ.copy()
        for name in list(env):
            if name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS") or name.startswith("CARGO_PROFILE_RELEASE_") or (name.startswith("CARGO_TARGET_") and name.endswith("_RUSTFLAGS")):
                env.pop(name)
        env["RUSTFLAGS"] = "-C target-feature=+simd128"
        command = [
            "cargo", "build", "--manifest-path", str(HERE / "Cargo.toml"),
            "--locked", "--release", "--target", "wasm32-wasip1", "--target-dir", str(target),
        ]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        if source_hashes() != before:
            raise RuntimeError("source changed during the build; run again after edits stop")
        wasm = target / "wasm32-wasip1/release/commp-simd.wasm"
        for size, expected in VECTORS:
            verify(runner, wasm, bytes([0x42]) * size, expected)
        if args.check:
            print("SIMD CommP WASI checks passed; frozen fixture unchanged")
            return
        input_file = ROOT / "benches/commp_bench_500k.bin"
        if sha256(input_file) != INPUT_SHA256:
            raise RuntimeError("the 500 KB profiling input does not match its pinned hash")
        verify(runner, wasm, input_file.read_bytes(), EXPECTED)
        metadata = {
            "upstream": "https://github.com/hugomrdias/commp",
            "revision": REVISION,
            "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
            "wasm_sha256": sha256(wasm),
            "expected_stdout": EXPECTED,
            "build": {
                "rustc": subprocess.check_output(["rustc", "--version", "--verbose"], text=True).strip(),
                "command": command,
                "rustflags": env["RUSTFLAGS"],
            },
            "source_sha256": before,
        }
        shutil.copyfile(wasm, HERE / "commp-simd.wasm")
        (HERE / "fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Frozen fixture: {HERE / 'commp-simd.wasm'} ({metadata['wasm_sha256']})")
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"commp-simd: {error}\n")


if __name__ == "__main__":
    main()
