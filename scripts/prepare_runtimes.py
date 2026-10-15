#!/usr/bin/env python3
"""Fetch and build the pinned external runtimes for comparison (Python 3.12+).

Wasmtime comes from its release archive; Wasmi is built from its crates.io
source with its own lock file, once per dispatch backend. Everything stays
under target/runtimes, described by target/runtimes/runtimes.json.
Use: prepare_runtimes.py
Then: profile.py external --cpu N --krasm target/release/krasm
"""

import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

from prepare_corpus import environment, extract, rust_toolchain, save, sha

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "target/runtimes"
PINS = ROOT / "scripts/runtime_sources.json"
MANIFEST = CACHE / "runtimes.json"


def download(spec):
    destination = CACHE / "downloads" / spec["archive"]
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        print(f"Fetching {spec['url']}", flush=True)
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "krasm prepare_runtimes.py"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.rename(destination)
    if sha(destination) != spec["sha256"]:
        raise RuntimeError(f"download digest mismatch: {destination}")
    return destination


def unpack(spec, directory):
    """Extract a verified archive once; a later run reuses the directory."""
    archive = download(spec)
    if not directory.exists():
        extract(archive, directory)
    return directory


def wasmtime(spec):
    directory = unpack(spec, CACHE / f"wasmtime-{spec['version']}")
    return [{"name": "wasmtime", "version": spec["version"], "binary": directory / spec["binary"]}]


def wasmi(spec, env):
    source = unpack(spec, CACHE / f"wasmi-{spec['version']}-source") / f"{spec['crate']}-{spec['version']}"
    built = []
    for backend, flags in spec["builds"].items():
        target = CACHE / f"wasmi-{spec['version']}-{backend}"
        command = ["cargo", "build", "--release", "--locked", *flags, "--target-dir", str(target)]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=source, env=env, check=True)
        built.append({"name": f"wasmi-{backend}", "version": spec["version"], "build": flags,
                      "binary": target / "release" / spec["binary"]})
    return built


def main():
    pins = json.loads(PINS.read_text())
    if pins.get("schema_version") != 1:
        raise RuntimeError("unsupported runtime pins version")
    rust_toolchain(pins)
    runtimes = pins["runtimes"]
    built = wasmtime(runtimes["wasmtime"]) + wasmi(runtimes["wasmi"], environment())
    manifest = {
        "pins_sha256": sha(PINS),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "runtimes": {
            entry["name"]: {**entry, "binary": str(entry["binary"].relative_to(ROOT)), "sha256": sha(entry["binary"])}
            for entry in built
        },
    }
    save(MANIFEST, manifest)
    for name, entry in manifest["runtimes"].items():
        print(f"{name} {entry['version']}: {entry['binary']}")


if __name__ == "__main__":
    main()
