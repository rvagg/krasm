#!/usr/bin/env python3
"""Fetch, build and qualify local Preview 1 profiling guests (Python 3.12+).

Artifacts and generated registry stay under target/corpus. Profiling never fetches.
Use: prepare_corpus.py fetch|build|qualify [--program NAME ...]
Then: profile.py collect --registry target/corpus/registry.json --corpus qualified
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "target/corpus"
PINS = ROOT / "scripts/corpus_sources.json"
PROGRAMS = ("nbody", "markdown", "zstd", "jq", "sqlite")


def sha(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def archive(name, pins):
    spec = pins["sources"][name]
    path = CACHE / "downloads" / spec["archive"]
    algorithm = "sha3_256" if "sha3_256" in spec else "sha256"
    if sha(path, algorithm) != spec[algorithm]:
        raise RuntimeError(f"source digest mismatch: {path}")
    return path


def extract(path, destination):
    """Only extract confined data; ZIP links/devices are not accepted."""
    destination.mkdir(parents=True, exist_ok=False)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as bundle:
            for member in bundle.infolist():
                relative = Path(member.filename)
                kind = (member.external_attr >> 16) & 0o170000
                if relative.is_absolute() or ".." in relative.parts or kind not in (0, 0o040000, 0o100000):
                    raise RuntimeError(f"unsafe ZIP member: {member.filename}")
            bundle.extractall(destination)
    else:
        with tarfile.open(path) as bundle:
            bundle.extractall(destination, filter="data")


def environment():
    env = os.environ.copy()
    for key in list(env):
        if key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS",
                   "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
                   "CC", "CXX", "AR", "RANLIB", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "LIBS") or key.startswith(("CARGO_TARGET_", "CARGO_PROFILE_")):
            env.pop(key)
    env.update({"LC_ALL": "C", "TZ": "UTC", "SOURCE_DATE_EPOCH": "0"})
    return env


def run(command, cwd, env, record):
    command = [str(part) for part in command]
    build_env = {key: value for key, value in env.items()
                 if key in ("CC", "CXX", "AR", "RANLIB", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "LIBS",
                            "LC_ALL", "TZ", "SOURCE_DATE_EPOCH", "RUSTFLAGS", "RUSTC", "RUSTUP_TOOLCHAIN")
                 or key.startswith(("CARGO_TARGET_", "CARGO_PROFILE_", "CARGO_BUILD_"))}
    record.append({"argv": command, "cwd": str(cwd), "environment": build_env})
    print("Running:", " ".join(command), flush=True)
    with (Path(cwd) / "corpus-build.log").open("ab") as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {cwd}/corpus-build.log")


def rust_toolchain(pins):
    version = subprocess.check_output(["rustc", "--version"], text=True).split()[1]
    if version != pins["rust_version"]:
        raise RuntimeError(f"select Rust {pins['rust_version']} before building; found {version}")


def fetch(names, pins):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("the pinned SDK archive requires Linux x86_64")
    for name in ["sdk", *names]:
        spec = pins["sources"][name]
        destination = CACHE / "downloads" / spec["archive"]
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + ".partial")
            print(f"Fetching {spec['url']}", flush=True)
            with urllib.request.urlopen(spec["url"], timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            algorithm = "sha3_256" if "sha3_256" in spec else "sha256"
            if sha(partial, algorithm) != spec[algorithm]:
                raise RuntimeError(f"download digest mismatch: {partial}")
            partial.rename(destination)
        archive(name, pins)
        print(f"Verified {name}: {destination}", flush=True)
    if "markdown" in names:
        rust_toolchain(pins)
        directory = new_build("cargo-fetch")
        extract(archive("markdown", pins), directory / "source")
        source = directory / "source" / pins["sources"]["markdown"]["root"]
        run(["cargo", "fetch", "--locked"], source, environment(), [])


def new_build(name):
    directory = CACHE / "build"
    directory.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=name + "-", dir=directory))


def sdk_path(pins):
    bundle = archive("sdk", pins)
    destination = CACHE / "toolchains" / pins["sources"]["sdk"]["sha256"]
    if not destination.exists():
        temporary = Path(tempfile.mkdtemp(prefix="sdk-", dir=new_build("toolchain"))) / "unpack"
        extract(bundle, temporary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
    return destination / pins["sources"]["sdk"]["root"]


def build(name, pins, jobs):
    directory = new_build(name)
    shutil.copyfile(__file__, directory / "prepare_corpus.py")
    shutil.copyfile(PINS, directory / "corpus_sources.json")
    extract(archive(name, pins), directory / "source")
    source = directory / "source" / pins["sources"][name]["root"]
    sdk = sdk_path(pins)
    env = environment()
    commands = []
    native = directory / name
    wasm = directory / (name + ".wasm")
    cc = sdk / "bin/clang"
    cflags = ["--target=wasm32-wasip1", "-O2", "-Wl,-z,stack-size=8388608"]
    if name == "nbody":
        member = source / pins["sources"][name]["member"]
        if sha(member) != pins["sources"][name]["member_sha256"]:
            raise RuntimeError("n-body member digest mismatch")
        for compiler, flags, output in ((cc, cflags, wasm), ("cc", ["-O2"], native)):
            run([compiler, *flags, "-ffp-contract=off", "-x", "c", member, "-lm", "-o", output], directory, env, commands)
    elif name == "sqlite":
        options = ["-DSQLITE_THREADSAFE=0", "-DSQLITE_OMIT_LOAD_EXTENSION", "-DSQLITE_TEMP_STORE=3"]
        for compiler, flags, output in ((cc, cflags, wasm), ("cc", ["-O2"], native)):
            run([compiler, *flags, *options, "-I", source, ROOT / "scripts/corpus_sqlite.c", source / "sqlite3.c", "-lm", "-o", output], directory, env, commands)
    elif name == "markdown":
        rust_toolchain(pins)
        env["CARGO_TARGET_DIR"] = str(directory / "cargo")
        env["CARGO_TARGET_WASM32_WASIP1_LINKER"] = str(sdk / "bin/clang")
        env["RUSTFLAGS"] = f"--remap-path-prefix={directory}=/corpus/markdown"
        for target, output in (("wasm32-wasip1", wasm), ("x86_64-unknown-linux-gnu", native)):
            run(["cargo", "build", "--offline", "--locked", "--release", "--target", target,
                 "-p", "pulldown-cmark", "--bin", "pulldown-cmark"], source, env, commands)
            suffix = ".wasm" if target == "wasm32-wasip1" else ""
            shutil.copyfile(directory / "cargo" / target / "release" / ("pulldown-cmark" + suffix), output)
            output.chmod(0o755)
    elif name == "zstd":
        for guest, output in ((True, wasm), (False, native)):
            destination = directory / ("wasm-build" if guest else "native-build")
            options = ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_FLAGS_RELEASE=-O2",
                       "-DZSTD_BUILD_PROGRAMS=OFF", "-DZSTD_BUILD_STATIC=ON"]
            options += [f"-D{flag}=OFF" for flag in ("ZSTD_MULTITHREAD_SUPPORT", "ZSTD_BUILD_SHARED", "ZSTD_BUILD_TESTS", "ZSTD_BUILD_CONTRIB", "ZSTD_ZLIB_SUPPORT", "ZSTD_LZMA_SUPPORT", "ZSTD_LZ4_SUPPORT")]
            if guest:
                options += [f"-DCMAKE_TOOLCHAIN_FILE={sdk}/share/cmake/wasi-sdk-p1.cmake"]
            run(["cmake", "-S", source / "build/cmake", "-B", destination, *options], directory, env, commands)
            run(["cmake", "--build", destination, "--target", "libzstd_static", "-j", jobs], directory, env, commands)
            compiler, flags = (cc, cflags) if guest else ("cc", ["-O2"])
            run([compiler, *flags, "-I", source / "lib", ROOT / "scripts/corpus_zstd.c",
                 destination / "lib/libzstd.a", "-o", output], directory, env, commands)
    elif name == "jq":
        for guest, output in ((True, wasm), (False, native)):
            destination = directory / ("wasm-build" if guest else "native-build")
            destination.mkdir()
            build_env = env.copy()
            options = ["--disable-docs", "--disable-valgrind", "--disable-maintainer-mode", "--disable-shared", "--enable-static", "--enable-all-static", "--without-oniguruma"]
            if guest:
                build_env.update({"CC": f"{cc} --target=wasm32-wasip1", "AR": str(sdk / "bin/llvm-ar"),
                                  "RANLIB": str(sdk / "bin/llvm-ranlib"), "CFLAGS": "-O2 -D_WASI_EMULATED_SIGNAL",
                                  "LDFLAGS": "-Wl,-z,stack-size=8388608", "LIBS": "-lwasi-emulated-signal"})
                options += ["--host=wasm32-unknown-wasi"]
            else:
                build_env.update({"CC": "cc", "CFLAGS": "-O2"})
            run([Path(os.path.relpath(source / "configure", destination)), *options], destination, build_env, commands)
            run(["make", "-j", jobs], destination, build_env, commands)
            shutil.copy2(destination / "jq", output)
    record = {
        "program": name, "source": pins["sources"][name], "sdk": pins["sources"]["sdk"],
        "commands": commands, "environment": {key: env[key] for key in ("LC_ALL", "TZ", "SOURCE_DATE_EPOCH")},
        "compiler": subprocess.check_output([str(cc), "--version"], text=True),
        "native_compiler": subprocess.check_output(["cc", "--version"], text=True),
        "rustc": subprocess.check_output(["rustc", "--version", "--verbose"], text=True),
        "build_tools": {tool: subprocess.check_output([tool, "--version"], text=True).splitlines()[0]
                        for tool in ("make", "cmake", "cargo")},
        "recipe_sha256": sha(directory / "prepare_corpus.py"),
        "pins_sha256": sha(directory / "corpus_sources.json"),
        "adapter_sha256": sha(ROOT / f"scripts/corpus_{name}.c") if name in ("sqlite", "zstd") else None,
        "module": str(wasm.relative_to(ROOT)), "module_sha256": sha(wasm),
        "native": str(native), "native_sha256": sha(native), "source_directory": str(source),
    }
    record_path = CACHE / "artifacts" / f"{name}.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        record["previous_module_sha256"] = previous["module_sha256"]
        record["reproduced"] = previous["module_sha256"] == record["module_sha256"]
        print(f"{name}: clean rebuild identical: {record['reproduced']}")
    save(directory / "build-record.json", record)
    save(record_path, record)
    print(f"Built {name}: {wasm}", flush=True)


def scenarios(name, record):
    if name == "nbody":
        return [("nbody-short", "numeric", ["1000"], b""), ("nbody-long", "numeric", ["100000"], b"")]
    if name == "markdown":
        base = Path(record["source_directory"]) / "pulldown-cmark/third_party"
        prose = (base / "xi-editor/crdt.md").read_bytes()
        nested = ((base / "markdown-it/block-list-nested.md").read_bytes() + b"\n\n"
                  + (base / "markdown-it/inline-em-nested.md").read_bytes() + b"\n\n") * 128
        return [("markdown-prose", "markdown", [], prose), ("markdown-nested", "markdown", [], nested)]
    if name == "jq":
        records = [{"id": i, "group": i % 97, "value": (i * 17) % 1009, "active": i % 3 != 0,
                    "detail": {"label": f"record-{i}", "tags": ["corpus", str(i % 11)]}} for i in range(10000)]
        encode = lambda value: (json.dumps(value, separators=(",", ":")) + "\n").encode()
        project = ".[] | select(.active) | {id, label: .detail.label, value}"
        group = "group_by(.group) | map({group: .[0].group, total: (map(.value) | add)}) | sort_by(.total, .group)"
        return [("jq-project", "json", ["-c", "-S", project], encode(records[:100])), ("jq-group", "json", ["-c", "-S", group], encode(records))]
    if name == "zstd":
        data = "".join(f"event={i} group={i % 97} value={(i * 17) % 1009} action=store status=success\n" for i in range(10000)).encode()
        frame = subprocess.run([record["native"], "compress"], input=data, capture_output=True, check=True).stdout
        return [("zstd-compress", "compression", ["compress"], data),
                ("zstd-decompress", "compression", ["decompress"], frame)]
    sql = """PRAGMA temp_store=MEMORY;
BEGIN;
CREATE TABLE records(id INTEGER PRIMARY KEY, category INTEGER, value INTEGER);
WITH RECURSIVE n(i) AS (VALUES(0) UNION ALL SELECT i+1 FROM n WHERE i<9999)
INSERT INTO records SELECT i, i%97, (i*17)%1009 FROM n;
CREATE INDEX categories ON records(category);
COMMIT;
"""
    indexed = "SELECT id,value FROM records WHERE category=42 ORDER BY id;"
    aggregate = "SELECT a.category,count(*),sum(a.value+b.value) FROM records a JOIN records b ON b.id=a.id/2 GROUP BY a.category ORDER BY a.category;"
    return [("sqlite-index", "database", [], (sql + indexed).encode()), ("sqlite-aggregate", "database", [], (sql + aggregate).encode())]


def qualify(name, krasm, registry):
    record = json.loads((CACHE / "artifacts" / f"{name}.json").read_text())
    module = ROOT / record["module"]
    if sha(module) != record["module_sha256"] or sha(Path(record["native"])) != record["native_sha256"]:
        raise RuntimeError("built artifact changed since its build record")
    report_dir = new_build(name + "-qualification")
    dump = subprocess.run([str(krasm), "dump", str(module)], capture_output=True, check=True)
    (report_dir / "module.txt").write_bytes(dump.stdout)
    imports = [line.rsplit(" <- ", 1)[1] for line in dump.stdout.decode().splitlines()
               if line.startswith(" - ") and " <- " in line]
    if not imports or any(not symbol.startswith("wasi_snapshot_preview1.") for symbol in imports):
        raise RuntimeError(f"guest imports are not exclusively Preview 1: {imports}")
    save(report_dir / "imports.json", imports)
    entries = {}
    for identifier, family, args, data in scenarios(name, record):
        input_path = report_dir / (identifier + ".input")
        input_path.write_bytes(data)
        oracle = subprocess.run([record["native"], *args], input=data, capture_output=True, timeout=120, check=True).stdout
        timings = {}
        output = None
        for engine in ("structured", "flat"):
            start = time.perf_counter()
            result = subprocess.run([str(krasm), "run", str(module), "--engine", engine, "--", *args], input=data, capture_output=True, timeout=120)
            (report_dir / f"{identifier}-{engine}.stderr").write_bytes(result.stderr)
            if result.returncode:
                raise RuntimeError(f"{identifier}/{engine}: {result.stderr.decode(errors='replace')}")
            if identifier == "zstd-compress":
                decoded = subprocess.run([record["native"], "decompress"], input=result.stdout, capture_output=True, check=True).stdout
                if decoded != data:
                    raise RuntimeError("native zstd cannot round-trip guest compression")
            elif result.stdout != oracle:
                raise RuntimeError(f"native output mismatch: {identifier}/{engine}")
            if output is not None and output != result.stdout:
                raise RuntimeError(f"engine output mismatch: {identifier}")
            output = result.stdout
            timings[engine] = time.perf_counter() - start
        expected = hashlib.sha256(output).hexdigest()
        previous = registry["workloads"].get(identifier)
        if previous and previous["expected_stdout_sha256"] != expected:
            raise RuntimeError(f"accepted output changed for {identifier}; inspect before using a new registry")
        entries[identifier] = {"family": family, "compiler": {"language": "Rust" if name == "markdown" else "C", "version": record["rustc"] if name == "markdown" else record["compiler"]},
                               "module": record["module"], "module_sha256": record["module_sha256"],
                               "input": str(input_path.relative_to(ROOT)), "input_sha256": sha(input_path), "args": args,
                               "expected_stdout_sha256": expected, "build_provenance": record}
        (report_dir / (identifier + ".stdout")).write_bytes(output)
        save(report_dir / (identifier + ".json"), {"seconds": timings, "stdout_sha256": expected, "oracle": "native round-trip" if identifier == "zstd-compress" else "native exact bytes"})
        print(f"Qualified {identifier}: {timings}", flush=True)
    registry["workloads"].update(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "build", "qualify"))
    parser.add_argument("--program", action="append", choices=PROGRAMS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--krasm", type=Path, default=ROOT / "target/release/krasm")
    parser.add_argument("--registry", type=Path, default=CACHE / "registry.json")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    pins = json.loads(PINS.read_text())
    names = args.program or PROGRAMS
    if args.action == "fetch":
        fetch(names, pins)
        return
    registry = json.loads(args.registry.read_text()) if args.registry.exists() else json.loads((ROOT / "scripts/profile_workloads.json").read_text())
    failures = {}
    for name in names:
        try:
            if args.action == "build":
                build(name, pins, args.jobs)
            else:
                qualify(name, args.krasm.resolve(), registry)
                registry["corpora"]["qualified"] = list(registry["workloads"])
                registry["corpora"]["training"] = [key for key, value in registry["workloads"].items() if value["family"] != "database"]
                registry["corpora"]["holdout"] = [key for key, value in registry["workloads"].items() if value["family"] == "database"]
                save(args.registry, registry)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            failures[name] = str(error)
            print(f"FAILED {name}: {error}", flush=True)
    save(CACHE / f"{args.action}-failures.json", failures)
    if failures:
        parser.exit(1, f"{len(failures)} program(s) failed; see {CACHE}/{args.action}-failures.json\n")


if __name__ == "__main__":
    main()
