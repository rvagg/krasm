#!/usr/bin/env python3
"""Repeatable Linux CommP timing and intrusive GDB sampling (Python 3.9+)."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
COMMP_WASM = ROOT / "examples/commp/commp.wasm"
INPUT = ROOT / "benches/commp_bench_500k.bin"
COMMP_EXPECTED = "c1bb8f1985dbf4bf34d06c7190d10a916d228dccd668ba87a10cb1cf0cf3b523"
COMMP_FIXTURE_HASHES = {
    COMMP_WASM: "23bc184b68ad3edd8e4e444fc4cbaf6193d31cf86cdbefda7f8464fe36b83fdc",
    INPUT: "ec8a9c811f2abd8f233c256a019e8686506b73669439c084479ab4f60fd28979",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def load_workload(name):
    if name == "commp":
        return {
            "name": name,
            "wasm": COMMP_WASM,
            "expected_stdout": COMMP_EXPECTED,
            "fixture_hashes": COMMP_FIXTURE_HASHES,
            "metadata": None,
            "metadata_sha256": None,
            "source_hashes": None,
        }

    fixture = ROOT / "examples/commp-simd/fixture.json"
    try:
        metadata = json.loads(fixture.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(f"SIMD workload fixture metadata is missing: {fixture}") from None
    except json.JSONDecodeError as error:
        raise RuntimeError(f"SIMD workload fixture metadata is invalid: {fixture}: {error}") from None
    if not isinstance(metadata, dict):
        raise RuntimeError(f"SIMD workload fixture metadata must be an object: {fixture}")
    required = ("upstream", "revision", "upstream_source_sha256", "wasm_sha256", "expected_stdout", "build", "source_sha256")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise RuntimeError(f"SIMD workload fixture metadata is missing {', '.join(missing)}: {fixture}")
    if not all(isinstance(metadata[key], str) and metadata[key] for key in ("upstream", "revision")):
        raise RuntimeError(f"SIMD workload fixture metadata has invalid pinned fields: {fixture}")
    if not isinstance(metadata["build"], (str, dict)) or not metadata["build"]:
        raise RuntimeError(f"SIMD workload fixture metadata has invalid build provenance: {fixture}")
    if not all(is_sha256(metadata[key]) for key in ("upstream_source_sha256", "wasm_sha256")):
        raise RuntimeError(f"SIMD workload fixture metadata has invalid SHA-256 fields: {fixture}")
    if not is_sha256(metadata["expected_stdout"]):
        raise RuntimeError(f"SIMD workload fixture metadata has invalid expected stdout: {fixture}")
    if not isinstance(metadata["source_sha256"], dict) or not metadata["source_sha256"]:
        raise RuntimeError(f"SIMD workload fixture metadata has no source SHA-256 entries: {fixture}")
    if not all(isinstance(path, str) and is_sha256(value) for path, value in metadata["source_sha256"].items()):
        raise RuntimeError(f"SIMD workload fixture metadata has invalid source SHA-256 entries: {fixture}")
    wasm = ROOT / "examples/commp-simd/commp-simd.wasm"
    return {
        "name": name,
        "wasm": wasm,
        "expected_stdout": metadata["expected_stdout"],
        "fixture_hashes": {wasm: metadata["wasm_sha256"], INPUT: COMMP_FIXTURE_HASHES[INPUT]},
        "metadata": metadata,
        "metadata_sha256": digest(fixture),
        "metadata_path": fixture,
        "source_hashes": metadata["source_sha256"],
    }


def verify_hash(path, expected, label):
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    if digest(path) != expected:
        raise RuntimeError(f"{label} differs from its recorded SHA-256: {path}")


def workload_source_paths(workload):
    if workload["source_hashes"] is None:
        return {}
    paths = {}
    for name, expected in workload["source_hashes"].items():
        relative = Path(name)
        if relative.is_absolute() or not name or ".." in relative.parts:
            raise RuntimeError(f"SIMD workload fixture metadata has unsafe source path: {name!r}")
        paths[relative] = expected
    return paths


def verify_workload(workload):
    if workload["metadata"] and digest(workload["metadata_path"]) != workload["metadata_sha256"]:
        raise RuntimeError("SIMD workload fixture metadata changed; run again when edits stop")
    for path, expected in workload["fixture_hashes"].items():
        verify_hash(path, expected, "fixture")
    example = workload["wasm"].parent
    for relative, expected in workload_source_paths(workload).items():
        verify_hash(example / relative, expected, "SIMD workload source")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.rs"))
    paths += [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.yml")]
    paths += sorted((ROOT / "scripts").glob("profile*.py"))
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def checked_output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def snapshot_workload_sources(output, workload):
    if workload["source_hashes"] is None:
        return
    example = workload["wasm"].parent
    destination = output / "source" / example.relative_to(ROOT)
    for relative, expected in workload_source_paths(workload).items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(example / relative, target)
        verify_hash(target, expected, "captured SIMD workload source")
    metadata = destination / workload["metadata_path"].name
    metadata.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(workload["metadata_path"], metadata)
    if digest(metadata) != workload["metadata_sha256"]:
        raise RuntimeError("SIMD workload metadata changed while taking the snapshot; run again when edits stop")


def prepare(args, output, workload):
    verify_workload(workload)
    before = source_hashes()
    for name in before:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    if {name: digest(output / "source" / name) for name in before} != before:
        raise RuntimeError("source changed while taking the snapshot; run again when edits stop")
    snapshot_workload_sources(output, workload)

    env = os.environ.copy()
    cleared = [
        "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS",
        "CARGO_PROFILE_RELEASE_DEBUG", "CARGO_PROFILE_RELEASE_STRIP",
    ]
    cleared += sorted(name for name in env if name.startswith("CARGO_TARGET_") and name.endswith("_RUSTFLAGS"))
    for name in cleared:
        env.pop(name, None)
    env["CARGO_TARGET_DIR"] = str(output / "build")
    if args.mode == "sample":
        env["CARGO_PROFILE_RELEASE_DEBUG"] = "2"
        env["CARGO_PROFILE_RELEASE_STRIP"] = "none"
        env["RUSTFLAGS"] = "-C force-frame-pointers=yes"
    command = ["cargo", "build", "--release", "--offline", "--locked", "--verbose", "--bin", "krasm"]
    print(f"Building {args.mode} binary; artifacts: {output}", flush=True)
    with (output / "build.log").open("w") as log:
        build = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if build.returncode:
        raise RuntimeError(f"build failed; see {output / 'build.log'}")
    if source_hashes() != before:
        raise RuntimeError("source changed during the build; run again when edits stop")
    verify_workload(workload)
    binary = output / "krasm"
    shutil.copy2(Path(env["CARGO_TARGET_DIR"]) / "release/krasm", binary)
    for path, expected in workload["fixture_hashes"].items():
        shutil.copyfile(path, output / path.name)
        verify_hash(output / path.name, expected, "captured fixture")

    build_keys = {
        "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_TARGET_DIR", "RUSTC",
        "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTUP_TOOLCHAIN", "CARGO_BUILD_TARGET",
    }
    build_keys.update(name for name in env if name.startswith("CARGO_PROFILE_RELEASE_"))
    governor = Path(f"/sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_governor")
    manifest = {
        "mode": args.mode,
        "workload": workload["name"],
        "workload_metadata": workload["metadata"],
        "workload_metadata_sha256": workload["metadata_sha256"],
        "engine": args.engine,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.uname()._asdict(),
        "cpu": args.cpu,
        "governor": governor.read_text().strip() if governor.exists() else None,
        "rustc": checked_output(["rustc", "--version", "--verbose"]),
        "cargo": checked_output(["cargo", "--version"]),
        "build_command": command,
        "build_environment": {name: env[name] for name in sorted(build_keys) if name in env},
        "cleared_build_environment": list(cleared),
        "source_root": str(ROOT),
        "source_sha256": before,
        "binary_sha256": digest(binary),
        "fixture_sha256": {path.name: value for path, value in workload["fixture_hashes"].items()},
        "expected_stdout": workload["expected_stdout"],
        "runs": args.runs,
    }
    save_json(output / "manifest.json", manifest)
    return binary, manifest


def invocation(binary, output, engine, workload):
    return [str(binary), "run", str(output / workload["wasm"].name), "--engine", engine]


def validate_output(stdout, label, workload):
    if stdout.strip() != workload["expected_stdout"].encode():
        raise RuntimeError(f"incorrect {workload['name']} output from {label}: {stdout!r}")

def benchmark(args, output, binary, manifest, workload):
    binaries = {"current": binary}
    baseline_provenance = None
    if args.baseline:
        baseline = args.baseline.resolve()
        shutil.copy2(baseline, output / "baseline")
        binaries["baseline"] = output / "baseline"
        manifest_path = baseline.parent / "manifest.json"
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text())
            required = {
                "binary_sha256": digest(binaries["baseline"]),
                "mode": "bench",
                "engine": args.engine,
                "fixture_sha256": manifest["fixture_sha256"],
            }
            for key, expected in required.items():
                if previous.get(key) != expected:
                    raise RuntimeError(f"baseline manifest {key} does not match this comparison: {manifest_path}")
            previous_workload = previous.get("workload")
            legacy_commp = (
                previous_workload is None
                and workload["name"] == "commp"
                and previous.get("fixture_sha256") == {path.name: value for path, value in COMMP_FIXTURE_HASHES.items()}
            )
            if previous_workload != workload["name"] and not legacy_commp:
                raise RuntimeError(f"baseline manifest workload does not match this comparison: {manifest_path}")
            save_json(output / "baseline-manifest.json", previous)
            current_env = {key: value for key, value in manifest["build_environment"].items() if key != "CARGO_TARGET_DIR"}
            previous_env = {key: value for key, value in previous["build_environment"].items() if key != "CARGO_TARGET_DIR"}
            settings_match = current_env == previous_env and all(
                previous[key] == manifest[key] for key in ("rustc", "cargo")
            )
            baseline_provenance = {"manifest_verified": True, "recorded_build_settings_match": settings_match}
            if not settings_match:
                print("Warning: baseline compiler/build settings differ; this is not an isolated code-change comparison.")
        else:
            if workload["name"] != "commp":
                raise RuntimeError(f"{workload['name']} comparisons require an adjacent baseline manifest.json")
            baseline_provenance = {"manifest_verified": False, "reason": "no adjacent manifest.json"}
            print("Warning: baseline provenance unverified; only its binary hash is recorded.")
    commands = {name: invocation(path, output, args.engine, workload) for name, path in binaries.items()}
    data = (output / INPUT.name).read_bytes()
    report = {
        "method": "whole-process wall time, including launch and WASI I/O; no debugger",
        "workload": workload["name"],
        "warmups_per_binary": 1,
        "commands": commands,
        "binary_sha256": {name: digest(path) for name, path in binaries.items()},
        "baseline_origin": str(args.baseline.resolve()) if args.baseline else None,
        "baseline_provenance": baseline_provenance,
        "round_order": [],
        "samples_ms": {name: [] for name in binaries},
    }

    def measure(name):
        start = time.perf_counter_ns()
        result = subprocess.run(commands[name], input=data, capture_output=True, timeout=60, check=True)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        validate_output(result.stdout, name, workload)
        return elapsed

    for name in binaries:
        measure(name)
    for number in range(args.runs):
        order = list(binaries)
        if number % 2:
            order.reverse()
        report["round_order"].append(order)
        for name in order:
            report["samples_ms"][name].append(measure(name))
    report["stats"] = {}
    for name, samples in report["samples_ms"].items():
        stats = {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples)}
        report["stats"][name] = stats
        print(f"{name}: {stats['median_ms']:.3f} ms median ({stats['min_ms']:.3f}-{stats['max_ms']:.3f})")
    if args.baseline:
        report["speedup"] = report["stats"]["baseline"]["median_ms"] / report["stats"]["current"]["median_ms"]
        print(f"baseline/current: {report['speedup']:.3f}x")
    report["correct_outputs"] = len(binaries) * (args.runs + 1)
    save_json(output / "timings.json", report)


def sample(args, output, binary, manifest, workload):
    manifest["gdb"] = checked_output(["gdb", "--version"])
    manifest["interval_ms"] = args.interval_ms
    save_json(output / "manifest.json", manifest)
    samples = []
    worker = output / "source/scripts/profile_gdb.py"
    for number in range(args.runs):
        config = {
            "binary": str(binary),
            "source_root": str(ROOT),
            "input": str(output / INPUT.name),
            "stdout": str(output / f"stdout-{number}.txt"),
            "report": str(output / f"stacks-{number}.json"),
            "delay": 0.06 + number * 0.04,
            "interval": args.interval_ms / 1000,
        }
        config_path = output / f"gdb-{number}.json"
        save_json(config_path, config)
        env = os.environ.copy()
        env["KRASM_PROFILE_CONFIG"] = str(config_path)
        env["DEBUGINFOD_URLS"] = ""
        command = [
            "gdb", "-q", "-nx", "-batch", "-iex", "set auto-load python-scripts off",
            "-ex", f"python exec(compile(open({str(worker)!r}).read(), {str(worker)!r}, 'exec'))",
            "--args", *invocation(binary, output, args.engine, workload),
        ]
        with (output / f"gdb-{number}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
        report = json.loads(Path(config["report"]).read_text())
        validate_output(Path(config["stdout"]).read_bytes(), f"GDB run {number}", workload)
        if report["exit_code"] != 0 or not report["samples"]:
            raise RuntimeError(f"GDB run {number} did not complete with samples; see {output}")
        samples.extend(report["samples"])
        print(f"GDB run {number + 1}: {len(report['samples'])} samples, correct hash", flush=True)

    leaves = Counter()
    runtime = Counter()
    locations = Counter()
    for item in samples:
        frames = item["frames"]
        if frames:
            leaves[frames[0]["function"]] += 1
        for frame in frames:
            if frame["function"].startswith("krasm::runtime::"):
                runtime[frame["function"]] += 1
                kind = "instruction" if frame["pc"] == frames[0]["pc"] else "caller_return"
                locations[(frame["file"], frame["line"], frame["elf_pc"], kind)] += 1
                break
    summary = {
        "method": "intrusive periodic GDB stops of a debuginfo/frame-pointer build; counts are not CPU percentages or timings",
        "samples": len(samples),
        "runtime_samples": sum(runtime.values()),
        "leaf_functions": leaves.most_common(),
        "first_runtime_functions": runtime.most_common(),
        "runtime_locations": [
            {"file": file, "line": line, "elf_pc": pc, "kind": kind, "count": count}
            for (file, line, pc, kind), count in locations.most_common()
        ],
    }
    save_json(output / "summary.json", summary)
    print(f"\n{summary['samples']} intrusive samples; {summary['runtime_samples']} have a runtime frame")
    print("First runtime frame (sample counts, not CPU percentages):")
    for name, count in runtime.most_common(12):
        print(f"{count:5}  {name}")
    print(f"Raw stacks and load-bias evidence: {output}/stacks-*.json")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("bench", "sample"):
        mode = modes.add_parser(name)
        mode.add_argument("--cpu", type=int, required=True, help="allowed logical CPU to pin measured processes to")
        mode.add_argument("--engine", choices=("flat", "structured"), default="flat")
        mode.add_argument("--workload", choices=("commp", "commp-simd"), default="commp")
        mode.add_argument("--runs", type=positive_int, default=10 if name == "bench" else 4)
        mode.add_argument("--output", type=Path, help="new artifact directory (default: target/profiles/<timestamp>-<mode>)")
        if name == "bench":
            mode.add_argument("--baseline", type=Path, help="frozen krasm binary for alternating paired measurements")
        else:
            mode.add_argument("--interval-ms", type=positive_int, default=6)
    args = parser.parse_args()
    if not hasattr(os, "sched_getaffinity"):
        parser.error("Linux CPU affinity is required")
    if args.cpu not in os.sched_getaffinity(0):
        parser.error(f"CPU {args.cpu} is not in the allowed affinity set")
    if args.mode == "bench" and args.baseline and args.runs % 2:
        parser.error("paired measurements require an even --runs count for balanced AB/BA order")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or ROOT / "target/profiles" / f"{stamp}-{args.mode}").resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
        workload = load_workload(args.workload)
        binary, manifest = prepare(args, output, workload)
        os.sched_setaffinity(0, {args.cpu})
        if args.mode == "bench":
            benchmark(args, output, binary, manifest, workload)
        else:
            sample(args, output, binary, manifest, workload)
        print(f"Results: {output}")
    except (OSError, RuntimeError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"profile: {error}\nArtifacts: {output}\n")


if __name__ == "__main__":
    main()
