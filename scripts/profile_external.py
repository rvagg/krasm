"""Wall time and user-mode counters for krasm against external runtimes.

Every runtime runs the same module, arguments and stdin on one CPU, in rotating
interleaved rounds, and every output is checked. Times are whole processes:
launch, module loading, any compilation, and execution. `wasmtime-jit`
compiles on every run; `wasmtime-aot` runs a module precompiled once per
workload, so compilation is excluded. Wasmi translates lazily (its default),
built with tail-call dispatch (`wasmi-tail`) and with its portable loop
(`wasmi-loop`). Runtimes come from `prepare_runtimes.py`; nothing is fetched.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

from profile_counters import base_event, orders, parse_perf_csv

EVENTS = ("instructions:u", "cycles:u")


def runtime_commands(krasm, manifest, root):
    """Each runtime's command, as a function of the module and guest arguments."""
    runtimes = {name: str(root / entry["binary"]) for name, entry in manifest["runtimes"].items()}
    wasmtime = runtimes["wasmtime"]
    return {
        "krasm": lambda module, args: [str(krasm), "run", str(module), *(["--", *args] if args else [])],
        "wasmi-tail": lambda module, args: [runtimes["wasmi-tail"], "run", str(module), *args],
        "wasmi-loop": lambda module, args: [runtimes["wasmi-loop"], "run", str(module), *args],
        "wasmtime-jit": lambda module, args: [wasmtime, "run", "-C", "cache=n", str(module), *args],
        "wasmtime-aot": lambda module, args: [wasmtime, "run", "--allow-precompiled",
                                              str(module.with_suffix(".cwasm")), *args],
    }


def measure(command, data, scratch, label, workload, validate_output):
    counters = scratch / "perf.csv"
    wrapped = ["perf", "stat", "-x", ",", "-o", str(counters), "-e", ",".join(EVENTS), "--", *command]
    start = time.perf_counter_ns()
    result = subprocess.run(wrapped, input=data, capture_output=True, timeout=600)
    elapsed = (time.perf_counter_ns() - start) / 1_000_000
    if result.returncode:
        raise RuntimeError(f"{label} failed on {workload['name']}: {result.stderr[-400:]!r}")
    validate_output(result.stdout, label, workload)
    counts = parse_perf_csv(counters.read_text())
    if any(counts.get(base_event(event)) is None for event in EVENTS):
        raise RuntimeError(f"perf did not count {', '.join(EVENTS)} for {label}")
    return {"wall_ms": elapsed, **{event: counts[base_event(event)] for event in EVENTS}}


def summarise(samples):
    stats = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        stats[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return stats


def external(krasm, manifest_path, workloads, runs, output, cpu, root, digest, validate_output, save_json):
    manifest = json.loads(manifest_path.read_text())
    for name, entry in manifest["runtimes"].items():
        if digest(root / entry["binary"]) != entry["sha256"]:
            raise RuntimeError(f"{name} differs from {manifest_path}; run prepare_runtimes.py again")
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "krasm"
    shutil.copy2(krasm, frozen)
    commands = runtime_commands(frozen, manifest, root)
    labels = list(commands)
    governor = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    report = {
        "method": "whole-process wall time and perf stat user-mode counters per invocation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.uname()._asdict(),
        "cpu": cpu,
        "governor": governor.read_text().strip() if governor.exists() else None,
        "krasm": {"origin": str(Path(krasm).resolve()), "sha256": digest(frozen)},
        "runtimes": manifest,
        "warmups_per_runtime": 1,
        "runs": runs,
        "workloads": {},
    }
    scratch = output / "scratch"
    scratch.mkdir()
    wasmtime = root / manifest["runtimes"]["wasmtime"]["binary"]
    for workload in workloads:
        module = scratch / workload["wasm"].name
        shutil.copy2(workload["wasm"], module)
        subprocess.run([str(wasmtime), "compile", "-o", str(module.with_suffix(".cwasm")), str(module)], check=True)
        data = workload["input"].read_bytes()
        entry = {"commands": {label: command(module, workload["args"]) for label, command in commands.items()},
                 "round_order": orders(labels, runs), "samples": {label: [] for label in labels}}
        for label in labels:
            measure(entry["commands"][label], data, scratch, label, workload, validate_output)
        for order in entry["round_order"]:
            for label in order:
                entry["samples"][label].append(
                    measure(entry["commands"][label], data, scratch, label, workload, validate_output))
        entry["stats"] = {label: summarise(samples) for label, samples in entry["samples"].items()}
        report["workloads"][workload["name"]] = entry
        print_workload(workload["name"], entry["stats"])
        save_json(output / "external.json", report)
    shutil.rmtree(scratch)
    print_summary(report)
    save_json(output / "external.json", report)
    return report


def print_workload(name, stats):
    krasm = stats["krasm"]["wall_ms"]["median"]
    print(f"\n{name}")
    print(f"  {'runtime':13} {'wall ms (median, min-max)':>28} {'cycles':>9} {'instructions':>15}  krasm/runtime")
    for label, stat in stats.items():
        wall = stat["wall_ms"]
        span = f"{wall['median']:,.1f} ({wall['min']:,.1f}-{wall['max']:,.1f})"
        print(f"  {label:13} {span:>28} {stat['cycles:u']['median'] / 1e6:8,.0f}M "
              f"{stat['instructions:u']['median']:15,.0f}  {krasm / wall['median']:8.2f}x")


def print_summary(report):
    """Geometric mean, across workloads, of krasm's wall time over each runtime's."""
    labels = list(next(iter(report["workloads"].values()))["stats"])
    print("\ngeometric mean krasm/runtime wall time")
    for label in labels[1:]:
        ratios = [entry["stats"]["krasm"]["wall_ms"]["median"] / entry["stats"][label]["wall_ms"]["median"]
                  for entry in report["workloads"].values()]
        print(f"  {label:13} {statistics.geometric_mean(ratios):6.2f}x")
