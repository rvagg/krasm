"""Hardware-counter comparisons of frozen krasm binaries across workloads.

Each measured invocation runs under `perf stat`, counting user-mode events only:
syscall and page-fault work in the kernel is excluded from instruction counts.
Instructions retired is the primary metric for interpreter mechanics; cycles and
wall time are secondary because code layout and frequency move them more.

Instructions per op divides instructions retired by a fixed work unit: the
total flat dispatches in an existing unfused instruction capture of the same
module and input. The denominator does not change when the IR does, so the
ratio stays comparable across builds that dispatch differently.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

EVENTS = ("instructions:u", "cycles:u", "branches:u", "branch-misses:u", "L1-dcache-load-misses:u")


def base_event(name):
    """perf may omit the `:u` modifier from some event names in its output."""
    return name.split(":", 1)[0]


def parse_perf_csv(text):
    """Parse `perf stat -x ,` output into {base event: count}; uncounted events are None."""
    counts = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) < 3:
            continue
        value, event = fields[0], fields[2]
        counts[base_event(event)] = int(value) if value.isdigit() else None
    return counts


def dispatch_total(capture, workload):
    """Total flat dispatches recorded for this workload's module, or None without a capture."""
    path = capture / workload["name"] / "instructions.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if report.get("module_sha256") != workload["fixture_hashes"][workload["wasm"]]:
        raise RuntimeError(f"dispatch capture is for a different module: {path}")
    return sum(sum(function["counts"]) for function in report["profile"]["functions"])


def freeze(arms, output, digest):
    """Copy each labelled binary into the capture so later rebuilds cannot alter it."""
    frozen = {}
    for label, path in arms.items():
        destination = output / "bin" / label
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        frozen[label] = {"path": destination, "origin": str(path.resolve()), "sha256": digest(destination)}
    return frozen


def orders(labels, runs):
    """Rotate the arm order every round and reverse alternate rounds, balancing position."""
    rounds = []
    for number in range(runs):
        shift = (number // 2) % len(labels)
        order = labels[shift:] + labels[:shift]
        rounds.append(order if number % 2 == 0 else order[::-1])
    return rounds


def measure(binary, workload, scratch, events, validate_output):
    """Run once under perf stat; return wall milliseconds and counter values."""
    counters = scratch / "perf.csv"
    command = ["perf", "stat", "-x", ",", "-o", str(counters), "-e", ",".join(events), "--",
               str(binary), "run", str(workload["wasm"]), "--engine", "flat"]
    if workload["args"]:
        command += ["--", *workload["args"]]
    data = workload["input"].read_bytes()
    start = time.perf_counter_ns()
    result = subprocess.run(command, input=data, capture_output=True, timeout=300)
    elapsed = (time.perf_counter_ns() - start) / 1_000_000
    if result.returncode:
        raise RuntimeError(f"{binary} failed on {workload['name']}: {result.stderr[-400:]!r}")
    validate_output(result.stdout, str(binary), workload)
    counts = parse_perf_csv(counters.read_text())
    missing = [event for event in events if counts.get(base_event(event)) is None]
    if missing:
        raise RuntimeError(f"perf did not count {', '.join(missing)}; reduce --events or free a counter")
    return {"wall_ms": elapsed, **{event: counts[base_event(event)] for event in events}}


def summarise(samples, dispatches):
    stats = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        stats[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    if dispatches:
        stats["instructions_per_op"] = stats["instructions:u"]["median"] / dispatches
    return stats


def compare(arms, workloads, runs, output, cpu, dispatch_capture, events, digest, validate_output, save_json):
    output.mkdir(parents=True, exist_ok=False)
    frozen = freeze(arms, output, digest)
    labels = list(frozen)
    governor = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    report = {
        "method": "perf stat user-mode counters and whole-process wall time per invocation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.uname()._asdict(),
        "cpu": cpu,
        "governor": governor.read_text().strip() if governor.exists() else None,
        "perf": subprocess.check_output(["perf", "--version"], text=True).strip(),
        "events": list(events),
        "arms": {label: {key: str(value) for key, value in arm.items()} for label, arm in frozen.items()},
        "warmups_per_arm": 1,
        "runs": runs,
        "dispatch_capture": str(dispatch_capture) if dispatch_capture else None,
        "workloads": {},
    }
    scratch = output / "scratch"
    scratch.mkdir()
    for workload in workloads:
        dispatches = dispatch_total(dispatch_capture, workload) if dispatch_capture else None
        entry = {"dispatches": dispatches, "round_order": orders(labels, runs),
                 "samples": {label: [] for label in labels}}
        for label in labels:
            measure(frozen[label]["path"], workload, scratch, events, validate_output)
        for order in entry["round_order"]:
            for label in order:
                entry["samples"][label].append(measure(frozen[label]["path"], workload, scratch, events, validate_output))
        entry["stats"] = {label: summarise(entry["samples"][label], dispatches) for label in labels}
        report["workloads"][workload["name"]] = entry
        print_workload(workload["name"], entry, labels)
        save_json(output / "counters.json", report)
    shutil.rmtree(scratch)
    return report


def print_workload(name, entry, labels):
    base = entry["stats"][labels[0]]
    print(f"\n{name}" + (f" ({entry['dispatches']:,} ops)" if entry["dispatches"] else ""))
    print(f"  {'arm':14} {'instructions':>16} {'insn/op':>8} {'cycles':>16} {'IPC':>5} {'br-miss':>12} {'wall ms':>9}  vs {labels[0]}")
    for label in labels:
        stats = entry["stats"][label]
        insns, cycles = stats["instructions:u"]["median"], stats["cycles:u"]["median"]
        per_op = f"{stats['instructions_per_op']:.1f}" if "instructions_per_op" in stats else "-"
        delta = ""
        if label != labels[0]:
            delta = (f"insn {100 * (insns / base['instructions:u']['median'] - 1):+.1f}%  "
                     f"cyc {100 * (cycles / base['cycles:u']['median'] - 1):+.1f}%  "
                     f"wall {100 * (stats['wall_ms']['median'] / base['wall_ms']['median'] - 1):+.1f}%")
        misses = stats.get("branch-misses:u", {}).get("median")
        misses = f"{misses:12,.0f}" if misses is not None else f"{'-':>12}"
        print(f"  {label:14} {insns:16,.0f} {per_op:>8} {cycles:16,.0f} {insns / cycles:5.2f} "
              f"{misses} {stats['wall_ms']['median']:9.1f}  {delta}")
