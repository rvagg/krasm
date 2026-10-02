"""Mine dynamically weighted straight-line Op sequences from frozen captures."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

# Control operations are counted in totals, but excluded from fusion candidates.
BARRIERS = {"Br", "BrIf", "BrTable", "Call", "CallIndirect", "Return", "End", "Label", "Unreachable"}
LOCAL_OPS = {"LocalGet", "LocalSet", "LocalTee"}


def validate_profile(profile):
    if (profile.get("schema_version"), profile.get("op_format_version"), profile.get("count_semantics")) != (1, 1, "dispatch-attempts"):
        raise ValueError("unsupported instruction profile schema, Op format or count semantics")
    seen = set()
    for function in profile["functions"]:
        index = function["function_index"]
        if index in seen:
            raise ValueError("duplicate function index")
        seen.add(index)
        ops, counts = function["ops"], function["counts"]
        if len(ops) != len(counts) or any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("invalid per-PC counts")
        if any(not isinstance(op.get("opcode"), str) for op in ops):
            raise ValueError("invalid Op record")


def regions(ops):
    """Exclude control operations and split at every possible entry point."""
    leaders = {0, len(ops)}
    for pc, op in enumerate(ops):
        name, immediate = op["opcode"], op.get("immediates", {})
        if name in {"Br", "BrIf"}:
            leaders.add(immediate["target"])
        elif name == "BrTable":
            leaders.update(target["pc"] for target in immediate["targets"])
            leaders.add(immediate["default"]["pc"])
        elif name == "Label":
            leaders.add(immediate["end_target"])
        if name in BARRIERS:
            leaders.update((pc, pc + 1))
    if any(type(pc) is not int or not 0 <= pc <= len(ops) for pc in leaders):
        raise ValueError("branch target outside function")
    boundaries = sorted(leaders)
    for start, end in zip(boundaries, boundaries[1:]):
        if start < end and ops[start]["opcode"] not in BARRIERS:
            yield start, end


def pattern(ops, operand_aware):
    locals_seen = {}
    tokens = []
    for op in ops:
        name = op["opcode"]
        if not operand_aware or "immediates" not in op:
            tokens.append(name)
        elif name in LOCAL_OPS:
            index = op["immediates"]["index"]
            slot = locals_seen.setdefault(index, len(locals_seen))
            tokens.append(f"{name} ${slot}")
        else:
            tokens.append(name + " " + json.dumps(op["immediates"], sort_keys=True, separators=(",", ":")))
    return tuple(tokens)


def mine(profile, operand_aware, max_length=6):
    validate_profile(profile)
    patterns = {}
    total = 0
    for function in profile["functions"]:
        ops, counts = function["ops"], function["counts"]
        total += sum(counts)
        for start, end in regions(ops):
            if any(counts[pc] < counts[pc + 1] for pc in range(start, end - 1)):
                raise ValueError("counts increase inside a single-entry straight-line region")
            # For a fixed pattern length, earliest-first gives a non-overlapping
            # placement with maximal weight: counts cannot increase in a region.
            next_free = {}
            for pc in range(start, end):
                for length in range(2, min(max_length, end - pc) + 1):
                    executions = counts[pc + length - 1]
                    if not executions:
                        continue
                    key = pattern(ops[pc:pc + length], operand_aware)
                    entry = patterns.setdefault(key, {"executions": 0, "non_overlapping_executions": 0, "sites": 0, "examples": []})
                    entry["executions"] += executions
                    entry["sites"] += 1
                    if len(entry["examples"]) < 3:
                        entry["examples"].append({"function_index": function["function_index"], "pc": pc, "executions": executions})
                    if pc >= next_free.get(key, start):
                        entry["non_overlapping_executions"] += executions
                        next_free[key] = pc + length
    for key, entry in patterns.items():
        entry["dispatches_removed"] = entry["non_overlapping_executions"] * (len(key) - 1)
        entry["dispatch_fraction"] = entry["dispatches_removed"] / total if total else 0
    return total, patterns


def rankings(patterns, top):
    by_length = {}
    for length in range(2, 7):
        rows = [(key, value) for key, value in patterns.items() if len(key) == length]
        rows.sort(key=lambda row: (-row[1]["dispatch_fraction"], -row[1]["dispatches_removed"], row[0]))
        by_length[str(length)] = [{"pattern": list(key), **value} for key, value in rows[:top]]
    return by_length


def aggregate(workloads, top=20):
    if not workloads:
        raise ValueError("no instruction captures found")
    ids = [item["workload"] for item in workloads]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate workload IDs; give distinct input scenarios distinct IDs")
    families = defaultdict(list)
    for item in workloads:
        families[item["family"]].append(item["workload"])
    result = {
        "schema_version": 1,
        "method": "per-candidate non-overlapping dispatch reduction; not predicted speedup; candidates cannot be summed",
        "weighting": "equal families, then equal workloads within each family; absent patterns contribute zero",
        "barriers": sorted(BARRIERS),
        "families": dict(families),
        "workloads": [],
        "aggregate": {},
    }
    mined = {"shape": {}, "operands": {}}
    for item in workloads:
        row = {key: item[key] for key in ("workload", "family", "capture")}
        row["rankings"] = {}
        for view, operand_aware in (("shape", False), ("operands", True)):
            total, patterns = mine(item["profile"], operand_aware)
            row["dispatches"] = total
            row["rankings"][view] = rankings(patterns, top)
            mined[view][item["workload"]] = patterns
        result["workloads"].append(row)
    for view, per_workload in mined.items():
        combined = {}
        for item in workloads:
            name, family = item["workload"], item["family"]
            weight = 1 / len(families) / len(families[family])
            for key, counts in per_workload[name].items():
                row = combined.setdefault(key, {"executions": 0, "dispatches_removed": 0, "dispatch_fraction": 0, "workloads": {}})
                row["executions"] += counts["executions"]
                row["dispatches_removed"] += counts["dispatches_removed"]
                row["dispatch_fraction"] += weight * counts["dispatch_fraction"]
                row["workloads"][name] = counts["dispatch_fraction"]
        for row in combined.values():
            row["workload_coverage"] = len(row["workloads"])
        result["aggregate"][view] = rankings(combined, top)
    return result


def load_captures(paths):
    captures = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            captures.append(path)
        elif path.is_dir():
            captures.extend(sorted(path.rglob("instructions.json")))
        else:
            raise ValueError(f"capture path does not exist: {path}")
    workloads = []
    for capture in captures:
        manifest = json.loads((capture.parent / "manifest.json").read_text())
        expected = manifest.get("instruction_profile_sha256")
        if not expected or hashlib.sha256(capture.read_bytes()).hexdigest() != expected:
            raise ValueError(f"missing or mismatched instruction capture hash: {capture}")
        if manifest.get("mode") != "collect" or manifest.get("engine") != "flat":
            raise ValueError(f"not a flat instruction capture: {capture}")
        raw = json.loads(capture.read_text())
        module_hash = manifest["fixture_sha256"].get(Path(raw["module"]).name)
        if not module_hash or raw.get("module_sha256") != module_hash:
            raise ValueError(f"instruction capture module disagrees with manifest: {capture}")
        if raw["outcome"] != {"exit_code": 0, "trap": None}:
            raise ValueError(f"failed execution is not eligible for corpus aggregation: {capture}")
        workloads.append({
            "workload": manifest["workload"], "family": manifest["family"],
            "capture": str(capture.resolve()), "profile": raw["profile"],
        })
    return workloads


def analyse(paths, output, top=20):
    report = aggregate(load_captures(paths), top)
    with output.open("x") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print("Potential dispatch reductions, not speedups; rows overlap and cannot be summed.")
    for view, lengths in report["aggregate"].items():
        print(f"\n{view}: equal-family weighted rankings")
        for length, rows in lengths.items():
            print(f"  length {length}")
            for row in rows[:5]:
                print(f"    {row['dispatch_fraction']:7.2%}  {row['workload_coverage']}/{len(report['workloads'])} workloads  " + "; ".join(row["pattern"]))
    print(f"Report: {output}")
