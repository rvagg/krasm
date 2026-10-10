"""Build one krasm source in several code layouts.

Interpreter performance moves with code placement: identical instructions can
take a quarter more cycles when functions land at different addresses. Each
seed relinks the binary with lld's `--shuffle-sections`, which reorders every
function deterministically; seed 0 keeps the natural order. Comparing groups of
layouts, instead of single binaries, separates code changes from placement luck.

Only the final binary is rebuilt per seed (`cargo rustc` passes the link flag to
it alone), so dependencies compile once per source.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

MANIFEST = "layouts.json"


def link_args(seed):
    return [] if seed == 0 else [f"-Clink-arg=-Wl,--shuffle-sections=*={seed}"]


def checkout(root, rev, destination):
    """Extract a revision without touching the repository's index or worktree."""
    commit = subprocess.check_output(["git", "rev-parse", "--verify", f"{rev}^{{commit}}"], cwd=root, text=True).strip()
    destination.mkdir(parents=True)
    archive = subprocess.Popen(["git", "archive", commit], cwd=root, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(destination)], stdin=archive.stdout, check=True)
    if archive.wait():
        raise RuntimeError(f"git archive failed for {rev}")
    return commit


def working_tree(root):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    return head, status.splitlines()


def build_environment(target_dir):
    env = os.environ.copy()
    for name in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS"]:
        env.pop(name, None)
    for name in [name for name in env if name.startswith("CARGO_PROFILE_RELEASE_")]:
        env.pop(name)
    for name in [name for name in env if name.startswith("CARGO_TARGET_") and name.endswith("_RUSTFLAGS")]:
        env.pop(name)
    env["CARGO_TARGET_DIR"] = str(target_dir)
    return env


def linker(binary):
    """The linker recorded in the binary's .comment section, if any."""
    comment = subprocess.run(["readelf", "-p", ".comment", str(binary)], capture_output=True, text=True).stdout
    return next((line.split("Linker:", 1)[1].strip() for line in comment.splitlines() if "Linker:" in line), None)


def build_layouts(root, output, seeds, rev, features, digest, save_json):
    output.mkdir(parents=True, exist_ok=False)
    if rev:
        source_dir = output / "source"
        source = {"rev": rev, "commit": checkout(root, rev, source_dir)}
    else:
        source_dir = root
        head, dirty = working_tree(root)
        source = {"working_tree": str(root), "head": head, "uncommitted": dirty}
    env = build_environment(output / "build")
    built = output / "build" / "release" / "krasm"
    layouts = []
    log = (output / "build.log").open("w")
    try:
        for seed in range(seeds):
            command = ["cargo", "rustc", "--release", "--offline", "--locked", "--bin", "krasm"]
            if features:
                command += ["--features", features]
            command += ["--", *link_args(seed)]
            print(f"Building layout {seed + 1}/{seeds}", flush=True)
            log.write(f"$ {' '.join(command)}\n")
            log.flush()
            if subprocess.run(command, cwd=source_dir, env=env, stdout=log, stderr=subprocess.STDOUT).returncode:
                raise RuntimeError(f"build failed; see {output / 'build.log'}")
            binary = output / f"krasm-seed{seed}"
            shutil.copy2(built, binary)
            layouts.append({"seed": seed, "binary": binary.name, "sha256": digest(binary), "link_args": link_args(seed)})
    finally:
        log.close()
    hashes = [layout["sha256"] for layout in layouts]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("layouts are not distinct; the linker may not support --shuffle-sections")
    manifest = {
        "kind": "layouts",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "features": features,
        "rustc": subprocess.check_output(["rustc", "--version", "--verbose"], text=True).strip(),
        "linker": linker(output / layouts[0]["binary"]),
        "layouts": layouts,
    }
    save_json(output / MANIFEST, manifest)
    return manifest


def expand_arm(label, path):
    """Variants for an arm: one per layout in a build directory, or a lone binary."""
    path = Path(path)
    if path.is_dir():
        manifest_path = path / MANIFEST
        if not manifest_path.is_file():
            raise RuntimeError(f"{path} has no {MANIFEST}; build it with `profile.py build`")
        manifest = json.loads(manifest_path.read_text())
        return [(f"{label}#{layout['seed']}", path / layout["binary"]) for layout in manifest["layouts"]], manifest
    if not path.is_file():
        raise RuntimeError(f"arm binary is missing: {path}")
    return [(label, path)], None
