"""GDB-side sampler, loaded by profile.py inside GDB's Python interpreter."""

import json
import os
from pathlib import Path
import re
import shlex
import signal
import threading

import gdb

config = json.loads(Path(os.environ["KRASM_PROFILE_CONFIG"]).read_text())
samples = []
started = False
running = threading.Event()
finished = threading.Event()
exit_code = None
mapping = None


def pulse(pidfd):
    try:
        if finished.wait(config["delay"]):
            return
        while not finished.is_set():
            if running.is_set():
                try:
                    signal.pidfd_send_signal(pidfd, signal.SIGUSR1)
                except ProcessLookupError:
                    break
            finished.wait(config["interval"])
    finally:
        os.close(pidfd)


def on_continue(event):
    global started
    running.set()
    if not started:
        started = True
        # A pidfd cannot signal an unrelated process if the inferior exits.
        pidfd = os.pidfd_open(gdb.selected_inferior().pid)
        threading.Thread(target=pulse, args=(pidfd,), daemon=True).start()


def on_exit(event):
    global exit_code
    running.clear()
    finished.set()
    exit_code = getattr(event, "exit_code", None)


def executable_mapping():
    with Path(config["binary"]).open("rb") as binary:
        header = binary.read(64)
    if header[:4] != b"\x7fELF" or header[4] not in (1, 2) or header[5] not in (1, 2):
        raise RuntimeError("sampling requires an ELF executable")
    width = 8 if header[4] == 2 else 4
    endian = "little" if header[5] == 1 else "big"
    elf_entry = int.from_bytes(header[24:24 + width], endian)
    files = gdb.execute("info files", to_string=True)
    match = re.search(r"Entry point:\s*(0x[0-9a-fA-F]+)", files)
    if not match:
        raise RuntimeError("GDB did not report the relocated executable entry point")
    runtime_entry = int(match.group(1), 16)
    maps = Path(f"/proc/{gdb.selected_inferior().pid}/maps").read_text()
    ranges = []
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == config["binary"]:
            start, end = fields[0].split("-")
            ranges.append((int(start, 16), int(end, 16)))
    if not any(start <= runtime_entry < end for start, end in ranges):
        raise RuntimeError("GDB's executable entry point does not match the frozen binary's mappings")
    return {
        "elf_entry": elf_entry,
        "runtime_entry": runtime_entry,
        "load_bias": runtime_entry - elf_entry,
        "ranges": ranges,
        "proc_maps": maps,
        "gdb_info_files": files,
    }


def capture():
    global mapping
    running.clear()
    if mapping is None:
        mapping = executable_mapping()
    frames = []
    frame = gdb.newest_frame()
    while frame is not None and len(frames) < 32:
        pc = frame.pc()
        sal = frame.find_sal()
        filename = sal.symtab.fullname() if sal.symtab else None
        if filename:
            try:
                filename = str(Path(filename).relative_to(config["source_root"]))
            except ValueError:
                pass
        in_binary = any(start <= pc < end for start, end in mapping["ranges"])
        frames.append({
            "function": frame.name() or "??",
            "pc": hex(pc),
            "elf_pc": hex(pc - mapping["load_bias"]) if in_binary else None,
            "file": filename,
            "line": sal.line,
            "inline": frame.type() == gdb.INLINE_FRAME,
        })
        frame = frame.older()
    samples.append({"frames": frames})


gdb.execute("set pagination off")
gdb.execute("set confirm off")
gdb.execute("set debuginfod enabled off")
gdb.execute("set print thread-events off")
gdb.execute("handle SIGUSR1 stop noprint nopass")
gdb.events.cont.connect(on_continue)
gdb.events.stop.connect(lambda event: running.clear())
gdb.events.exited.connect(on_exit)
gdb.execute("catch signal SIGUSR1")
gdb.execute("commands\n silent\n python capture()\n continue\nend")
gdb.execute(f"run {gdb.parameter('args')} < {shlex.quote(config['input'])} > {shlex.quote(config['stdout'])}")
finished.set()
Path(config["report"]).write_text(json.dumps({
    "exit_code": exit_code,
    "mapping": mapping,
    "samples": samples,
}, indent=2) + "\n")
