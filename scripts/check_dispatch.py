"""Check that a release krasm binary's flat executor dispatches by jump.

Each handler must end by calling the next in tail position, which the optimiser
compiles to an indirect jump. A handler that calls its successor instead grows
the native stack on every op it executes, and a long loop overflows it. The
only indirect calls a handler may make are to external functions, such as libm,
whose addresses come from the global offset table.

Handlers are found by symbol name, so the binary must keep its symbols. The
check reads x86-64 disassembly from `objdump` and skips other architectures.
"""

import re
import subprocess
import sys

HANDLER = "flat_executor::handler_for::{closure"
MIN_HANDLERS = 100
MIN_JUMPING = 0.9

SYMBOL = re.compile(r"^[0-9a-f]+ <(.*)>:$")
INSTRUCTION = re.compile(r"^\s*([0-9a-f]+):\s+(?:(?:notrack|bnd)\s+)?(\S+)\s*([^#]*)")


def functions(disassembly):
    """Each function's instructions, as (address, mnemonic, operands), by symbol."""
    found = {}
    body = None
    for line in disassembly.splitlines():
        symbol = SYMBOL.match(line)
        if symbol:
            body = found.setdefault(symbol.group(1), [])
        elif body is not None and (insn := INSTRUCTION.match(line)):
            body.append((insn.group(1), insn.group(2), insn.group(3).strip()))
    return found


def indirect(mnemonic, operands, kind):
    """Whether this is a register- or memory-indirect `kind` not through `%rip`."""
    return mnemonic.startswith(kind) and operands.startswith("*") and "(%rip)" not in operands


def loaded_from_got(body, index, register):
    """Whether `register` was last written, before `index`, by a `%rip`-relative load."""
    for _, mnemonic, operands in reversed(body[:index]):
        if operands.endswith("," + register):
            return mnemonic.startswith("mov") and "(%rip)" in operands
    return False


def stray_calls(body):
    """The indirect calls in a handler that are not to external functions."""
    return [
        (address, f"{mnemonic} {operands}")
        for i, (address, mnemonic, operands) in enumerate(body)
        if indirect(mnemonic, operands, "call")
        and not (re.fullmatch(r"\*%\w+", operands) and loaded_from_got(body, i, operands[1:]))
    ]


def check(disassembly):
    """Problems with the handlers in this disassembly, as messages."""
    handlers = {name: body for name, body in functions(disassembly).items() if HANDLER in name}
    if len(handlers) < MIN_HANDLERS:
        return [f"found {len(handlers)} handlers, expected at least {MIN_HANDLERS}: stripped, or renamed?"]
    stray = [
        f"{name}: {address}: {call}"
        for name, body in sorted(handlers.items())
        for address, call in stray_calls(body)
    ]
    if stray:
        return stray
    jumping = sum(1 for body in handlers.values() if any(indirect(m, o, "jmp") for _, m, o in body))
    if jumping < MIN_JUMPING * len(handlers):
        return [f"{jumping} of {len(handlers)} handlers dispatch by jump: is this a trampoline build?"]
    return []


def main(argv):
    if len(argv) != 2:
        sys.exit(f"usage: {argv[0]} BINARY")
    binary = argv[1]
    header = subprocess.run(["objdump", "-f", binary], check=True, capture_output=True, text=True).stdout
    if "x86-64" not in header:
        print(f"dispatch check skipped: {binary} is not x86-64")
        return
    disassembly = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", "-C", "-j", ".text", binary],
        check=True, capture_output=True, text=True,
    ).stdout
    problems = check(disassembly)
    if problems:
        print(f"{binary}: handlers that do not dispatch by jump:", *problems, sep="\n  ")
        sys.exit(1)
    print(f"{binary}: every handler dispatches by jump")


if __name__ == "__main__":
    main(sys.argv)
