from pathlib import Path
import unittest

from profile_external import runtime_commands


class ExternalTests(unittest.TestCase):
    def test_guest_arguments_follow_each_runtime_convention(self):
        manifest = {"runtimes": {name: {"binary": name} for name in ("wasmtime", "wasmi-tail", "wasmi-loop")}}
        commands = runtime_commands("krasm", manifest, Path("/r"))
        module = Path("/w/jq.wasm")
        args = ["-c", ".[]"]
        self.assertEqual(commands["krasm"](module, args), ["krasm", "run", "/w/jq.wasm", "--", "-c", ".[]"])
        self.assertEqual(commands["krasm"](module, []), ["krasm", "run", "/w/jq.wasm"])
        self.assertEqual(commands["wasmi-tail"](module, args), ["/r/wasmi-tail", "run", "/w/jq.wasm", "-c", ".[]"])
        self.assertEqual(commands["wasmtime-jit"](module, args),
                         ["/r/wasmtime", "run", "-C", "cache=n", "/w/jq.wasm", "-c", ".[]"])
        self.assertEqual(commands["wasmtime-aot"](module, args),
                         ["/r/wasmtime", "run", "--allow-precompiled", "/w/jq.cwasm", "-c", ".[]"])


if __name__ == "__main__":
    unittest.main()
