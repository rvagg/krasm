import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from profile import load_workload, validate_output, verify_workload
from prepare_corpus import extract


class CorpusTests(unittest.TestCase):
    def test_binary_oracle_preserves_whitespace_and_non_utf8_bytes(self):
        data = b"\n\x00\xffpayload\t "
        workload = {"name": "binary", "expected_stdout_sha256": hashlib.sha256(data).hexdigest()}
        validate_output(data, "guest", workload)
        for changed in (data.strip(), data + b"\n", data[:-1]):
            with self.assertRaises(RuntimeError):
                validate_output(changed, "guest", workload)

    def test_explicit_registry_freezes_input_and_registry_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            module = directory / "guest.wasm"
            data = directory / "stdin.bin"
            module.write_bytes(b"wasm fixture")
            data.write_bytes(b"input")
            registry = directory / "registry.json"
            spec = {"family": "binary", "compiler": {}, "args": [], "module": str(module), "input": str(data),
                    "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                    "input_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                    "expected_stdout_sha256": hashlib.sha256(b"output").hexdigest()}
            registry.write_text(json.dumps({"schema_version": 1, "workloads": {"custom": spec}}))
            workload = load_workload("custom", registry)
            verify_workload(workload)
            data.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "recorded SHA-256"):
                verify_workload(workload)
            data.write_bytes(b"input")
            registry.write_text(registry.read_text() + "\n")
            with self.assertRaisesRegex(RuntimeError, "workload registry differs"):
                verify_workload(workload)

    def test_archive_members_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tar_path = directory / "source.tar"
            with tarfile.open(tar_path, "w") as bundle:
                member = tarfile.TarInfo("../escaped")
                member.size = 1
                bundle.addfile(member, io.BytesIO(b"x"))
            with self.assertRaises(tarfile.FilterError):
                extract(tar_path, directory / "tar")
            zip_path = directory / "source.zip"
            with zipfile.ZipFile(zip_path, "w") as bundle:
                bundle.writestr("../escaped", b"x")
            with self.assertRaisesRegex(RuntimeError, "unsafe ZIP member"):
                extract(zip_path, directory / "zip")
            self.assertFalse((directory / "escaped").exists())


if __name__ == "__main__":
    unittest.main()
