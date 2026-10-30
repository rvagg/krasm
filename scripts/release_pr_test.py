import base64
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile
from contextlib import redirect_stdout

SCRIPT = Path(__file__).with_name('release_pr.py')
spec = importlib.util.spec_from_file_location('release_pr', SCRIPT)
release_pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_pr)
BASE, OLD, PREPARED, ANCESTOR, HUMAN = (character * 40 for character in 'abcdf')


def metadata(version='0.0.2'):
    return {
        'Cargo.toml': (f'[package]\nname = "krasm"\nversion = "{version}"\n'
                       '[profile.release]\ncodegen-units = 1\n'
                       '[dependencies]\nserde = "1"\n').encode(),
        'Cargo.lock': (f'version = 4\n[[package]]\nname = "krasm"\nversion = "{version}"\n'
                       'dependencies = ["serde"]\n[[package]]\nname = "serde"\nversion = "1.0.0"\n'
                       'source = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "approved"\n').encode(),
        'CHANGELOG.md': f'## [{version}]\n\nPrepared release notes.\n'.encode(),
    }


class MetadataTests(unittest.TestCase):
    def test_accepts_root_version_only_and_unpublished_bootstrap(self):
        self.assertEqual(release_pr.validate_candidate(metadata('0.0.3'), metadata()), '0.0.3')
        candidate = metadata()
        candidate['CHANGELOG.md'] += b'\nAdditional changes.\n'
        self.assertEqual(release_pr.validate_candidate(candidate, metadata()), '0.0.2')

    def test_rejects_dependency_and_lock_checksum_changes(self):
        for path, old, new in (('Cargo.toml', b'serde = "1"', b'serde = "2"'),
                               ('Cargo.lock', b'checksum = "approved"', b'checksum = "changed"')):
            candidate = metadata('0.0.3')
            candidate[path] = candidate[path].replace(old, new)
            with self.subTest(path=path), self.assertRaises(SystemExit):
                release_pr.validate_candidate(candidate, metadata())

    def test_rejects_toml_value_type_changes(self):
        for path, old, new in (('Cargo.toml', b'codegen-units = 1', b'codegen-units = true'),
                               ('Cargo.lock', b'version = 4', b'version = 4.0')):
            candidate = metadata('0.0.3')
            candidate[path] = candidate[path].replace(old, new)
            with self.subTest(path=path), self.assertRaises(SystemExit):
                release_pr.validate_candidate(candidate, metadata())

    def test_rejects_mismatched_lock_and_version_regression(self):
        mismatched = metadata('0.0.3')
        mismatched['Cargo.lock'] = metadata()['Cargo.lock']
        for candidate in (mismatched, metadata('0.0.1')):
            with self.subTest(candidate=candidate), self.assertRaises(SystemExit):
                release_pr.validate_candidate(candidate, metadata())

    def test_rejects_ambiguous_root_and_changelog_sections(self):
        root = metadata('0.0.3')
        root['Cargo.lock'] += b'\n[[package]]\nname = "krasm"\nversion = "0.0.3"\n'
        notes = metadata('0.0.3')
        notes['CHANGELOG.md'] *= 2
        for candidate in (root, notes):
            with self.subTest(candidate=candidate), self.assertRaises(SystemExit):
                release_pr.validate_candidate(candidate, metadata())


class GitHubModel:
    def __init__(self, existing=True):
        self.master = BASE
        self.branch = OLD if existing else None
        self.race = False
        self.human_pr = False
        self.orphan_branch = False
        self.foreign_change = False
        self.mutations = []
        self.files = None
        self.pulls = []
        self.labelled = False

    def api(self, method, path, payload=None, missing=False):
        repo = f'repos/{release_pr.REPOSITORY}'
        if method != 'GET':
            self.mutations.append((method, path))
        if path == repo:
            return {'full_name': release_pr.REPOSITORY, 'owner': {'id': 495647},
                    'default_branch': 'master', 'node_id': 'fixture-repository'}
        if path == f'{repo}/git/ref/heads/master':
            return {'object': {'sha': self.master}}
        if path == f'{repo}/git/ref/heads/{release_pr.BRANCH}':
            return None if self.branch is None else {'object': {'sha': self.branch}}
        if path.startswith(f'{repo}/pulls?'):
            if not self.branch or self.orphan_branch:
                return []
            return [{'number': 7, 'base': {'ref': 'master'},
                     'head': {'repo': {'full_name': release_pr.REPOSITORY}},
                     'user': {'login': 'rvagg' if self.human_pr else 'github-actions[bot]'}}]
        if path.startswith(f'{repo}/compare/'):
            name = 'src/runtime/compiler.rs' if self.foreign_change else 'Cargo.toml'
            return {'files': [{'filename': name, 'status': 'modified'}],
                    'merge_base_commit': {'sha': ANCESTOR}}
        if path.startswith(f'{repo}/contents/'):
            relative, _, sha = path.removeprefix(f'{repo}/contents/').partition('?ref=')
            data = metadata('0.0.1' if sha == ANCESTOR else '0.0.2')[relative]
            return {'type': 'file', 'encoding': 'base64', 'size': len(data),
                    'content': base64.b64encode(data).decode()}
        if path == f'{repo}/git/commits/{BASE}':
            return {'tree': {'sha': 'fixture-base-tree'}}
        if path == f'{repo}/git/trees':
            self.files = {entry['path']: entry['content'].encode() for entry in payload['tree']}
            return {'sha': 'fixture-prepared-tree'}
        if path == f'{repo}/git/commits':
            return {'sha': PREPARED}
        if path == 'graphql':
            update = payload['variables']['updates'][0]
            if self.race:
                self.branch = HUMAN
            if 'beforeOid' in update and update['beforeOid'] != self.branch:
                raise SystemExit('fixture lease rejected')
            self.branch = update['afterOid']
            return {'data': {'updateRefs': {'clientMutationId': None}}}
        if path == f'{repo}/git/refs':
            if self.branch is not None:
                raise SystemExit('fixture existing branch')
            self.branch = payload['sha']
            return {'ref': payload['ref']}
        if path == f'{repo}/pulls' or path == f'{repo}/pulls/7':
            self.pulls.append(copy.deepcopy(payload))
            return {'number': 7, 'html_url': 'https://github.com/rvagg/krasm/pull/7'}
        if path.startswith(f'{repo}/labels/'):
            return {'name': 'autorelease: pending'}
        if path == f'{repo}/issues/7/labels':
            self.labelled = True
            return []
        raise AssertionError((method, path, payload))


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifact = self.root / 'artifact.zip'
        self.candidate = metadata('0.0.3')
        self.write_artifact()
        self.environment = {'GITHUB_SHA': BASE, 'GITHUB_REPOSITORY': release_pr.REPOSITORY,
                            'GITHUB_SERVER_URL': 'https://github.com', 'GITHUB_REF': 'refs/heads/master',
                            'GITHUB_EVENT_NAME': 'push'}
        self.environment.update(GITHUB_RUN_ID='42', GITHUB_RUN_ATTEMPT='2',
                                GITHUB_OUTPUT=str(self.root / 'outputs'))

    def write_artifact(self, extra=None):
        payload = {'base': BASE, 'files': {path: base64.b64encode(data).decode()
                                        for path, data in self.candidate.items()}}
        payload.update(extra or {})
        with zipfile.ZipFile(self.artifact, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('release-metadata.json', json.dumps(payload))

    def propose(self, model, environment=None):
        with mock.patch.dict(os.environ, environment or self.environment, clear=True), \
             mock.patch.object(release_pr, 'GitHub', return_value=model), \
             mock.patch.object(release_pr, 'git_source', side_effect=lambda source, base, path: metadata()[path]), \
             redirect_stdout(io.StringIO()):
            release_pr.propose(self.root, self.artifact, BASE)

    def test_creates_and_refreshes_metadata_branch_without_changing_master(self):
        for existing in (False, True):
            model = GitHubModel(existing)
            self.propose(model)
            self.assertEqual(model.master, BASE)
            self.assertEqual(model.branch, PREPARED)
            self.assertTrue(model.labelled)

    def test_branch_lease_preserves_concurrent_human_commit(self):
        model = GitHubModel()
        model.race = True
        with self.assertRaises(SystemExit):
            self.propose(model)
        self.assertEqual(model.branch, HUMAN)
        self.assertFalse(model.pulls)
        self.assertFalse(model.labelled)

    def test_master_drift_stops_before_any_mutation(self):
        model = GitHubModel()
        model.master = HUMAN
        with self.assertRaises(SystemExit):
            self.propose(model)
        self.assertFalse(model.mutations)

    def test_preserves_non_metadata_changes_and_human_created_pr(self):
        for field in ('foreign_change', 'human_pr', 'orphan_branch'):
            model = GitHubModel()
            setattr(model, field, True)
            with self.subTest(field=field), self.assertRaises(SystemExit):
                self.propose(model)
            self.assertFalse(model.mutations)
            self.assertEqual(model.branch, OLD)

    def test_rejects_other_run_artifact_and_extra_file(self):
        for change in ({'base': HUMAN}, {'files': {'../Cargo.toml': 'YQ=='}}):
            self.write_artifact(change)
            model = GitHubModel()
            with self.subTest(change=change), self.assertRaises(SystemExit):
                self.propose(model)
            self.assertFalse(model.mutations)

    def test_rejects_archive_bombs_and_foreign_members_before_writes(self):
        for members in ((('release-metadata.json', b'x' * 1025),),
                        (('../release-metadata.json', b'{}'),),
                        (('release-metadata.json', b'{}'), ('extra', b'x'))):
            with zipfile.ZipFile(self.artifact, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in members:
                    archive.writestr(name, data)
            model = GitHubModel()
            with self.subTest(members=tuple(name for name, _ in members)), \
                    mock.patch.object(release_pr, 'MAX_ARTIFACT', 1024), self.assertRaises(SystemExit):
                self.propose(model)
            self.assertFalse(model.mutations)
            self.assertEqual(model.branch, OLD)

    def test_artifact_selection_rejects_wrong_run_and_size_before_download(self):
        valid = {'id': 17, 'size_in_bytes': 512, 'expired': False,
                 'name': 'release-metadata-42-2', 'workflow_run': {'id': 42, 'head_sha': BASE}}
        for change in ({'size_in_bytes': release_pr.MAX_ARTIFACT + 1},
                       {'workflow_run': {'id': 43, 'head_sha': BASE}},
                       {'workflow_run': {'id': 42, 'head_sha': HUMAN}},
                       {'expired': True}, {'name': 'release-metadata-42-1'}):
            response = {'total_count': 1, 'artifacts': [valid | change]}
            github = mock.Mock()
            github.api.return_value = response
            with self.subTest(change=change), \
                    mock.patch.dict(os.environ, self.environment, clear=True), \
                    mock.patch.object(release_pr, 'GitHub', return_value=github), self.assertRaises(SystemExit):
                release_pr.locate_artifact(BASE)
            self.assertFalse(Path(self.environment['GITHUB_OUTPUT']).exists())

    def test_rejects_unapproved_workflow_context(self):
        changed = self.environment | {'GITHUB_REF': 'refs/heads/unreviewed'}
        model = GitHubModel()
        with self.assertRaises(SystemExit):
            self.propose(model, changed)
        self.assertFalse(model.mutations)


if __name__ == '__main__':
    unittest.main()
