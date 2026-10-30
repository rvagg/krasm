#!/usr/bin/python3
"""Carry release-plz metadata across the read-only and PR-writer jobs."""
import argparse
import base64
import copy
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import tomllib
from urllib.parse import quote
import zipfile

REPOSITORY = "rvagg/krasm"
BRANCH = "release-plz--master"
FILES = ("Cargo.toml", "Cargo.lock", "CHANGELOG.md")
MAX_FILE = 4 * 1024 * 1024
MAX_ARTIFACT = 20 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{40}\Z")
VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")


def fail(message):
    raise SystemExit(message)


def read_file(path, limit=MAX_FILE):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        fail("release metadata file is invalid")
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        fail("release metadata exceeds its size limit")
    return data


def version_metadata(files):
    try:
        manifest = tomllib.loads(files["Cargo.toml"].decode("utf-8"))
        lock = tomllib.loads(files["Cargo.lock"].decode("utf-8"))
        package = manifest["package"]
        version = package["version"]
        roots = [entry for entry in lock["package"] if entry.get("name") == "krasm" and "source" not in entry]
    except (KeyError, TypeError, AttributeError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        fail("release Cargo metadata is invalid")
    if (package.get("name") != "krasm" or not isinstance(version, str) or not VERSION.fullmatch(version)
            or len(roots) != 1 or roots[0].get("version") != version or "checksum" in roots[0]):
        fail("release package identity or lock version differs")
    return manifest, lock, version


def same_metadata(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_metadata(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(same_metadata(a, b) for a, b in zip(left, right))
    return left == right


def validate_candidate(files, baseline):
    if set(files) != set(FILES):
        fail("release metadata inventory differs")
    current, lock, version = version_metadata(files)
    previous, old_lock, old_version = version_metadata(baseline)
    current = copy.deepcopy(current)
    previous = copy.deepcopy(previous)
    current["package"].pop("version")
    previous["package"].pop("version")
    if not same_metadata(current, previous):
        fail("release Cargo.toml changes more than package.version")
    lock = copy.deepcopy(lock)
    old_lock = copy.deepcopy(old_lock)
    for document in (lock, old_lock):
        for entry in document["package"]:
            if entry.get("name") == "krasm" and "source" not in entry:
                entry.pop("version")
    if not same_metadata(lock, old_lock) or tuple(map(int, version.split("."))) < tuple(map(int, old_version.split("."))):
        fail("release lock dependencies changed or version decreased")
    try:
        changelog = files["CHANGELOG.md"].decode("utf-8")
    except UnicodeDecodeError:
        fail("release changelog is not UTF-8")
    heading = re.compile(r"(?m)^## \[?v?" + re.escape(version) + r"\]?(?:\(|\s|$)")
    if len(heading.findall(changelog)) != 1:
        fail("release changelog has no unique version section")
    return version


def context(base):
    if (not SHA.fullmatch(base) or os.environ.get("GITHUB_SHA") != base
            or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
            or os.environ.get("GITHUB_SERVER_URL") != "https://github.com"
            or os.environ.get("GITHUB_REF") != "refs/heads/master"
            or os.environ.get("GITHUB_EVENT_NAME") not in {"push", "workflow_dispatch"}):
        fail("release preparation requires the approved master context")


def git_source(source, base, path):
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(source), "show", f"{base}:{path}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
    )
    if result.returncode or len(result.stdout) > MAX_FILE:
        fail("cannot read release baseline")
    return result.stdout


def export_candidate(source, output, base):
    context(base)
    baseline = {path: git_source(source, base, path) for path in FILES}
    files = {path: read_file(source / path) for path in FILES}
    validate_candidate(files, baseline)
    if files == baseline:
        print("No release metadata changed.")
        return
    payload = {"base": base, "files": {path: base64.b64encode(data).decode("ascii") for path, data in files.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    destination = os.environ.get("GITHUB_OUTPUT")
    if destination:
        with open(destination, "a", encoding="utf-8") as target:
            target.write("changed=true\n")


class GitHub:
    def __init__(self, home):
        token = os.environ.get("GH_TOKEN")
        if not token:
            fail("release PR writer has no GitHub token")
        self.environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home),
                            "GH_CONFIG_DIR": str(home / ".config/gh"), "GH_HOST": "github.com",
                            "GH_PROMPT_DISABLED": "1", "GH_TOKEN": token}

    def api(self, method, path, payload=None, missing=False):
        if not (path.startswith(f"repos/{REPOSITORY}/") or path == f"repos/{REPOSITORY}" or path == "graphql"):
            fail("release PR endpoint is outside its repository")
        command = ["/usr/bin/gh", "api", "--hostname", "github.com", "--method", method, path]
        if payload is not None:
            command += ["--input", "-"]
        result = subprocess.run(command, input=None if payload is None else json.dumps(payload),
                                env=self.environment, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False, timeout=120)
        if result.returncode:
            if missing and re.search(r"\bHTTP(?:/\S+)?\s+404\b", result.stderr):
                return None
            fail("release PR GitHub operation failed")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError:
            fail("release PR GitHub response is invalid")
        if path == "graphql" and (not isinstance(value, dict) or value.get("errors")):
            fail("release PR branch lease was rejected")
        return value


def remote_files(github, sha):
    if not isinstance(sha, str) or not SHA.fullmatch(sha):
        fail("release baseline identity is invalid")
    files = {}
    for path in FILES:
        value = github.api("GET", f"repos/{REPOSITORY}/contents/{path}?ref={sha}")
        if (value.get("type") != "file" or value.get("encoding") != "base64"
                or value.get("size", MAX_FILE + 1) > MAX_FILE):
            fail("release branch metadata is invalid")
        try:
            files[path] = base64.b64decode("".join(value["content"].split()), validate=True)
        except (KeyError, TypeError, ValueError):
            fail("release branch metadata is invalid")
        if len(files[path]) > MAX_FILE:
            fail("release branch metadata exceeds its size limit")
    return files


def locate_artifact(base):
    context(base)
    run, attempt = (os.environ.get(key, "") for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"))
    if not re.fullmatch(r"[1-9][0-9]*", run) or not re.fullmatch(r"[1-9][0-9]*", attempt):
        fail("release artifact run identity is invalid")
    name = f"release-metadata-{run}-{attempt}"
    with tempfile.TemporaryDirectory(prefix="release-artifact-") as temporary:
        github = GitHub(Path(temporary))
        response = github.api("GET", f"repos/{REPOSITORY}/actions/runs/{run}/artifacts?name={name}&per_page=100")
    if not isinstance(response, dict) or response.get("total_count") != 1:
        fail("release artifact selection is ambiguous")
    artifacts = response.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1 or not isinstance(artifacts[0], dict):
        fail("release artifact selection is invalid")
    artifact = artifacts[0]
    identifier, size = artifact.get("id"), artifact.get("size_in_bytes")
    workflow = artifact.get("workflow_run")
    if (type(identifier) is not int or identifier < 1 or artifact.get("name") != name
            or type(size) is not int or not 0 < size <= MAX_ARTIFACT or artifact.get("expired") is not False
            or not isinstance(workflow, dict) or workflow.get("id") != int(run) or workflow.get("head_sha") != base):
        fail("release artifact size or source run differs")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"artifact-id={identifier}\n")


def read_artifact(artifact):
    if artifact.is_symlink():
        fail("release artifact path is invalid")
    if artifact.is_dir():
        entries = list(artifact.iterdir())
        if len(entries) != 1:
            fail("release artifact inventory differs")
        artifact = entries[0]
    raw = read_file(artifact, MAX_ARTIFACT)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) != 1:
                fail("release artifact inventory differs")
            member = members[0]
            mode = stat.S_IFMT(member.external_attr >> 16)
            if (member.filename != "release-metadata.json" or member.is_dir()
                    or member.flag_bits & 65 or mode not in (0, stat.S_IFREG)
                    or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or not 0 < member.file_size <= MAX_ARTIFACT or member.compress_size > MAX_ARTIFACT):
                fail("release artifact entry is invalid or oversized")
            with archive.open(member) as source:
                payload = source.read(MAX_ARTIFACT + 1)
            if len(payload) != member.file_size:
                fail("release artifact entry is incomplete")
            return json.loads(payload)
    except (OSError, EOFError, ValueError, zipfile.BadZipFile, RuntimeError, NotImplementedError):
        fail("release artifact archive is invalid")


def propose(source, artifact, base):
    context(base)
    try:
        payload = read_artifact(artifact)
        if set(payload) != {"base", "files"} or payload["base"] != base or set(payload["files"]) != set(FILES):
            fail("release artifact does not match its source run")
        files = {path: base64.b64decode(payload["files"][path], validate=True) for path in FILES}
    except (KeyError, TypeError, ValueError):
        fail("release artifact is invalid")
    if any(len(data) > MAX_FILE for data in files.values()):
        fail("release metadata exceeds its size limit")
    baseline = {path: git_source(source, base, path) for path in FILES}
    version = validate_candidate(files, baseline)
    if files == baseline:
        fail("release artifact contains no metadata changes")
    with tempfile.TemporaryDirectory(prefix="release-pr-") as temporary:
        github = GitHub(Path(temporary))
        repo = github.api("GET", f"repos/{REPOSITORY}")
        if (repo.get("full_name") != REPOSITORY or repo.get("owner", {}).get("id") != 495647
                or repo.get("default_branch") != "master"):
            fail("release repository identity differs")
        master = github.api("GET", f"repos/{REPOSITORY}/git/ref/heads/master")
        if master["object"]["sha"] != base:
            fail("master moved during release preparation")
        pulls = github.api("GET", f"repos/{REPOSITORY}/pulls?state=open&head={quote('rvagg:' + BRANCH)}")
        if (len(pulls) > 1 or any(pull["base"]["ref"] != "master"
                                or pull["head"]["repo"]["full_name"] != REPOSITORY
                                or pull["user"]["login"] != "github-actions[bot]" for pull in pulls)):
            fail("release PR selection is ambiguous or not automation-owned")
        branch = github.api("GET", f"repos/{REPOSITORY}/git/ref/heads/{BRANCH}", missing=True)
        if branch:
            if not pulls:
                fail("existing release branch has no open automation-owned PR; inspect it before retrying")
            old = branch["object"]["sha"]
            if not isinstance(old, str) or not SHA.fullmatch(old):
                fail("release branch identity is invalid")
            comparison = github.api("GET", f"repos/{REPOSITORY}/compare/{base}...{old}")
            if any(change.get("filename") not in FILES or change.get("status") != "modified"
                   for change in comparison["files"]):
                fail("release branch contains changes outside generated metadata")
            if comparison["files"]:
                merge_base = comparison["merge_base_commit"]["sha"]
                validate_candidate(remote_files(github, old), remote_files(github, merge_base))
        commit = github.api("GET", f"repos/{REPOSITORY}/git/commits/{base}")
        tree = github.api("POST", f"repos/{REPOSITORY}/git/trees", {
            "base_tree": commit["tree"]["sha"],
            "tree": [{"path": path, "mode": "100644", "type": "blob", "content": data.decode("utf-8")}
                     for path, data in files.items()],
        })
        prepared = github.api("POST", f"repos/{REPOSITORY}/git/commits", {
            "message": f"chore: release {version}", "tree": tree["sha"], "parents": [base],
        })
        if branch:
            query = ("mutation($repository:ID!,$updates:[RefUpdate!]!){updateRefs(input:{"
                     "repositoryId:$repository,refUpdates:$updates}){clientMutationId}}")
            github.api("POST", "graphql", {"query": query, "variables": {
                "repository": repo["node_id"], "updates": [{"name": f"refs/heads/{BRANCH}",
                "beforeOid": old, "afterOid": prepared["sha"], "force": True}],
            }})
        else:
            github.api("POST", f"repos/{REPOSITORY}/git/refs", {
                "ref": f"refs/heads/{BRANCH}", "sha": prepared["sha"],
            })
        final = github.api("GET", f"repos/{REPOSITORY}/git/ref/heads/{BRANCH}")
        master = github.api("GET", f"repos/{REPOSITORY}/git/ref/heads/master")
        if final["object"]["sha"] != prepared["sha"] or master["object"]["sha"] != base:
            fail("release branch or master moved during preparation")
        body = (f"Release-plz prepared version `{version}` from `{base}`.\n\n"
                "Review the Cargo metadata and changelog diff. Leave this PR open; "
                "the owner-operated `release-repo krasm` command checks CI, merges and tags it.")
        values = {"title": f"chore: release {version}", "body": body}
        if pulls:
            pull = github.api("PATCH", f"repos/{REPOSITORY}/pulls/{pulls[0]['number']}", values)
        else:
            pull = github.api("POST", f"repos/{REPOSITORY}/pulls", values | {"head": BRANCH, "base": "master"})
        label = github.api("GET", f"repos/{REPOSITORY}/labels/{quote('autorelease: pending', safe='')}", missing=True)
        if label is None:
            github.api("POST", f"repos/{REPOSITORY}/labels", {"name": "autorelease: pending", "color": "ededed"})
        github.api("POST", f"repos/{REPOSITORY}/issues/{pull['number']}/labels", {"labels": ["autorelease: pending"]})
        print(pull["html_url"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "artifact", "propose"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    if args.command == "artifact":
        locate_artifact(args.base)
        return
    if args.artifact is None:
        parser.error("--artifact is required for export and propose")
    if args.command == "export":
        export_candidate(args.source, args.artifact, args.base)
    else:
        propose(args.source, args.artifact, args.base)


if __name__ == "__main__":
    main()
