"""Direct unit tests for the backend-only frontend-skip decision.

``frontend_skip`` (issue #7132, the repo-visible follow-up to PR #7123's
deferral) is the pure, stdlib-only helper the sync runner consults at runtime to
decide whether a backend-only Pull+Build may skip BOTH frontend steps -- the
``npm ci`` reinstall and the vite build+stage. These tests pin the DECISION:
that it skips only on strong, positive evidence of both conditions and is
conservative everywhere else.

The module is loaded BY FILE PATH, exactly as the sync runner loads its
snapshot, rather than through ``kiro_crew.apps.builtins.dev_fleet`` -- the
dotted import would execute the package ``__init__`` chain (which pulls in
croniter and the rest of the runtime), and the module imports nothing but the
standard library, so it needs no package context. Loading by path is also what
makes these runnable without the project's runtime dependencies installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

_HELPER = (
    Path(__file__).resolve().parents[1]
    / "src" / "kiro_crew" / "apps" / "builtins" / "dev_fleet" / "frontend_skip.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("frontend_skip", _HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fs = _load()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _stage_dist(repo: Path, *, index: bool = True) -> None:
    """Populate the served bundle at src/kiro_crew/static/dist.

    ``index=True`` writes the ``index.html`` resolution marker frontend.py
    requires; ``index=False`` leaves the directory present but empty (the
    interrupted-stage case)."""
    static_dist = repo / "src" / "kiro_crew" / "static" / "dist"
    static_dist.mkdir(parents=True, exist_ok=True)
    if index:
        (static_dist / "index.html").write_bytes(b"<!doctype html>")


def _fake_repo(
    tmp_path: Path, *, lockfile: bytes, hidden: bytes | None, dist: bool = True
) -> Path:
    """A checkout whose website/ carries a lockfile and (optionally) an installed
    tree recording the lockfile it was installed from.

    ``dist=True`` also stages a usable built bundle at static/dist so the
    dist-presence precondition is satisfied; pass ``dist=False`` to exercise the
    absent-dist guard."""
    website = tmp_path / "website"
    _write(website / "package-lock.json", lockfile)
    if hidden is not None:
        _write(website / "node_modules" / ".package-lock.json", hidden)
    if dist:
        _stage_dist(tmp_path)
    return tmp_path


class _FakeGit:
    """Stands in for the git binary: records the last argv and replays canned
    output for ``git show`` and ``git diff``.

    Installed as the ``git`` argument, which the helper passes straight to
    ``subprocess.run``; we intercept by monkeypatching ``subprocess.run`` on the
    loaded module so no real git process is ever spawned.
    """

    def __init__(self, *, show: bytes | None, diff_names: bytes, show_rc: int = 0,
                 diff_rc: int = 0):
        self.show = show
        self.diff_names = diff_names
        self.show_rc = show_rc
        self.diff_rc = diff_rc
        self.calls: list[list[str]] = []

    def run(self, argv, capture_output=True, timeout=None, check=False):
        self.calls.append(list(argv))
        sub = argv[3] if len(argv) > 3 else ""

        class _Proc:
            pass

        p = _Proc()
        if sub == "show":
            if self.show is None:
                p.returncode = 1
                p.stdout = b""
            else:
                p.returncode = self.show_rc
                p.stdout = self.show
        elif sub == "diff":
            p.returncode = self.diff_rc
            p.stdout = self.diff_names
        else:  # pragma: no cover - defensive
            p.returncode = 1
            p.stdout = b""
        return p


@pytest.fixture
def patch_git(monkeypatch):
    def _install(fake: _FakeGit):
        monkeypatch.setattr(fs.subprocess, "run", fake.run)
        return fake
    return _install


LOCK = b'{"name":"website","lockfileVersion":3,"packages":{}}\n'
OTHER = b'{"name":"website","lockfileVersion":3,"packages":{"x":{}}}\n'


def test_skips_when_website_unchanged_and_tree_matches(patch_git, tmp_path):
    """The one case a skip is safe: empty website/ diff AND the on-disk tree's
    hidden lockfile byte-matches both the working tree and the incoming ref."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=LOCK)
    patch_git(_FakeGit(show=LOCK, diff_names=b""))
    assert fs.may_skip_frontend("git", str(repo), "refs/x") is True


def test_does_not_skip_when_website_changed(patch_git, tmp_path):
    """A non-empty website/ diff is a definite change -> build."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=LOCK)
    patch_git(_FakeGit(show=LOCK, diff_names=b"website/package.json\n"))
    assert fs.may_skip_frontend("git", str(repo), "refs/x") is False


def test_does_not_skip_when_hidden_lockfile_absent(patch_git, tmp_path):
    """No node_modules/.package-lock.json means the tree is not verifiably the
    lockfile's (or node_modules is absent) -- npm ci must run to repair it."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=None)
    patch_git(_FakeGit(show=LOCK, diff_names=b""))
    assert fs.may_skip_frontend("git", str(repo), "refs/x") is False


def test_does_not_skip_when_hidden_lockfile_mismatches(patch_git, tmp_path):
    """A partially-populated tree: the hidden lockfile does not match, so the
    stronger-than-#7123 verification refuses to skip."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=OTHER)
    patch_git(_FakeGit(show=LOCK, diff_names=b""))
    assert fs.may_skip_frontend("git", str(repo), "refs/x") is False


def test_does_not_skip_when_incoming_lockfile_differs(patch_git, tmp_path):
    """Even with an empty diff, an incoming lockfile that differs from the tree's
    means do not skip -- the direct three-way match is required."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=LOCK)
    patch_git(_FakeGit(show=OTHER, diff_names=b""))
    assert fs.may_skip_frontend("git", str(repo), "refs/x") is False


def test_does_not_skip_when_git_diff_unavailable(patch_git, tmp_path):
    """git failing to answer the diff (rc != 0) is unobtainable evidence -> the
    conservative fallback is to build, never to skip."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=LOCK)
    patch_git(_FakeGit(show=LOCK, diff_names=b"", diff_rc=128))
    assert fs.website_diff_is_empty("git", str(repo), "refs/x") is None
    assert fs.may_skip_frontend("git", str(repo), "refs/x") is False


def test_does_not_skip_when_incoming_lockfile_unreadable(patch_git, tmp_path):
    """git show of the incoming lockfile failing -> no strong evidence -> build."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=LOCK)
    patch_git(_FakeGit(show=None, diff_names=b""))
    assert fs.node_modules_matches_lockfile("git", str(repo), "refs/x") is False
    assert fs.may_skip_frontend("git", str(repo), "refs/x") is False


def test_website_diff_uses_head_and_ref_scoped_to_website(patch_git, tmp_path):
    """The diff is HEAD..ref restricted to website/ -- the same ref pair the sync
    merges (git merge --ff-only <ref>), scoped to the frontend subdir."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=LOCK)
    fake = patch_git(_FakeGit(show=LOCK, diff_names=b""))
    fs.website_diff_is_empty("git", str(repo), "refs/kirocrew/sync-base-99")
    diff_calls = [c for c in fake.calls if len(c) > 3 and c[3] == "diff"]
    assert diff_calls, fake.calls
    argv = diff_calls[0]
    assert "--name-only" in argv
    assert "HEAD" in argv and "refs/kirocrew/sync-base-99" in argv
    assert "--" in argv and "website" in argv


def test_does_not_skip_when_built_dist_absent(patch_git, tmp_path):
    """Issue #7132's explicit case: the website/ diff is empty and all three
    lockfiles match, but no built bundle exists at static/dist. Skipping would
    leave the dashboard with no assets where every prior stock Pull+Build
    rebuilt them, so the presence gate forces do-not-skip."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=LOCK, dist=False)
    patch_git(_FakeGit(show=LOCK, diff_names=b""))
    # The lockfile evidence is fully satisfied on its own...
    assert fs.node_modules_matches_lockfile("git", str(repo), "refs/x") is True
    assert fs.website_diff_is_empty("git", str(repo), "refs/x") is True
    # ...yet the missing dist forces a build.
    assert fs.built_dist_is_present(str(repo)) is False
    assert fs.may_skip_frontend("git", str(repo), "refs/x") is False


def test_does_not_skip_when_built_dist_has_no_index(patch_git, tmp_path):
    """An interrupted stage can leave static/dist present but without the
    index.html frontend.py resolves the bundle by. That is not a usable bundle,
    so the presence gate still forces do-not-skip."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=LOCK, dist=False)
    _stage_dist(repo, index=False)  # empty dist directory, no index.html
    patch_git(_FakeGit(show=LOCK, diff_names=b""))
    assert fs.built_dist_is_present(str(repo)) is False
    assert fs.may_skip_frontend("git", str(repo), "refs/x") is False


def test_built_dist_present_follows_symlink(tmp_path):
    """On a source-tree install static/dist is a symlink to website/dist; the
    Path.is_file() probe follows the link, so a symlinked bundle counts as
    present -- matching how frontend.ensure_dev_dist_symlink resolves it."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=LOCK, dist=False)
    website_dist = repo / "website" / "dist"
    website_dist.mkdir(parents=True, exist_ok=True)
    (website_dist / "index.html").write_bytes(b"<!doctype html>")
    static_parent = repo / "src" / "kiro_crew" / "static"
    static_parent.mkdir(parents=True, exist_ok=True)
    try:
        (static_parent / "dist").symlink_to(website_dist, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - symlink-less FS
        pytest.skip("filesystem does not support symlinks")
    assert fs.built_dist_is_present(str(repo)) is True


def test_hidden_lockfile_match_is_a_real_sha256(patch_git, tmp_path):
    """Guard the mechanism, not just the branch: a one-byte change to the tree's
    lockfile flips the verdict, so the check is a genuine content comparison."""
    repo = _fake_repo(tmp_path, lockfile=LOCK, hidden=LOCK)
    patch_git(_FakeGit(show=LOCK, diff_names=b""))
    assert fs.node_modules_matches_lockfile("git", str(repo), "refs/x") is True
    # Perturb one byte of the on-disk tree's lockfile.
    (repo / "website" / "package-lock.json").write_bytes(LOCK + b" ")
    assert fs.node_modules_matches_lockfile("git", str(repo), "refs/x") is False
    # sanity: the constant really is the sha256 the helper compares on
    assert hashlib.sha256(LOCK).hexdigest() != hashlib.sha256(LOCK + b" ").hexdigest()
