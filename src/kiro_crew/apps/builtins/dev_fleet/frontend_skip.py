"""Runtime decision: may a backend-only Pull+Build skip the frontend half?

Follow-up to PR #7123, tracked as issue #7132. That PR removed the pre-merge
installability probe's scratch ``npm ci`` on a backend-only sync by asking one
question -- ``git diff --name-only <ref> -- website`` empty plus a populated
``node_modules`` -- but DEFERRED the larger share: the real ``npm ci --prefix
website`` (which deletes and reinstalls the checkout's own tree) and the full
``npm run build`` + dist stage. This module answers whether those two may be
skipped, and it is deliberately its own file for three reasons the issue spells
out:

* **It runs at RUNTIME, inside the generated sync runner.** The only evidence a
  sync changes nothing under ``website/`` is a diff against the fetched base
  ref, and that ref (``refs/kirocrew/sync-base-<pid>``) does not exist on disk
  until the sync's own ``fetch`` step has run. The step list is assembled BEFORE
  fetch, so the decision cannot be made at assembly time; it is made here, in
  the runner, after fetch.

* **It needs STRONGER evidence than #7123.** ``npm ci`` is also what REPAIRS a
  partially populated tree, so "node_modules is non-empty" is not enough to
  skip it -- an interrupted earlier install would then never be healed. This
  module verifies the on-disk tree against the incoming lockfile: npm writes a
  hidden lockfile at ``node_modules/.package-lock.json`` recording exactly what
  it last installed the tree from, so a byte match between that, the working
  tree's ``package-lock.json``, and the lockfile in the incoming ref is strong
  evidence the tree already IS the one ``npm ci`` would produce.

  It does NOT lean on ``npm ci --dry-run``: as :mod:`npm_preflight` documents,
  a dry run never fetches and passes lockfiles a real install would fail, so it
  is the wrong tool for asserting a tree is complete.

  It ALSO requires a usable built ``dist`` to already exist. The build+stage
  step's job is to populate ``src/kiro_crew/static/dist``, and before this
  change it ran on every stock Pull+Build -- repairing an absent dist as a side
  effect. ``node_modules`` and the built bundle are independent artifacts, so
  the tree can match the lockfile while the dist is missing; skipping the build
  without confirming a bundle is on disk would leave the dashboard with no
  assets. So the skip also gates on ``dist/index.html`` presence -- the same
  marker ``frontend.py`` resolves the runtime bundle by.

* **Skipping is CONSERVATIVE.** When any evidence is weak, missing, or
  unobtainable without the network, this returns "do not skip" and the sync
  runs ``npm ci`` exactly as it does today. Skipping is only ever the answer on
  strong, positive evidence, because the failure mode of a wrong skip (a stale
  or partial tree served as if fresh) is worse than the cost it saves.

This module imports ONLY the standard library. The sync runner is a stdlib-only
``python -c`` program that must not import ``kiro_crew`` (that would drag in the
package ``__init__`` chain, which imports croniter and the rest of the runtime),
so this helper is snapshotted at import and executed BY PATH the same way
``dep_sync`` and ``npm_preflight`` are -- see ``server._sync_start_locked``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess  # nosec B404 - reading git is this module's purpose
from pathlib import Path

#: The checkout subdirectory holding the frontend half, matching
#: :mod:`npm_preflight`'s ``_FRONTEND_SUBDIR``.
_FRONTEND_SUBDIR = "website"

#: The lockfile the tree must match, both in the working tree and in the ref.
_LOCKFILE = "package-lock.json"

#: npm's OWN record of what it last installed ``node_modules`` from. npm writes
#: this hidden lockfile into the tree on every ``npm ci``/``npm install``, so a
#: byte match between it and the incoming ``package-lock.json`` is evidence the
#: on-disk tree already IS what ``npm ci`` would reproduce -- far stronger than
#: "the directory is non-empty", which #7123's bar allowed and which cannot tell
#: a complete tree from one an interrupted install left half-written.
_HIDDEN_LOCKFILE = os.path.join("node_modules", ".package-lock.json")

#: The SERVED frontend bundle, relative to the repo root: the build+stage step's
#: whole job is to populate this directory (``<repo>/src/kiro_crew/static/dist``)
#: so the gateway can serve the SPA. On a packaged install it is a real directory
#: shipped in the wheel; on a source-tree run it is a symlink to
#: ``website/dist``. Either way ``frontend.py`` resolves the runtime bundle from
#: here and treats ``index.html`` as the marker of a usable dist -- see
#: ``frontend.ensure_dev_dist_symlink`` / ``_resolve_website_dist``. Because
#: ``Path`` stat calls follow symlinks, one ``index.html`` probe on this path
#: covers BOTH layouts.
_STATIC_DIST = os.path.join("src", "kiro_crew", "static", "dist")

#: The resolution marker ``frontend.py`` requires before it will serve a bundle;
#: an absent ``index.html`` is exactly what makes it fall back to the "not built"
#: guidance page.
_DIST_INDEX = "index.html"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_show(git: str, repo: str, ref: str, rel: str) -> bytes | None:
    """Return ``<ref>:<rel>`` bytes, or ``None`` if it cannot be read.

    Reads out of the fetched ref rather than the working tree -- the same move
    :mod:`npm_preflight` makes -- so the answer is about the revision the sync is
    about to land, not whatever happens to be checked out. Any failure (git
    missing, ref absent, path not in the ref, timeout) collapses to ``None``,
    which the caller treats as "evidence unobtainable -> do not skip".
    """
    try:
        proc = subprocess.run(  # nosec B603 - argv list, no shell
            [git, "-C", repo, "show", f"{ref}:{rel}"],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _read_bytes(path: Path) -> bytes | None:
    """Return *path* bytes, or ``None`` on any read failure."""
    try:
        return path.read_bytes()
    except OSError:
        return None


def website_diff_is_empty(git: str, repo: str, ref: str) -> bool | None:
    """Does the incoming sync change nothing under ``website/``?

    Compares the fetched base ref against the working tree's ``HEAD`` -- the same
    ref pair the sync merges (``git merge --ff-only <ref>``), so "what changes
    under website/" is exactly the ff-only merge's website/ delta. ``True`` when
    the diff is empty, ``False`` when it lists anything, ``None`` when git could
    not be run at all (unobtainable evidence, which the caller treats as "do not
    skip").
    """
    try:
        proc = subprocess.run(  # nosec B603 - argv list, no shell
            [git, "-C", repo, "diff", "--name-only", "HEAD", ref,
             "--", _FRONTEND_SUBDIR],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return not proc.stdout.strip()


def node_modules_matches_lockfile(git: str, repo: str, ref: str) -> bool:
    """Does the on-disk ``node_modules`` verifiably satisfy the incoming lockfile?

    STRONGER than #7123's "node_modules is non-empty", because ``npm ci`` is also
    what repairs a partially populated tree -- skipping it safely requires the
    tree to be VERIFIED, not merely present. The verification is a three-way byte
    match, and each leg is required:

      1. The incoming ref's ``package-lock.json`` == the working tree's
         ``package-lock.json``. When ``website/`` is unchanged this holds by
         definition, but checking it directly means this function is correct even
         if a caller ever invokes it without the diff gate: it will not skip
         against a tree built from a different lockfile.
      2. The working tree's ``package-lock.json`` == ``node_modules``'s hidden
         ``.package-lock.json`` (npm's own record of what it installed the tree
         from). This is what proves the tree matches the lockfile rather than
         merely coexisting with it, and it is what excludes the partial-install
         case that ``npm ci`` exists to repair: an interrupted install leaves the
         hidden lockfile out of sync with (or absent from) the tree.

    Any missing or unreadable input returns ``False`` (conservative: no strong
    evidence means do not skip). It NEVER returns ``None``: an unobtainable input
    here is simply weak evidence, indistinguishable in consequence from a
    mismatch -- either way, do not skip.
    """
    incoming = _git_show(git, repo, ref, f"{_FRONTEND_SUBDIR}/{_LOCKFILE}")
    if not incoming:
        return False
    root = Path(repo) / _FRONTEND_SUBDIR
    worktree_lock = _read_bytes(root / _LOCKFILE)
    if worktree_lock is None:
        return False
    hidden_lock = _read_bytes(root / _HIDDEN_LOCKFILE)
    if hidden_lock is None:
        # No hidden lockfile means either node_modules is absent or it was not
        # installed by a modern npm ci/install -- in both cases the tree is not
        # verifiably the lockfile's, so npm ci must run.
        return False
    incoming_hash = _sha256(incoming)
    return (
        _sha256(worktree_lock) == incoming_hash
        and _sha256(hidden_lock) == incoming_hash
    )


def built_dist_is_present(repo: str) -> bool:
    """Is a USABLE built frontend bundle already on disk?

    The build+stage step exists to produce ``<repo>/src/kiro_crew/static/dist``,
    the directory the gateway serves the SPA from. Skipping that step is only
    safe when a usable bundle is ALREADY staged there -- otherwise a backend-only
    sync on a checkout whose dist was never built (or whose stage was interrupted)
    would ``continue`` past the build and leave the dashboard with no assets,
    where every prior stock Pull+Build rebuilt it as a side effect.

    The lockfile evidence in :func:`node_modules_matches_lockfile` does NOT cover
    this: ``node_modules`` and the built dist are INDEPENDENT artifacts, so the
    tree can match the lockfile while ``static/dist`` is absent. Dist freshness
    (stale CONTENT) rides on the diff-empty condition -- a sync that changes
    nothing under ``website/`` would rebuild a byte-identical bundle -- but dist
    PRESENCE does not, so it needs its own gate.

    We mirror how ``frontend.py`` resolves the runtime bundle: it treats
    ``static/dist/index.html`` as the marker of a usable dist (see
    ``ensure_dev_dist_symlink`` / ``_resolve_website_dist``), and its ``Path``
    probes follow symlinks, so this one ``index.html`` check covers both the
    packaged real directory and the source-tree symlink to ``website/dist``. We
    do NOT re-implement its deeper asset-completeness scan here: this gate is a
    CONSERVATIVE presence floor, and ``build_and_stage`` remains the safety net
    that produces a complete bundle whenever this returns ``False`` and the
    build runs. Any read failure collapses to ``False`` (do not skip -> build).
    """
    try:
        index = Path(repo) / _STATIC_DIST / _DIST_INDEX
        return index.is_file()
    except OSError:
        return False


def may_skip_frontend(git: str, repo: str, ref: str) -> bool:
    """The one decision the runner consults: skip BOTH frontend steps?

    The two steps -- ``npm ci`` and ``npm run build`` + stage -- are COUPLED on
    purpose: they share one cause (no ``website/`` change), and skipping the
    build without the install (or vice versa) has no coherent meaning. A build
    over a reinstalled tree and a build over the existing tree produce the same
    bundle only when the tree already matches the lockfile, which is exactly the
    condition checked here. So one evidence check gates both; see
    ``server._sync_start_locked`` where the two steps carry the same skip marker.

    Returns ``True`` only on STRONG evidence of ALL THREE conditions:

      * the incoming sync changes nothing under ``website/``, AND
      * the on-disk ``node_modules`` verifiably satisfies the incoming lockfile,
        AND
      * a usable built bundle is already staged at ``src/kiro_crew/static/dist``.

    Everything else -- a website/ change, an unverifiable tree, a missing built
    dist, or git being unavailable to answer either question -- returns ``False``
    and the sync builds as it does today.

    Dist freshness (stale CONTENT) rides on the diff-empty condition: when this
    sync changes nothing under ``website/``, the bundle it would build is
    byte-for-byte the one already staged, so declining to rebuild serves the same
    bytes. A dist left stale for a reason UNRELATED to this sync is a pre-existing
    condition this sync neither created nor is obligated to repair -- and this
    sync would not have changed it either way, because it touches nothing under
    ``website/``.

    Dist PRESENCE, however, does NOT ride on the diff: ``node_modules`` and the
    built dist are independent artifacts, so the tree can match the lockfile
    while ``static/dist`` is absent or incomplete. Before this change the
    build+stage step ran on every stock Pull+Build and repaired a missing dist as
    a side effect; skipping it removes that repair, so we add an explicit
    presence gate (see :func:`built_dist_is_present`) so the skip fires only when
    a usable bundle already exists. The gate covers the whole coupled verdict
    rather than only the build leg: the two steps skip together or neither does,
    so gating the shared verdict on dist presence keeps that coupling intact --
    an absent dist forces BOTH ``npm ci`` and the build to run, exactly as today.
    """
    if website_diff_is_empty(git, repo, ref) is not True:
        # None (git unavailable) and False (website/ changed) both mean do not
        # skip. Only a definite empty diff clears this gate.
        return False
    if not built_dist_is_present(repo):
        # No usable served bundle on disk -> must build+stage to produce one.
        return False
    return node_modules_matches_lockfile(git, repo, ref)
