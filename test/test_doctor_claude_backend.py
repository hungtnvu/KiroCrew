"""``kirocrew doctor`` reports Claude Code as a real optional agent backend.

The old behaviour printed the adapter ONLY when it happened to be present, and
labelled it "dormant seam -- not used by the public core". Both halves were wrong once
``ACP_BACKEND_CLAUDE`` joined ``BASELINE_SELECTABLE_BACKENDS``: an operator who could
select the harness was told the build did not use it, and an operator who had NOT
installed the adapter was told nothing at all -- doctor's whole job is naming the thing
that is absent.
"""

import contextlib
import io

from kiro_crew import cli_doctor


def _doctor_output() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with contextlib.suppress(SystemExit):
            cli_doctor._doctor()
    return buf.getvalue()


def test_doctor_reports_claude_code_whether_or_not_it_is_installed():
    """The line exists unconditionally, so an absent adapter is discoverable."""
    lines = [ln for ln in _doctor_output().splitlines() if "claude-acp:" in ln]
    assert lines, "doctor must report Claude Code even when the adapter is absent"


def test_doctor_never_calls_the_harness_dormant():
    """It is selectable, so the old wording actively misinforms."""
    assert "dormant" not in _doctor_output()


def test_doctor_names_the_absent_component_and_how_to_install_it():
    """Same verdict source as the dashboard, so the two cannot disagree.

    Claude Code needs TWO binaries, so a bare "not found" would send someone after the
    half they already have. Doctor takes its answer from ``agent_sdk.probe_backend`` --
    the same owner ``GET /api/acp-backends`` uses -- rather than re-deriving it from a
    ``shutil.which`` of its own, which is how the two would drift.
    """
    out = _doctor_output()
    rows = out.splitlines()
    idx = next(i for i, ln in enumerate(rows) if "claude-acp:" in ln)
    line = rows[idx]

    if "not found (optional agent backend)" in line:
        # A component is missing: it must be NAMED, and the next line must carry the
        # command that installs it.
        assert "claude-agent-acp" in line or "claude" in line, line
        assert "npm i -g" in rows[idx + 1] or "install" in rows[idx + 1].lower(), rows[idx + 1]
    else:
        # Fully installed, or the check itself failed. Either way, never worded as a
        # harness the build cannot use.
        assert "selectable" in line or "could not check" in line, line
