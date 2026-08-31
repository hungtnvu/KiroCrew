"""The doctor Credentials section — the self-service answer to "AWS is unavailable".

Advisory by construction: an unconfigured AWS profile is not a Kiro Crew fault, so
every case here also asserts that ``issues`` stays empty. A regression that made
this section blocking would turn ``doctor`` red on every host that does not use
AWS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiro_crew import cli_doctor


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(cli_doctor, "_credential_vendor_line", lambda: "")
    return tmp_path


class TestCredentialsSection:
    def test_no_aws_config_is_reported_without_failing(self, fake_home, capsys):
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        out = capsys.readouterr().out
        assert "Credentials" in out
        assert "no ~/.aws config" in out
        assert issues == []

    def test_profiles_are_listed(self, fake_home, capsys):
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "config").write_text(
            "[default]\nregion = us-west-2\n\n[profile build]\nregion = us-east-1\n"
        )
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        out = capsys.readouterr().out
        assert "default" in out
        assert "build" in out
        assert issues == []

    def test_credential_process_is_called_out(self, fake_home, capsys):
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "config").write_text("[profile p]\ncredential_process = /usr/bin/vend\n")
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        out = capsys.readouterr().out
        assert "credential_process configured" in out
        assert issues == []

    def test_absent_credential_process_is_reported(self, fake_home, capsys):
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "config").write_text("[profile p]\nregion = us-west-2\n")
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        assert "no credential_process" in capsys.readouterr().out
        assert issues == []

    def test_credentials_file_alone_still_reports_a_profile(self, fake_home, capsys):
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "credentials").write_text("[default]\n")
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        assert "default" in capsys.readouterr().out
        assert issues == []

    def test_no_secret_value_is_printed(self, fake_home, capsys):
        """The credentials file is probed for existence only, never read."""
        aws = fake_home / ".aws"
        aws.mkdir()
        (aws / "config").write_text("[profile p]\nregion = us-west-2\n")
        (aws / "credentials").write_text("[p]\naws_secret_access_key = SUPERSECRETVALUE\n")
        cli_doctor._doctor_credentials([])
        out = capsys.readouterr().out
        assert "SUPERSECRETVALUE" not in out
        assert "aws_secret_access_key" not in out

    def test_the_misdiagnosis_note_is_always_present(self, fake_home, capsys):
        """This line is the point of the section — it must not be conditional."""
        cli_doctor._doctor_credentials([])
        out = capsys.readouterr().out
        assert "cannot READ credential files" in out
        assert "blocked-commands.md" in out

    def test_vendor_line_is_shown_when_the_edition_has_one(self, fake_home, monkeypatch, capsys):
        monkeypatch.setattr(
            cli_doctor, "_credential_vendor_line", lambda: "may vend credentials (creds-agent)"
        )
        issues: list[str] = []
        cli_doctor._doctor_credentials(issues)
        out = capsys.readouterr().out
        assert "vending MCP" in out
        assert "creds-agent" in out
        assert issues == []


class TestProfileHeaderScan:
    def test_strips_the_profile_prefix(self, tmp_path):
        path = tmp_path / "config"
        path.write_text("[profile alpha]\n[beta]\n")
        assert cli_doctor._aws_profile_names(path) == ["alpha", "beta"]

    def test_deduplicates(self, tmp_path):
        path = tmp_path / "config"
        path.write_text("[profile a]\n[profile a]\n")
        assert cli_doctor._aws_profile_names(path) == ["a"]

    def test_ignores_key_lines(self, tmp_path):
        path = tmp_path / "config"
        path.write_text("[profile a]\nregion = us-west-2\ncredential_process = /bin/x\n")
        assert cli_doctor._aws_profile_names(path) == ["a"]

    def test_unreadable_file_yields_nothing(self, tmp_path):
        assert cli_doctor._aws_profile_names(tmp_path / "missing") == []


class TestVendorLineIsFailSoft:
    def test_public_edition_yields_no_line(self):
        """The public default reports available() False, so nothing is probed."""
        assert cli_doctor._credential_vendor_line() == ""

    def test_a_lookup_error_degrades_to_no_line(self, monkeypatch):
        import kiro_crew.platform.context as ctx_mod

        def _boom(fn, **kw):
            raise RuntimeError("composition exploded")

        monkeypatch.setattr(ctx_mod, "safe_context_call", _boom, raising=True)
        assert cli_doctor._credential_vendor_line() == ""
