"""Remediation guidance attached to a denied tool call — the "do this instead" half.

A refusal that states only WHY leaves the model to invent a way forward, and for
credential work the invention is systematically wrong in a way that costs the
user the capability entirely: the agent re-tries the same shape under a
different reader (``cat`` → ``head`` → ``python open``), each of which the same
rule family blocks, and then reports that the host has no AWS access at all.
The sanctioned path was available the whole time — nothing ever told it.

Guidance is keyed by the CLASS of thing the gate refused, not by the individual
rule. Per-rule text cannot cover the tiers that matter: an edition overlay
contributes bare fnmatch globs carrying no id, category or description, so for
exactly the rules an enterprise adds there is nothing to hang text on. The class
is instead recovered from the refusal text, whose anchor phrases every producer
in :mod:`kiro_crew.security` already shares. ``test_deny_guidance.py`` drives
those real producers rather than asserting on copied strings, so an anchor that
drifts fails there instead of silently degrading to no guidance.

Every string here is static prose. None of it is interpolated from the command,
which is what keeps it safe to hand back to a model that may be acting on
injected content: it names the sanctioned path, never a way around the rule.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

#: Deny classes. The split follows what the caller must DO differently, which is
#: why AWS and enterprise-SSO credentials are separate: one has a local
#: resolution the agent can drive itself (the SDK reads the profile), the other
#: can only be re-established by the human in their own terminal.
DENY_CLASS_AWS_CREDENTIAL = "aws_credential"
DENY_CLASS_SSO_CREDENTIAL = "sso_credential"
DENY_CLASS_SECRET_FILE = "secret_file"
DENY_CLASS_TRUST_ROOT = "trust_root"
DENY_CLASS_EXFIL_SHAPE = "exfil_shape"
DENY_CLASS_SELF_PROTECTION = "self_protection"

#: Ordered (class, anchors) rules, matched case-insensitively as substrings of
#: the refusal text. Order is precedence and is load bearing: a command can
#: satisfy two classes at once (reading a credential file INTO an outbound
#: request body is both), and the narrower verdict is the one worth acting on.
#: The trust root comes first because it is the one class where the answer is
#: "you cannot, and neither can a workaround" — offering a credential remedy
#: there would send the model looking for a path that must not exist.
_CLASS_ANCHORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        DENY_CLASS_TRUST_ROOT,
        ("governance trust-root", "write-protected config path"),
    ),
    (
        DENY_CLASS_SSO_CREDENTIAL,
        ("sso", "cookie"),
    ),
    (
        DENY_CLASS_AWS_CREDENTIAL,
        (
            ".aws",
            "aws_secret",
            "aws_access",
            "aws_session",
            "aws credentials from environment",
            "imds endpoint",
            "169.254.169.254",
            "boto3",
            "botocore",
        ),
    ),
    (
        DENY_CLASS_EXFIL_SHAPE,
        ("data-exfiltration pattern",),
    ),
    (
        DENY_CLASS_SELF_PROTECTION,
        ("matched structurally on the command's argv",),
    ),
    # Widest credential anchor last: every more specific credential class above
    # also matches these phrases, so leading with them would collapse the whole
    # taxonomy into one generic answer.
    #
    # Every anchor in this table is deliberately GENERIC. The public core must not
    # carry any edition's credential-tool or identity-store names, so a refusal
    # naming one of those degrades to this widest class — whose prose is written to
    # stay true for aws, git and ssh alike — rather than being classified by a
    # marker this file is not allowed to know. An edition that wants a sharper
    # answer supplies it through its own adapter.
    (
        DENY_CLASS_SECRET_FILE,
        (
            "sensitive credential path",
            "sensitive path",
            "credentials",
            ".ssh",
            ".gnupg",
            ".netrc",
            ".npmrc",
            ".pypirc",
            "git-credentials",
        ),
    ),
)

#: class → what the caller should do instead. Written as an instruction to the
#: agent, in the present tense, naming the sanctioned path concretely enough to
#: act on without a further round-trip to the user.
REMEDIATION: dict[str, str] = {
    DENY_CLASS_AWS_CREDENTIAL: (
        "You do not need to read AWS credential material, and no reader of it is "
        "allowed — trying head/less/python instead of cat hits the same rule. AWS "
        "CLI calls themselves are NOT blocked: the SDK resolves credentials on its "
        "own, so run the command you actually wanted. To list configured profiles "
        "use `aws configure list-profiles`; to confirm the identity in effect use "
        "`aws sts get-caller-identity`. If neither works because nothing is "
        "configured, that is the user's action to take in their own terminal (for "
        "example `aws sso login`) — say so instead of concluding this host has no "
        "AWS access."
    ),
    DENY_CLASS_SSO_CREDENTIAL: (
        "This is a live enterprise SSO bearer credential: holding it would let you "
        "act as the user against every SSO-gated service, so it is fenced for "
        "reading as well as writing, and copying it into a cookie jar is blocked "
        "on the same grounds. You cannot authenticate on the user's behalf and "
        "must not try to re-mint the session yourself. Ask the user to run their "
        "host's SSO login command in their own terminal, then retry the request "
        "that needed it."
    ),
    DENY_CLASS_SECRET_FILE: (
        "This path holds credential or key material, so no reader of it is "
        "allowed — trying a different reader hits the same rule family. You "
        "almost never need the bytes: run the command that USES it instead, "
        "because aws, git and ssh all resolve their own credentials, and if the "
        "task genuinely requires the secret's value, ask the user for it rather "
        "than reading the file."
    ),
    DENY_CLASS_TRUST_ROOT: (
        "This path is the security ceiling you are governed BY, so it is "
        "deliberately unreachable from inside a tool call — that is the property "
        "which makes the ceiling un-disableable, not a misconfiguration to work "
        "around. Do not look for another writer or a temp-file rename. If the "
        "policy genuinely needs to change, state what needs changing and let the "
        "user edit it themselves."
    ),
    DENY_CLASS_EXFIL_SHAPE: (
        "The refusal is about the SHAPE of the request, not its destination: "
        "reading a local file into an outbound body (`-d @file`, `--upload-file`, "
        "a `/dev/tcp/` redirect) is indistinguishable from exfiltration, so every "
        "spelling of it is blocked. Send the payload inline instead — for example "
        '`curl -d \'{"k":"v"}\' https://example.invalid` — or use a tool that '
        "takes the request body as an argument."
    ),
    DENY_CLASS_SELF_PROTECTION: (
        "This refusal is about what the action would DO — reach the product's own "
        "credential mint, or stop the supervisor that is running you — so it is not "
        "a spelling problem and must not be re-spelled. The same program reached by "
        "any other invocation form is the same action, so a form that got past the "
        "check would mean the control was defeated rather than satisfied; do not go "
        "looking for one. If what you actually needed was unrelated and importing "
        "the product merely tripped the shape, get it another way that does not run "
        "product code — a file-reading tool, a CLI subcommand's own output, or an "
        "ordinary package query. If you genuinely need this exact action, say so "
        "and let the user run it."
    ),
}

#: class → commands the prose above tells the caller to run. Pinned so
#: ``test_deny_guidance.py`` can prove each one is actually ALLOWED. Guidance
#: that walks the agent into a second wall is worse than none: it spends a turn
#: and teaches it that the advice is untrustworthy.
SUGGESTED_COMMANDS: dict[str, tuple[str, ...]] = {
    DENY_CLASS_AWS_CREDENTIAL: (
        "aws configure list-profiles",
        "aws sts get-caller-identity",
    ),
    DENY_CLASS_EXFIL_SHAPE: ('curl -d \'{"k":"v"}\' https://example.invalid',),
}

#: Substrings that identify an installed MCP server as a credential vendor.
#: Deliberately generic: the public core must not name any edition's server, and
#: a keyword match keeps a host-specific vendor discoverable without one. Chosen
#: to be narrow enough not to sweep in unrelated servers — a bare "auth" would
#: match "author", and a bare "aws" would match every AWS-adjacent tool.
_CREDENTIAL_SERVER_KEYWORDS: tuple[str, ...] = (
    "credential",
    "creds",
    "sso",
    "sts",
    "iam",
)

#: Fields of a capability-manager row consulted for the keyword match.
_SERVER_TEXT_FIELDS: tuple[str, ...] = ("server_id", "name", "title", "description")

#: TTL for the installed-server snapshot. The lookup shells out to the edition's
#: package manager, so it is cached rather than run per refusal; denials are rare
#: enough that a stale-by-minutes hint costs nothing, while an uncached call
#: would put a subprocess on a path that fires during an already-failing turn.
_HINT_TTL_SECS = 300.0

_hint_cache: str = ""
_hint_cache_ts: float = 0.0


def classify_deny(reason: str, subject: str = "") -> str:
    """The deny class named by *reason*, or "" when none applies.

    *subject* is the refused thing itself — the tool title, which for a shell
    call carries the command and for a file read is the path. It is needed
    because the sensitive-path tier refuses with a deliberately GENERIC reason
    ("accesses sensitive credential path") that names no path, so reason alone
    cannot tell an AWS profile from an SSH key from an SSO cookie — three
    refusals with three different sanctioned paths. Consulted as display text
    only: it selects WHICH remediation prose is shown and can never make
    something allowed, so an LLM-authored title steering it costs nothing.

    "" is a first-class answer, not a failure: most denials (a destructive rm, a
    protected-branch push) are self-explanatory, and inventing guidance for them
    would bury the classes where the agent genuinely cannot infer the next step.
    """
    text = f"{reason or ''} {subject or ''}".lower().strip()
    if not text:
        return ""
    for deny_class, anchors in _CLASS_ANCHORS:
        if any(anchor in text for anchor in anchors):
            return deny_class
    return ""


def remediation_for(reason: str, subject: str = "", *, credential_tool_hint: str = "") -> str:
    """Guidance for *reason*, with the host's credential-vendor hint folded in.

    *credential_tool_hint* is appended only for the two credential classes that
    a vending tool can actually resolve. Appending it to, say, a trust-root
    refusal would suggest a credential tool could reach the security ceiling.
    """
    deny_class = classify_deny(reason, subject)
    if not deny_class:
        return ""
    text = REMEDIATION.get(deny_class, "")
    hint = (credential_tool_hint or "").strip()
    if hint and deny_class in (DENY_CLASS_AWS_CREDENTIAL, DENY_CLASS_SSO_CREDENTIAL):
        text = f"{text} {hint}"
    return text


def credential_tool_hint(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hint naming installed MCP servers that look like credential vendors.

    Pure, so the keyword policy is testable without a platform context. Returns
    "" when nothing matches — which is the public edition's normal state, and the
    reason the hint is additive rather than part of the base prose.
    """
    names: list[str] = []
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        server_id = str(row.get("server_id") or row.get("name") or "").strip()
        if not server_id:
            continue
        haystack = " ".join(str(row.get(field) or "") for field in _SERVER_TEXT_FIELDS).lower()
        if any(keyword in haystack for keyword in _CREDENTIAL_SERVER_KEYWORDS):
            if server_id not in names:
                names.append(server_id)
    if not names:
        return ""
    listed = ", ".join(sorted(names))
    return (
        "This host also has MCP server(s) that may vend credentials directly "
        f"({listed}). Prefer one of those and then run the command normally — that "
        "SUPERSEDES the profile guidance above, because a credential-vending host "
        "commonly makes the profile files unreadable even to commands that are "
        "otherwise allowed, and may reject an explicit --profile. If the vendor "
        "reports no configured profile, that is the user's setup step, not a "
        "missing capability."
    )


async def resolve_credential_tool_hint() -> str:
    """Cached :func:`credential_tool_hint` for the composed edition.

    Costs nothing on a host with no capability manager: the public default
    reports ``available() == False``, so this returns "" without spawning
    anything. Fail-soft in every direction — a hint is an enhancement to a
    refusal that already works, so a lookup error degrades to "" rather than
    turning a clean policy block into a turn error.
    """
    global _hint_cache, _hint_cache_ts
    now = time.monotonic()
    if _hint_cache_ts and now - _hint_cache_ts < _HINT_TTL_SECS:
        return _hint_cache
    hint = ""
    try:
        from kiro_crew.platform.capability_bound import bind_capability_manager
        from kiro_crew.platform.context import current_context, safe_context_call
        from kiro_crew.platform.defaults import DefaultCapabilityManager

        manager = safe_context_call(
            lambda: current_context().capability_manager,
            fallback_factory=lambda: bind_capability_manager(DefaultCapabilityManager()),
            log_message="capability_manager lookup failed; skipping credential-tool hint",
        )
        if manager.available():
            hint = credential_tool_hint(await manager.list_mcp())
    except Exception:
        # Includes PlatformCompositionError: a composition fault must not be
        # re-raised onto the refusal path, whose job is to explain a block that
        # already happened. The single write below still caches "" so a broken
        # host is not probed once per denial.
        logger.debug("credential-tool hint lookup failed", exc_info=True)
    _hint_cache = hint
    _hint_cache_ts = now
    return hint


def reset_credential_tool_hint_cache() -> None:
    """Drop the cached hint. For tests, and for a capability mutation."""
    global _hint_cache, _hint_cache_ts
    _hint_cache = ""
    _hint_cache_ts = 0.0
