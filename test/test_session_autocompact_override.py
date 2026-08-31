"""Per-session auto-compact threshold override.

Three layers, each pinned where its defect would live:

- **SessionManager override map** — the override moves the compaction gate for
  exactly its own session while every other session keeps the global; values
  clamp into the documented range (an out-of-range override must degrade to the
  nearest firing value, never silently disable the backstop) and ``None``
  restores the global.
- **HTTP endpoint** (``/api/chat/slots/{slot}/autocompact``) — the validation
  mirrors the global knob's PATCH handler: out-of-range and NaN are rejected,
  null clears, and a successful set reaches both the slot (persistence) and the
  SessionManager (live gate).
- **Persistence validator** — a tampered/corrupted metadata file cannot seed a
  non-numeric or NaN override, and finite out-of-range values clamp exactly as
  the facade would clamp them.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.config.loader import (
    AUTOCOMPACT_PCT_MAX,
    AUTOCOMPACT_PCT_MIN,
    KiroCrewConfig,
)
from kiro_crew.dashboard.chat import api_chat_slot_autocompact
from kiro_crew.dashboard.chat_persistence import _validate_autocompact_pct
from kiro_crew.dashboard.state import DashboardState, _ChatSlot
from kiro_crew.session import SessionManager


def _manager() -> SessionManager:
    return SessionManager(KiroCrewConfig(), provider_factory=lambda *a, **k: object())


class TestSessionManagerOverride:
    def test_override_moves_the_gate_for_its_session_only(self) -> None:
        mgr = _manager()
        glob = mgr._cfg.session.autocompact_pct
        pct = glob - 10.0  # below global, above the override we set
        mgr.set_autocompact_pct("mine", pct - 5.0)

        # The overridden session is now over ITS threshold…
        assert mgr._compaction_gate_decision("mine", object(), pct) != "below_threshold"
        # …while a sibling session at the same usage still declines on the global.
        assert mgr._compaction_gate_decision("other", object(), pct) == "below_threshold"

    def test_effective_pct_falls_back_to_global(self) -> None:
        mgr = _manager()
        assert mgr.effective_autocompact_pct("k") == mgr._cfg.session.autocompact_pct
        mgr.set_autocompact_pct("k", 42.0)
        assert mgr.effective_autocompact_pct("k") == 42.0
        assert mgr.autocompact_pct_override("k") == 42.0
        mgr.set_autocompact_pct("k", None)
        assert mgr.effective_autocompact_pct("k") == mgr._cfg.session.autocompact_pct
        assert mgr.autocompact_pct_override("k") is None

    def test_values_clamp_into_the_documented_range(self) -> None:
        mgr = _manager()
        mgr.set_autocompact_pct("k", 200.0)
        assert mgr.effective_autocompact_pct("k") == AUTOCOMPACT_PCT_MAX
        mgr.set_autocompact_pct("k", 0.5)
        assert mgr.effective_autocompact_pct("k") == AUTOCOMPACT_PCT_MIN

    def test_nan_is_ignored_not_stored(self) -> None:
        # NaN survives min/max unchanged (every comparison is False), so a
        # stored NaN would make ``pct >= threshold`` never fire — silently
        # disabling the backstop. The facade drops it instead.
        mgr = _manager()
        mgr.set_autocompact_pct("k", 42.0)
        mgr.set_autocompact_pct("k", float("nan"))
        assert mgr.effective_autocompact_pct("k") == 42.0

    def test_clearing_an_absent_override_is_a_noop(self) -> None:
        mgr = _manager()
        mgr.set_autocompact_pct("never-set", None)
        assert mgr.autocompact_pct_override("never-set") is None


def _make_app(state: DashboardState) -> web.Application:
    app = web.Application()
    app["state"] = state
    app.router.add_get("/api/chat/slots/{slot}/autocompact", api_chat_slot_autocompact)
    app.router.add_post("/api/chat/slots/{slot}/autocompact", api_chat_slot_autocompact)
    return app


def _mock_state(slot: _ChatSlot | None = None) -> DashboardState:
    state = MagicMock(spec=DashboardState)
    state._slots = {}
    if slot:
        state._slots[slot.key] = slot
    state.sessions = MagicMock()
    return state


class TestAutocompactEndpoint:
    @pytest.mark.asyncio
    async def test_get_reports_override_global_and_range(self) -> None:
        slot = _ChatSlot("test")
        slot.autocompact_pct = 55.0
        state = _mock_state(slot)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/test/autocompact")
            assert resp.status == 200
            data = await resp.json()
            assert data["pct"] == 55.0
            assert data["min"] == AUTOCOMPACT_PCT_MIN
            assert data["max"] == AUTOCOMPACT_PCT_MAX
            assert AUTOCOMPACT_PCT_MIN < data["global_pct"] <= AUTOCOMPACT_PCT_MAX

    @pytest.mark.asyncio
    async def test_post_sets_slot_and_live_session(self) -> None:
        slot = _ChatSlot("test")
        state = _mock_state(slot)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/test/autocompact", json={"pct": 60})
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True and data["pct"] == 60.0
        assert slot.autocompact_pct == 60.0
        # The live gate reads the SessionManager map, so the set must reach it.
        state.sessions.set_autocompact_pct.assert_called_once()
        assert state.sessions.set_autocompact_pct.call_args.args[1] == 60.0
        # Persisted via the dirty flush.
        assert slot._dirty_flag is True

    @pytest.mark.asyncio
    async def test_post_null_clears_back_to_global(self) -> None:
        slot = _ChatSlot("test")
        slot.autocompact_pct = 60.0
        state = _mock_state(slot)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/test/autocompact", json={"pct": None})
            assert resp.status == 200
        assert slot.autocompact_pct is None
        assert state.sessions.set_autocompact_pct.call_args.args[1] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "pct",
        [AUTOCOMPACT_PCT_MIN - 1, AUTOCOMPACT_PCT_MAX + 1, float("nan"), "80", True, [80]],
    )
    async def test_post_rejects_invalid_values(self, pct: object) -> None:
        slot = _ChatSlot("test")
        state = _mock_state(slot)
        payload = '{"pct": NaN}' if isinstance(pct, float) and math.isnan(pct) else None
        async with TestClient(TestServer(_make_app(state))) as client:
            if payload is not None:
                resp = await client.post(
                    "/api/chat/slots/test/autocompact",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
            else:
                resp = await client.post("/api/chat/slots/test/autocompact", json={"pct": pct})
            assert resp.status == 400
        assert slot.autocompact_pct is None
        state.sessions.set_autocompact_pct.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_without_pct_key_is_rejected(self) -> None:
        slot = _ChatSlot("test")
        state = _mock_state(slot)
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post("/api/chat/slots/test/autocompact", json={})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_unknown_slot_is_404(self) -> None:
        state = _mock_state()
        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.get("/api/chat/slots/nope/autocompact")
            assert resp.status == 404


class TestPersistedValueValidator:
    def test_valid_values_round_trip(self) -> None:
        assert _validate_autocompact_pct(55.0) == 55.0
        assert _validate_autocompact_pct(60) == 60.0

    def test_out_of_range_clamps_like_the_facade(self) -> None:
        assert _validate_autocompact_pct(500.0) == AUTOCOMPACT_PCT_MAX
        assert _validate_autocompact_pct(1.0) == AUTOCOMPACT_PCT_MIN

    @pytest.mark.parametrize("raw", ["80", True, [80], {}, float("nan")])
    def test_garbage_is_discarded(self, raw: object) -> None:
        assert _validate_autocompact_pct(raw) is None

    def test_none_is_discarded_silently(self) -> None:
        assert _validate_autocompact_pct(None) is None


class TestOverrideLifecycle:
    """The two leak paths the review mirrors caught on the first pass."""

    def test_slot_save_owns_the_key_so_a_clear_erases_it(self) -> None:
        """A cleared override must not resurrect from stale metadata.

        The slot save rebuilds its metadata line and then carries forward every
        key it does NOT own (``carry_unowned_metadata``). If ``autocompact_pct``
        is not in ``SLOT_OWNED_META_KEYS``, a clear (slot field None, key
        omitted from the rebuilt line) is undone by the carry — the old value
        rides back in and the next restart resurrects the cleared override.
        """
        from kiro_crew.history import SLOT_OWNED_META_KEYS, carry_unowned_metadata

        assert "autocompact_pct" in SLOT_OWNED_META_KEYS
        rebuilt = {"_type": "meta", "model": "m"}  # cleared: key absent
        existing = {"_type": "meta", "model": "m", "autocompact_pct": 85.0}
        merged = carry_unowned_metadata(rebuilt, existing, SLOT_OWNED_META_KEYS)
        assert "autocompact_pct" not in merged

    @pytest.mark.asyncio
    async def test_destroy_clears_the_override(self) -> None:
        """A destroyed session's override must not leak to a same-key successor.

        ``destroy`` is permanent (unlike reset/recycle, which the override
        deliberately survives): a session recreated on the key afterwards is a
        new conversation, and silently inheriting the deleted one's threshold
        while the endpoint reports "following global" is state divergence.
        """
        mgr = _manager()
        mgr.set_autocompact_pct("gone", 50.0)
        assert mgr.autocompact_pct_override("gone") == 50.0
        await mgr.destroy("gone")
        assert mgr.autocompact_pct_override("gone") is None
