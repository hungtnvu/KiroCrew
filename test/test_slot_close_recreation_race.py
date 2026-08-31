"""Close-vs-recreate race on the shared dashboard slot-close teardown.

The defect these pin (issue #7191): both ``api_chat_slot_delete`` and
``api_chat_slots_cleanup`` pop ``name`` out of ``state._slots`` and then run a
sequence of AWAITS — cancel the task, ``save_slot_off_loop(..., closed=True)``,
``state.sessions.remove(_history_key_for(name))``. A concurrent same-key
recreate (a POST /api/chat, or the session_close MCP verb) can mint a
REPLACEMENT slot for the same key inside that window. The original, still in
flight, then (a) writes ITS transcript over the shared history key as closed and
(b) tears down the session the replacement now uses. The failure arms compound
it: they blindly ``state._slots[name] = <original>`` over whatever now owns the
key.

The fix re-checks identity — ``state._slots.get(name) is <popped>`` — before the
closed=True save and again before ``sessions.remove``, at BOTH sites, and guards
the failure-arm restores so they never clobber a live replacement. These tests
interleave a recreate across the teardown awaits (via an ``asyncio.Event`` the
monkeypatched ``save_slot_off_loop`` parks on) and assert the replacement's
identity, history, and session survive, and that the destructive step was
skipped. Each would FAIL if its guard were reverted.
"""

from __future__ import annotations

import asyncio

import pytest
from chat_test_helpers import _make_state

from kiro_crew import autonudge
from kiro_crew.autonudge import AutoNudgeService
from kiro_crew.dashboard import chat_handlers as handlers

NAME = "chat-1-1785"


@pytest.fixture(autouse=True)
def _no_nudge_service(monkeypatch):
    """No auto-nudge service: these tests isolate the teardown-race guard.

    The nudge-loop retirement and its rollbacks are pinned by
    test_slot_close_nudge_race.py; here the close path must find nothing to
    retire so the only variable is the post-pop identity re-check.
    """
    monkeypatch.setattr(autonudge, "_INSTANCE", None)


class _Req:
    """Minimal stand-in for the aiohttp request the handlers read.

    The race tests drive the handlers directly rather than through a client: the
    interleaving of the concurrent recreate has to be scheduled deterministically
    inside the teardown window, and a client's own awaits would let it run before
    the handler even reached the pop.
    """

    def __init__(self, state, slot: str = NAME, body: dict | None = None) -> None:
        self.app = {"state": state}
        self.match_info = {"slot": slot}
        self._body = body if body is not None else {}

    def get(self, key: str, default: str = "") -> str:
        del key
        return default

    async def json(self) -> dict:
        return self._body


def _state_with_slot(tmp_path, name: str = NAME):
    state = _make_state(tmp_path)
    slot = state.get_or_create_slot(name)
    slot.append("user", "watch the PR")
    slot.append("assistant", "watching")
    slot.drain()
    return state


def _arm_running_turn(slot, entered: asyncio.Event, release: asyncio.Event):
    """Give ``slot`` a live, parkable turn so its task-cancel await blocks.

    The close pops the slot and then, if ``slot.running``, cancels ``slot.task``
    and waits on ``asyncio.wait_for(asyncio.shield(slot.task), 2.0)``. To land a
    recreate in THAT await (the window the first pre-save guard protects) the
    task has to still be pending when the recreate is minted. This turn swallows
    the handler's cancel, signals ``entered`` so the test can mint the
    replacement, then parks on ``release`` before returning — so the handler's
    shielded wait is still blocked at the exact instant the key changes owner.
    ``slot.running`` is ``self.task is not None and not self.task.done()``, so
    assigning this task is all it takes to make the slot look busy.
    """

    async def _turn() -> None:
        try:
            await asyncio.Event().wait()  # block until cancelled
        except asyncio.CancelledError:
            entered.set()
            await release.wait()

    slot.task = asyncio.create_task(_turn())
    return slot.task


# --------------------------------------------------------------------------- #
# api_chat_slot_delete
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_delete_recreate_during_save_preserves_replacement(tmp_path, monkeypatch) -> None:
    """(a) A recreate landing inside the closed=True save must survive intact.

    While the close is parked inside ``save_slot_off_loop`` a concurrent
    ``get_or_create_slot(NAME)`` mints a replacement. After the close returns the
    replacement must still own the key, and ``sessions.remove`` must NOT have been
    called for its key (the second identity re-check, before the remove, must see
    the key is no longer ours and skip the destructive teardown). Without the
    guard the close would run ``sessions.remove`` and tear down the session the
    replacement now uses.
    """
    state = _state_with_slot(tmp_path)
    original = state._slots[NAME]

    entered = asyncio.Event()
    release = asyncio.Event()

    async def _persist(*_a, **_kw) -> None:
        # Park so the recreate can interleave INSIDE the teardown window.
        entered.set()
        await release.wait()

    removed_keys: list[str] = []

    async def _remove(key) -> None:
        removed_keys.append(key)

    monkeypatch.setattr(handlers, "save_slot_off_loop", _persist)
    state.sessions.remove = _remove  # type: ignore[assignment]

    close = asyncio.create_task(handlers.api_chat_slot_delete(_Req(state, NAME)))
    await entered.wait()  # close is parked inside the persist
    # The concurrent same-key recreate mints a fresh slot object under NAME.
    replacement = state.get_or_create_slot(NAME)
    assert replacement is not original
    release.set()
    resp = await close

    assert resp.status == 200
    assert state._slots.get(NAME) is replacement, "the replacement was clobbered by the close"
    assert removed_keys == [], "sessions.remove tore down the live replacement's session"


@pytest.mark.asyncio
async def test_delete_recreate_during_task_cancel_hits_first_guard(tmp_path, monkeypatch) -> None:
    """(a2) A recreate landing in the task-cancel await hits the FIRST guard.

    This is the ONLY window where the pre-save guard fires: the recreate is
    minted while the close is parked in ``asyncio.wait_for(asyncio.shield(
    slot.task), 2.0)`` — BEFORE ``save_slot_off_loop`` is reached — so the first
    ``_slot_still_ours`` check (immediately after the cancel block) sees the key
    is no longer ours and takes the early ``return {"ok": True}``. That means the
    closed=True save is NEVER attempted for the original and ``sessions.remove``
    is NEVER called: the replacement keeps its slot, its (unclosed) history, and
    its session. Reverting ONLY the first guard would let the close fall through
    to the save and the remove, so this case fails without it — the other cases
    park inside the save and so exercise only the second/failure-arm guards.
    """
    state = _state_with_slot(tmp_path)
    original = state._slots[NAME]

    entered = asyncio.Event()
    release = asyncio.Event()
    _arm_running_turn(original, entered, release)
    assert original.running, "the turn must be live so the cancel-wait actually blocks"

    saved: list[bool] = []
    removed_keys: list[str] = []

    async def _persist(_state, _slot, *_a, **kw) -> None:
        saved.append(bool(kw.get("closed")))

    async def _remove(key) -> None:
        removed_keys.append(key)

    monkeypatch.setattr(handlers, "save_slot_off_loop", _persist)
    state.sessions.remove = _remove  # type: ignore[assignment]

    close = asyncio.create_task(handlers.api_chat_slot_delete(_Req(state, NAME)))
    await entered.wait()  # close is parked in the shielded task-cancel wait
    # The concurrent same-key recreate mints a fresh slot while the close is
    # still short of the pre-save guard.
    replacement = state.get_or_create_slot(NAME)
    assert replacement is not original
    release.set()  # let the cancelled turn finish so the wait returns
    resp = await close

    assert resp.status == 200
    assert state._slots.get(NAME) is replacement, "the first guard did not preserve the replacement"
    assert saved == [], "the closed=True save ran past the first guard onto the replacement's key"
    assert removed_keys == [], "sessions.remove ran past the first guard on the replacement's key"


@pytest.mark.asyncio
async def test_delete_recreate_between_save_and_remove_skips_remove(tmp_path, monkeypatch) -> None:
    """(b) A recreate landing between the save and sessions.remove skips remove.

    ``sessions.remove`` tears down the session backing the reused key, which the
    replacement now uses; the second identity re-check must skip it.
    """
    state = _state_with_slot(tmp_path)
    original = state._slots[NAME]

    removed_keys: list[str] = []

    async def _remove(key) -> None:
        removed_keys.append(key)

    # The recreate lands AFTER the save completes but BEFORE sessions.remove.
    # The second identity re-check, immediately before the remove on the same
    # frame, is what must catch it. Recreating as the save's final act reproduces
    # exactly that interleaving deterministically.
    async def _persist_then_recreate(*_a, **_kw) -> None:
        state.get_or_create_slot(NAME)

    monkeypatch.setattr(handlers, "save_slot_off_loop", _persist_then_recreate)
    state.sessions.remove = _remove  # type: ignore[assignment]

    resp = await handlers.api_chat_slot_delete(_Req(state, NAME))

    assert resp.status == 200
    replacement = state._slots.get(NAME)
    assert replacement is not None and replacement is not original, "replacement lost"
    assert removed_keys == [], "sessions.remove tore down the live replacement's session"


@pytest.mark.asyncio
async def test_delete_failure_arm_does_not_clobber_replacement(tmp_path, monkeypatch) -> None:
    """(c) A persist that raises WHILE a replacement owns the key must not restore.

    The failure arm's ``state._slots[name] = slot`` would overwrite the live
    replacement with the failed original; the guard must leave the replacement.
    """
    state = _state_with_slot(tmp_path)
    original = state._slots[NAME]

    entered = asyncio.Event()
    release = asyncio.Event()

    async def _persist(*_a, **_kw) -> None:
        entered.set()
        await release.wait()
        raise RuntimeError("disk wedged")

    monkeypatch.setattr(handlers, "save_slot_off_loop", _persist)

    close = asyncio.create_task(handlers.api_chat_slot_delete(_Req(state, NAME)))
    await entered.wait()
    replacement = state.get_or_create_slot(NAME)
    assert replacement is not original
    release.set()
    resp = await close

    # The close still fails (its own persist raised), but the restore must NOT
    # have clobbered the live replacement.
    assert resp.status == 500
    assert state._slots.get(NAME) is replacement, "the failure arm clobbered the replacement"


@pytest.mark.asyncio
async def test_delete_failure_arm_compensation_runs_even_when_restore_skipped(
    tmp_path, monkeypatch
) -> None:
    """(c2) The failure arm's NON-_slots compensation still fires past the guard.

    The guard skips only the ``state._slots[name] = slot`` write when a
    replacement owns the key. Everything else the failed close owes the ORIGINAL
    must still run: the retired auto-nudge loop has to be put back (a restored
    session with no clock is an abandoned worker) and the app must be told the
    dismissal was undone (the notify already SUCCEEDED, durably pausing a crew).
    This pins that both compensations fire even though the ``_slots`` restore was
    skipped in favour of the live replacement — a fix that guarded the restore
    but also short-circuited the compensation would leave a disabled worker and a
    clockless loop behind the surviving replacement.
    """
    state = _state_with_slot(tmp_path)
    original = state._slots[NAME]
    original._app = "issue-radar"

    svc = AutoNudgeService(base_dir=tmp_path)
    monkeypatch.setattr(autonudge, "_INSTANCE", svc)
    await svc.add(NAME, "check the PR", idle_secs=300, max_cycles=24)

    undone: list[str] = []

    async def _told(_app: str, _slot_key: str) -> bool:
        return True

    async def _undo(_app: str, slot_key: str) -> bool:
        undone.append(slot_key)
        return True

    entered = asyncio.Event()
    release = asyncio.Event()

    async def _persist(*_a, **_kw) -> None:
        entered.set()
        await release.wait()
        raise RuntimeError("disk wedged")

    monkeypatch.setattr("kiro_crew.apps.teardown.notify_slot_closed", _told)
    monkeypatch.setattr("kiro_crew.apps.teardown.notify_slot_close_undone", _undo)
    monkeypatch.setattr(handlers, "save_slot_off_loop", _persist)

    close = asyncio.create_task(handlers.api_chat_slot_delete(_Req(state, NAME)))
    await entered.wait()
    replacement = state.get_or_create_slot(NAME)
    assert replacement is not original
    release.set()
    resp = await close

    assert resp.status == 500
    # The _slots restore was skipped: the live replacement is untouched.
    assert state._slots.get(NAME) is replacement, "the failure arm clobbered the replacement"
    # ...but the compensation the original is owed still ran.
    assert undone == [NAME], "app-notify-undo did not fire when the _slots restore was skipped"
    assert svc.get_by_slot(NAME) is not None, "the retired nudge loop was not restored"
    svc.stop()


@pytest.mark.asyncio
async def test_delete_ordinary_close_still_saves_and_removes(tmp_path, monkeypatch) -> None:
    """(f) With NO recreate, the guard is inert: pop, save closed=True, remove.

    Proves the guard does not change the common path.
    """
    state = _state_with_slot(tmp_path)

    saved_closed: list[bool] = []
    removed_keys: list[str] = []

    async def _persist(_state, _slot, *_a, **kw) -> None:
        saved_closed.append(bool(kw.get("closed")))

    async def _remove(key) -> None:
        removed_keys.append(key)

    monkeypatch.setattr(handlers, "save_slot_off_loop", _persist)
    state.sessions.remove = _remove  # type: ignore[assignment]

    resp = await handlers.api_chat_slot_delete(_Req(state, NAME))

    assert resp.status == 200
    assert NAME not in state._slots, "ordinary close did not remove the slot"
    assert saved_closed == [True], "ordinary close did not persist closed=True"
    assert removed_keys == [f"dashboard:{NAME}"], "ordinary close did not tear down the session"


# --------------------------------------------------------------------------- #
# api_chat_slots_cleanup (bulk archive)
# --------------------------------------------------------------------------- #


def _make_stale(state, name: str = NAME):
    """Age the slot's last activity past the 3-day cleanup cutoff."""
    slot = state._slots[name]
    slot.created_at = "2000-01-01T00:00:00+00:00"
    for m in slot.messages:
        m["ts"] = "2000-01-01T00:00:00+00:00"
    return slot


@pytest.mark.asyncio
async def test_cleanup_recreate_during_save_preserves_replacement(tmp_path, monkeypatch) -> None:
    """(d) The bulk path: a recreate inside the archive save must survive.

    The replacement must not be archived-over, must remain in ``_slots``, and its
    session must not be removed.
    """
    state = _state_with_slot(tmp_path)
    original = _make_stale(state)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def _persist(*_a, **_kw) -> None:
        entered.set()
        await release.wait()

    removed_keys: list[str] = []

    async def _remove(key) -> None:
        removed_keys.append(key)

    monkeypatch.setattr(handlers, "save_slot_off_loop", _persist)
    state.sessions.remove = _remove  # type: ignore[assignment]

    close = asyncio.create_task(handlers.api_chat_slots_cleanup(_Req(state, NAME)))
    await entered.wait()  # parked inside the archive save for NAME
    replacement = state.get_or_create_slot(NAME)
    assert replacement is not original
    release.set()
    resp = await close

    payload = _json(resp)
    assert resp.status == 200
    assert NAME not in payload["keys"], "a live replacement was reported archived-over"
    assert state._slots.get(NAME) is replacement, "the replacement was clobbered by cleanup"
    assert removed_keys == [], "cleanup tore down the live replacement's session"


@pytest.mark.asyncio
async def test_cleanup_recreate_during_task_cancel_hits_first_guard(tmp_path, monkeypatch) -> None:
    """(d2) Bulk path: a recreate in the task-cancel await hits the FIRST guard.

    The stale slot has a live turn, so the per-iteration cancel block awaits
    ``asyncio.wait_for(asyncio.shield(removed.task), 2.0)``. A recreate minted in
    THAT await lands before the flush+save, so the first ``_slot_still_ours``
    check takes the early ``continue`` — the flush and closed=True save never run
    for the original, ``sessions.remove`` is never called, and NAME is NOT
    appended to the archived ``keys`` (a live replacement must never be reported
    archived-over). Reverting only the first guard would let the pass flush, save
    onto the replacement's key, remove its session, and report it archived.
    """
    state = _state_with_slot(tmp_path)
    original = _make_stale(state)

    entered = asyncio.Event()
    release = asyncio.Event()
    _arm_running_turn(original, entered, release)
    assert original.running, "the turn must be live so the cancel-wait actually blocks"

    saved: list[bool] = []
    removed_keys: list[str] = []

    async def _persist(_state, _slot, *_a, **kw) -> None:
        saved.append(bool(kw.get("closed")))

    async def _remove(key) -> None:
        removed_keys.append(key)

    monkeypatch.setattr(handlers, "save_slot_off_loop", _persist)
    state.sessions.remove = _remove  # type: ignore[assignment]

    close = asyncio.create_task(handlers.api_chat_slots_cleanup(_Req(state, NAME)))
    await entered.wait()  # parked in the shielded task-cancel wait for NAME
    replacement = state.get_or_create_slot(NAME)
    assert replacement is not original
    release.set()
    resp = await close

    payload = _json(resp)
    assert resp.status == 200
    assert NAME not in payload["keys"], "a live replacement was reported archived-over"
    assert state._slots.get(NAME) is replacement, "the first guard did not preserve the replacement"
    assert saved == [], "the closed=True save ran past the first guard onto the replacement's key"
    assert removed_keys == [], "sessions.remove ran past the first guard on the replacement's key"


@pytest.mark.asyncio
async def test_cleanup_failure_arm_does_not_clobber_replacement(tmp_path, monkeypatch) -> None:
    """(e) Bulk failure arm: a persist that raises while a replacement owns the key.

    ``state._slots[name] = removed`` must not overwrite the live replacement.
    """
    state = _state_with_slot(tmp_path)
    original = _make_stale(state)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def _persist(*_a, **_kw) -> None:
        entered.set()
        await release.wait()
        raise RuntimeError("disk wedged")

    monkeypatch.setattr(handlers, "save_slot_off_loop", _persist)

    close = asyncio.create_task(handlers.api_chat_slots_cleanup(_Req(state, NAME)))
    await entered.wait()
    replacement = state.get_or_create_slot(NAME)
    assert replacement is not original
    release.set()
    resp = await close

    payload = _json(resp)
    assert resp.status == 200
    assert NAME in payload["failed"], "the failed original should be reported failed"
    assert state._slots.get(NAME) is replacement, "the failure arm clobbered the replacement"


def _json(resp) -> dict:
    """Decode an aiohttp json_response body to a dict."""
    import json

    return json.loads(resp.body)
