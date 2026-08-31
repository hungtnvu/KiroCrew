"""The re-embed sweeps pace themselves; an explicitly-requested one does not.

Locks in the behaviour that keeps an unattended post-migration sweep from pinning
several cores for tens of minutes: every bulk row is followed by an idle window
proportional to the work it just did, taken on the SWEEP's thread (so it holds
neither the DB lock nor the model), and skipped entirely when a human asked for
the sweep and is waiting on it.
"""

from __future__ import annotations

import pytest

from kiro_crew import vector_memory as vm
from kiro_crew.vector_memory import VectorMemoryStore


@pytest.fixture()
def store(tmp_path):
    st = VectorMemoryStore(db_path=tmp_path / "memory.db", embedding_dim=4)
    st.init()
    return st


@pytest.fixture()
def sleeps(monkeypatch):
    """Record pace sleeps instead of performing them."""
    recorded: list[float] = []
    monkeypatch.setattr(vm.time, "sleep", lambda s: recorded.append(s))
    return recorded


def _deferred_rows(store, n):
    for i in range(n):
        assert store.write_episodic(
            f"deferred row {i}",
            conversation_id=f"c{i}",
            source="import",
            preserve_existing=True,
            defer_embedding=True,
        )


def test_each_bulk_row_is_paced(store, sleeps, monkeypatch):
    monkeypatch.setattr(vm, "bulk_pace_delay", lambda elapsed: 0.125)
    _deferred_rows(store, 3)
    store.embed_fn = lambda text: [0.5, 0.5, 0.5, 0.5]

    assert store.backfill_missing_embeddings() == 3
    assert sleeps == [0.125, 0.125, 0.125]


def test_pace_false_never_sleeps(store, sleeps, monkeypatch):
    monkeypatch.setattr(vm, "bulk_pace_delay", lambda elapsed: 0.125)
    _deferred_rows(store, 3)
    store.embed_fn = lambda text: [0.5, 0.5, 0.5, 0.5]

    assert store.backfill_missing_embeddings(pace=False) == 3
    assert sleeps == []


def test_zero_delay_does_not_call_sleep(store, sleeps, monkeypatch):
    """Pacing off (duty 1.0) must not add a syscall per row."""
    monkeypatch.setattr(vm, "bulk_pace_delay", lambda elapsed: 0.0)
    _deferred_rows(store, 2)
    store.embed_fn = lambda text: [0.5, 0.5, 0.5, 0.5]

    assert store.backfill_missing_embeddings() == 2
    assert sleeps == []


def test_a_failed_row_is_still_paced_by_measured_time(store, sleeps, monkeypatch):
    """A row that returns no vector still ran the model; pace on elapsed, not success."""
    monkeypatch.setattr(vm, "bulk_pace_delay", lambda elapsed: 0.05)
    _deferred_rows(store, 2)
    store.embed_fn = lambda text: None

    assert store.backfill_missing_embeddings() == 0
    assert sleeps == [0.05, 0.05]


def test_delay_is_derived_from_the_row_s_own_elapsed_time(store, monkeypatch):
    seen: list[float] = []

    def _record(elapsed):
        seen.append(elapsed)
        return 0.0

    monkeypatch.setattr(vm, "bulk_pace_delay", _record)
    _deferred_rows(store, 1)
    store.embed_fn = lambda text: [0.5, 0.5, 0.5, 0.5]

    store.backfill_missing_embeddings()
    assert len(seen) == 1
    assert seen[0] >= 0.0


def test_semantic_and_lesson_sweeps_are_paced_too(store, sleeps, monkeypatch):
    """All three sweeps share the paced helper, not just the episodic one."""
    monkeypatch.setattr(vm, "bulk_pace_delay", lambda elapsed: 0.01)
    store.embed_fn = lambda text: [0.5, 0.5, 0.5, 0.5]
    store.set_semantic_if_absent("user.city", "Seattle", 0.9, "user_explicit")
    store.write_lesson("always pace bulk work")
    # Clear the vectors the write paths just stored so the sweeps have work.
    with store._db_lock:
        store.db.execute("UPDATE semantic_memory SET embedding = NULL")
        store.db.commit()

    assert store._backfill_lesson_embeddings(pace=True) >= 1
    assert store._backfill_semantic_kv_embeddings(pace=True) >= 1
    assert sleeps and all(s == 0.01 for s in sleeps)


def test_the_pause_sits_between_rows_not_inside_a_write(store, monkeypatch):
    """The pause follows the row's INFERENCE, before its write lands.

    That ordering is what makes the pause safe to interrupt: a sweep killed
    mid-pause leaves that row's ``embedding`` NULL, and the next sweep re-embeds
    it — the same idempotent contract every unfinished row already has. It also
    holds no DB lock, so a concurrent writer is never blocked by pacing.
    """
    visible: list[int] = []

    def _probe(_seconds):
        rows = store._fetch_all_locked(
            "SELECT count(*) AS n FROM episodic_memories WHERE embedding IS NOT NULL"
        )
        visible.append(int(rows[0]["n"]))

    monkeypatch.setattr(vm, "bulk_pace_delay", lambda elapsed: 0.01)
    monkeypatch.setattr(vm.time, "sleep", _probe)
    _deferred_rows(store, 2)
    store.embed_fn = lambda text: [0.5, 0.5, 0.5, 0.5]

    assert store.backfill_missing_embeddings() == 2
    # One pause per row, and each sees only the rows already finished — never a
    # partially-written one.
    assert visible == [0, 1]
    remaining = store._fetch_all_locked(
        "SELECT count(*) AS n FROM episodic_memories WHERE embedding IS NULL"
    )
    assert int(remaining[0]["n"]) == 0
