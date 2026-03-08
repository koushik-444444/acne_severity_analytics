"""Integration tests for BridgeStore database operations."""
import json
from datetime import timedelta

import pytest

from api_bridge import BridgeStore, utcnow_iso, utcnow


def make_session(session_id, **overrides):
    """Create a minimal session payload with sensible defaults."""
    base = {
        'session_id': session_id,
        'profile_id': 'default-profile',
        'timestamp': utcnow_iso(),
        'severity': 'Mild',
        'gags_score': 5,
        'lesion_count': 3,
        'symmetry_delta': 1.0,
        'results_json': json.dumps({'clinical_analysis': {'total_lesions': 3}}),
        'note': '',
        'diagnostic_image_path': None,
        'original_image_path': None,
        'privacy_mode': False,
        'retention_hours': 72,
    }
    base.update(overrides)
    return base


def test_upsert_and_get_session(store):
    payload = make_session('s1')
    store.upsert_session(payload)
    row = store.get_session_row('s1')
    assert row is not None
    assert row['session_id'] == 's1'
    assert row['severity'] == 'Mild'


def test_get_session_row_nonexistent(store):
    assert store.get_session_row('nonexistent') is None


def test_upsert_updates_existing(store):
    store.upsert_session(make_session('s1', severity='Mild'))
    store.upsert_session(make_session('s1', severity='Severe'))
    row = store.get_session_row('s1')
    assert row['severity'] == 'Severe'


def test_history_ordering(store):
    # Insert in non-chronological order
    store.upsert_session(make_session('s1', timestamp='2025-01-01T00:00:00+00:00'))
    store.upsert_session(make_session('s2', timestamp='2025-01-03T00:00:00+00:00'))
    store.upsert_session(make_session('s3', timestamp='2025-01-02T00:00:00+00:00'))
    items, _ = store.history(limit=10)
    ids = [item['session_id'] for item in items]
    assert ids == ['s2', 's3', 's1']


def test_history_limit_clamping(store):
    for i in range(5):
        store.upsert_session(make_session(f's{i}'))
    items, _ = store.history(limit=3)
    assert len(items) == 3


def test_history_limit_upper_bound(store):
    for i in range(5):
        store.upsert_session(make_session(f's{i}'))
    # Limit above 200 should be clamped
    items, _ = store.history(limit=9999)
    assert len(items) == 5  # Only 5 exist


def test_set_and_get_status(store):
    status = store.set_status('s1', 'segmenting', 'Running BiSeNet', 30)
    assert status['stage'] == 'segmenting'
    assert status['progress'] == 30

    retrieved = store.get_status('s1')
    assert retrieved is not None
    assert retrieved['stage'] == 'segmenting'


def test_get_status_nonexistent(store):
    assert store.get_status('nonexistent') is None


def test_latest_status(store):
    store.set_status('s1', 'queued', 'Waiting', 0)
    store.set_status('s2', 'completed', 'Done', 100)
    latest = store.latest_status()
    assert latest is not None
    assert latest['session_id'] == 's2'


def test_purge_removes_session_and_status(store):
    store.upsert_session(make_session('s1'))
    store.set_status('s1', 'completed', 'Done', 100)

    assert store.purge('s1') is True
    assert store.get_session_row('s1') is None
    assert store.get_status('s1') is None


def test_purge_nonexistent_returns_false(store):
    assert store.purge('nonexistent') is False


def test_cleanup_expired(store):
    # Session that expired 1 hour ago
    expired_ts = (utcnow() - timedelta(hours=2)).isoformat()
    store.upsert_session(make_session('expired', timestamp=expired_ts, retention_hours=1))

    # Session that should still be live
    live_ts = utcnow_iso()
    store.upsert_session(make_session('live', timestamp=live_ts, retention_hours=72))

    result = store.cleanup_expired()
    assert result['purged_sessions'] >= 1
    assert store.get_session_row('expired') is None
    assert store.get_session_row('live') is not None


def test_session_payload_includes_status(store):
    store.upsert_session(make_session('s1'))
    store.set_status('s1', 'completed', 'Done', 100)
    payload = store.session_payload('s1')
    assert payload is not None
    assert payload['status'] is not None
    assert payload['status']['stage'] == 'completed'


def test_previous_session(store):
    store.upsert_session(make_session('s1', timestamp='2025-01-01T00:00:00+00:00'))
    store.upsert_session(make_session('s2', timestamp='2025-01-02T00:00:00+00:00'))
    prev = store.previous_session('s2')
    assert prev is not None
    assert prev['session_id'] == 's1'


def test_previous_session_none_when_first(store):
    store.upsert_session(make_session('s1', timestamp='2025-01-01T00:00:00+00:00'))
    prev = store.previous_session('s1')
    assert prev is None


def test_update_note(store):
    store.upsert_session(make_session('s1', note='initial'))
    with store.lock:
        store.conn.execute(
            'UPDATE sessions SET note = ? WHERE session_id = ?',
            ('updated note', 's1'),
        )
        store.conn.commit()
    row = store.get_session_row('s1')
    assert row['note'] == 'updated note'


def test_history_profile_id_filter(store):
    store.upsert_session(make_session('s1', profile_id='profile-a'))
    store.upsert_session(make_session('s2', profile_id='profile-b'))
    store.upsert_session(make_session('s3', profile_id='profile-a'))
    items, _ = store.history(limit=10, profile_id='profile-a')
    assert len(items) == 2
    assert all(item['profile_id'] == 'profile-a' for item in items)


def test_history_no_profile_returns_all(store):
    store.upsert_session(make_session('s1', profile_id='profile-a'))
    store.upsert_session(make_session('s2', profile_id='profile-b'))
    items, _ = store.history(limit=10)
    assert len(items) == 2


def test_history_cursor_pagination(store):
    store.upsert_session(make_session('s1', timestamp='2025-01-01T00:00:00+00:00'))
    store.upsert_session(make_session('s2', timestamp='2025-01-02T00:00:00+00:00'))
    store.upsert_session(make_session('s3', timestamp='2025-01-03T00:00:00+00:00'))
    store.upsert_session(make_session('s4', timestamp='2025-01-04T00:00:00+00:00'))
    store.upsert_session(make_session('s5', timestamp='2025-01-05T00:00:00+00:00'))

    # First page: 2 items
    items_1, cursor_1 = store.history(limit=2)
    assert len(items_1) == 2
    assert items_1[0]['session_id'] == 's5'
    assert items_1[1]['session_id'] == 's4'
    assert cursor_1 is not None

    # Second page using cursor
    items_2, cursor_2 = store.history(limit=2, cursor=cursor_1)
    assert len(items_2) == 2
    assert items_2[0]['session_id'] == 's3'
    assert items_2[1]['session_id'] == 's2'
    assert cursor_2 is not None

    # Third page: only 1 item left, no more cursor
    items_3, cursor_3 = store.history(limit=2, cursor=cursor_2)
    assert len(items_3) == 1
    assert items_3[0]['session_id'] == 's1'
    assert cursor_3 is None


def test_history_cursor_with_profile_filter(store):
    store.upsert_session(make_session('s1', timestamp='2025-01-01T00:00:00+00:00', profile_id='p1'))
    store.upsert_session(make_session('s2', timestamp='2025-01-02T00:00:00+00:00', profile_id='p2'))
    store.upsert_session(make_session('s3', timestamp='2025-01-03T00:00:00+00:00', profile_id='p1'))
    store.upsert_session(make_session('s4', timestamp='2025-01-04T00:00:00+00:00', profile_id='p1'))

    items, cursor = store.history(limit=1, profile_id='p1')
    assert len(items) == 1
    assert items[0]['session_id'] == 's4'
    assert cursor is not None

    items_2, cursor_2 = store.history(limit=1, profile_id='p1', cursor=cursor)
    assert len(items_2) == 1
    assert items_2[0]['session_id'] == 's3'
    assert cursor_2 is not None

    items_3, cursor_3 = store.history(limit=1, profile_id='p1', cursor=cursor_2)
    assert len(items_3) == 1
    assert items_3[0]['session_id'] == 's1'
    assert cursor_3 is None
