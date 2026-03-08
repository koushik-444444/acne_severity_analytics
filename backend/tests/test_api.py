"""Integration tests for FastAPI endpoints via TestClient."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Set env vars before importing app
os.environ['ROBOFLOW_API_KEY'] = 'test-key-for-api-tests'


from api_bridge import app, BridgeStore, DB_PATH


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Replace the app's store with a temporary database for each test."""
    test_db = tmp_path / 'test.db'
    test_uploads = tmp_path / 'uploads'
    test_outputs = tmp_path / 'outputs'
    test_reports = tmp_path / 'reports'
    test_uploads.mkdir()
    test_outputs.mkdir()
    test_reports.mkdir()

    monkeypatch.setattr('api_bridge.UPLOAD_DIR', test_uploads)
    monkeypatch.setattr('api_bridge.OUTPUT_DIR', test_outputs)
    monkeypatch.setattr('api_bridge.REPORT_DIR', test_reports)

    store = BridgeStore(test_db)
    app.state.resources = {
        'store': store,
        'pipeline': None,
        'cloud_engine': None,
        'startup_cleanup': {'purged_sessions': 0, 'purged_files': 0},
    }
    yield
    store.close()


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_root(client):
    r = client.get('/')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert 'version' in body


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert body['roboflow_api_key_configured'] is True


def test_version(client):
    r = client.get('/version')
    assert r.status_code == 200
    body = r.json()
    assert 'version' in body
    assert 'model_a_id' in body


def test_privacy(client):
    r = client.get('/privacy')
    assert r.status_code == 200
    body = r.json()
    assert body['privacy_mode_supported'] is True
    assert 'default_retention_hours' in body


def test_analysis_start_creates_session(client):
    r = client.post('/analysis/start', json={
        'privacy_mode': False,
        'retention_hours': 72,
    })
    assert r.status_code == 200
    body = r.json()
    assert 'session_id' in body
    assert body['privacy_mode'] is False


def test_analysis_start_custom_session_id(client):
    r = client.post('/analysis/start', json={
        'session_id': 'my-custom-session',
        'privacy_mode': False,
        'retention_hours': 48,
    })
    assert r.status_code == 200
    assert r.json()['session_id'] == 'my-custom-session'


def test_analysis_start_duplicate_409(client):
    payload = {'session_id': 'dup-test', 'privacy_mode': False, 'retention_hours': 72}
    r1 = client.post('/analysis/start', json=payload)
    assert r1.status_code == 200
    r2 = client.post('/analysis/start', json=payload)
    assert r2.status_code == 409


def test_analysis_start_invalid_session_id(client):
    r = client.post('/analysis/start', json={
        'session_id': 'invalid session!',
        'privacy_mode': False,
        'retention_hours': 72,
    })
    assert r.status_code == 400


def test_history_empty(client):
    r = client.get('/history')
    assert r.status_code == 200
    assert r.json()['items'] == []


def test_history_returns_sessions(client):
    # Create a session first
    client.post('/analysis/start', json={
        'session_id': 'hist-test',
        'privacy_mode': False,
        'retention_hours': 72,
    })
    r = client.get('/history')
    assert r.status_code == 200
    items = r.json()['items']
    assert len(items) >= 1
    assert items[0]['session_id'] == 'hist-test'


def test_profiles_empty(client):
    r = client.get('/profiles')
    assert r.status_code == 200
    assert r.json()['items'] == []


def test_session_detail_not_found(client):
    r = client.get('/session/nonexistent')
    assert r.status_code == 404


def test_purge_not_found(client):
    r = client.delete('/privacy/purge/nonexistent')
    assert r.status_code == 404


def test_purge_existing_session(client):
    client.post('/analysis/start', json={
        'session_id': 'purge-me',
        'privacy_mode': False,
        'retention_hours': 72,
    })
    r = client.delete('/privacy/purge/purge-me')
    assert r.status_code == 200
    assert r.json()['purged'] is True

    # Verify it is gone
    r2 = client.get('/session/purge-me')
    assert r2.status_code == 404


def test_notes_update(client):
    client.post('/analysis/start', json={
        'session_id': 'notes-test',
        'privacy_mode': False,
        'retention_hours': 72,
    })
    r = client.post('/session/notes-test/notes', json={'note': 'Test observation'})
    assert r.status_code == 200
    assert r.json()['note'] == 'Test observation'


def test_notes_update_session_not_found(client):
    r = client.post('/session/nonexistent/notes', json={'note': 'hello'})
    assert r.status_code == 404


def test_status_not_found(client):
    r = client.get('/status/nonexistent')
    assert r.status_code == 404


def test_status_latest_idle(client):
    r = client.get('/status/latest')
    assert r.status_code == 200
    body = r.json()
    assert body['status']['stage'] == 'idle'


def test_security_headers(client):
    r = client.get('/')
    assert r.headers.get('X-Content-Type-Options') == 'nosniff'
    assert r.headers.get('X-Frame-Options') == 'DENY'
    assert r.headers.get('X-XSS-Protection') == '1; mode=block'
    assert r.headers.get('Referrer-Policy') == 'strict-origin-when-cross-origin'


def test_compare_session_not_found(client):
    r = client.get('/compare/nonexistent')
    assert r.status_code == 404


def test_history_profile_filter(client):
    client.post('/analysis/start', json={
        'session_id': 'pf-1',
        'profile_id': 'profile-x',
        'privacy_mode': False,
        'retention_hours': 72,
    })
    client.post('/analysis/start', json={
        'session_id': 'pf-2',
        'profile_id': 'profile-y',
        'privacy_mode': False,
        'retention_hours': 72,
    })
    r = client.get('/history?profile_id=profile-x')
    assert r.status_code == 200
    items = r.json()['items']
    assert len(items) == 1
    assert items[0]['session_id'] == 'pf-1'


def test_history_cursor_pagination(client):
    """The /history endpoint supports cursor-based pagination."""
    for i in range(4):
        client.post('/analysis/start', json={
            'session_id': f'cursor-{i}',
            'privacy_mode': False,
            'retention_hours': 72,
        })
    # Request 2 items per page
    r1 = client.get('/history?limit=2')
    assert r1.status_code == 200
    body1 = r1.json()
    assert len(body1['items']) == 2
    assert 'next_cursor' in body1

    # Second page
    r2 = client.get(f'/history?limit=2&cursor={body1["next_cursor"]}')
    assert r2.status_code == 200
    body2 = r2.json()
    assert len(body2['items']) == 2
    # No overlap between pages
    page1_ids = {item['session_id'] for item in body1['items']}
    page2_ids = {item['session_id'] for item in body2['items']}
    assert page1_ids.isdisjoint(page2_ids)
