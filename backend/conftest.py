"""Shared fixtures for backend tests."""
import os
import sys
from pathlib import Path

import pytest

# Ensure the backend directory is on sys.path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Override env vars before importing any app code
os.environ.setdefault('ROBOFLOW_API_KEY', 'test-key-for-unit-tests')


@pytest.fixture()
def tmp_db(tmp_path: Path):
    """Return a path to a temporary SQLite database file."""
    return tmp_path / 'test_sessions.db'


@pytest.fixture()
def store(tmp_db: Path):
    """Create a BridgeStore backed by an in-memory-like temp database."""
    from api_bridge import BridgeStore
    return BridgeStore(tmp_db)


@pytest.fixture()
def sample_session():
    """Return a minimal valid session payload dict."""
    return {
        'session_id': 'test-session-001',
        'profile_id': 'default-profile',
        'timestamp': '2025-01-15T12:00:00+00:00',
        'severity': 'Mild',
        'gags_score': 8,
        'lesion_count': 5,
        'symmetry_delta': 3.2,
        'results_json': '{"clinical_analysis": {"total_lesions": 5}}',
        'note': 'Test note',
        'diagnostic_image_path': None,
        'original_image_path': None,
        'privacy_mode': False,
        'retention_hours': 72,
    }


@pytest.fixture()
def populated_store(store, sample_session):
    """Store with one session already inserted."""
    store.upsert_session(sample_session)
    return store
