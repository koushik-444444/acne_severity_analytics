"""Tests for Phase 5 metrics: consensus_summary, usage_tracker, /metrics endpoint."""
import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Set env vars before importing app
os.environ['ROBOFLOW_API_KEY'] = 'test-key-for-metrics-tests'

from api_bridge import app, BridgeStore, consensus_summary
import usage_tracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def _isolate_usage_db(tmp_path, monkeypatch):
    """Point usage_tracker at a temporary database."""
    monkeypatch.setattr(usage_tracker, '_DB_PATH', tmp_path / 'usage.db')
    # Reset the cached connection so the next call re-creates
    monkeypatch.setattr(usage_tracker, '_conn', None)


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# consensus_summary() unit tests
# ---------------------------------------------------------------------------

class TestConsensusSummary:
    """Validate consensus_summary() correctly handles assignments."""

    def test_basic_counts(self):
        assignments = {
            'forehead': [
                {'confidence': 0.9, 'class_name': 'papule'},
                {'confidence': 0.7, 'class_name': 'pustule'},
            ],
            'left_cheek': [
                {'confidence': 0.8, 'class_name': 'papule'},
            ],
            'unassigned': [
                {'confidence': 0.3, 'class_name': 'acne'},
            ],
        }
        result = consensus_summary(assignments)
        assert result['verified_lesions'] == 3
        assert result['unassigned_count'] == 1
        assert len(result['lesions']) == 3
        assert result['type_counts'] == {'papule': 2, 'pustule': 1}

    def test_empty_assignments(self):
        result = consensus_summary({})
        assert result['verified_lesions'] == 0
        assert result['average_confidence'] == 0.0
        assert result['lesions'] == []
        assert result['summary'] == 'No verified lesions detected'

    def test_skips_pipeline_metrics_key(self):
        """_pipeline_metrics is a dict (not a list of dicts) and must be ignored."""
        assignments = {
            'forehead': [
                {'confidence': 0.9, 'class_name': 'papule'},
            ],
            '_pipeline_metrics': {
                'raw_detections': 50,
                'post_nms': 30,
                'post_gating': 20,
                'proximity_propagated': 5,
                'type_coverage': {'papule': 15, 'pustule': 5},
                'raw_by_stream': {'preds_a_1280': 40, 'preds_b': 10},
            },
        }
        result = consensus_summary(assignments)
        assert result['verified_lesions'] == 1
        assert len(result['lesions']) == 1
        # _pipeline_metrics should not appear in lesions or type_counts
        assert result['type_counts'] == {'papule': 1}

    def test_skips_any_underscore_prefixed_key(self):
        """Any key starting with _ should be filtered out."""
        assignments = {
            'chin': [
                {'confidence': 0.85, 'class_name': 'comedone'},
            ],
            '_internal_debug': 'some debug string',
        }
        result = consensus_summary(assignments)
        assert result['verified_lesions'] == 1
        assert len(result['lesions']) == 1

    def test_skips_non_list_values(self):
        """Non-list values (even without underscore prefix) are filtered."""
        assignments = {
            'nose': [
                {'confidence': 0.75, 'class_name': 'papule'},
            ],
            'metadata': {'key': 'value'},  # not a list
        }
        result = consensus_summary(assignments)
        assert result['verified_lesions'] == 1

    def test_average_confidence(self):
        assignments = {
            'forehead': [
                {'confidence': 0.8, 'class_name': 'papule'},
                {'confidence': 0.6, 'class_name': 'papule'},
            ],
        }
        result = consensus_summary(assignments)
        assert result['average_confidence'] == 0.7

    def test_top_regions_ordering(self):
        assignments = {
            'forehead': [{'confidence': 0.9}] * 5,
            'left_cheek': [{'confidence': 0.9}] * 3,
            'right_cheek': [{'confidence': 0.9}] * 8,
            'chin': [{'confidence': 0.9}] * 1,
        }
        result = consensus_summary(assignments)
        top = result['top_regions']
        assert len(top) == 3
        assert top[0]['region'] == 'right_cheek'
        assert top[0]['count'] == 8
        assert top[1]['region'] == 'forehead'
        assert top[1]['count'] == 5

    def test_lesions_have_region_tag(self):
        assignments = {
            'forehead': [
                {'confidence': 0.9, 'class_name': 'papule', 'x': 10, 'y': 20},
            ],
        }
        result = consensus_summary(assignments)
        assert result['lesions'][0]['region'] == 'forehead'
        assert result['lesions'][0]['x'] == 10


# ---------------------------------------------------------------------------
# usage_tracker unit tests
# ---------------------------------------------------------------------------

class TestUsageTracker:
    """Test usage_tracker module with isolated temp database."""

    def test_log_and_count(self, _isolate_usage_db):
        usage_tracker.log_api_call('model-a/v1', 'success')
        usage_tracker.log_api_call('model-a/v1', 'success')
        usage_tracker.log_api_call('model-b/v1', 'error', error='timeout')
        assert usage_tracker.get_usage_stats() == 3

    def test_summary_structure(self, _isolate_usage_db):
        usage_tracker.log_api_call('model-a', 'success', latency_ms=100.0)
        usage_tracker.log_api_call('model-a', 'success', latency_ms=200.0)
        usage_tracker.log_api_call('model-b', 'error', latency_ms=500.0, error='500 Internal')
        summary = usage_tracker.get_usage_summary()
        assert summary['total_calls'] == 3
        assert summary['calls_by_model'] == {'model-a': 2, 'model-b': 1}
        assert summary['calls_by_status'] == {'success': 2, 'error': 1}
        assert summary['error_rate'] == round(1 / 3, 4)
        assert len(summary['recent_errors']) == 1
        assert summary['recent_errors'][0]['error'] == '500 Internal'

    def test_latency_stats(self, _isolate_usage_db):
        for ms in [100.0, 200.0, 300.0, 400.0, 500.0]:
            usage_tracker.log_api_call('model-a', 'success', latency_ms=ms)
        summary = usage_tracker.get_usage_summary()
        stats = summary['latency_stats']
        assert stats['count'] == 5
        assert stats['min_ms'] == 100.0
        assert stats['max_ms'] == 500.0
        assert stats['mean_ms'] == 300.0
        assert stats['p50_ms'] == 300.0  # median of [100,200,300,400,500]

    def test_empty_db(self, _isolate_usage_db):
        summary = usage_tracker.get_usage_summary()
        assert summary['total_calls'] == 0
        assert summary['latency_stats'] is None
        assert summary['error_rate'] == 0.0
        assert summary['recent_errors'] == []

    def test_no_latency_rows(self, _isolate_usage_db):
        usage_tracker.log_api_call('model-a', 'success')
        summary = usage_tracker.get_usage_summary()
        assert summary['latency_stats'] is None


# ---------------------------------------------------------------------------
# /metrics endpoint integration tests
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    """Test the /metrics GET endpoint."""

    def test_metrics_empty(self, client, _isolate_usage_db):
        r = client.get('/metrics')
        assert r.status_code == 200
        body = r.json()
        assert 'api_usage' in body
        assert 'session_stats' in body
        assert 'timing' in body
        assert 'pipeline_metrics' in body
        assert body['session_stats']['total_sessions'] == 0

    def test_metrics_with_sessions(self, client, _isolate_usage_db):
        """Create sessions with results, verify metrics aggregation."""
        store = app.state.resources['store']
        for i in range(3):
            sess = {
                'session_id': f'metric-test-{i}',
                'profile_id': 'default',
                'timestamp': f'2025-01-1{i}T12:00:00+00:00',
                'severity': 'Mild',
                'gags_score': 5 + i,
                'lesion_count': 10 + i,
                'symmetry_delta': 2.0,
                'results_json': json.dumps({
                    'timing_ms': {
                        'bisenet': 1000 + i * 100,
                        'landmarks': 100 + i * 10,
                        'geometry': 50,
                        'combine': 20,
                        'total': 1170 + i * 110,
                    },
                    'clinical_analysis': {
                        'total_lesions': 10 + i,
                        'gags_score': 5 + i,
                    },
                }),
                'note': None,
                'diagnostic_image_path': None,
                'original_image_path': None,
                'privacy_mode': False,
                'retention_hours': 72,
            }
            store.upsert_session(sess)

        r = client.get('/metrics')
        assert r.status_code == 200
        body = r.json()
        assert body['session_stats']['total_sessions'] == 3
        assert body['session_stats']['sessions_with_results'] == 3
        assert body['session_stats']['detection_counts']['mean'] == 11.0
        assert body['timing']['sample_count'] == 3
        assert body['timing']['local_pipeline']['bisenet_mean'] is not None

    def test_metrics_with_api_usage(self, client, _isolate_usage_db):
        """Verify that API usage data flows into /metrics."""
        usage_tracker.log_api_call('model-a/v2', 'success', latency_ms=150.0)
        usage_tracker.log_api_call('model-b/v1', 'success', latency_ms=300.0)
        r = client.get('/metrics')
        assert r.status_code == 200
        body = r.json()
        assert body['api_usage']['total_calls'] == 2
        assert body['api_usage']['calls_by_model']['model-a/v2'] == 1

    def test_metrics_pipeline_metrics(self, client, _isolate_usage_db):
        """Sessions with pipeline_metrics should aggregate properly."""
        store = app.state.resources['store']
        sess = {
            'session_id': 'pm-test',
            'profile_id': 'default',
            'timestamp': '2025-01-15T12:00:00+00:00',
            'severity': 'Moderate',
            'gags_score': 12,
            'lesion_count': 20,
            'symmetry_delta': 4.0,
            'results_json': json.dumps({
                'timing_ms': {'total': 1500},
                'clinical_analysis': {'total_lesions': 20, 'gags_score': 12},
                'pipeline_metrics': {
                    'raw_detections': 50,
                    'post_nms': 30,
                    'post_gating': 25,
                    'proximity_propagated': 5,
                    'type_coverage': {'papule': 15, 'pustule': 10},
                },
            }),
            'note': None,
            'diagnostic_image_path': None,
            'original_image_path': None,
            'privacy_mode': False,
            'retention_hours': 72,
        }
        store.upsert_session(sess)
        r = client.get('/metrics')
        assert r.status_code == 200
        body = r.json()
        pm = body['pipeline_metrics']
        assert pm is not None
        assert pm['sample_count'] == 1
        assert pm['total_raw_detections'] == 50
        assert pm['total_post_nms'] == 30
        assert pm['nms_reduction_pct'] == 40.0
        assert pm['type_coverage_aggregate'] == {'papule': 15, 'pustule': 10}
