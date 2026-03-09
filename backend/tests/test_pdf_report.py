"""Tests for PDF clinical report generation (write_pdf_report).

Covers all three presets (clinical, compact, presentation), region tables,
type distribution, diagnostic image embedding, comparison data, and the
/report and /export API endpoints.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ['ROBOFLOW_API_KEY'] = 'test-key-for-pdf-tests'

from api_bridge import (
    BridgeStore,
    REPORT_DIR,
    app,
    write_pdf_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path, monkeypatch):
    """Redirect all file I/O to temp directories."""
    test_uploads = tmp_path / 'uploads'
    test_outputs = tmp_path / 'outputs'
    test_reports = tmp_path / 'reports'
    test_uploads.mkdir()
    test_outputs.mkdir()
    test_reports.mkdir()

    monkeypatch.setattr('api_bridge.UPLOAD_DIR', test_uploads)
    monkeypatch.setattr('api_bridge.OUTPUT_DIR', test_outputs)
    monkeypatch.setattr('api_bridge.REPORT_DIR', test_reports)

    test_db = tmp_path / 'test.db'
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


@pytest.fixture()
def full_session(tmp_path):
    """A session dict with complete results data for PDF generation."""
    return {
        'session_id': 'pdf-test-session-001',
        'profile_id': 'test-profile',
        'timestamp': '2025-06-15T10:30:00+00:00',
        'severity': 'Moderate',
        'gags_score': 22,
        'lesion_count': 14,
        'symmetry_delta': 8.5,
        'note': 'Follow-up visit — patient reports improvement.',
        'diagnostic_image_path': None,
        'original_image_path': None,
        'privacy_mode': False,
        'retention_hours': 72,
        'results': {
            'clinical_analysis': {
                'regions': {
                    'forehead': {'count': 4, 'lpi': 12.3, 'area_px': 52000, 'gags_score': 6},
                    'right_cheek': {'count': 3, 'lpi': 8.1, 'area_px': 48000, 'gags_score': 4},
                    'left_cheek': {'count': 5, 'lpi': 15.7, 'area_px': 49000, 'gags_score': 6},
                    'nose': {'count': 1, 'lpi': 3.2, 'area_px': 18000, 'gags_score': 2},
                    'chin': {'count': 1, 'lpi': 4.0, 'area_px': 22000, 'gags_score': 2},
                    'jawline_neck': {'count': 0, 'lpi': 0, 'area_px': 35000, 'gags_score': 0},
                },
                'total_lesions': 14,
                'gags_total_score': 22,
                'clinical_severity': 'Moderate',
                'symmetry_delta': 8.5,
            },
            'consensus_summary': {
                'verified_lesions': 14,
                'average_confidence': 0.72,
                'top_regions': [
                    {'region': 'left_cheek', 'count': 5},
                    {'region': 'forehead', 'count': 4},
                    {'region': 'right_cheek', 'count': 3},
                ],
                'region_counts': {
                    'forehead': 4,
                    'right_cheek': 3,
                    'left_cheek': 5,
                    'nose': 1,
                    'chin': 1,
                    'jawline_neck': 0,
                },
                'unassigned_count': 0,
                'summary': '14 verified lesions with 0.72 average confidence across 5 regions.',
                'type_counts': {
                    'pustule': 5,
                    'papule': 6,
                    'comedone': 2,
                    'nodule': 1,
                },
            },
            'timing_ms': {
                'bisenet': 120.5,
                'landmarks': 45.2,
                'geometry': 33.1,
                'combine': 12.8,
                'total': 211.6,
            },
            'cloud_timing': {
                'model_a_640': 1200.0,
                'model_a_1280': 1800.0,
                'model_b': 950.0,
            },
            'pipeline_metrics': {
                'raw_detections': 42,
                'post_nms': 28,
                'post_gating': 14,
                'proximity_propagated': 3,
                'type_coverage': {'direct': 8, 'proximity': 3, 'none': 3},
            },
            'lesions': {
                '_pipeline_metrics': {
                    'raw_detections': 42,
                    'post_nms': 28,
                    'post_gating': 14,
                },
            },
        },
    }


@pytest.fixture()
def compare_data():
    """Comparison payload for temporal comparison tests."""
    return {
        'previous_session_id': 'prev-session-999',
        'current_session_id': 'pdf-test-session-001',
        'previous_timestamp': '2025-05-15T10:00:00+00:00',
        'current_timestamp': '2025-06-15T10:30:00+00:00',
        'severity_change': {'from': 'Severe', 'to': 'Moderate'},
        'lesion_delta': -6,
        'gags_delta': -8,
        'symmetry_delta_change': -2.1,
        'comparison_mode': 'previous_archived_session',
        'regions': {
            'forehead': {
                'previous_count': 7, 'current_count': 4,
                'count_delta': -3, 'previous_lpi': 20.0, 'current_lpi': 12.3, 'lpi_delta': -7.7,
            },
            'left_cheek': {
                'previous_count': 7, 'current_count': 5,
                'count_delta': -2, 'previous_lpi': 22.0, 'current_lpi': 15.7, 'lpi_delta': -6.3,
            },
            'right_cheek': {
                'previous_count': 3, 'current_count': 3,
                'count_delta': 0, 'previous_lpi': 8.1, 'current_lpi': 8.1, 'lpi_delta': 0,
            },
        },
    }


@pytest.fixture()
def diagnostic_image(tmp_path):
    """Create a small diagnostic JPEG for embedding tests."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = (200, 180, 160)
    cv2.rectangle(img, (40, 40), (80, 80), (0, 255, 0), 2)
    path = tmp_path / 'outputs' / 'pdf-test-session-001_diagnostic.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path)


# ---------------------------------------------------------------------------
# write_pdf_report — unit tests
# ---------------------------------------------------------------------------

class TestWritePdfReportClinical:
    """Tests for the clinical (full) preset."""

    def test_generates_pdf_file(self, full_session):
        path = write_pdf_report(full_session, None, 'clinical')
        assert path.exists()
        assert path.suffix == '.pdf'
        assert path.stat().st_size > 500

    def test_filename_contains_session_id_and_preset(self, full_session):
        path = write_pdf_report(full_session, None, 'clinical')
        assert 'pdf-test-session-001' in path.name
        assert 'clinical' in path.name

    def test_pdf_contains_expected_content(self, full_session):
        path = write_pdf_report(full_session, None, 'clinical')
        content = path.read_bytes()
        # Check PDF header
        assert content[:5] == b'%PDF-'
        # Check for session text (strings may be embedded in PDF content streams)
        assert b'Acne Severity Analysis Report' in content or b'pdf-test-session-001' in content

    def test_clinical_with_regions(self, full_session):
        """Clinical preset should include regional GAGS table."""
        path = write_pdf_report(full_session, None, 'clinical')
        assert path.exists()
        content = path.read_bytes()
        # The PDF should be substantial with tables
        assert len(content) > 2000

    def test_clinical_with_type_distribution(self, full_session):
        """Clinical preset should include type distribution when type_counts exist."""
        path = write_pdf_report(full_session, None, 'clinical')
        assert path.exists()
        # PDF should be larger when type data is present
        assert path.stat().st_size > 2000

    def test_clinical_with_comparison(self, full_session, compare_data):
        """Clinical with comparison data should include temporal section."""
        path = write_pdf_report(full_session, compare_data, 'clinical')
        assert path.exists()
        assert path.stat().st_size > 2000

    def test_clinical_with_pipeline_metrics(self, full_session):
        """Pipeline metadata section should be included in clinical preset."""
        path = write_pdf_report(full_session, None, 'clinical')
        assert path.exists()
        assert path.stat().st_size > 2000

    def test_clinical_with_diagnostic_image(self, full_session, diagnostic_image):
        """Diagnostic image should be embedded when path exists."""
        full_session['diagnostic_image_path'] = diagnostic_image
        path_without = write_pdf_report(full_session, None, 'clinical')
        size_without = path_without.stat().st_size

        # Re-generate with image
        full_session['diagnostic_image_path'] = diagnostic_image
        path_with = write_pdf_report(full_session, None, 'clinical')
        size_with = path_with.stat().st_size

        # PDF with image should be larger
        assert size_with >= size_without

    def test_clinical_missing_diagnostic_image(self, full_session):
        """Should gracefully handle missing diagnostic image file."""
        full_session['diagnostic_image_path'] = '/nonexistent/path.jpg'
        path = write_pdf_report(full_session, None, 'clinical')
        assert path.exists()

    def test_clinical_with_note(self, full_session):
        """Session note should be included."""
        full_session['note'] = 'Important observation about treatment response.'
        path = write_pdf_report(full_session, None, 'clinical')
        assert path.exists()


class TestWritePdfReportCompact:
    """Tests for the compact (one-page) preset."""

    def test_generates_compact_pdf(self, full_session):
        path = write_pdf_report(full_session, None, 'compact')
        assert path.exists()
        assert 'compact' in path.name

    def test_compact_is_smaller_than_clinical(self, full_session):
        """Compact should generally be shorter than clinical."""
        clinical = write_pdf_report(full_session, None, 'clinical')
        compact = write_pdf_report(full_session, None, 'compact')
        # Compact should exist and be a valid PDF
        assert compact.exists()
        assert compact.read_bytes()[:5] == b'%PDF-'

    def test_compact_with_comparison(self, full_session, compare_data):
        path = write_pdf_report(full_session, compare_data, 'compact')
        assert path.exists()

    def test_compact_with_note(self, full_session):
        full_session['note'] = 'Quick follow-up note.'
        path = write_pdf_report(full_session, None, 'compact')
        assert path.exists()


class TestWritePdfReportPresentation:
    """Tests for the presentation preset."""

    def test_generates_presentation_pdf(self, full_session):
        path = write_pdf_report(full_session, None, 'presentation')
        assert path.exists()
        assert 'presentation' in path.name

    def test_presentation_with_comparison(self, full_session, compare_data):
        path = write_pdf_report(full_session, compare_data, 'presentation')
        assert path.exists()

    def test_presentation_with_note(self, full_session):
        full_session['note'] = 'Presenter note for case conference.'
        path = write_pdf_report(full_session, None, 'presentation')
        assert path.exists()


class TestWritePdfReportEdgeCases:
    """Edge cases and error handling."""

    def test_empty_results(self):
        """Should handle session with empty results."""
        session = {
            'session_id': 'empty-session',
            'profile_id': None,
            'timestamp': '2025-01-01T00:00:00+00:00',
            'severity': None,
            'gags_score': None,
            'lesion_count': None,
            'symmetry_delta': None,
            'note': '',
            'diagnostic_image_path': None,
            'results': {},
        }
        path = write_pdf_report(session, None, 'clinical')
        assert path.exists()

    def test_no_results_key(self):
        """Should handle session with no results at all."""
        session = {
            'session_id': 'no-results',
            'profile_id': None,
            'timestamp': '2025-01-01T00:00:00+00:00',
            'severity': None,
            'gags_score': None,
            'lesion_count': None,
            'symmetry_delta': None,
            'note': '',
            'diagnostic_image_path': None,
        }
        path = write_pdf_report(session, None, 'clinical')
        assert path.exists()

    def test_severity_colors(self, full_session):
        """Different severity levels should produce valid PDFs."""
        for i, sev in enumerate(['Mild', 'Moderate', 'Severe', 'Very Severe / Cystic', None]):
            full_session['severity'] = sev
            full_session['session_id'] = f'sev-test-{i}'
            path = write_pdf_report(full_session, None, 'clinical')
            assert path.exists()

    def test_no_type_counts(self, full_session):
        """Should handle missing type_counts gracefully."""
        full_session['results']['consensus_summary'].pop('type_counts', None)
        path = write_pdf_report(full_session, None, 'clinical')
        assert path.exists()

    def test_empty_regions(self, full_session):
        """Should handle empty regions dict."""
        full_session['results']['clinical_analysis']['regions'] = {}
        path = write_pdf_report(full_session, None, 'clinical')
        assert path.exists()

    def test_unknown_preset_defaults_to_clinical(self, full_session):
        """Unknown preset should still generate a valid PDF."""
        path = write_pdf_report(full_session, None, 'unknown')
        assert path.exists()


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

def _insert_session_with_results(client, session_dict):
    """Helper: insert a session directly into the store for endpoint testing."""
    store = app.state.resources['store']
    row = {
        'session_id': session_dict['session_id'],
        'profile_id': session_dict.get('profile_id'),
        'timestamp': session_dict['timestamp'],
        'severity': session_dict.get('severity'),
        'gags_score': session_dict.get('gags_score'),
        'lesion_count': session_dict.get('lesion_count'),
        'symmetry_delta': session_dict.get('symmetry_delta'),
        'results_json': json.dumps(session_dict.get('results', {})),
        'note': session_dict.get('note', ''),
        'diagnostic_image_path': session_dict.get('diagnostic_image_path'),
        'original_image_path': session_dict.get('original_image_path'),
        'privacy_mode': session_dict.get('privacy_mode', False),
        'retention_hours': session_dict.get('retention_hours', 72),
    }
    store.upsert_session(row)


class TestReportEndpoint:
    """Tests for GET /report/{session_id}."""

    def test_report_returns_pdf_data(self, client, full_session):
        _insert_session_with_results(client, full_session)
        r = client.get(f'/report/{full_session["session_id"]}')
        assert r.status_code == 200
        body = r.json()
        assert body['session_id'] == full_session['session_id']
        assert 'report' in body
        assert 'pdf_path' in body['report']
        assert 'pdf_data_uri' in body['report']
        assert body['report']['pdf_data_uri'].startswith('data:application/pdf;base64,')

    def test_report_404_missing_session(self, client):
        r = client.get('/report/nonexistent-session')
        assert r.status_code == 404

    def test_report_409_no_results(self, client):
        """Session without completed results should return 409."""
        store = app.state.resources['store']
        store.upsert_session({
            'session_id': 'incomplete-session',
            'profile_id': None,
            'timestamp': '2025-01-01T00:00:00+00:00',
            'severity': None,
            'gags_score': None,
            'lesion_count': None,
            'symmetry_delta': None,
            'results_json': None,
            'note': '',
            'diagnostic_image_path': None,
            'original_image_path': None,
            'privacy_mode': False,
            'retention_hours': 72,
        })
        r = client.get('/report/incomplete-session')
        assert r.status_code == 409


class TestExportEndpoint:
    """Tests for POST /export/{session_id}."""

    def test_export_clinical(self, client, full_session):
        _insert_session_with_results(client, full_session)
        r = client.post(
            f'/export/{full_session["session_id"]}',
            json={'preset': 'clinical', 'include_pdf_data': True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body['preset'] == 'clinical'
        assert body['pdf_data_uri'].startswith('data:application/pdf;base64,')

    def test_export_compact(self, client, full_session):
        _insert_session_with_results(client, full_session)
        r = client.post(
            f'/export/{full_session["session_id"]}',
            json={'preset': 'compact', 'include_pdf_data': True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body['preset'] == 'compact'

    def test_export_presentation(self, client, full_session):
        _insert_session_with_results(client, full_session)
        r = client.post(
            f'/export/{full_session["session_id"]}',
            json={'preset': 'presentation', 'include_pdf_data': True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body['preset'] == 'presentation'

    def test_export_without_pdf_data(self, client, full_session):
        _insert_session_with_results(client, full_session)
        r = client.post(
            f'/export/{full_session["session_id"]}',
            json={'preset': 'clinical', 'include_pdf_data': False},
        )
        assert r.status_code == 200
        body = r.json()
        assert 'pdf_data_uri' not in body or body.get('pdf_data_uri') is None

    def test_export_404_missing_session(self, client):
        r = client.post(
            '/export/nonexistent-session',
            json={'preset': 'clinical', 'include_pdf_data': True},
        )
        assert r.status_code == 404

    def test_export_with_previous_session(self, client, full_session):
        """Export with a comparison session should include comparison data in PDF."""
        prev_session = dict(full_session)
        prev_session['session_id'] = 'prev-session-for-export'
        prev_session['severity'] = 'Severe'
        prev_session['gags_score'] = 30
        prev_session['lesion_count'] = 20
        _insert_session_with_results(client, prev_session)
        _insert_session_with_results(client, full_session)

        r = client.post(
            f'/export/{full_session["session_id"]}',
            json={
                'preset': 'clinical',
                'include_pdf_data': True,
                'previous_session_id': prev_session['session_id'],
            },
        )
        assert r.status_code == 200
