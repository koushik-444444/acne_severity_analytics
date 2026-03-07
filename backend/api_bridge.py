import asyncio
import base64
import json
import os
import re
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

load_dotenv()

APP_VERSION = '2.0.0'
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR / 'outputs'
REPORT_DIR = OUTPUT_DIR / 'reports'
DB_PATH = BASE_DIR / 'sessions.db'
MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', str(10 * 1024 * 1024)))
DEFAULT_RETENTION_HOURS = int(os.getenv('DEFAULT_RETENTION_HOURS', '72'))
MAX_RETENTION_HOURS = int(os.getenv('MAX_RETENTION_HOURS', '720'))
MODEL_A_ID = os.getenv('MODEL_A_ID', 'runner-e0dmy/acne-ijcab/2')
MODEL_B_ID = os.getenv('MODEL_B_ID', 'acne-project-2auvb/acne-detection-v2/1')
ROBOFLOW_API_KEY: Optional[str] = os.getenv('ROBOFLOW_API_KEY')

if not ROBOFLOW_API_KEY:
    raise RuntimeError('ROBOFLOW_API_KEY environment variable is required')

assert ROBOFLOW_API_KEY is not None
API_KEY = ROBOFLOW_API_KEY

CloudInferenceEngine = None
EnsembleLesionMapper = None
FaceSegmentationPipeline = None
draw_region_masks = None
draw_lesion_boxes = None


def ensure_runtime_imports() -> None:
    global CloudInferenceEngine, EnsembleLesionMapper, FaceSegmentationPipeline, draw_region_masks, draw_lesion_boxes
    if CloudInferenceEngine is None:
        from cloud_inference import CloudInferenceEngine as runtime_cloud_engine
        CloudInferenceEngine = runtime_cloud_engine
    if EnsembleLesionMapper is None:
        from face_segmentation.ensemble_mapper import EnsembleLesionMapper as runtime_mapper
        EnsembleLesionMapper = runtime_mapper
    if FaceSegmentationPipeline is None:
        from face_segmentation.pipeline import FaceSegmentationPipeline as runtime_pipeline
        FaceSegmentationPipeline = runtime_pipeline
    if draw_region_masks is None:
        from face_segmentation.utils.visualization import draw_region_masks as runtime_draw_region_masks
        draw_region_masks = runtime_draw_region_masks
    if draw_lesion_boxes is None:
        from face_segmentation.utils.visualization import draw_lesion_boxes as runtime_draw_lesion_boxes
        draw_lesion_boxes = runtime_draw_lesion_boxes


class AnalysisStartRequest(BaseModel):
    session_id: Optional[str] = None
    profile_id: Optional[str] = None
    privacy_mode: bool = False
    retention_hours: int = Field(
        default=DEFAULT_RETENTION_HOURS,
        ge=1,
        le=MAX_RETENTION_HOURS,
    )


class ExportRequest(BaseModel):
    include_pdf_data: bool = True
    preset: str = 'clinical'
    previous_session_id: Optional[str] = None


class NotesRequest(BaseModel):
    note: str = ''


model_init_lock = threading.Lock()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat()


def parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def ensure_dirs() -> None:
    for path in (UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def validate_session_id(session_id: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', session_id):
        raise HTTPException(status_code=400, detail='Invalid session_id format')
    return session_id


def normalize_retention(hours: int) -> int:
    return max(1, min(int(hours), MAX_RETENTION_HOURS))


def safe_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        file_path = Path(path)
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
    except OSError:
        pass


def absolute_managed_path(path: Path) -> str:
    return str(path.resolve())


def bytes_to_data_uri(payload: bytes, mime: str) -> str:
    encoded = base64.b64encode(payload).decode('ascii')
    return f'data:{mime};base64,{encoded}'


def file_to_data_uri(path: Optional[str], mime: str) -> Optional[str]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = (BASE_DIR / file_path).resolve()
    if not file_path.exists():
        return None
    return bytes_to_data_uri(file_path.read_bytes(), mime)


def image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(status_code=500, detail='Failed to encode image output')
    return buf.tobytes()


def save_image(path: Path, image: np.ndarray) -> str:
    if not cv2.imwrite(str(path), image):
        raise HTTPException(status_code=500, detail=f'Failed to save image: {path.name}')
    return str(path)


def validate_upload(upload: UploadFile, payload: bytes) -> None:
    if upload.content_type not in {'image/jpeg', 'image/png', 'image/webp'}:
        raise HTTPException(
            status_code=415,
            detail='Only JPEG, PNG, and WEBP images are supported',
        )
    if not payload:
        raise HTTPException(status_code=400, detail='Uploaded file is empty')
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f'File exceeds {MAX_UPLOAD_BYTES} byte limit',
        )


def decode_image(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail='Failed to decode image')
    return image


def consensus_summary(assignments: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    region_counts = {
        region: len(items)
        for region, items in assignments.items()
        if region != 'unassigned' and items
    }
    confidence_values = [
        item.get('confidence', 0.0)
        for items in assignments.values()
        for item in items
    ]
    total = sum(region_counts.values())
    top_regions = sorted(region_counts.items(), key=lambda item: item[1], reverse=True)[:3]
    summary = 'No verified lesions detected'
    if total:
        summary = f"{total} verified lesions concentrated in {', '.join(name for name, _ in top_regions)}"
    return {
        'verified_lesions': total,
        'average_confidence': round(sum(confidence_values) / len(confidence_values), 3)
        if confidence_values
        else 0.0,
        'top_regions': [{'region': name, 'count': count} for name, count in top_regions],
        'region_counts': region_counts,
        'unassigned_count': len(assignments.get('unassigned', [])),
        'summary': summary,
    }


def compact_session(session: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'session_id': session['session_id'],
        'profile_id': session.get('profile_id'),
        'timestamp': session['timestamp'],
        'severity': session.get('severity'),
        'gags_score': session.get('gags_score'),
        'lesion_count': session.get('lesion_count'),
        'symmetry_delta': session.get('symmetry_delta'),
        'privacy_mode': session.get('privacy_mode'),
        'retention_hours': session.get('retention_hours'),
        'status': session.get('status'),
    }


def row_to_session(row: sqlite3.Row, status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        'session_id': row['session_id'],
        'profile_id': row['profile_id'] if 'profile_id' in row.keys() else None,
        'timestamp': row['timestamp'],
        'severity': row['severity'],
        'gags_score': row['gags_score'],
        'lesion_count': row['lesion_count'],
        'symmetry_delta': row['symmetry_delta'],
        'results': json.loads(row['results_json']) if row['results_json'] else None,
        'note': row['note'] if 'note' in row.keys() else '',
        'diagnostic_image_path': row['diagnostic_image_path'],
        'original_image_path': row['original_image_path'],
        'diagnostic_image': file_to_data_uri(row['diagnostic_image_path'], 'image/jpeg'),
        'original_image': file_to_data_uri(row['original_image_path'], 'image/jpeg'),
        'privacy_mode': bool(row['privacy_mode']),
        'retention_hours': row['retention_hours'],
        'status': status,
    }


def compare_payload(
    previous: Optional[Dict[str, Any]],
    current: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not previous or not previous.get('results') or not current.get('results'):
        return None
    prev = previous['results'].get('clinical_analysis', {})
    curr = current['results'].get('clinical_analysis', {})
    prev_regions = prev.get('regions', {})
    curr_regions = curr.get('regions', {})
    regions = {}
    for region in sorted(set(prev_regions) | set(curr_regions)):
        prev_region = prev_regions.get(region, {})
        curr_region = curr_regions.get(region, {})
        regions[region] = {
            'previous_count': prev_region.get('count', 0),
            'current_count': curr_region.get('count', 0),
            'count_delta': curr_region.get('count', 0) - prev_region.get('count', 0),
            'previous_lpi': prev_region.get('lpi', 0),
            'current_lpi': curr_region.get('lpi', 0),
            'lpi_delta': round(curr_region.get('lpi', 0) - prev_region.get('lpi', 0), 2),
        }
    return {
        'previous_session_id': previous['session_id'],
        'current_session_id': current['session_id'],
        'previous_timestamp': previous['timestamp'],
        'current_timestamp': current['timestamp'],
        'severity_change': {
            'from': prev.get('clinical_severity', 'Unknown'),
            'to': curr.get('clinical_severity', 'Unknown'),
        },
        'lesion_delta': curr.get('total_lesions', 0) - prev.get('total_lesions', 0),
        'gags_delta': curr.get('gags_total_score', 0) - prev.get('gags_total_score', 0),
        'symmetry_delta_change': round(
            curr.get('symmetry_delta', 0) - prev.get('symmetry_delta', 0),
            2,
        ),
        'regions': regions,
    }


def annotate_compare_payload(
    compare: Optional[Dict[str, Any]],
    explicit_previous: bool,
) -> Optional[Dict[str, Any]]:
    if not compare:
        return None
    compare['comparison_mode'] = 'selected_baseline' if explicit_previous else 'previous_archived_session'
    return compare


def get_delta_status(value: float, better_when_lower: bool = True) -> str:
    if value == 0:
        return 'stable'
    improved = value < 0 if better_when_lower else value > 0
    return 'improved' if improved else 'worsened'


def format_delta_number(value: Any) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(round(numeric, 2))


def format_signed_delta(value: Any, suffix: str = '') -> str:
    numeric = float(value)
    prefix = '+' if numeric > 0 else ''
    return f'{prefix}{format_delta_number(numeric)}{suffix}'


def describe_delta(value: float, label: str, better_when_lower: bool = True) -> str:
    status = get_delta_status(value, better_when_lower)
    if status == 'stable':
        return f'{label} remained stable'
    magnitude = format_delta_number(abs(float(value)))
    return f'{label} {status} by {magnitude}'


def compare_target_label(compare: Dict[str, Any]) -> str:
    if compare.get('comparison_mode') == 'selected_baseline':
        return 'the selected baseline session'
    return 'the previous archived session'


def comparison_summary(compare: Dict[str, Any]) -> str:
    return (
        f"{describe_delta(compare.get('lesion_delta', 0), 'Lesion burden')} and "
        f"{describe_delta(compare.get('gags_delta', 0), 'GAGS')} versus "
        f"{compare_target_label(compare)}."
    )


def top_region_changes(compare: Dict[str, Any], limit: int = 3) -> List[Any]:
    region_items = list((compare.get('regions') or {}).items())
    ranked = sorted(
        region_items,
        key=lambda item: abs(item[1].get('count_delta', 0)),
        reverse=True,
    )
    return [item for item in ranked if item[1].get('count_delta', 0) != 0][:limit]


class BridgeStore:
    def __init__(self, db_path: Path):
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self.lock:
            self.conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    profile_id TEXT,
                    timestamp TEXT NOT NULL,
                    severity TEXT,
                    gags_score INTEGER,
                    lesion_count INTEGER,
                    symmetry_delta REAL,
                    results_json TEXT,
                    note TEXT,
                    diagnostic_image_path TEXT,
                    original_image_path TEXT,
                    privacy_mode INTEGER NOT NULL DEFAULT 0,
                    retention_hours INTEGER NOT NULL DEFAULT 72
                )
                '''
            )
            try:
                self.conn.execute('ALTER TABLE sessions ADD COLUMN profile_id TEXT')
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute('ALTER TABLE sessions ADD COLUMN retention_hours INTEGER NOT NULL DEFAULT 72')
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute('ALTER TABLE sessions ADD COLUMN privacy_mode INTEGER NOT NULL DEFAULT 0')
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute('ALTER TABLE sessions ADD COLUMN original_image_path TEXT')
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute('ALTER TABLE sessions ADD COLUMN note TEXT')
            except sqlite3.OperationalError:
                pass
            self.conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS statuses (
                    session_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    status_json TEXT NOT NULL
                )
                '''
            )
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def upsert_session(self, payload: Dict[str, Any]) -> None:
        with self.lock:
            self.conn.execute(
                '''
                INSERT INTO sessions (
                    session_id, profile_id, timestamp, severity, gags_score, lesion_count,
                    symmetry_delta, results_json, note, diagnostic_image_path,
                    original_image_path, privacy_mode, retention_hours
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    profile_id=excluded.profile_id,
                    timestamp=excluded.timestamp,
                    severity=excluded.severity,
                    gags_score=excluded.gags_score,
                    lesion_count=excluded.lesion_count,
                    symmetry_delta=excluded.symmetry_delta,
                    results_json=excluded.results_json,
                    note=excluded.note,
                    diagnostic_image_path=excluded.diagnostic_image_path,
                    original_image_path=excluded.original_image_path,
                    privacy_mode=excluded.privacy_mode,
                    retention_hours=excluded.retention_hours
                ''',
                (
                    payload['session_id'],
                    payload.get('profile_id'),
                    payload['timestamp'],
                    payload.get('severity'),
                    payload.get('gags_score'),
                    payload.get('lesion_count'),
                    payload.get('symmetry_delta'),
                    payload.get('results_json'),
                    payload.get('note', ''),
                    payload.get('diagnostic_image_path'),
                    payload.get('original_image_path'),
                    1 if payload.get('privacy_mode') else 0,
                    payload.get('retention_hours', DEFAULT_RETENTION_HOURS),
                ),
            )
            self.conn.commit()

    def set_status(
        self,
        session_id: str,
        stage: str,
        detail: str,
        progress: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        status = {
            'session_id': session_id,
            'stage': stage,
            'detail': detail,
            'progress': int(max(0, min(progress, 100))),
            'updated_at': utcnow_iso(),
        }
        if extra:
            status.update(extra)
        with self.lock:
            self.conn.execute(
                '''
                INSERT INTO statuses (session_id, updated_at, status_json)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    status_json=excluded.status_json
                ''',
                (session_id, status['updated_at'], json.dumps(status)),
            )
            self.conn.commit()
        return status

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            'SELECT status_json FROM statuses WHERE session_id = ?',
            (session_id,),
        ).fetchone()
        return json.loads(row['status_json']) if row else None

    def latest_status(self) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            'SELECT status_json FROM statuses ORDER BY updated_at DESC LIMIT 1'
        ).fetchone()
        return json.loads(row['status_json']) if row else None

    def get_session_row(self, session_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            'SELECT * FROM sessions WHERE session_id = ?',
            (session_id,),
        ).fetchone()

    def session_payload(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self.get_session_row(session_id)
        if not row:
            return None
        return row_to_session(row, self.get_status(session_id))

    def previous_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        current = self.get_session_row(session_id)
        if not current:
            return None
        row = self.conn.execute(
            '''
            SELECT * FROM sessions
            WHERE timestamp < ?
              AND session_id != ?
              AND results_json IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
            ''',
            (current['timestamp'], session_id),
        ).fetchone()
        if not row:
            return None
        return row_to_session(row, self.get_status(row['session_id']))

    def history(self, limit: int = 25) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            'SELECT * FROM sessions ORDER BY timestamp DESC LIMIT ?',
            (max(1, min(limit, 200)),),
        ).fetchall()
        return [row_to_session(row, self.get_status(row['session_id'])) for row in rows]

    def purge(self, session_id: str) -> bool:
        row = self.get_session_row(session_id)
        if not row:
            return False
        safe_unlink(row['diagnostic_image_path'])
        safe_unlink(row['original_image_path'])
        safe_unlink(str(REPORT_DIR / f'{session_id}_report.pdf'))
        with self.lock:
            self.conn.execute('DELETE FROM statuses WHERE session_id = ?', (session_id,))
            self.conn.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
            self.conn.commit()
        return True

    def cleanup_expired(self) -> Dict[str, int]:
        now = utcnow()
        rows = self.conn.execute(
            '''
            SELECT session_id, timestamp, retention_hours, diagnostic_image_path, original_image_path
            FROM sessions
            '''
        ).fetchall()
        purged_sessions = 0
        live_paths: Set[str] = set()
        for row in rows:
            try:
                expires_at = parse_ts(row['timestamp']) + timedelta(
                    hours=int(row['retention_hours'] or DEFAULT_RETENTION_HOURS)
                )
            except Exception:
                expires_at = now - timedelta(seconds=1)
            if expires_at <= now:
                if self.purge(row['session_id']):
                    purged_sessions += 1
                continue
            for key in ('diagnostic_image_path', 'original_image_path'):
                if row[key]:
                    live_paths.add(str(Path(row[key]).resolve()))
            live_paths.add(str((REPORT_DIR / f"{row['session_id']}_report.pdf").resolve()))
        with self.lock:
            self.conn.execute('DELETE FROM statuses WHERE session_id NOT IN (SELECT session_id FROM sessions)')
            self.conn.commit()
        purged_files = self._cleanup_files(live_paths)
        return {'purged_sessions': purged_sessions, 'purged_files': purged_files}

    def _cleanup_files(self, live_paths: Set[str]) -> int:
        cutoff = utcnow() - timedelta(hours=DEFAULT_RETENTION_HOURS)
        removed = 0
        for root in (UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR):
            if not root.exists():
                continue
            for path in root.rglob('*'):
                if not path.is_file():
                    continue
                resolved = str(path.resolve())
                if resolved in live_paths:
                    continue
                try:
                    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                except OSError:
                    continue
                managed = path.name.endswith('_original.jpg') or path.name.endswith('_diagnostic.jpg')
                managed = managed or path.name.endswith('_report.pdf')
                if managed and modified <= cutoff:
                    safe_unlink(str(path))
                    removed += 1
        return removed


def session_stub(session_id: str, privacy_mode: bool, retention_hours: int) -> Dict[str, Any]:
    return {
        'session_id': session_id,
        'profile_id': None,
        'timestamp': utcnow_iso(),
        'severity': None,
        'gags_score': None,
        'lesion_count': None,
        'symmetry_delta': None,
        'results_json': None,
        'note': '',
        'diagnostic_image_path': None,
        'original_image_path': None,
        'privacy_mode': privacy_mode,
        'retention_hours': normalize_retention(retention_hours),
    }


def require_session(session_id: str) -> Dict[str, Any]:
    session = get_store().session_payload(validate_session_id(session_id))
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')
    return session


def summarize_stream_provenance(cloud_results: Dict[str, Any]) -> Dict[str, Any]:
    streams = {
        'model_a_640': len(cloud_results.get('preds_a_640', [])),
        'model_a_1280': len(cloud_results.get('preds_a_1280', [])),
        'model_b_native': len(cloud_results.get('preds_b', [])),
    }
    strongest_stream = max(streams, key=lambda key: streams[key]) if streams else None
    return {
        'streams': streams,
        'strongest_stream': strongest_stream,
        'stream_total': sum(streams.values()),
    }


def write_pdf_report(session: Dict[str, Any], compare: Optional[Dict[str, Any]], preset: str = 'clinical') -> Path:
    report_path = REPORT_DIR / f"{session['session_id']}_{preset}_report.pdf"
    results = session.get('results') or {}
    clinical = results.get('clinical_analysis', {})
    consensus = results.get('consensus_summary', {})
    regions = clinical.get('regions', {})
    pdf = canvas.Canvas(str(report_path), pagesize=letter)
    pdf.setTitle(f"Acne Diagnostic Report - {session['session_id']}")
    _, height = letter
    content_width = 508
    dominant_regions = sorted(
        regions.items(),
        key=lambda item: (item[1].get('gags_score', 0), item[1].get('count', 0)),
        reverse=True,
    )[:3]
    region_changes = top_region_changes(compare) if compare else []

    def line(
        y: float,
        text: str,
        font: str = 'Helvetica',
        size: int = 10,
        color: tuple[float, float, float] = (0.92, 0.95, 0.98),
    ) -> float:
        pdf.setFillColorRGB(*color)
        pdf.setFont(font, size)
        pdf.drawString(52, y, text)
        return y - (size + 5)

    def ensure_space(y: float, min_y: float = 96) -> float:
        if y >= min_y:
            return y
        pdf.showPage()
        return height - 48

    def wrap_text(text: str, font: str = 'Helvetica', size: int = 10) -> List[str]:
        words = str(text).split()
        if not words:
            return ['']

        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f'{current} {word}'
            if pdf.stringWidth(candidate, font, size) <= content_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def block(
        y: float,
        text: str,
        font: str = 'Helvetica',
        size: int = 10,
        color: tuple[float, float, float] = (0.82, 0.86, 0.9),
    ) -> float:
        for chunk in wrap_text(text, font, size):
            y = ensure_space(y)
            y = line(y, chunk, font, size, color)
        return y - 2

    def section(y: float, label: str, size: int = 12) -> float:
        y = ensure_space(y, 120)
        return line(y, label, 'Helvetica-Bold', size, (0.0, 0.84, 0.9))

    def bullet(y: float, text: str, size: int = 10) -> float:
        return block(y, f'- {text}', size=size)

    def metric(y: float, label: str, value: str) -> float:
        return line(y, f'{label}: {value}', 'Helvetica', 10, (0.92, 0.95, 0.98))

    title = {
        'clinical': 'Clinical Acne Analysis Report',
        'compact': 'Compact Acne Summary',
        'presentation': 'Presentation Review Deck',
    }.get(preset, 'Clinical Acne Analysis Report')
    subtitle = {
        'clinical': 'Full chart-ready diagnostic worksheet for detailed review.',
        'compact': 'One-page follow-up snapshot for quick clinician handoff.',
        'presentation': 'Narrative summary for review meetings and case presentations.',
    }.get(preset, 'Full chart-ready diagnostic worksheet for detailed review.')

    y = height - 48
    y = line(y, title, 'Helvetica-Bold', 18, (0.98, 0.99, 1.0))
    y = block(y, subtitle, 'Helvetica', 10, (0.0, 0.84, 0.9))
    y = metric(y, 'Session ID', session['session_id'])
    y = metric(y, 'Profile ID', session.get('profile_id') or 'default-profile')
    y = metric(y, 'Timestamp', session['timestamp'])

    if preset == 'compact':
        y = section(y, 'At a Glance', 13)
        y = metric(y, 'Severity', str(session.get('severity') or 'Unknown'))
        y = metric(y, 'GAGS Score', str(session.get('gags_score') or 0))
        y = metric(y, 'Lesion Count', str(session.get('lesion_count') or 0))
        y = metric(y, 'Symmetry Delta', f"{session.get('symmetry_delta') or 0}%")
        y = metric(y, 'Consensus', str(consensus.get('summary', 'N/A')))

        if dominant_regions:
            y = section(y, 'Top Burden Regions', 12)
            for region, values in dominant_regions[:2]:
                y = bullet(
                    y,
                    (
                        f"{region}: {values.get('count', 0)} lesions, "
                        f"GAGS {values.get('gags_score', 0)}"
                    ),
                )

        if compare:
            y = section(y, 'Longitudinal Snapshot', 12)
            y = block(y, comparison_summary(compare), 'Helvetica', 10, (0.92, 0.95, 0.98))
            y = bullet(
                y,
                (
                    f"Lesions {format_signed_delta(compare.get('lesion_delta', 0))} "
                    f"({get_delta_status(float(compare.get('lesion_delta', 0)))})"
                ),
            )
            y = bullet(
                y,
                (
                    f"GAGS {format_signed_delta(compare.get('gags_delta', 0))} "
                    f"({get_delta_status(float(compare.get('gags_delta', 0)))})"
                ),
            )

        if session.get('note'):
            y = section(y, 'Session Note', 12)
            y = block(y, str(session.get('note')), 'Helvetica', 10, (0.92, 0.95, 0.98))

        pdf.save()
        return report_path

    y = section(y, 'Clinical Snapshot', 13)
    y = metric(y, 'Severity', str(session.get('severity') or 'Unknown'))
    y = metric(y, 'GAGS Score', str(session.get('gags_score') or 0))
    y = metric(y, 'Lesion Count', str(session.get('lesion_count') or 0))
    y = metric(y, 'Symmetry Delta', f"{session.get('symmetry_delta') or 0}%")
    y = metric(y, 'Privacy Mode', 'Enabled' if session.get('privacy_mode') else 'Disabled')
    y = metric(y, 'Retention Hours', str(session.get('retention_hours')))
    y = block(y, f"Consensus Summary: {consensus.get('summary', 'N/A')}", 'Helvetica', 10, (0.92, 0.95, 0.98))

    if session.get('note'):
        y = section(y, 'Session Note', 12)
        y = block(y, str(session.get('note')), 'Helvetica', 10, (0.92, 0.95, 0.98))

    y = section(y, 'Regional Findings', 12)
    for region, values in regions.items():
        y = block(
            y,
            (
                f"- {region}: count={values.get('count', 0)}, lpi={values.get('lpi', 0)}, "
                f"area_px={values.get('area_px', 0)}, gags={values.get('gags_score', 0)}"
            ),
            'Helvetica',
            10,
            (0.82, 0.86, 0.9),
        )

    if compare:
        y = section(y, 'Temporal Comparison', 12)
        y = metric(y, 'Comparison Target', compare_target_label(compare))
        y = metric(y, 'Previous Session', str(compare.get('previous_session_id', 'N/A')))
        y = block(y, f"Clinical Summary: {comparison_summary(compare)}", 'Helvetica', 10, (0.92, 0.95, 0.98))
        severity = compare.get('severity_change', {})
        y = metric(y, 'Severity Change', f"{severity.get('from', 'Unknown')} -> {severity.get('to', 'Unknown')}")
        lesion_delta = float(compare.get('lesion_delta', 0))
        gags_delta = float(compare.get('gags_delta', 0))
        symmetry_delta = float(compare.get('symmetry_delta_change', 0))
        y = metric(y, 'Lesion Delta', f"{format_signed_delta(lesion_delta)} ({get_delta_status(lesion_delta)})")
        y = metric(y, 'GAGS Delta', f"{format_signed_delta(gags_delta)} ({get_delta_status(gags_delta)})")
        y = metric(y, 'Symmetry Delta Change', f"{format_signed_delta(symmetry_delta, '%')} ({get_delta_status(symmetry_delta)})")
        region_changes = top_region_changes(compare)
        if region_changes:
            y = section(y, 'Key Regional Changes', 11)
            for region, values in region_changes:
                y = block(
                    y,
                    (
                        f"- {region}: {get_delta_status(values.get('count_delta', 0))} "
                        f"({format_signed_delta(values.get('count_delta', 0))} lesions, "
                        f"LPI {format_signed_delta(values.get('lpi_delta', 0))})"
                    ),
                    'Helvetica',
                    10,
                    (0.82, 0.86, 0.9),
                )

    if preset == 'presentation':
        pdf.showPage()
        y = height - 48
        y = line(y, 'Presentation Review Deck', 'Helvetica-Bold', 18, (0.98, 0.99, 1.0))
        y = block(y, 'High-level narrative for case conferences, slide handoff, and executive review.', 'Helvetica', 10, (0.0, 0.84, 0.9))
        y = section(y, 'Clinical Headline', 14)
        y = block(
            y,
            (
                f"This case is currently graded {session.get('severity') or 'Unknown'} with a GAGS score of "
                f"{session.get('gags_score') or 0} across {session.get('lesion_count') or 0} verified lesions."
            ),
            'Helvetica',
            11,
            (0.92, 0.95, 0.98),
        )
        y = section(y, 'Why This Case Matters', 13)
        y = bullet(y, f"Consensus readout: {consensus.get('summary', 'N/A')}", 10)
        if dominant_regions:
            y = bullet(
                y,
                'Top burden regions: ' + ', '.join(
                    f"{region} (GAGS {values.get('gags_score', 0)})"
                    for region, values in dominant_regions
                ),
                10,
            )
        if compare:
            y = section(y, 'Longitudinal Story', 13)
            y = block(y, comparison_summary(compare), 'Helvetica', 11, (0.92, 0.95, 0.98))
            y = bullet(
                y,
                (
                    f"Severity shifted from {compare.get('severity_change', {}).get('from', 'Unknown')} "
                    f"to {compare.get('severity_change', {}).get('to', 'Unknown')}"
                ),
                10,
            )
            for region, values in region_changes[:2]:
                y = bullet(
                    y,
                    (
                        f"{region} trend: {get_delta_status(values.get('count_delta', 0))} with "
                        f"{format_signed_delta(values.get('count_delta', 0))} lesion change"
                    ),
                    10,
                )
        if session.get('note'):
            y = section(y, 'Presenter Note', 13)
            y = block(y, str(session.get('note')), 'Helvetica', 10, (0.92, 0.95, 0.98))

    pdf.save()
    return report_path


async def build_app_state() -> Dict[str, Any]:
    ensure_dirs()
    store = BridgeStore(DB_PATH)
    startup_cleanup = store.cleanup_expired()
    return {
        'store': store,
        'pipeline': None,
        'cloud_engine': None,
        'startup_cleanup': startup_cleanup,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.resources = await build_app_state()
    try:
        yield
    finally:
        app.state.resources['store'].close()


app = FastAPI(title='Acne V7 API Bridge', version=APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def get_store() -> BridgeStore:
    return app.state.resources['store']


@app.get('/')
async def root() -> Dict[str, Any]:
    return {
        'app': 'Acne V7 API Bridge',
        'status': 'ok',
        'version': APP_VERSION,
        'docs': '/docs',
        'health': '/health',
    }


def get_pipeline() -> Any:
    pipeline = app.state.resources['pipeline']
    if pipeline is None:
        with model_init_lock:
            pipeline = app.state.resources['pipeline']
            if pipeline is None:
                ensure_runtime_imports()
                runtime_pipeline = FaceSegmentationPipeline
                print('[Bridge] Initializing FaceSegmentationPipeline...')
                pipeline = runtime_pipeline(
                    bisenet_weights='weights/79999_iter.pth',
                    smooth_edges=True,
                )
                app.state.resources['pipeline'] = pipeline
                print('[Bridge] FaceSegmentationPipeline ready.')
    return pipeline


def get_cloud_engine() -> Any:
    cloud_engine = app.state.resources['cloud_engine']
    if cloud_engine is None:
        with model_init_lock:
            cloud_engine = app.state.resources['cloud_engine']
            if cloud_engine is None:
                ensure_runtime_imports()
                runtime_cloud_engine = CloudInferenceEngine
                print('[Bridge] Initializing CloudInferenceEngine...')
                cloud_engine = runtime_cloud_engine(api_key=API_KEY)
                app.state.resources['cloud_engine'] = cloud_engine
                print('[Bridge] CloudInferenceEngine ready.')
    return cloud_engine


@app.get('/health')
async def health() -> Dict[str, Any]:
    return {
        'status': 'ok',
        'version': APP_VERSION,
        'roboflow_api_key_configured': True,
        'pipeline_initialized': app.state.resources['pipeline'] is not None,
        'cloud_engine_initialized': app.state.resources['cloud_engine'] is not None,
        'cleanup': get_store().cleanup_expired(),
    }


@app.get('/version')
async def version() -> Dict[str, Any]:
    return {
        'app': 'Acne V7 API Bridge',
        'version': APP_VERSION,
        'model_a_id': MODEL_A_ID,
        'model_b_id': MODEL_B_ID,
        'max_upload_bytes': MAX_UPLOAD_BYTES,
    }


@app.get('/privacy')
async def privacy() -> Dict[str, Any]:
    return {
        'privacy_mode_supported': True,
        'default_retention_hours': DEFAULT_RETENTION_HOURS,
        'max_retention_hours': MAX_RETENTION_HOURS,
        'purge_endpoint': '/privacy/purge/{session_id}',
        'stored_fields': [
            'session_id',
            'timestamp',
            'severity',
            'gags_score',
            'lesion_count',
            'symmetry_delta',
            'results_json',
            'diagnostic_image_path',
            'original_image_path',
            'privacy_mode',
            'retention_hours',
        ],
    }


@app.post('/privacy/purge/{session_id}')
async def purge_session(session_id: str) -> Dict[str, Any]:
    session_id = validate_session_id(session_id)
    if not get_store().purge(session_id):
        raise HTTPException(status_code=404, detail='Session not found')
    return {'purged': True, 'session_id': session_id}


@app.post('/analysis/start')
async def analysis_start(request: AnalysisStartRequest) -> Dict[str, Any]:
    store = get_store()
    store.cleanup_expired()
    session_id = validate_session_id(request.session_id or uuid.uuid4().hex)
    if store.get_session_row(session_id):
        raise HTTPException(status_code=409, detail='Session already exists')
    stub = session_stub(session_id, request.privacy_mode, request.retention_hours)
    stub['profile_id'] = request.profile_id or 'default-profile'
    store.upsert_session(stub)
    status = store.set_status(session_id, 'queued', 'Analysis session created', 0)
    return {
        'session_id': session_id,
        'profile_id': stub['profile_id'],
        'privacy_mode': request.privacy_mode,
        'retention_hours': normalize_retention(request.retention_hours),
        'status': status,
    }


@app.get('/status/stream/{session_id}')
async def session_status_stream(session_id: str) -> StreamingResponse:
    session_id = validate_session_id(session_id)

    async def event_generator():
        previous = None
        while True:
            current = get_store().get_status(session_id)
            if current and current != previous:
                previous = current
                yield f"data: {json.dumps(current)}\n\n"
                if current.get('completed') or current.get('failed'):
                    break
            await asyncio.sleep(0.6)

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.get('/status/latest')
async def status_latest() -> Dict[str, Any]:
    status = get_store().latest_status()
    return {'status': status or {'stage': 'idle', 'detail': 'No analyses yet', 'progress': 0}}


@app.get('/status/{session_id}')
async def session_status(session_id: str) -> Dict[str, Any]:
    status = get_store().get_status(validate_session_id(session_id))
    if not status:
        raise HTTPException(status_code=404, detail='Status not found')
    return status


@app.get('/history')
async def history(limit: int = 25, profile_id: Optional[str] = None) -> Dict[str, Any]:
    items = [compact_session(item) for item in get_store().history(limit)]
    if profile_id:
        items = [item for item in items if item.get('profile_id') == profile_id]
    return {'items': items}


@app.get('/profiles')
async def profiles() -> Dict[str, Any]:
    items = get_store().history(500)
    grouped: Dict[str, Dict[str, Any]] = {}
    for item in items:
        profile_id = item.get('profile_id') or 'default-profile'
        current = grouped.setdefault(
            profile_id,
            {
                'profile_id': profile_id,
                'sessions': 0,
                'latest_timestamp': item['timestamp'],
                'latest_severity': item.get('severity'),
            },
        )
        current['sessions'] += 1
        if item['timestamp'] > current['latest_timestamp']:
            current['latest_timestamp'] = item['timestamp']
            current['latest_severity'] = item.get('severity')
    return {'items': list(grouped.values())}


@app.get('/session/{session_id}')
async def session_detail(session_id: str) -> Dict[str, Any]:
    return require_session(session_id)


@app.post('/session/{session_id}/notes')
async def update_session_notes(session_id: str, request: NotesRequest) -> Dict[str, Any]:
    session_id = validate_session_id(session_id)
    session = require_session(session_id)
    store = get_store()
    payload = {
        'session_id': session['session_id'],
        'profile_id': session.get('profile_id'),
        'timestamp': session['timestamp'],
        'severity': session.get('severity'),
        'gags_score': session.get('gags_score'),
        'lesion_count': session.get('lesion_count'),
        'symmetry_delta': session.get('symmetry_delta'),
        'results_json': json.dumps(session.get('results')) if session.get('results') else None,
        'note': request.note,
        'diagnostic_image_path': session.get('diagnostic_image_path'),
        'original_image_path': session.get('original_image_path'),
        'privacy_mode': session.get('privacy_mode'),
        'retention_hours': session.get('retention_hours'),
    }
    store.upsert_session(payload)
    return {'session_id': session_id, 'note': request.note}


@app.get('/compare/{current_session_id}')
async def compare(current_session_id: str, previous_session_id: Optional[str] = None) -> Dict[str, Any]:
    current = require_session(current_session_id)
    if previous_session_id and previous_session_id == current['session_id']:
        previous = None
    else:
        previous = require_session(previous_session_id) if previous_session_id else get_store().previous_session(current['session_id'])
    return {
        'current_session_id': current['session_id'],
        'compare': annotate_compare_payload(compare_payload(previous, current), bool(previous_session_id)),
    }


@app.get('/report/{session_id}')
async def report(session_id: str, previous_session_id: Optional[str] = None) -> Dict[str, Any]:
    session = require_session(session_id)
    if not session.get('results'):
        raise HTTPException(status_code=409, detail='Analysis not completed for this session')
    if previous_session_id and previous_session_id == session['session_id']:
        previous = None
    else:
        previous = require_session(previous_session_id) if previous_session_id else get_store().previous_session(session['session_id'])
    compare_data = annotate_compare_payload(compare_payload(previous, session), bool(previous_session_id))
    pdf_path = write_pdf_report(session, compare_data, 'clinical')
    return {
        'session_id': session['session_id'],
        'report': {
            'clinical_analysis': session['results'].get('clinical_analysis', {}),
            'consensus_summary': session['results'].get('consensus_summary', {}),
            'compare': compare_data,
            'pdf_path': str(pdf_path),
            'pdf_data_uri': file_to_data_uri(str(pdf_path), 'application/pdf'),
        },
    }


@app.post('/export/{session_id}')
async def export(session_id: str, request: ExportRequest) -> Dict[str, Any]:
    session = require_session(session_id)
    if not session.get('results'):
        raise HTTPException(status_code=409, detail='Analysis not completed for this session')
    if request.previous_session_id and request.previous_session_id == session['session_id']:
        previous = None
    else:
        previous = require_session(request.previous_session_id) if request.previous_session_id else get_store().previous_session(session['session_id'])
    compare_data = annotate_compare_payload(compare_payload(previous, session), bool(request.previous_session_id))
    pdf_path = write_pdf_report(session, compare_data, request.preset)
    payload: Dict[str, Any] = {
        'session_id': session['session_id'],
        'pdf_path': str(pdf_path),
        'preset': request.preset,
    }
    if request.include_pdf_data:
        payload['pdf_data_uri'] = file_to_data_uri(str(pdf_path), 'application/pdf')
    return payload


@app.post('/analyze')
async def analyze(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    profile_id: Optional[str] = Form(None),
    privacy_mode: bool = Form(False),
    retention_hours: int = Form(DEFAULT_RETENTION_HOURS),
) -> Dict[str, Any]:
    store = get_store()
    store.cleanup_expired()
    payload = await file.read()
    validate_upload(file, payload)
    image = decode_image(payload)

    session_id = validate_session_id(session_id or uuid.uuid4().hex)
    retention_hours = normalize_retention(retention_hours)
    existing = store.get_session_row(session_id)
    if existing:
        privacy_mode = bool(existing['privacy_mode'])
        retention_hours = int(existing['retention_hours'])
        profile_id = existing['profile_id'] if 'profile_id' in existing.keys() else profile_id
    else:
        stub = session_stub(session_id, privacy_mode, retention_hours)
        stub['profile_id'] = profile_id or 'default-profile'
        store.upsert_session(stub)
        profile_id = stub['profile_id']

    original_path = (UPLOAD_DIR / f'{session_id}_original.jpg').resolve()
    diagnostic_path = (OUTPUT_DIR / f'{session_id}_diagnostic.jpg').resolve()
    original_jpeg = image_to_jpeg_bytes(image)
    save_image(original_path, image)

    try:
        store.set_status(session_id, 'received', 'Upload accepted', 5)
        store.set_status(session_id, 'segmenting', 'Running face segmentation', 20)
        segmentation = get_pipeline().segment(image, return_intermediates=True)

        store.set_status(session_id, 'cloud_inference', 'Fetching ensemble detections', 45)
        cloud_results = get_cloud_engine().fetch_multi_scale_consensus(
            image,  # type: ignore[arg-type]
            MODEL_A_ID,
            MODEL_B_ID,
        )

        store.set_status(session_id, 'mapping', 'Assigning lesions to regions', 70)
        height, width = image.shape[:2]
        mapper = EnsembleLesionMapper(segmentation['masks'])
        assignments = mapper.ensemble_map_multi_scale(
            cloud_results['preds_a_640'],
            cloud_results['preds_a_1280'],
            cloud_results['preds_b'],
            (height, width),
            image=image,
        )
        clinical_report = mapper.get_clinical_report(assignments)

        store.set_status(session_id, 'rendering', 'Rendering diagnostic overlay', 82)
        ensure_runtime_imports()
        overlay = draw_lesion_boxes(
            image,
            lesions=assignments,
            clinical_report=clinical_report,
        )
        diagnostic_jpeg = image_to_jpeg_bytes(overlay)
        save_image(diagnostic_path, overlay)

        timestamp = utcnow_iso()
        results = dict(segmentation.get('metadata', {}))
        results['timestamp'] = timestamp
        results['session_id'] = session_id
        results['privacy_mode'] = privacy_mode
        results['retention_hours'] = retention_hours
        results['clinical_analysis'] = clinical_report
        results['lesions'] = assignments
        results['consensus_summary'] = consensus_summary(assignments)
        results['cloud_results'] = cloud_results
        results['source_stream_provenance'] = summarize_stream_provenance(cloud_results)

        original_image_path = absolute_managed_path(original_path)
        diagnostic_image_path = absolute_managed_path(diagnostic_path)
        if privacy_mode:
            safe_unlink(original_image_path)
            safe_unlink(diagnostic_image_path)
            original_image_path = None
            diagnostic_image_path = None

        session_record = {
            'session_id': session_id,
            'profile_id': profile_id or 'default-profile',
            'timestamp': timestamp,
            'severity': clinical_report.get('clinical_severity', 'Unknown'),
            'gags_score': int(clinical_report.get('gags_total_score', 0)),
            'lesion_count': int(clinical_report.get('total_lesions', 0)),
            'symmetry_delta': float(clinical_report.get('symmetry_delta', 0.0)),
            'results_json': json.dumps(results),
            'diagnostic_image_path': diagnostic_image_path,
            'original_image_path': original_image_path,
            'privacy_mode': privacy_mode,
            'retention_hours': retention_hours,
        }
        store.set_status(session_id, 'saving', 'Persisting analysis outputs', 92)
        store.upsert_session(session_record)
        session_data = store.session_payload(session_id)
        if not session_data:
            raise HTTPException(status_code=500, detail='Failed to reload saved session')
        previous = store.previous_session(session_id)
        compare_data = compare_payload(previous, session_data)
        final_status = store.set_status(
            session_id,
            'completed',
            'Analysis complete',
            100,
            {'completed': True},
        )

        return {
            'session_id': session_id,
            'status': final_status,
            'severity': session_data['severity'],
            'gags_score': session_data['gags_score'],
            'lesion_count': session_data['lesion_count'],
            'symmetry_delta': session_data['symmetry_delta'],
            'results': session_data['results'],
            'compare': compare_data,
            'original_image': bytes_to_data_uri(original_jpeg, 'image/jpeg'),
            'diagnostic_image': bytes_to_data_uri(diagnostic_jpeg, 'image/jpeg'),
        }
    except HTTPException:
        store.set_status(session_id, 'failed', 'Analysis failed', 100, {'failed': True})
        raise
    except Exception as exc:
        store.set_status(session_id, 'failed', str(exc), 100, {'failed': True})
        raise HTTPException(status_code=500, detail=f'Analysis failed: {exc}') from exc
