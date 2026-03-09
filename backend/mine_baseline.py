"""Mine existing data from sessions.db and api_usage_log.json to build metrics baseline.

Run:  python mine_baseline.py
Output: metrics_baseline.json
"""
import sqlite3
import json
import sys
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / 'sessions.db'
LOG_PATH = BASE / 'api_usage_log.json'
OUT_PATH = BASE / 'metrics_baseline.json'


def mine_sessions():
    """Extract timing, detection, and scoring baselines from sessions.db."""
    if not DB_PATH.exists():
        return {'error': 'sessions.db not found', 'sessions': []}

    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        'SELECT session_id, timestamp, severity, gags_score, lesion_count, '
        'symmetry_delta, results_json FROM sessions'
    ).fetchall()
    conn.close()

    sessions = []
    all_timing = []
    all_lesion_counts = []
    all_gags = []
    cloud_stats = {'model_a_detections': [], 'model_b_detections': []}

    for sid, ts, severity, gags, lcount, sym_delta, rj in rows:
        entry = {
            'session_id': sid,
            'timestamp': ts,
            'severity': severity,
            'gags_score': gags,
            'lesion_count': lcount,
            'symmetry_delta': sym_delta,
        }

        if rj:
            data = json.loads(rj)
            schema_keys = list(data.keys())
            entry['schema_version'] = 'v2' if 'cloud_results' in schema_keys else 'v1'
            entry['schema_keys'] = schema_keys

            # Timing
            timing = data.get('timing_ms', {})
            entry['timing_ms'] = timing
            if timing:
                all_timing.append(timing)

            # Clinical analysis
            ca = data.get('clinical_analysis', {})
            entry['total_lesions'] = ca.get('total_lesions')
            entry['clinical_gags'] = ca.get('gags_score')
            entry['clinical_severity'] = ca.get('severity_grade')
            if ca.get('total_lesions') is not None:
                all_lesion_counts.append(ca['total_lesions'])
            if ca.get('gags_score') is not None:
                all_gags.append(ca['gags_score'])

            # Cloud results (v2 only) — values are lists of detection dicts
            cr = data.get('cloud_results', {})
            for model_key, dets in cr.items():
                if not isinstance(dets, list):
                    dets = dets.get('detections', []) if isinstance(dets, dict) else []
                entry.setdefault('cloud_detections', {})[model_key] = len(dets)
                if 'preds_a' in model_key or 'model_a' in model_key:
                    cloud_stats['model_a_detections'].append(len(dets))
                elif 'preds_b' in model_key or 'model_b' in model_key:
                    cloud_stats['model_b_detections'].append(len(dets))
                # Collect class distribution from typed model
                if 'preds_b' in model_key:
                    class_dist = {}
                    for d in dets:
                        cls = d.get('class', 'unknown')
                        class_dist[cls] = class_dist.get(cls, 0) + 1
                    entry['model_b_class_distribution'] = class_dist

            # Consensus summary
            cs = data.get('consensus_summary', {})
            if cs:
                entry['consensus_summary'] = cs

            # Source stream provenance
            sp = data.get('source_stream_provenance', {})
            if sp:
                entry['source_stream_provenance'] = sp

            # Region stats
            regions = data.get('regions', {})
            region_summary = {}
            for rname, rdata in regions.items():
                if isinstance(rdata, dict) and rdata.get('count', 0) > 0:
                    region_summary[rname] = {
                        'count': rdata.get('count', 0),
                        'gags_score': rdata.get('gags_score', 0),
                    }
            if region_summary:
                entry['active_regions'] = region_summary

        sessions.append(entry)

    # Compute timing averages
    timing_avg = {}
    if all_timing:
        timing_keys = set()
        for t in all_timing:
            timing_keys.update(t.keys())
        for k in sorted(timing_keys):
            vals = [t[k] for t in all_timing if k in t]
            timing_avg[k] = {
                'mean_ms': round(sum(vals) / len(vals), 2),
                'min_ms': round(min(vals), 2),
                'max_ms': round(max(vals), 2),
                'count': len(vals),
            }

    return {
        'session_count': len(sessions),
        'sessions': sessions,
        'timing_averages': timing_avg,
        'detection_stats': {
            'lesion_counts': all_lesion_counts,
            'mean_lesions': round(sum(all_lesion_counts) / len(all_lesion_counts), 1) if all_lesion_counts else None,
            'gags_scores': all_gags,
            'mean_gags': round(sum(all_gags) / len(all_gags), 1) if all_gags else None,
        },
        'cloud_stats': {
            k: {
                'mean': round(sum(v) / len(v), 1) if v else None,
                'values': v,
            }
            for k, v in cloud_stats.items()
        },
    }


def mine_usage_log():
    """Parse api_usage_log.json for historical call patterns."""
    if not LOG_PATH.exists():
        return {'error': 'api_usage_log.json not found'}

    with open(LOG_PATH, 'r') as f:
        data = json.load(f)

    history = data.get('history', [])
    total = data.get('total_calls', len(history))

    # Model breakdown
    model_counts = {}
    status_counts = {}
    timestamps = []
    for entry in history:
        model = entry.get('model', 'unknown')
        status = entry.get('status', 'unknown')
        model_counts[model] = model_counts.get(model, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        ts = entry.get('timestamp')
        if ts:
            timestamps.append(ts)

    # Time range
    timestamps.sort()
    time_range = None
    if timestamps:
        time_range = {
            'first': timestamps[0],
            'last': timestamps[-1],
        }

    return {
        'total_calls': total,
        'entries_in_history': len(history),
        'model_breakdown': model_counts,
        'status_breakdown': status_counts,
        'time_range': time_range,
    }


def main():
    baseline = {
        'generated_at': datetime.now().isoformat(),
        'description': 'Pre-Phase 5 metrics baseline mined from existing data',
        'sessions': mine_sessions(),
        'api_usage': mine_usage_log(),
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f'Baseline saved to {OUT_PATH}')
    print(f'Sessions: {baseline["sessions"]["session_count"]}')
    ta = baseline['sessions']['timing_averages']
    for k, v in ta.items():
        print(f'  {k}: mean={v["mean_ms"]:.0f}ms  range=[{v["min_ms"]:.0f}, {v["max_ms"]:.0f}]ms')
    ds = baseline['sessions']['detection_stats']
    print(f'Detection: mean_lesions={ds["mean_lesions"]}, mean_gags={ds["mean_gags"]}')
    print(f'API calls: {baseline["api_usage"]["total_calls"]}')
    print(f'Model breakdown: {baseline["api_usage"]["model_breakdown"]}')


if __name__ == '__main__':
    main()
