"""Phase 8 benchmark analysis: breakdown by seg_mode, comparison with Phase 7."""

import json


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def compute_metrics(images):
    """Compute aggregate metrics from a list of per-image results."""
    tp = sum(i['tp'] for i in images)
    fp = sum(i['fp'] for i in images)
    fn = sum(i['fn'] for i in images)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ious = [i['mean_iou'] for i in images if i['mean_iou'] > 0]
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    gt_total = sum(i['gt_count'] for i in images)
    pred_total = sum(i['pred_count'] for i in images)

    raw_total = sum(i.get('pipeline_metrics', {}).get('raw_detections', 0) for i in images)
    post_nms = sum(i.get('pipeline_metrics', {}).get('post_nms', 0) for i in images)
    post_gate = sum(i.get('pipeline_metrics', {}).get('post_gating', 0) for i in images)

    return {
        'images': len(images),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'mean_iou': round(mean_iou, 4),
        'gt_total': gt_total,
        'pred_total': pred_total,
        'raw_detections': raw_total,
        'post_nms': post_nms,
        'post_gating': post_gate,
        'nms_drop_rate': round(1 - post_nms / raw_total, 4) if raw_total > 0 else 0,
        'gate_drop_rate': round(1 - post_gate / post_nms, 4) if post_nms > 0 else 0,
    }


def print_metrics(label, m):
    sep = '=' * 60
    print(f'\n{sep}')
    print(f'  {label}')
    print(sep)
    print(f'  Images:       {m["images"]}')
    print(f'  GT labels:    {m["gt_total"]}')
    print(f'  Predictions:  {m["pred_total"]}')
    print(f'  TP: {m["tp"]}  FP: {m["fp"]}  FN: {m["fn"]}')
    print(f'  Precision:    {m["precision"]:.4f}')
    print(f'  Recall:       {m["recall"]:.4f}')
    print(f'  F1 Score:     {m["f1"]:.4f}')
    print(f'  Mean IoU:     {m["mean_iou"]:.4f}')
    print(f'  Pipeline:     raw={m["raw_detections"]} -> NMS={m["post_nms"]} -> gate={m["post_gating"]}')
    print(f'  NMS drop:     {m["nms_drop_rate"]:.1%}')
    print(f'  Gate drop:    {m["gate_drop_rate"]:.1%}')


def analyze_zero_tp(images, label):
    """Find images with zero TP but nonzero ground truth."""
    with_gt = [i for i in images if i['gt_count'] > 0]
    zero_tp = [i for i in with_gt if i['tp'] == 0]
    zero_tp.sort(key=lambda x: x['gt_count'], reverse=True)
    pct = 100 * len(zero_tp) / max(1, len(with_gt))
    print(f'\n  Zero-TP images ({label}): {len(zero_tp)} / {len(with_gt)} ({pct:.0f}%)')
    if zero_tp:
        print('  Top 5 by GT count:')
        for img in zero_tp[:5]:
            pm = img.get('pipeline_metrics', {})
            name = img['image'][:50]
            gt = img['gt_count']
            raw = pm.get('raw_detections', 0)
            nms = pm.get('post_nms', 0)
            gate = pm.get('post_gating', 0)
            print(f'    {name:50s}  gt={gt:3d}  raw={raw:2d} -> nms={nms:2d} -> gate={gate:2d}')


def main():
    p8 = load_json('validation_benchmark_phase8_116.json')
    p7 = load_json('validation_benchmark_phase7_50.json')

    p8_imgs = p8['per_image']
    p7_imgs = p7['per_image']

    full_imgs = [i for i in p8_imgs if i.get('seg_mode') == 'full']
    fb_imgs = [i for i in p8_imgs if i.get('seg_mode') == 'fallback']

    p8_all = compute_metrics(p8_imgs)
    p8_full = compute_metrics(full_imgs)
    p8_fb = compute_metrics(fb_imgs)
    p7_all = compute_metrics(p7_imgs)

    print_metrics('Phase 7: 50 random images (all fallback, IoU=0.35)', p7_all)
    print_metrics('Phase 8: 116 face-detected images (all modes, IoU=0.45)', p8_all)
    print_metrics('Phase 8: FULL segmentation only (90 images)', p8_full)
    print_metrics('Phase 8: FALLBACK only (26 images)', p8_fb)

    # --- Overlapping images ---
    p7_by_name = {i['image']: i for i in p7_imgs}
    p8_by_name = {i['image']: i for i in p8_imgs}
    overlap = set(p7_by_name.keys()) & set(p8_by_name.keys())
    print(f'\n{"=" * 60}')
    print(f'  Overlapping images (in both Phase 7 and Phase 8): {len(overlap)}')
    print('=' * 60)
    if overlap:
        p7_overlap = [p7_by_name[n] for n in overlap]
        p8_overlap = [p8_by_name[n] for n in overlap]
        m7 = compute_metrics(p7_overlap)
        m8 = compute_metrics(p8_overlap)
        print(f'  Phase 7 (overlap subset): F1={m7["f1"]:.4f} P={m7["precision"]:.4f} R={m7["recall"]:.4f}')
        print(f'  Phase 8 (overlap subset): F1={m8["f1"]:.4f} P={m8["precision"]:.4f} R={m8["recall"]:.4f}')
        print('\n  Per-image comparison:')
        for name in sorted(overlap):
            i7 = p7_by_name[name]
            i8 = p8_by_name[name]
            mode = i8.get('seg_mode', '?')
            print(f'    {name[:45]:45s}  mode={mode:8s}  '
                  f'P7: tp={i7["tp"]} fp={i7["fp"]} fn={i7["fn"]}  '
                  f'P8: tp={i8["tp"]} fp={i8["fp"]} fn={i8["fn"]}')

    # --- IoU threshold impact ---
    print(f'\n{"=" * 60}')
    print('  KEY DIFFERENCES between Phase 7 and Phase 8')
    print('=' * 60)
    print(f'  IoU threshold:  Phase 7 = 0.35,  Phase 8 = 0.45')
    print(f'  Image set:      Phase 7 = 50 random (all fallback)')
    print(f'                  Phase 8 = 116 Haar-detected (90 full + 26 fallback)')
    print(f'  Seg mode mix:   Phase 7 = 100% fallback')
    print(f'                  Phase 8 = {len(full_imgs)/len(p8_imgs)*100:.0f}% full, '
          f'{len(fb_imgs)/len(p8_imgs)*100:.0f}% fallback')

    # --- Zero-TP analysis ---
    analyze_zero_tp(full_imgs, 'full seg')
    analyze_zero_tp(fb_imgs, 'fallback')

    # --- Gate A impact on full seg ---
    print(f'\n{"=" * 60}')
    print('  Gate A (anatomical) impact analysis')
    print('=' * 60)
    full_gate_drop = p8_full['gate_drop_rate']
    fb_gate_drop = p8_fb['gate_drop_rate']
    print(f'  Full seg gate drop rate:     {full_gate_drop:.1%}')
    print(f'  Fallback gate drop rate:     {fb_gate_drop:.1%}')
    print(f'  Difference: Full seg drops {full_gate_drop - fb_gate_drop:+.1%} more')
    print('  -> Full segmentation creates tighter face masks, rejecting more detections')

    # --- False positives on 0-GT images ---
    zero_gt = [i for i in p8_imgs if i['gt_count'] == 0]
    if zero_gt:
        fp_on_empty = sum(i['fp'] for i in zero_gt)
        print(f'\n  False positives on 0-GT images: {fp_on_empty} FP across {len(zero_gt)} images')

    # --- Summary ---
    print(f'\n{"=" * 60}')
    print('  SUMMARY')
    print('=' * 60)
    print(f'  Phase 8 full-seg F1 ({p8_full["f1"]:.4f}) vs fallback F1 ({p8_fb["f1"]:.4f})')
    if p8_full['f1'] < p8_fb['f1']:
        print('  -> Full segmentation HURTS accuracy vs fallback on this dataset')
        print('  -> Gate A drops more detections when face regions are precisely mapped')
    else:
        print('  -> Full segmentation helps accuracy vs fallback')

    print(f'  Phase 7 F1 ({p7_all["f1"]:.4f}) vs Phase 8 F1 ({p8_all["f1"]:.4f})')
    print(f'  -> IoU threshold change (0.35 -> 0.45) makes matching stricter')
    print(f'  -> Different image set (random vs face-detected) changes difficulty')

    # Save results as JSON
    results = {
        'phase7': p7_all,
        'phase8_all': p8_all,
        'phase8_full': p8_full,
        'phase8_fallback': p8_fb,
        'overlap_count': len(overlap),
        'key_findings': [
            f'Full seg gate drop rate ({full_gate_drop:.1%}) vs fallback ({fb_gate_drop:.1%})',
            f'Phase 8 uses stricter IoU=0.45 vs Phase 7 IoU=0.35',
            f'{len(zero_gt)} images with 0 ground truth labels (FP-only testing)',
        ]
    }
    with open('phase8_analysis_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved analysis to phase8_analysis_report.json')


if __name__ == '__main__':
    main()
