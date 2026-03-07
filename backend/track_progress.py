"""
Temporal Tracking Utility - Analyzes progress between two clinical sessions.
Calculates improvement in LPI (Lesion Density) and GAGS score.
"""
import json
import argparse
import pandas as pd
from pathlib import Path

def track_progress(session1_path, session2_path, output_path=None):
    with open(session1_path, 'r') as f:
        s1 = json.load(f)
    with open(session2_path, 'r') as f:
        s2 = json.load(f)

    # Extract clinical analysis
    c1 = s1.get("clinical_analysis", {})
    c2 = s2.get("clinical_analysis", {})
    
    # Global Metrics
    report = {
        "overall": {
            "session1_date": s1.get("timestamp", "N/A"),
            "session2_date": s2.get("timestamp", "N/A"),
            "s1_lesions": c1.get("total_lesions", 0),
            "s2_lesions": c2.get("total_lesions", 0),
            "s1_gags": c1.get("gags_total_score", 0),
            "s2_gags": c2.get("gags_total_score", 0),
            "s1_severity": c1.get("clinical_severity", "N/A"),
            "s2_severity": c2.get("clinical_severity", "N/A"),
        }
    }

    # Calculate Improvement %
    l_diff = report["overall"]["s1_lesions"] - report["overall"]["s2_lesions"]
    l_pct = (l_diff / report["overall"]["s1_lesions"] * 100) if report["overall"]["s1_lesions"] > 0 else 0
    report["overall"]["lesion_reduction_pct"] = round(l_pct, 2)
    
    g_diff = report["overall"]["s1_gags"] - report["overall"]["s2_gags"]
    report["overall"]["gags_improvement"] = g_diff

    # Regional Metrics
    report["regions"] = {}
    r1 = c1.get("regions", {})
    r2 = c2.get("regions", {})
    
    # Get skin health metrics (erythema) from metadata
    m1 = s1.get("regions", {})
    m2 = s2.get("regions", {})

    all_regions = set(r1.keys()).union(set(r2.keys()))
    
    for reg in all_regions:
        d1 = r1.get(reg, {})
        d2 = r2.get(reg, {})
        
        # LPI tracking
        lpi1 = d1.get("lpi", 0)
        lpi2 = d2.get("lpi", 0)
        lpi_improvement = round(((lpi1 - lpi2) / lpi1 * 100), 1) if lpi1 > 0 else 0
        
        # Erythema tracking
        ei1 = m1.get(reg, {}).get("erythema_index", 0)
        ei2 = m2.get(reg, {}).get("erythema_index", 0)
        ei_improvement = round(ei1 - ei2, 2)

        report["regions"][reg] = {
            "lpi_s1": lpi1,
            "lpi_s2": lpi2,
            "lpi_improvement_pct": lpi_improvement,
            "erythema_reduction": ei_improvement
        }

    # Print Summary
    print("\n=== TEMPORAL PROGRESS REPORT ===")
    print(f"Severity: {report['overall']['s1_severity']} -> {report['overall']['s2_severity']}")
    print(f"Total Lesion Change: {report['overall']['s1_lesions']} -> {report['overall']['s2_lesions']} ({l_pct:.1f}% reduction)")
    print(f"GAGS Score Change:   {report['overall']['s1_gags']} -> {report['overall']['s2_gags']}")
    
    print("\nRegional Density Improvement (LPI):")
    for reg, data in report["regions"].items():
        if data['lpi_s1'] > 0:
            print(f"  {reg:>13s}: {data['lpi_improvement_pct']:>5.1f}% improvement")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"\nReport saved to: {output_path}")
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", required=True, help="Session 1 results.json")
    parser.add_argument("--s2", required=True, help="Session 2 results.json")
    parser.add_argument("--output", help="Path to save comparison JSON")
    args = parser.parse_args()
    
    track_progress(args.s1, args.s2, args.output)
