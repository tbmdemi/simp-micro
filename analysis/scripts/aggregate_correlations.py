"""
aggregate_correlations.py
──────────────────────────
Aggregate correlation data from all 30 per-config JSON files in Phase 1.
Outputs a compact JSON with: seed, objective, correlations (6 params),
best_obj_value, top-3 params.

Usage:
    python -m analysis.scripts.aggregate_correlations
"""

import json
import os
import glob
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────
PHASE1_DIR = "outputs/pipeline/phase1"
OUTPUT_FILE = "outputs/pipeline/phase1/_all_correlations.json"

PARAM_NAMES = ["volfrac", "penal", "rmin", "move", "void_size_frac", "rotation_deg"]


def main() -> None:
    """Find all per-config JSONs, aggregate correlations, write output."""
    records = []
    pattern = os.path.join(PHASE1_DIR, "*", "*", "phase1_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[WARN] No phase1_*.json files found under {PHASE1_DIR}")
        return

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        seed = data["metadata"]["seed"]
        obj = data["metadata"]["objective"]
        best_obj = min(r["obj_value"] for r in data["results"]
                       if r.get("success") and r.get("obj_value") is not None)
        corr = data["analysis"]["correlations"]
        pvals = data["analysis"]["p_values"]
        top3 = [(t[0], t[1], t[2]) for t in data["analysis"]["top_3"]]

        records.append({
            "seed": seed,
            "objective": obj,
            "best_obj_value": best_obj,
            "n_success": data["metadata"]["n_success"],
            "n_converged": sum(1 for r in data["results"] if r.get("converged")),
            "corr": corr,
            "pval": pvals,
            "top3": top3,
        })

    out = {"param_names": PARAM_NAMES, "configs": records}

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Aggregated {len(records)} configs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
