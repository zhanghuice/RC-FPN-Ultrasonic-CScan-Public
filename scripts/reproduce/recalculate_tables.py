"""Verify paper values and three-seed mean/sample-SD from archived CSV files."""
from __future__ import annotations
import csv, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def close(a, b, tol=5e-13):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol):
        raise AssertionError((a, b))

def main():
    raw = rows(ROOT / "results/three_seed/three_seed_results.csv")
    saved = {r["method"]: r for r in rows(ROOT / "results/three_seed/three_seed_mean_std.csv")}
    metrics = ("recall", "precision", "dice", "iou", "pixel_ap", "pixel_auroc", "aupro")
    for method in ("E0", "E1", "E2", "E3"):
        group = [r for r in raw if r["method"] == method]
        assert [int(r["seed"]) for r in group] == [42, 2026, 3407]
        for metric in metrics:
            values = [float(r[metric]) for r in group]
            close(statistics.mean(values), saved[method][f"{metric}_mean"])
            close(statistics.stdev(values), saved[method][f"{metric}_sd"])
    main_a0 = next(csv.DictReader((ROOT / "experiments/archived_main/A0/test_metrics.csv").open(encoding="utf-8")))
    close(round(float(main_a0["recall"]), 4), 0.4482, 1e-4)
    close(round(float(main_a0["aupro_normalized"]), 4), 0.3476, 1e-4)
    print("PASS: 12 runs, ddof=1 summaries, and archived FPN paper values verified.")

if __name__ == "__main__":
    main()
