# Results and interpretation

The corrected archived main comparison is FPN Recall 0.4482/AUPRO 0.3476 versus RC-FPN Recall 0.6277/AUPRO 0.5774. The latter comes from one sigma=1 pass over the frozen final RC-FPN checkpoint. Historical stored values and evaluator differences are preserved in `EVALUATION_AUDIT.md`.

The matched three-seed study reports E0 0.4890±0.0542/0.4207±0.0323, E1 0.5276±0.0364/0.4278±0.0520, E2 0.4954±0.0766/0.3937±0.0296, and E3 0.5591±0.0438/0.5594±0.0441 for Recall/AUPRO. These are descriptive statistics over seeds 42, 2026, and 3407 with sample SD (`ddof=1`), not significance claims.

Classwise test counts are Hole 215, Crack 9, Layer 2. The small Crack and Layer groups permit only descriptive comparison. The validation-only sigma sweep found maximum AUPRO at sigma=2.0; sigma=1.0 remains the archived historical test configuration and is not claimed as the seven-value optimum.
