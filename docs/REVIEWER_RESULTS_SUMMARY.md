# Reviewer revision results

**Current revision convention.** The paper revision uses Recall=0.6277 and AUPRO=0.5774 for the final sigma=1 RC-FPN test result. Both metrics are recomputed from the same once-smoothed continuous probability maps, with Recall thresholded at 0.5 and AUPRO evaluated over 101 thresholds. The former manuscript values 0.6276 and 0.5773 are retained only as historical values in the audit. The complete insertion-ready text, corrected ablation interpretation, reviewer responses, and modification locations are provided in `paper_revision_package.md`.

## Classwise comparison on the test set

| Defect type | Test images | FPN Recall | RC-FPN Recall | Δ Recall | FPN AUPRO | RC-FPN AUPRO | Δ AUPRO |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hole | 215 | 0.4284 | 0.5845 | +0.1561 | 0.3420 | 0.5694 | +0.2274 |
| Crack | 9 | 0.6904 | 0.9143 | +0.2239 | 0.5248 | 0.7223 | +0.1974 |
| Layer | 2 | 0.2220 | 0.5980 | +0.3760 | 0.2046 | 0.6459 | +0.4413 |

Using the fixed 226-image test split and original ground-truth masks, RC-FPN outperformed the standard FPN baseline in both Recall and AUPRO for all three defect types. The largest absolute gains were observed for Layer (+0.3760 Recall and +0.4413 AUPRO), followed by substantial gains for Crack and Hole. These subgroup results indicate consistent improvements in pixel recovery and low-FPR region coverage, but they do not establish equal generalization strength across defect types because the test distribution is highly imbalanced. The Crack and Layer subgroups contain fewer than 30 test images, so their estimates should be interpreted cautiously. Recall was computed from class-level pooled TP and FN counts, whereas AUPRO was recomputed independently for each class using all 4-connected ground-truth regions and the complete FPR-limited threshold sweep. The classwise AUPRO values were not averaged to obtain an overall score.

## Gaussian-smoothing sensitivity on the validation set

| Sigma | Validation images | Recall | AUPRO |
|---:|---:|---:|---:|
| 0 | 113 | 0.6294 | 0.5729 |
| 0.5 | 113 | 0.6292 | 0.5809 |
| 0.75 | 113 | 0.6290 | 0.5888 |
| 1 | 113 | 0.6287 | 0.5981 |
| 1.25 | 113 | 0.6282 | 0.6094 |
| 1.5 | 113 | 0.6276 | 0.6167 |
| 2 | 113 | 0.6264 | 0.6321 |

Across the seven post-processing settings, Recall decreased slightly from 0.6294 without smoothing to 0.6264 at sigma=2, whereas AUPRO increased monotonically from 0.5729 to 0.6321. The highest validation AUPRO was therefore observed at sigma=2, with a corresponding Recall of 0.6264. This pattern shows that stronger smoothing improved low-FPR connected-region coverage at the cost of a small reduction in thresholded pixel recall. All settings used the same RC-FPN checkpoint, the same 113 validation images, original ground truth, threshold 0.5 for Recall, and the same 101-threshold AUPRO implementation. Gaussian filtering used SciPy reflect-mode boundaries and a 4.0-sigma kernel truncation. This sweep is a post hoc sensitivity analysis requested during review; the archived original records compare raw output with sigma=1 but do not contain a contemporaneous seven-value selection sweep.

## Figure captions

**Figure X.** Sensitivity of validation Recall to the Gaussian smoothing parameter sigma. Recall was computed after smoothing the continuous RC-FPN probability maps and applying a fixed threshold of 0.5. The checkpoint, validation split, image size, and ground-truth masks were held constant.

**Figure Y.** Sensitivity of validation AUPRO to the Gaussian smoothing parameter sigma. AUPRO was calculated using 4-connected ground-truth regions, 101 thresholds, an FPR limit of 0.3, and normalized trapezoidal integration. Sigma=0 denotes no smoothing.

## Response to Comment 2

Thank you for this suggestion. We added a defect-type-specific comparison for Hole, Crack, and Layer on the fixed 226-image test set. For each category, Recall was calculated by pooling TP and FN pixels across all images in that category, and AUPRO was recomputed from all 4-connected ground-truth regions using the same 101-threshold, FPR<=0.3 normalized integration procedure as in the main experiments. The new results are reported in Table X. We also state the number of test images per category and caution that estimates for small subgroups have greater statistical uncertainty. Original, non-expanded masks were used throughout evaluation, and the category-specific AUPRO values were not averaged to produce the overall AUPRO.

## Response to Comment 4

Thank you for requesting a sensitivity analysis of Gaussian smoothing. We evaluated sigma in {0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0} on the fixed 113-image validation set using one frozen RC-FPN checkpoint. Smoothing was applied once to each continuous probability map with SciPy's Gaussian filter (`mode='reflect'`, `truncate=4.0`), after which Recall was measured at threshold 0.5 and AUPRO was recomputed over the full threshold sweep. Table Y and Figures X-Y present the results. This experiment uses no test-set feedback. We describe it explicitly as a post hoc reviewer-requested sensitivity analysis because the archived original records document raw-versus-sigma=1 validation evaluation but do not preserve a seven-value sweep from the initial model-selection stage.

## Overall-result consistency note

The locked checkpoint reproduces sigma=1 test Recall=0.6277 and AUPRO=0.5774 under the unified evaluation convention. These are the values used in the current revision package. The archived legacy CSV stored raw-probability Recall while selecting smoothed AUPRO, and its smoothed AUPRO differs from the current re-evaluation by approximately 0.000095. The historical files remain unchanged and the difference is disclosed in the audit. The archived A2 and final A3 checkpoint hashes also differ; accordingly, the revision no longer interprets their difference as a controlled smoothing-only effect.
