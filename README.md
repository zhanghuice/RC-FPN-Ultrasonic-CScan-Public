# RC-FPN for Ultrasonic C-Scan Defect Segmentation

> **Publication candidate — not yet released.** A data-split audit found byte-identical images across the historical training, validation, and test partitions. Historical results are retained for transparency, but they must not be described as leakage-free independent-test estimates. Model-weight redistribution and source-code licensing are also awaiting written clearance.

This repository is a clean, history-free candidate assembled from audited project files. It contains source code, configurations, logs, aggregate metrics, documentation, and plots. It contains no UltraChip images, masks, filename manifest, split membership, data-derived hashes, or checkpoint binaries.

## Dataset availability

UltraChip is available for academic research upon request from the dataset authors.

- Official website: https://iiplab.net/ultrachip/
- Agreement: https://iiplab.net/ultrachip/UltraChip_agreement.pdf
- Contact: `luk@ucas.ac.cn`

Applicants should follow the official access procedure and sign the required agreement. The original images and pixel-level annotations are not redistributed here. The agreement restricts redistribution and reproduction of dataset images; users must independently comply with its terms.

The official site lists 1,126 images (1,079 Hole, 37 Crack, 10 Layer), whereas the historical local corpus contains 1,130 (1,079 Hole, 40 Crack, 11 Layer). The four-record difference remains unresolved and must not be “fixed” by guessing.

## Leakage status

The private content audit found 17 byte-identical image pairs. Ten cross the historical fixed partitions: five Train–Validation and five Train–Test; none are Validation–Test. One additional exact image-and-mask pair is duplicated inside Test. Four pairs use identical image bytes with different mask bytes.

The frozen-checkpoint sensitivity analysis shows measurable inflation in absolute metrics. Excluding the five known Train–Test duplicate records changed FPN Recall/AUPRO from 0.4481/0.3476 to 0.4434/0.3421 and RC-FPN sigma=1 from 0.6277/0.5774 to 0.6069/0.5691. This exclusion is diagnostic only: validation leakage and training exposure remain. A corrected validity claim requires group-disjoint splitting and retraining.

Before any local training or evaluation, run:

```bash
python scripts/audit/check_split_leakage.py --split-dir PATH_TO_PRIVATE_SPLITS
```

Training and evaluation entry points also refuse to proceed when byte-identical image content crosses splits.

## Repository contents

- `src/train_ultrachip_fpn.py`: FPN/RC-FPN training and evaluation implementation.
- `scripts/train/run_direct_comparison.py`: controlled E0–E3 comparison.
- `scripts/evaluate/run_reviewer_evaluation.py`: frozen-checkpoint evaluation.
- `scripts/reproduce/`: split-path preparation and aggregate-table recalculation.
- `experiments/` and `results/`: preserved historical logs and aggregate results.
- `docs/`: method, protocol, audit summary, licensing status, and limitations.

The separate comparison-model implementation was omitted pending a source-provenance review.

## Historical experiment labels

- E0: original masks with BCE + Dice + Focal Tversky.
- E1: E0 plus signed-distance Boundary Loss.
- E2: boundary-label relaxation for the BCE term.
- E3 / RC-FPN: 3×3 expansion of training masks only.

All controlled runs used FPN-ResNet50 initialized from ImageNet weights, 384×384 inputs, seeds 42/2026/3407, and validation AUPRO checkpoint selection. These details describe the archived experiments; they do not resolve the leakage finding.

## Pretrained model weights

No weights are distributed at present. All 16 candidates remain on hold while three independent issues are resolved: author/institution ownership approval, exact upstream pretrained-weight provenance/terms, and written clarification about distributing models trained on UltraChip. Dataset access, source-code licensing, and model-weight redistribution are separate questions.

## Reproduction status

The repository supports inspection and local audit, but does not currently support a defensible leak-free reproduction of the reported paper metrics. Users with authorized data may create a new group-disjoint split, run the leakage checker, and then train from scratch. Historical aggregate tables can be recalculated without distributing the dataset.

## Citation

The manuscript has not yet been formally published. Do not invent a DOI, proceedings title, or page range. Use `CITATION.cff` only as a provisional software/manuscript citation and cite the UltraChip paper listed by the official dataset site.

## License status and limitations

No repository-wide open-source license is granted yet. See `LICENSE_STATUS.md` and `THIRD_PARTY_NOTICES.md`. Subject to contributor/institution confirmation, identified author-owned code may later be released under MIT. That would not license UltraChip, third-party dependencies, pretrained weights, or trained checkpoints.

The historical corpus is small and class-imbalanced, the local/official version mismatch is unresolved, and the fixed historical split is content-leaked. Do not use these models for safety-critical decisions.
