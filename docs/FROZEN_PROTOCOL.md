# RI027 minimal direct comparison: frozen protocol

Frozen before any new test-set inference on 2026-11-03 (Asia/Shanghai).

## Scope and invariants

- Fixed split files: `results/splits_ultrachip_seed42/{train,val,test}.txt`; counts 791/113/226. Their pre-existing SHA-256 values are verified before every action.
- Training seed: 42. The split is never regenerated from this seed.
- All methods use the same SMP FPN with an ImageNet-pretrained ResNet50 encoder, one binary output channel, 384 x 384 input, batch size 4, 100 epochs, AMP, AdamW (learning rate 1e-4, weight decay 1e-4), and CosineAnnealingLR with T_max=100.
- Historical preprocessing is retained: RGB conversion, bilinear image resize, nearest-neighbor mask resize, `mask > 0` binarization, ImageNet normalization. The audited historical trainer contains no random image augmentation, so none is introduced.
- Seed 42 is reset before each method. cuDNN deterministic mode is enabled and benchmark mode disabled. Every method starts from the same seeded ImageNet-pretrained model construction and uses the same seeded shuffle order. Initial state hashes are recorded.
- Validation and test always use complete original masks. The only permitted supervision differences are below.

## Methods

- E0: `BCEWithLogits(mean) + SMP Dice(from_logits=True) + FocalTversky(alpha=0.3,beta=0.7,gamma=0.75,eps=1e-7)` against original M.
- E1: E0 plus `0.1 * mean(sigmoid(logits) * normalized_signed_distance(M))`. The signed distance is negative inside and positive outside. It is normalized per nonempty image by its maximum absolute value. Empty masks receive an all-zero distance map. Lambda=0.1 is preregistered; lambda=1.0 will not be tried unless the initial recorded loss/gradient diagnostic shows severe scale collapse, and any such choice would use validation evidence only.
- E2: Dice and Focal Tversky are identical to E0 and use original M. BCE is unreduced and averaged only over pixels outside `B=(dilate_3x3(M) != erode_3x3(M))`: `sum((1-B)*BCE)/max(sum(1-B),1)`. This is a binary adaptation of boundary-label relaxation, not a reproduction of the full Zhu et al. method.
- E3: one 3 x 3 all-ones dilation is applied to training M only. BCE, Dice, and Focal Tversky all use the dilated training target. Validation/test use original M.

## Model selection and locked evaluation

- Each epoch is evaluated on all 113 validation images. Sigmoid probabilities are filtered exactly once by SciPy `gaussian_filter(sigma=1.0, mode='reflect', truncate=4.0)`.
- The checkpoint with maximum validation normalized AUPRO is selected. Ties retain the earlier epoch.
- AUPRO uses the same smoothed maps, 4-connected GT regions, 101 thresholds, global background FPR <= 0.3, trapezoidal integration, and division by 0.3.
- Recall, Precision, Dice, and IoU use the same smoothed maps at fixed threshold 0.5. Pixel-AP and Pixel-AUROC use the same continuous smoothed maps.
- After training, the selected checkpoint is re-evaluated on validation, hashed with SHA-256, and written to `checkpoint_frozen_before_test.json`. Only then is the 226-image test split evaluated once. Test results cannot alter a checkpoint, lambda, threshold, smoothing, seed, or display choice.

## Reporting limits

Stage A is one fixed training seed and supports a controlled descriptive comparison, not a significance claim. Stage B seeds 2026 and 3407 are outside the initial four-run commitment and will be reported only if actually completed. Historical A0--A3 values remain historical and are not overwritten.
