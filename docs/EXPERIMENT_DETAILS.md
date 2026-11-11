# Experiment details

RC-FPN retains the standard SMP FPN-ResNet50 topology. Its defining intervention is one 3x3 all-ones dilation of training masks. Evaluation always uses original masks. The main loss is BCE + Dice + Focal Tversky. Continuous inference probabilities are Gaussian-smoothed once at sigma=1.0 for the archived final result.

The controlled study compares original labels (E0), signed-distance Boundary Loss (E1), binary boundary-label relaxation (E2), and training-label region expansion (E3). All use 384x384 inputs, batch 4, 100 epochs, AdamW at 1e-4, weight decay 1e-4, cosine scheduling, and validation AUPRO checkpoint selection.

AUPRO uses 4-connected ground-truth components, 101 thresholds, global background FPR up to 0.3, trapezoidal integration, and division by 0.3. Recall/Precision/Dice/IoU use threshold 0.5; Pixel-AP and Pixel-AUROC use continuous probabilities.
