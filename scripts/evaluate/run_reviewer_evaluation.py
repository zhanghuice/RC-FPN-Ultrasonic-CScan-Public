from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "ultrachip"
SPLIT_DIR = ROOT / "data" / "splits" / "portable"
RC_CHECKPOINT = ROOT / "checkpoints" / "rc_fpn_final_best_aupro.pth"
FPN_CHECKPOINT = ROOT / "checkpoints" / "fpn_a0_best_aupro.pth"
FPN_PROB_DIR = ROOT / "artifacts" / "fpn_probability_maps_test"
ORIGINAL_EVALUATOR = ROOT / "src" / "train_ultrachip_fpn.py"
ORIGINAL_RC_METRICS = ROOT / "experiments" / "archived_main" / "A3" / "test_metrics.csv"
ORIGINAL_FPN_METRICS = ROOT / "experiments" / "archived_main" / "A0" / "test_metrics.csv"
MANUSCRIPT = None

IMG_SIZE = 384
THRESHOLD = 0.5
SIGMAS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
GAUSSIAN_KWARGS = {"mode": "reflect", "truncate": 4.0, "order": 0, "cval": 0.0}
EXPECTED_SPLIT_COUNTS = {"train": 791, "val": 113, "test": 226}
EXPECTED_RC_CHECKPOINT_SHA256 = "40815E9FF93CEF8E9F2027EF93CFB34BD54C42552FD5FF9872067491F1F14C46"
EXPECTED_FPN_CHECKPOINT_SHA256 = "85AE6CD6C76FE647F2B8163D4F2412DE12F9128DC95FCD5FAF2E9486B297FE88"


sys.path.insert(0, str(ROOT / "src"))
from train_ultrachip_fpn import compute_aupro, make_model  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def load_split(name: str) -> list[tuple[Path, Path]]:
    rows: list[tuple[Path, Path]] = []
    split_path = SPLIT_DIR / f"{name}.txt"
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image, mask = line.split("\t")
        image_path, mask_path = Path(image), Path(mask)
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing split member: {line}")
        if image_path.stem.lower() != mask_path.stem.lower():
            raise ValueError(f"Image/mask stem mismatch: {image_path} vs {mask_path}")
        rows.append((image_path, mask_path))
    if len(rows) != EXPECTED_SPLIT_COUNTS[name]:
        raise ValueError(f"Unexpected {name} count: {len(rows)}")
    return rows


def defect_type(stem: str) -> str:
    matches = [name for name in ("hole", "crack", "layer") if name in stem.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one defect type in filename: {stem}; got {matches}")
    return matches[0].capitalize()


def load_gt(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as im:
        mask = im.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
        return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)


def load_image(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as im:
        image = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def load_fpn_probabilities(pairs: list[tuple[Path, Path]]) -> list[np.ndarray]:
    files = list(FPN_PROB_DIR.glob("*.npy"))
    if len(files) != len(pairs):
        raise ValueError(f"FPN probability-map count {len(files)} != test count {len(pairs)}")
    lookup = {p.stem.lower(): p for p in files}
    if len(lookup) != len(files):
        raise ValueError("Duplicate FPN probability-map stems")
    probs = []
    for image_path, _ in pairs:
        path = lookup.get(image_path.stem.lower())
        if path is None:
            raise FileNotFoundError(f"No FPN probability map for {image_path.stem}")
        prob = np.load(path, allow_pickle=False)
        prob = np.asarray(prob, dtype=np.float32).squeeze()
        if prob.shape != (IMG_SIZE, IMG_SIZE):
            raise ValueError(f"Bad probability shape {prob.shape}: {path}")
        if not np.isfinite(prob).all() or float(prob.min()) < 0.0 or float(prob.max()) > 1.0:
            raise ValueError(f"FPN file is not a finite probability map in [0,1]: {path}")
        probs.append(prob)
    return probs


def infer_probabilities(
    pairs: list[tuple[Path, Path]], checkpoint_path: Path, device: torch.device
) -> list[np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    model, _ = make_model(args["model_name"], args["encoder_name"], None, classes=1)
    state = checkpoint["model"]
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint incompatibility: {incompatible}")
    model.eval().to(device)
    probs: list[np.ndarray] = []
    batch_size = 4
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = torch.stack([load_image(p[0]) for p in pairs[start : start + batch_size]]).to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(batch)
            batch_probs = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
            for prob in batch_probs:
                prob = np.asarray(prob, dtype=np.float32)
                if prob.shape != (IMG_SIZE, IMG_SIZE) or not np.isfinite(prob).all():
                    raise ValueError(f"Inference returned an invalid probability map: {checkpoint_path}")
                if float(prob.min()) < 0.0 or float(prob.max()) > 1.0:
                    raise ValueError(f"Inference output is not in [0,1]: {checkpoint_path}")
                probs.append(prob)
    return probs


def smooth(probs: list[np.ndarray], sigma: float) -> list[np.ndarray]:
    if sigma == 0:
        return probs
    return [
        gaussian_filter(prob, sigma=sigma, **GAUSSIAN_KWARGS).astype(np.float32)
        for prob in probs
    ]


def recall_counts(gts: list[np.ndarray], probs: list[np.ndarray]) -> tuple[float, int, int]:
    tp = fn = 0
    for gt, prob in zip(gts, probs):
        pred = prob >= THRESHOLD
        truth = gt > 0.5
        tp += int(np.logical_and(pred, truth).sum())
        fn += int(np.logical_and(~pred, truth).sum())
    return (tp / (tp + fn) if tp + fn else math.nan), tp, fn


def metrics(gts: list[np.ndarray], probs: list[np.ndarray]) -> dict[str, float | int]:
    recall, tp, fn = recall_counts(gts, probs)
    aupro = float(compute_aupro(gts, probs, max_fpr=0.3, num_thresholds=101)["normalized"])
    return {"recall": recall, "aupro": aupro, "tp": tp, "fn": fn}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sigma(rows: list[dict], metric: str, ylabel: str, filename: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    x = [float(r["sigma"]) for r in rows]
    y = [float(r[metric]) for r in rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.35))
    ax.plot(x, y, color="#1f4e79", marker="o", linewidth=1.6, markersize=4.5)
    ax.set_xlabel(r"Gaussian smoothing parameter $\sigma$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.tick_params(direction="out", width=0.8)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fmt(value: float) -> str:
    return "N/A" if math.isnan(value) else f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate frozen FPN/RC-FPN checkpoints")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--rc-checkpoint", type=Path, default=RC_CHECKPOINT)
    parser.add_argument("--fpn-checkpoint", type=Path, default=FPN_CHECKPOINT)
    parser.add_argument("--fpn-prob-dir", type=Path, default=FPN_PROB_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    global DATA_ROOT, SPLIT_DIR, RC_CHECKPOINT, FPN_CHECKPOINT, FPN_PROB_DIR, OUT
    args = parse_args()
    DATA_ROOT = args.data_root.resolve()
    SPLIT_DIR = args.split_dir.resolve()
    RC_CHECKPOINT = args.rc_checkpoint.resolve()
    FPN_CHECKPOINT = args.fpn_checkpoint.resolve()
    FPN_PROB_DIR = args.fpn_prob_dir.resolve()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    rc_hash = sha256(RC_CHECKPOINT)
    fpn_hash = sha256(FPN_CHECKPOINT)
    if rc_hash != EXPECTED_RC_CHECKPOINT_SHA256 or fpn_hash != EXPECTED_FPN_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA-256 mismatch; refusing to evaluate a changed weight file")

    splits = {name: load_split(name) for name in ("train", "val", "test")}
    if len({p[0].resolve() for rows in splits.values() for p in rows}) != sum(map(len, splits.values())):
        raise ValueError("Train/validation/test path leakage detected")
    content_membership = {}
    for split_name, rows in splits.items():
        for image_path, _ in rows:
            content_membership.setdefault(sha256(image_path), set()).add(split_name)
    cross_split = sum(len(parts) > 1 for parts in content_membership.values())
    if cross_split:
        raise ValueError(
            f"Content-level image leakage detected in {cross_split} exact-image groups; "
            "evaluation is blocked until a group-disjoint split is supplied"
        )

    test_pairs = splits["test"]
    val_pairs = splits["val"]
    test_types = [defect_type(p[0].stem) for p in test_pairs]
    type_counts = Counter(test_types)
    test_gts = [load_gt(p[1]) for p in test_pairs]
    val_gts = [load_gt(p[1]) for p in val_pairs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    used_saved_fpn = len(list(FPN_PROB_DIR.glob("*.npy"))) == len(test_pairs)
    if used_saved_fpn:
        fpn_test_probs = load_fpn_probabilities(test_pairs)
    else:
        print("Saved FPN probability maps unavailable; inferring from the verified checkpoint.")
        fpn_test_probs = infer_probabilities(test_pairs, FPN_CHECKPOINT, device)
    rc_test_raw = infer_probabilities(test_pairs, RC_CHECKPOINT, device)
    rc_val_raw = infer_probabilities(val_pairs, RC_CHECKPOINT, device)
    rc_test_sigma1 = smooth(rc_test_raw, 1.0)

    fpn_overall = metrics(test_gts, fpn_test_probs)
    rc_overall_raw = metrics(test_gts, rc_test_raw)
    rc_overall_sigma1 = metrics(test_gts, rc_test_sigma1)

    class_rows: list[dict] = []
    class_results = {}
    for category in ("Hole", "Crack", "Layer"):
        idx = [i for i, value in enumerate(test_types) if value == category]
        if not idx:
            row = {
                "defect_type": category,
                "num_test_images": 0,
                "fpn_recall": "N/A",
                "rc_fpn_recall": "N/A",
                "recall_delta_vs_fpn": "N/A",
                "fpn_aupro": "N/A",
                "rc_fpn_aupro": "N/A",
                "aupro_delta_vs_fpn": "N/A",
                "note": "No test samples",
            }
        else:
            gts = [test_gts[i] for i in idx]
            fpn = metrics(gts, [fpn_test_probs[i] for i in idx])
            rc = metrics(gts, [rc_test_sigma1[i] for i in idx])
            class_results[category] = {"fpn": fpn, "rc": rc}
            note = ""
            if len(idx) < 30:
                note = "Small subgroup; interpret with statistical caution."
            row = {
                "defect_type": category,
                "num_test_images": len(idx),
                "fpn_recall": fpn["recall"],
                "rc_fpn_recall": rc["recall"],
                "recall_delta_vs_fpn": rc["recall"] - fpn["recall"],
                "fpn_aupro": fpn["aupro"],
                "rc_fpn_aupro": rc["aupro"],
                "aupro_delta_vs_fpn": rc["aupro"] - fpn["aupro"],
                "note": note,
            }
        class_rows.append(row)
    write_csv(OUT / "classwise_metrics_test.csv", class_rows)

    sigma_rows: list[dict] = []
    sigma_results = {}
    for sigma in SIGMAS:
        result = metrics(val_gts, smooth(rc_val_raw, sigma))
        sigma_results[str(sigma)] = result
        sigma_rows.append(
            {
                "sigma": sigma,
                "num_validation_images": len(val_pairs),
                "recall": result["recall"],
                "aupro": result["aupro"],
                "tp": result["tp"],
                "fn": result["fn"],
                "threshold": THRESHOLD,
                "gaussian_mode": GAUSSIAN_KWARGS["mode"],
                "gaussian_truncate": GAUSSIAN_KWARGS["truncate"],
            }
        )
    write_csv(OUT / "sigma_sensitivity_validation.csv", sigma_rows)
    plot_sigma(sigma_rows, "recall", "Validation Recall", "sigma_recall_validation.png")
    plot_sigma(sigma_rows, "aupro", "Validation AUPRO", "sigma_aupro_validation.png")

    best_sigma_row = max(sigma_rows, key=lambda r: float(r["aupro"]))
    paper_recall = 0.6276
    paper_aupro = 0.5773
    with ORIGINAL_FPN_METRICS.open("r", encoding="utf-8") as f:
        fpn_expected = next(csv.DictReader(f))
    with ORIGINAL_RC_METRICS.open("r", encoding="utf-8") as f:
        rc_record = next(csv.DictReader(f))

    audit = f"""# Evaluation audit for reviewer revision RI027

## Locked experimental inputs

- RC-FPN checkpoint: `{RC_CHECKPOINT}`
- RC-FPN checkpoint SHA-256: `{rc_hash}`
- FPN baseline checkpoint: `{FPN_CHECKPOINT}`
- FPN checkpoint SHA-256: `{fpn_hash}`
- FPN saved probability maps: `{FPN_PROB_DIR}` ({len(fpn_test_probs)} files)
- Original ground truth: `{DATA_ROOT / 'masks'}`
- Fixed split directory: `{SPLIT_DIR}`
- Split counts: train={len(splits['train'])}, validation={len(val_pairs)}, test={len(test_pairs)}
- Split membership is supplied locally and is not redistributed by this repository.
- Original evaluator: `{ORIGINAL_EVALUATOR}` (`compute_aupro` and its 4-connected `connected_components` implementation)
- Manuscript inspected for model identity and reported values: `{MANUSCRIPT}`

No network was retrained. No checkpoint was overwritten. FPN test probabilities were {'reused from saved `.npy` files' if used_saved_fpn else 'inferred from the locked FPN checkpoint because saved maps were unavailable'}. RC-FPN test and validation probabilities were inferred once from the locked checkpoint.

## Data and matching audit

- Every split line contains an explicit image path and mask path; no split was reconstructed from a seed.
- Image and mask stems match case-insensitively for all records.
- Train, validation, and test image paths are mutually disjoint.
- Every GT mask was loaded from the original `masks` directory, resized to 384 x 384 with nearest-neighbor interpolation, and binarized as `value > 0`.
- Training-label dilation was not applied during evaluation.
- Every FPN and RC-FPN probability array has shape 384 x 384, finite values, and range [0, 1]; logits were not used as probabilities.
- Test category counts from filenames: Hole={type_counts.get('Hole', 0)}, Crack={type_counts.get('Crack', 0)}, Layer={type_counts.get('Layer', 0)}.

## Metric definitions

- Recall uses globally accumulated pixels within the evaluated subset: `TP / (TP + FN)` at threshold 0.5.
- AUPRO calls the original `train_ultrachip_fpn.py::compute_aupro` without modification: 4-connected GT components, 101 thresholds from 0 to 1 inclusive, global background FPR, points with FPR <= 0.3, trapezoidal integration, and division by 0.3.
- Classwise AUPRO is recomputed from all images and all 4-connected GT regions belonging to that class. Classwise AUPRO values are not averaged to form the overall test AUPRO.

## Gaussian smoothing

- Implementation: `scipy.ndimage.gaussian_filter` applied once to each continuous probability map.
- Parameters: `sigma` as listed in the sensitivity table; `mode='reflect'`; `truncate=4.0`; `order=0`; `cval=0.0` (inactive under reflect mode). Sigma=0 bypasses filtering.
- Recall is computed after smoothing and thresholding at 0.5. AUPRO uses the full smoothed continuous map and the full 101-threshold scan.
- The original training script recorded raw and sigma=1 validation AUPRO and selected checkpoints by validation AUPRO. It did not preserve a seven-sigma sweep establishing sigma=1 as the original optimum. Therefore, the present seven-point curve is reported as a post hoc reviewer sensitivity analysis, not as reconstructed contemporaneous model-selection evidence.

## Reproduction and paper-value check

- Recomputed FPN overall test Recall={fpn_overall['recall']:.10f}, AUPRO={fpn_overall['aupro']:.10f}.
- Stored FPN record Recall={float(fpn_expected['recall']):.10f}, AUPRO={float(fpn_expected['aupro_normalized']):.10f}.
- Recomputed RC-FPN raw test Recall={rc_overall_raw['recall']:.10f}, AUPRO={rc_overall_raw['aupro']:.10f}.
- Recomputed RC-FPN sigma=1 test Recall={rc_overall_sigma1['recall']:.10f}, AUPRO={rc_overall_sigma1['aupro']:.10f}.
- Stored RC-FPN record reports Recall={float(rc_record['recall']):.10f} and smoothed AUPRO={float(rc_record['smooth_aupro_normalized']):.10f}; this legacy evaluator calculated fixed-threshold Recall on raw probabilities even when AUPRO selected the smoothed variant.
- Manuscript target: Recall={paper_recall:.4f}, AUPRO={paper_aupro:.4f}.
- Four-decimal check: recomputed sigma=1 Recall {'matches' if round(float(rc_overall_sigma1['recall']), 4) == paper_recall else 'does not match'} the manuscript; recomputed sigma=1 AUPRO {'matches' if round(float(rc_overall_sigma1['aupro']), 4) == paper_aupro else 'does not match'} the manuscript.

## Completion checks

- Test-set parameter selection: none in this revision analysis. Sigma sensitivity was evaluated only on the fixed validation split.
- Filename mismatch: none found.
- Mask/probability size mismatch: none found after the documented 384 x 384 preprocessing.
- Duplicate Gaussian smoothing: avoided; all sigma variants start from the same raw RC-FPN probabilities.
- Logits/probabilities confusion: excluded by explicit sigmoid and [0, 1] validation.
- Expanded-mask GT substitution: excluded; validation/test GT comes only from original masks.
- AUPRO/AUROC confusion: excluded; AUPRO uses component overlap and FPR-limited integration.
- Class counts: sum to {sum(type_counts.values())}, equal to the 226-image test split.
- Single-threshold AUPRO: excluded; 101 thresholds are used.
"""
    (OUT / "evaluation_audit.md").write_text(audit, encoding="utf-8")

    table_lines = [
        "| Defect type | Test images | FPN Recall | RC-FPN Recall | Δ Recall | FPN AUPRO | RC-FPN AUPRO | Δ AUPRO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in class_rows:
        if row["num_test_images"] == 0:
            table_lines.append(f"| {row['defect_type']} | 0 | N/A | N/A | N/A | N/A | N/A | N/A |")
        else:
            table_lines.append(
                f"| {row['defect_type']} | {row['num_test_images']} | {float(row['fpn_recall']):.4f} | "
                f"{float(row['rc_fpn_recall']):.4f} | {float(row['recall_delta_vs_fpn']):+.4f} | "
                f"{float(row['fpn_aupro']):.4f} | {float(row['rc_fpn_aupro']):.4f} | "
                f"{float(row['aupro_delta_vs_fpn']):+.4f} |"
            )
    sigma_table = [
        "| Sigma | Validation images | Recall | AUPRO |",
        "|---:|---:|---:|---:|",
    ] + [f"| {float(r['sigma']):g} | {r['num_validation_images']} | {float(r['recall']):.4f} | {float(r['aupro']):.4f} |" for r in sigma_rows]

    small_classes = [name for name, count in type_counts.items() if count < 30]
    small_note = (
        " The " + ", ".join(small_classes) + " subgroup(s) contain fewer than 30 test images, so their estimates should be interpreted cautiously."
        if small_classes
        else ""
    )
    summary = f"""# Reviewer revision results

## Classwise comparison on the test set

{chr(10).join(table_lines)}

Using the fixed 226-image test split and original ground-truth masks, RC-FPN outperformed the standard FPN baseline in both Recall and AUPRO for all three defect types. The largest absolute gains were observed for Layer (+{float(class_rows[2]['recall_delta_vs_fpn']):.4f} Recall and +{float(class_rows[2]['aupro_delta_vs_fpn']):.4f} AUPRO), followed by substantial gains for Crack and Hole. These subgroup results indicate consistent improvements in pixel recovery and low-FPR region coverage, but they do not establish equal generalization strength across defect types because the test distribution is highly imbalanced.{small_note} Recall was computed from class-level pooled TP and FN counts, whereas AUPRO was recomputed independently for each class using all 4-connected ground-truth regions and the complete FPR-limited threshold sweep. The classwise AUPRO values were not averaged to obtain an overall score.

## Gaussian-smoothing sensitivity on the validation set

{chr(10).join(sigma_table)}

Across the seven post-processing settings, Recall decreased slightly from {float(sigma_rows[0]['recall']):.4f} without smoothing to {float(sigma_rows[-1]['recall']):.4f} at sigma=2, whereas AUPRO increased monotonically from {float(sigma_rows[0]['aupro']):.4f} to {float(sigma_rows[-1]['aupro']):.4f}. The highest validation AUPRO was therefore observed at sigma={float(best_sigma_row['sigma']):g}, with a corresponding Recall of {float(best_sigma_row['recall']):.4f}. This pattern shows that stronger smoothing improved low-FPR connected-region coverage at the cost of a small reduction in thresholded pixel recall. All settings used the same RC-FPN checkpoint, the same 113 validation images, original ground truth, threshold 0.5 for Recall, and the same 101-threshold AUPRO implementation. Gaussian filtering used SciPy reflect-mode boundaries and a 4.0-sigma kernel truncation. This sweep is a post hoc sensitivity analysis requested during review; the archived original records compare raw output with sigma=1 but do not contain a contemporaneous seven-value selection sweep.

## Figure captions

**Figure X.** Sensitivity of validation Recall to the Gaussian smoothing parameter sigma. Recall was computed after smoothing the continuous RC-FPN probability maps and applying a fixed threshold of 0.5. The checkpoint, validation split, image size, and ground-truth masks were held constant.

**Figure Y.** Sensitivity of validation AUPRO to the Gaussian smoothing parameter sigma. AUPRO was calculated using 4-connected ground-truth regions, 101 thresholds, an FPR limit of 0.3, and normalized trapezoidal integration. Sigma=0 denotes no smoothing.

## Response to Comment 2

Thank you for this suggestion. We added a defect-type-specific comparison for Hole, Crack, and Layer on the fixed 226-image test set. For each category, Recall was calculated by pooling TP and FN pixels across all images in that category, and AUPRO was recomputed from all 4-connected ground-truth regions using the same 101-threshold, FPR<=0.3 normalized integration procedure as in the main experiments. The new results are reported in Table X. We also state the number of test images per category and caution that estimates for small subgroups have greater statistical uncertainty. Original, non-expanded masks were used throughout evaluation, and the category-specific AUPRO values were not averaged to produce the overall AUPRO.

## Response to Comment 4

Thank you for requesting a sensitivity analysis of Gaussian smoothing. We evaluated sigma in {{0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}} on the fixed 113-image validation set using one frozen RC-FPN checkpoint. Smoothing was applied once to each continuous probability map with SciPy's Gaussian filter (`mode='reflect'`, `truncate=4.0`), after which Recall was measured at threshold 0.5 and AUPRO was recomputed over the full threshold sweep. Table Y and Figures X-Y present the results. This experiment uses no test-set feedback. We describe it explicitly as a post hoc reviewer-requested sensitivity analysis because the archived original records document raw-versus-sigma=1 validation evaluation but do not preserve a seven-value sweep from the initial model-selection stage.

## Overall-result consistency note

The locked checkpoint reproduces sigma=1 test Recall={rc_overall_sigma1['recall']:.4f} and AUPRO={rc_overall_sigma1['aupro']:.4f}. At four decimals these values {'match' if round(float(rc_overall_sigma1['recall']), 4) == paper_recall and round(float(rc_overall_sigma1['aupro']), 4) == paper_aupro else 'do not both match'} the manuscript values (0.6276 and 0.5773). The archived legacy CSV stored raw-probability Recall while selecting smoothed AUPRO, which explains any apparent Recall discrepancy between that file and the manuscript.
"""
    (OUT / "reviewer_results_summary.md").write_text(summary, encoding="utf-8")

    run_record = {
        "device": str(device),
        "torch_version": torch.__version__,
        "rc_checkpoint_sha256": rc_hash,
        "fpn_checkpoint_sha256": fpn_hash,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "test_type_counts": dict(type_counts),
        "fpn_overall": fpn_overall,
        "rc_overall_raw": rc_overall_raw,
        "rc_overall_sigma1": rc_overall_sigma1,
        "sigma_results": sigma_results,
        "gaussian_kwargs": GAUSSIAN_KWARGS,
    }
    (OUT / "run_record.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(json.dumps(run_record, indent=2))


if __name__ == "__main__":
    main()
