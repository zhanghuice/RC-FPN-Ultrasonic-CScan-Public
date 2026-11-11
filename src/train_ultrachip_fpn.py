import argparse
import csv
import hashlib
import math
import random
import re
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import auc, average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import segmentation_models_pytorch as smp


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}
IMAGE_DIR_NAMES = ["images", "image", "imgs", "img"]
MASK_DIR_NAMES = ["masks", "mask", "ground_truth", "groundtruth", "gt", "labels", "label"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

try:
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import label as scipy_label
except Exception:
    gaussian_filter = None
    scipy_label = None


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def log(msg: str):
    print(msg, flush=True)


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def list_images(directory: Path):
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )


def find_named_child(parent: Path, names):
    if not parent.exists():
        return None
    children = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for name in names:
        if name.lower() in children:
            return children[name.lower()]
    return None


def pair_files(image_dir: Path, mask_dir: Path, split_name: str):
    image_files = list_images(image_dir)
    mask_files = list_images(mask_dir)
    masks_by_stem = {}
    duplicate_mask_stems = set()
    for mask in mask_files:
        key = mask.stem.lower()
        if key in masks_by_stem:
            duplicate_mask_stems.add(key)
        else:
            masks_by_stem[key] = mask

    pairs = []
    unmatched_images = []
    for image in image_files:
        mask = masks_by_stem.get(image.stem.lower())
        if mask is None:
            unmatched_images.append(image.name)
        else:
            pairs.append((image, mask))

    image_stems = {p.stem.lower() for p in image_files}
    unmatched_masks = [p.name for p in mask_files if p.stem.lower() not in image_stems]

    if duplicate_mask_stems:
        log(f"[WARN] {split_name}: duplicate mask stems: {sorted(duplicate_mask_stems)[:20]}")
    if unmatched_images or unmatched_masks or len(image_files) != len(mask_files):
        log(f"[WARN] {split_name}: image/mask count mismatch or unmatched files.")
        log(f"[WARN] {split_name}: num_images={len(image_files)}, num_masks={len(mask_files)}, num_pairs={len(pairs)}")
        if unmatched_images:
            log(f"[WARN] {split_name}: unmatched images (first 50): {unmatched_images[:50]}")
        if unmatched_masks:
            log(f"[WARN] {split_name}: unmatched masks (first 50): {unmatched_masks[:50]}")
    return pairs


def find_split_dirs(data_root: Path):
    split_pairs = {}

    for split, aliases in SPLIT_ALIASES.items():
        found = None
        for alias in aliases:
            split_dir = data_root / alias
            if split_dir.exists() and split_dir.is_dir():
                image_dir = find_named_child(split_dir, IMAGE_DIR_NAMES)
                mask_dir = find_named_child(split_dir, MASK_DIR_NAMES)
                if image_dir and mask_dir:
                    found = (image_dir, mask_dir)
                    break
        if found:
            split_pairs[split] = found

    if "train" in split_pairs and ("val" in split_pairs or "test" in split_pairs):
        return split_pairs

    image_root = find_named_child(data_root, IMAGE_DIR_NAMES)
    mask_root = find_named_child(data_root, MASK_DIR_NAMES)
    if image_root and mask_root:
        split_pairs = {}
        for split, aliases in SPLIT_ALIASES.items():
            for alias in aliases:
                image_dir = image_root / alias
                mask_dir = mask_root / alias
                if image_dir.exists() and mask_dir.exists():
                    split_pairs[split] = (image_dir, mask_dir)
                    break
        if "train" in split_pairs and ("val" in split_pairs or "test" in split_pairs):
            return split_pairs

    return {}


def discover_pairs(data_root: Path, seed: int = 42):
    split_dirs = find_split_dirs(data_root)
    if split_dirs:
        log("[INFO] Found existing split directories:")
        split_pairs = {split: pair_files(img_dir, mask_dir, split) for split, (img_dir, mask_dir) in split_dirs.items()}
        if "val" not in split_pairs:
            split_pairs["val"] = []
        if "test" not in split_pairs:
            split_pairs["test"] = []
        return split_pairs

    image_dir = find_named_child(data_root, IMAGE_DIR_NAMES)
    mask_dir = find_named_child(data_root, MASK_DIR_NAMES)
    if not image_dir or not mask_dir:
        raise FileNotFoundError(
            f"Could not find image/mask directories under {data_root}. "
            f"Expected names like {IMAGE_DIR_NAMES} and {MASK_DIR_NAMES}."
        )

    all_pairs = pair_files(image_dir, mask_dir, "all")
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    n_total = len(all_pairs)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    split_pairs = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train : n_train + n_val],
        "test": all_pairs[n_train + n_val :],
    }
    log("[INFO] No train/val/test split found. Created 7:1:2 split with seed=42.")
    return split_pairs


def save_split_files(split_pairs, split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        with (split_dir / f"{split}.txt").open("w", encoding="utf-8") as f:
            for image_path, mask_path in split_pairs.get(split, []):
                f.write(f"{Path(image_path).resolve()}\t{Path(mask_path).resolve()}\n")


def load_split_files(split_dir: Path):
    split_pairs = {}
    for split in ["train", "val", "test"]:
        split_file = split_dir / f"{split}.txt"
        if not split_file.exists():
            return None
        pairs = []
        with split_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    raise ValueError(f"Invalid split line in {split_file}: {line}")
                image_path, mask_path = Path(parts[0]), Path(parts[1])
                if not image_path.exists() or not mask_path.exists():
                    raise FileNotFoundError(f"Split file points to missing file: {line}")
                pairs.append((image_path, mask_path))
        split_pairs[split] = pairs
    return split_pairs


def assert_content_disjoint(split_pairs):
    membership = {}
    for split, pairs in split_pairs.items():
        for image_path, _ in pairs:
            value = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
            membership.setdefault(value, set()).add(split)
    cross_split = sum(len(parts) > 1 for parts in membership.values())
    if cross_split:
        raise RuntimeError(
            f"Content-level image leakage detected in {cross_split} exact-image groups; "
            "use a group-disjoint split before training or evaluation"
        )


def get_or_create_splits(data_root: Path, split_dir: Path, seed: int):
    split_pairs = load_split_files(split_dir)
    if split_pairs is not None:
        assert_content_disjoint(split_pairs)
        log(f"[INFO] Loaded fixed split files from {split_dir.resolve()}")
        return split_pairs
    split_pairs = discover_pairs(data_root, seed=seed)
    assert_content_disjoint(split_pairs)
    save_split_files(split_pairs, split_dir)
    log(f"[INFO] Saved fixed split files to {split_dir.resolve()}")
    return split_pairs


def print_dataset_summary(split_pairs):
    for split in ["train", "val", "test"]:
        pairs = split_pairs.get(split, [])
        log(f"num_{split}_images={len(pairs)}")
        for idx, (image_path, mask_path) in enumerate(pairs[:3], start=1):
            log(f"[{split} sample {idx}] image={image_path}")
            log(f"[{split} sample {idx}] mask ={mask_path}")


class UltraChipDataset(Dataset):
    def __init__(self, pairs, img_size: int, mask_dilate: int = 0, boundary_width: int = 3):
        self.pairs = pairs
        self.img_size = img_size
        self.mask_dilate = mask_dilate
        self.boundary_width = boundary_width if boundary_width % 2 == 1 else boundary_width + 1

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        raw_mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
        raw_mask = raw_mask.point(lambda p: 255 if p > 0 else 0)
        mask = raw_mask
        if self.mask_dilate > 0:
            size = self.mask_dilate if self.mask_dilate % 2 == 1 else self.mask_dilate + 1
            mask = mask.filter(ImageFilter.MaxFilter(size=size))

        dilated_boundary = raw_mask.filter(ImageFilter.MaxFilter(size=self.boundary_width))
        eroded_boundary = raw_mask.filter(ImageFilter.MinFilter(size=self.boundary_width))

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        boundary_np = (np.asarray(dilated_boundary, dtype=np.uint8) != np.asarray(eroded_boundary, dtype=np.uint8)).astype(np.float32)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = (image_tensor - MEAN) / STD
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        boundary_tensor = torch.from_numpy(boundary_np).unsqueeze(0)
        return image_tensor, mask_tensor, boundary_tensor, str(image_path)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return torch.pow(1.0 - tversky, self.gamma).mean()


class CompositeSegmentationLoss(nn.Module):
    def __init__(self, loss_type: str, use_boundary: bool = False, boundary_weight: float = 0.2):
        super().__init__()
        self.loss_type = loss_type
        self.use_boundary = use_boundary
        self.boundary_weight = boundary_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)

    def segmentation_loss(self, logits, targets):
        if self.loss_type == "bce_dice":
            return self.bce(logits, targets) + self.dice(logits, targets)
        if self.loss_type == "bce_dice_tversky":
            return self.bce(logits, targets) + self.dice(logits, targets) + self.focal_tversky(logits, targets)
        if self.loss_type == "bce_focal_tversky":
            return self.bce(logits, targets) + self.focal_tversky(logits, targets)
        raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def boundary_loss(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)

    def split_logits(self, logits):
        mask_logits = logits[:, 0:1, :, :]
        boundary_logits = logits[:, 1:2, :, :] if self.use_boundary and logits.shape[1] > 1 else None
        return mask_logits, boundary_logits

    def forward(self, logits, seg_targets, boundary_targets=None):
        mask_logits, boundary_logits = self.split_logits(logits)
        seg_loss = self.segmentation_loss(mask_logits, seg_targets)
        boundary_loss = logits.new_tensor(0.0)
        if self.use_boundary:
            if boundary_logits is None:
                raise RuntimeError("Boundary loss requested but model output has no boundary channel.")
            if boundary_targets is None:
                raise RuntimeError("Boundary loss requested but boundary target is missing.")
            boundary_loss = self.boundary_loss(boundary_logits, boundary_targets)
        total_loss = seg_loss + self.boundary_weight * boundary_loss
        return total_loss, seg_loss, boundary_loss, mask_logits, boundary_logits


def binary_stats(probs, targets, threshold):
    preds = probs >= threshold
    targets = targets >= 0.5
    tp = np.logical_and(preds, targets).sum(dtype=np.float64)
    fp = np.logical_and(preds, ~targets).sum(dtype=np.float64)
    fn = np.logical_and(~preds, targets).sum(dtype=np.float64)
    return tp, fp, fn


def stats_to_metrics(tp, fp, fn, eps=1e-7):
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return iou, dice, precision, recall


def connected_components(mask):
    mask = mask.astype(bool)
    if scipy_label is not None:
        # Default SciPy structure is 4-connected in 2D, exactly matching the
        # historical BFS below while avoiding a Python loop over every pixel.
        labels, count = scipy_label(mask)
        return [np.argwhere(labels == component) for component in range(1, count + 1)]
    visited = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    components = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            coords = []
            queue = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(np.asarray(coords, dtype=np.int32))
    return components


def smooth_probability_maps(probs, sigma: float):
    if sigma <= 0:
        return probs
    if gaussian_filter is None:
        raise RuntimeError("prob_smooth_sigma requires scipy, but scipy.ndimage.gaussian_filter is unavailable.")
    return np.stack([gaussian_filter(prob, sigma=sigma) for prob in probs]).astype(np.float32)


def compute_aupro(gt_list, prob_list, max_fpr=0.3, num_thresholds=101):
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    fprs = []
    pros = []
    has_region = any(gt.astype(bool).any() for gt in gt_list)
    has_negative = any((~gt.astype(bool)).any() for gt in gt_list)
    if not has_region or not has_negative:
        return {
            "raw": math.nan,
            "normalized": math.nan,
            "percent": math.nan,
            "note": "AUPRO is nan because GT has no positive regions or no negative pixels.",
        }

    components_per_image = [connected_components(gt > 0.5) for gt in gt_list]
    total_negative = sum((gt <= 0.5).sum(dtype=np.float64) for gt in gt_list)

    for threshold in thresholds:
        fp = 0.0
        region_overlaps = []
        for gt, prob, components in zip(gt_list, prob_list, components_per_image):
            pred = prob >= threshold
            fp += np.logical_and(pred, gt <= 0.5).sum(dtype=np.float64)
            for comp in components:
                ys = comp[:, 0]
                xs = comp[:, 1]
                region_overlaps.append(pred[ys, xs].mean())
        fprs.append(fp / max(total_negative, 1.0))
        pros.append(float(np.mean(region_overlaps)) if region_overlaps else math.nan)

    points = sorted((f, p) for f, p in zip(fprs, pros) if not math.isnan(p) and f <= max_fpr)
    if not points:
        return {
            "raw": math.nan,
            "normalized": math.nan,
            "percent": math.nan,
            "note": f"AUPRO is nan because no threshold points had FPR <= {max_fpr}.",
        }
    if points[0][0] > 0.0:
        points.insert(0, (0.0, points[0][1]))
    if points[-1][0] < max_fpr:
        points.append((max_fpr, points[-1][1]))
    xs = np.asarray([p[0] for p in points], dtype=np.float64)
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    raw = float(auc(xs, ys))
    normalized = raw / max_fpr
    return {"raw": raw, "normalized": normalized, "percent": normalized * 100.0, "note": ""}


def compute_curve_metrics(y_true, y_score, gt_images, prob_images, prefix=""):
    out = {}
    notes = []
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        out[f"{prefix}pixel_ap"] = math.nan
        out[f"{prefix}pixel_auroc"] = math.nan
        notes.append(f"{prefix}Pixel-AP/Pixel-AUROC are nan because flattened GT has a single class or is empty.")
    else:
        out[f"{prefix}pixel_ap"] = float(average_precision_score(y_true, y_score))
        out[f"{prefix}pixel_auroc"] = float(roc_auc_score(y_true, y_score))

    aupro = compute_aupro(gt_images, prob_images)
    out[f"{prefix}aupro_raw"] = aupro["raw"]
    out[f"{prefix}aupro_normalized"] = aupro["normalized"]
    out[f"{prefix}aupro_percent"] = aupro["percent"]
    if aupro["note"]:
        notes.append(prefix + aupro["note"])
    out[f"{prefix}notes"] = " ".join(notes)
    return out


def evaluate(model, loader, criterion, device, threshold, collect_curves=False, prob_smooth_sigma=0.0):
    model.eval()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_boundary_loss = 0.0
    total_samples = 0
    tp = fp = fn = 0.0
    all_probs = []
    all_gts = []
    gt_images = []
    prob_images = []

    with torch.no_grad():
        for images, masks, boundaries, _ in tqdm(loader, desc="eval", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            boundaries = boundaries.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss, seg_loss, boundary_loss, mask_logits, _ = criterion(logits, masks, boundaries)
            probs = torch.sigmoid(mask_logits).float().cpu().numpy()[:, 0]
            gts = masks.float().cpu().numpy()[:, 0]
            total_loss += float(loss.item()) * images.size(0)
            total_seg_loss += float(seg_loss.item()) * images.size(0)
            total_boundary_loss += float(boundary_loss.item()) * images.size(0)
            total_samples += images.size(0)
            btp, bfp, bfn = binary_stats(probs, gts, threshold)
            tp += btp
            fp += bfp
            fn += bfn
            if collect_curves:
                all_probs.append(probs.reshape(-1))
                all_gts.append(gts.reshape(-1))
                prob_images.extend([p.astype(np.float32) for p in probs])
                gt_images.extend([g.astype(np.float32) for g in gts])

    iou, dice, precision, recall = stats_to_metrics(tp, fp, fn)
    out = {
        "loss": total_loss / max(total_samples, 1),
        "seg_loss": total_seg_loss / max(total_samples, 1),
        "boundary_loss": total_boundary_loss / max(total_samples, 1),
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }
    if collect_curves:
        y_true = np.concatenate(all_gts).astype(np.uint8) if all_gts else np.asarray([], dtype=np.uint8)
        y_score = np.concatenate(all_probs).astype(np.float32) if all_probs else np.asarray([], dtype=np.float32)
        raw = compute_curve_metrics(y_true, y_score, gt_images, prob_images, prefix="raw_")
        out.update(raw)

        smooth = {}
        if prob_smooth_sigma > 0:
            smooth_prob_images = [gaussian_filter(prob, sigma=prob_smooth_sigma).astype(np.float32) for prob in prob_images]
            smooth_y_score = np.concatenate([p.reshape(-1) for p in smooth_prob_images]).astype(np.float32)
            smooth = compute_curve_metrics(y_true, smooth_y_score, gt_images, smooth_prob_images, prefix="smooth_")
            out.update(smooth)
        else:
            out.update(
                {
                    "smooth_pixel_ap": math.nan,
                    "smooth_pixel_auroc": math.nan,
                    "smooth_aupro_raw": math.nan,
                    "smooth_aupro_normalized": math.nan,
                    "smooth_aupro_percent": math.nan,
                    "smooth_notes": "",
                }
            )

        raw_aupro = out["raw_aupro_normalized"]
        smooth_aupro = out["smooth_aupro_normalized"]
        use_smooth = prob_smooth_sigma > 0 and not math.isnan(smooth_aupro) and (math.isnan(raw_aupro) or smooth_aupro > raw_aupro)
        selected_prefix = "smooth_" if use_smooth else "raw_"
        out["prob_variant_used"] = "smooth" if use_smooth else "raw"
        out["pixel_ap"] = out[f"{selected_prefix}pixel_ap"]
        out["pixel_auroc"] = out[f"{selected_prefix}pixel_auroc"]
        out["aupro_raw"] = out[f"{selected_prefix}aupro_raw"]
        out["aupro_normalized"] = out[f"{selected_prefix}aupro_normalized"]
        out["aupro_percent"] = out[f"{selected_prefix}aupro_percent"]
        out["metric_notes"] = " ".join(note for note in [out.get("raw_notes", ""), out.get("smooth_notes", "")] if note)
    return out


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_boundary_loss = 0.0
    total_samples = 0
    for images, masks, boundaries, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        boundaries = boundaries.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss, seg_loss, boundary_loss, _, _ = criterion(logits, masks, boundaries)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.size(0)
        total_seg_loss += float(seg_loss.item()) * images.size(0)
        total_boundary_loss += float(boundary_loss.item()) * images.size(0)
        total_samples += images.size(0)
    return {
        "loss": total_loss / max(total_samples, 1),
        "seg_loss": total_seg_loss / max(total_samples, 1),
        "boundary_loss": total_boundary_loss / max(total_samples, 1),
    }


def denormalize(image_tensor):
    image = image_tensor.cpu() * STD + MEAN
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def colorize_heatmap(prob):
    prob = np.clip(prob, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * prob - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * prob - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * prob - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def save_visualizations(model, dataset, device, threshold, vis_dir: Path, max_images=20, use_boundary=False):
    vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    count = min(max_images, len(dataset))
    with torch.no_grad():
        for idx in range(count):
            image_tensor, mask_tensor, boundary_tensor, image_path = dataset[idx]
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(image_tensor.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits[:, 0:1, :, :])[0, 0].float().cpu().numpy()
            pred = (prob >= threshold).astype(np.uint8) * 255
            image = denormalize(image_tensor)
            gt = (mask_tensor[0].numpy() * 255).astype(np.uint8)
            gt_rgb = np.repeat(gt[:, :, None], 3, axis=2)
            pred_rgb = np.repeat(pred[:, :, None], 3, axis=2)
            heatmap = colorize_heatmap(prob)
            panels = [image, gt_rgb, pred_rgb, heatmap]
            if use_boundary:
                gt_boundary = (boundary_tensor[0].numpy() * 255).astype(np.uint8)
                gt_boundary_rgb = np.repeat(gt_boundary[:, :, None], 3, axis=2)
                if logits.shape[1] > 1:
                    pred_boundary = (torch.sigmoid(logits[:, 1:2, :, :])[0, 0].float().cpu().numpy() > 0.5).astype(np.uint8) * 255
                else:
                    pred_boundary = np.zeros_like(gt_boundary, dtype=np.uint8)
                pred_boundary_rgb = np.repeat(pred_boundary[:, :, None], 3, axis=2)
                panels.extend([gt_boundary_rgb, pred_boundary_rgb])
            panel = np.concatenate(panels, axis=1)
            stem = Path(image_path).stem
            Image.fromarray(panel).save(vis_dir / f"{idx:03d}_{stem}.png")


def make_model(model_name, encoder_name, encoder_weights, classes=1):
    model_name = model_name.lower()
    model_builders = {
        "fpn": smp.FPN,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "deeplabv3+": smp.DeepLabV3Plus,
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "unet++": smp.UnetPlusPlus,
    }
    if model_name not in model_builders:
        raise ValueError(f"Unsupported model_name={model_name}. Supported: {sorted(model_builders)}")
    builder = model_builders[model_name]

    try:
        return builder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=None,
        ), encoder_weights
    except Exception as exc:
        if encoder_weights is None:
            raise
        log(f"[WARN] Failed to create {model_name} with encoder={encoder_name}, encoder_weights={encoder_weights!r}: {exc}")
        log("[WARN] Falling back to encoder_weights=None. ImageNet pretrained weights were not used.")
        return builder(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=classes,
            activation=None,
        ), None


def write_metrics_csv(path: Path, row: dict):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def append_metrics_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    fieldnames = list(row.keys())
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        for name in old_fieldnames:
            if name not in fieldnames:
                fieldnames.append(name)
        if old_fieldnames != fieldnames:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow({name: old_row.get(name, "") for name in fieldnames})
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def is_better_metric(value, best_value):
    return not math.isnan(value) and (best_value is None or value > best_value)


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation models on UltraChip binary defect segmentation.")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_dir", type=str, default="results/fpn_resnet34_ultrachip")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name",
        type=str,
        default="fpn",
        choices=["fpn", "deeplabv3plus", "deeplabv3+", "unet", "unetplusplus", "unet++"],
    )
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--loss_type", type=str, default="bce_dice", choices=["bce_dice", "bce_dice_tversky", "bce_focal_tversky"])
    parser.add_argument("--train_mask_dilate", type=int, default=0)
    parser.add_argument("--prob_smooth_sigma", type=float, default=0.0)
    parser.add_argument("--use_boundary", action="store_true")
    parser.add_argument("--boundary_width", type=int, default=3)
    parser.add_argument("--boundary_weight", type=float, default=0.2)
    parser.add_argument("--split_dir", type=str, default="results/splits_ultrachip_seed42")
    parser.add_argument("--comparison_csv", type=str, default="results/ultrachip_aupro_comparison.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    data_root = Path(args.data_root) if args.data_root else Path("segmentation_models_pytorch/inputs/ultrachip")
    data_root = data_root.resolve()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log(f"[INFO] data_root={data_root}")
    log(f"[INFO] save_dir={save_dir.resolve()}")

    if args.prob_smooth_sigma > 0 and gaussian_filter is None:
        raise RuntimeError("--prob_smooth_sigma requires scipy, but scipy is not available in this environment.")

    split_pairs = get_or_create_splits(data_root, Path(args.split_dir), seed=args.seed)
    print_dataset_summary(split_pairs)
    if not split_pairs["train"] or not split_pairs["val"] or not split_pairs["test"]:
        raise RuntimeError("Train/val/test split must all contain at least one paired image and mask.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        log("[WARN] CUDA requested but unavailable. Falling back to CPU.")
    if args.batch_size > 4:
        log("[WARN] If RTX4060 runs out of memory, reduce --batch_size to 2.")

    train_dataset = UltraChipDataset(split_pairs["train"], args.img_size, mask_dilate=args.train_mask_dilate, boundary_width=args.boundary_width)
    val_dataset = UltraChipDataset(split_pairs["val"], args.img_size, mask_dilate=0, boundary_width=args.boundary_width)
    test_dataset = UltraChipDataset(split_pairs["test"], args.img_size, mask_dilate=0, boundary_width=args.boundary_width)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    encoder_weights = None if str(args.encoder_weights).lower() in {"none", "null", ""} else args.encoder_weights
    output_classes = 2 if args.use_boundary else 1
    model, used_encoder_weights = make_model(args.model_name, args.encoder_name, encoder_weights, classes=output_classes)
    model.to(device)

    criterion = CompositeSegmentationLoss(args.loss_type, use_boundary=args.use_boundary, boundary_weight=args.boundary_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    log_path = save_dir / "log.csv"
    log_fields = [
        "epoch",
        "train_loss",
        "train_seg_loss",
        "train_boundary_loss",
        "val_loss",
        "val_seg_loss",
        "val_boundary_loss",
        "val_iou",
        "val_dice",
        "val_precision",
        "val_recall",
        "val_pixel_auroc",
        "val_pixel_ap",
        "val_aupro_raw",
        "val_aupro_normalized",
        "val_aupro_percent",
        "val_prob_variant_used",
        "raw_val_pixel_auroc",
        "raw_val_pixel_ap",
        "raw_val_aupro_raw",
        "raw_val_aupro_normalized",
        "raw_val_aupro_percent",
        "smooth_val_pixel_auroc",
        "smooth_val_pixel_ap",
        "smooth_val_aupro_raw",
        "smooth_val_aupro_normalized",
        "smooth_val_aupro_percent",
        "lr",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    best_dice = None
    best_aupro = None
    best_pixel_ap = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.threshold,
            collect_curves=True,
            prob_smooth_sigma=args.prob_smooth_sigma,
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        log(
            f"epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.6f} train_seg_loss={train_metrics['seg_loss']:.6f} "
            f"train_boundary_loss={train_metrics['boundary_loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_seg_loss={val_metrics['seg_loss']:.6f} "
            f"val_boundary_loss={val_metrics['boundary_loss']:.6f} "
            f"val_iou={val_metrics['iou']:.6f} val_dice={val_metrics['dice']:.6f} "
            f"val_pixel_auroc={val_metrics['pixel_auroc']} val_pixel_ap={val_metrics['pixel_ap']} "
            f"val_aupro_norm={val_metrics['aupro_normalized']} val_aupro_percent={val_metrics['aupro_percent']} "
            f"variant={val_metrics['prob_variant_used']}"
        )
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_seg_loss": train_metrics["seg_loss"],
                    "train_boundary_loss": train_metrics["boundary_loss"],
                    "val_loss": val_metrics["loss"],
                    "val_seg_loss": val_metrics["seg_loss"],
                    "val_boundary_loss": val_metrics["boundary_loss"],
                    "val_iou": val_metrics["iou"],
                    "val_dice": val_metrics["dice"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_pixel_auroc": val_metrics["pixel_auroc"],
                    "val_pixel_ap": val_metrics["pixel_ap"],
                    "val_aupro_raw": val_metrics["aupro_raw"],
                    "val_aupro_normalized": val_metrics["aupro_normalized"],
                    "val_aupro_percent": val_metrics["aupro_percent"],
                    "val_prob_variant_used": val_metrics["prob_variant_used"],
                    "raw_val_pixel_auroc": val_metrics["raw_pixel_auroc"],
                    "raw_val_pixel_ap": val_metrics["raw_pixel_ap"],
                    "raw_val_aupro_raw": val_metrics["raw_aupro_raw"],
                    "raw_val_aupro_normalized": val_metrics["raw_aupro_normalized"],
                    "raw_val_aupro_percent": val_metrics["raw_aupro_percent"],
                    "smooth_val_pixel_auroc": val_metrics["smooth_pixel_auroc"],
                    "smooth_val_pixel_ap": val_metrics["smooth_pixel_ap"],
                    "smooth_val_aupro_raw": val_metrics["smooth_aupro_raw"],
                    "smooth_val_aupro_normalized": val_metrics["smooth_aupro_normalized"],
                    "smooth_val_aupro_percent": val_metrics["smooth_aupro_percent"],
                    "lr": lr,
                }
            )
        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "encoder_weights": used_encoder_weights,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        if is_better_metric(val_metrics["dice"], best_dice):
            best_dice = val_metrics["dice"]
            torch.save(checkpoint, save_dir / "best_dice_model.pth")
        if is_better_metric(val_metrics["aupro_normalized"], best_aupro):
            best_aupro = val_metrics["aupro_normalized"]
            torch.save(checkpoint, save_dir / "best_aupro_model.pth")
            torch.save(checkpoint, save_dir / "best_model.pth")
        if is_better_metric(val_metrics["pixel_ap"], best_pixel_ap):
            best_pixel_ap = val_metrics["pixel_ap"]
            torch.save(checkpoint, save_dir / "best_pixel_ap_model.pth")

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "best_val_dice": best_dice,
            "best_val_aupro_normalized": best_aupro,
            "best_val_pixel_ap": best_pixel_ap,
            "encoder_weights": used_encoder_weights,
        },
        save_dir / "last_model.pth",
    )

    best_path = save_dir / "best_aupro_model.pth"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        args.threshold,
        collect_curves=True,
        prob_smooth_sigma=args.prob_smooth_sigma,
    )
    if test_metrics.get("metric_notes"):
        log(f"[WARN] {test_metrics['metric_notes']}")

    metrics_row = {
        "model_name": args.model_name,
        "encoder_name": args.encoder_name,
        "loss_type": args.loss_type,
        "train_mask_dilate": args.train_mask_dilate,
        "prob_smooth_sigma": args.prob_smooth_sigma,
        "use_boundary": args.use_boundary,
        "boundary_width": args.boundary_width,
        "boundary_weight": args.boundary_weight,
        "best_metric_used": "val_aupro_normalized",
        "image_size": args.img_size,
        "threshold": args.threshold,
        "iou": test_metrics["iou"],
        "dice": test_metrics["dice"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "pixel_auroc": test_metrics["pixel_auroc"],
        "pixel_ap": test_metrics["pixel_ap"],
        "aupro_raw": test_metrics["aupro_raw"],
        "aupro_normalized": test_metrics["aupro_normalized"],
        "aupro_percent": test_metrics["aupro_percent"],
        "prob_variant_used": test_metrics["prob_variant_used"],
        "raw_pixel_auroc": test_metrics["raw_pixel_auroc"],
        "raw_pixel_ap": test_metrics["raw_pixel_ap"],
        "raw_aupro_raw": test_metrics["raw_aupro_raw"],
        "raw_aupro_normalized": test_metrics["raw_aupro_normalized"],
        "raw_aupro_percent": test_metrics["raw_aupro_percent"],
        "smooth_pixel_auroc": test_metrics["smooth_pixel_auroc"],
        "smooth_pixel_ap": test_metrics["smooth_pixel_ap"],
        "smooth_aupro_raw": test_metrics["smooth_aupro_raw"],
        "smooth_aupro_normalized": test_metrics["smooth_aupro_normalized"],
        "smooth_aupro_percent": test_metrics["smooth_aupro_percent"],
        "num_train_images": len(train_dataset),
        "num_val_images": len(val_dataset),
        "num_test_images": len(test_dataset),
        "save_dir": str(save_dir),
    }
    write_metrics_csv(save_dir / "test_metrics.csv", metrics_row)
    comparison_row = {
        "model_name": args.model_name,
        "encoder_name": args.encoder_name,
        "loss_type": args.loss_type,
        "train_mask_dilate": args.train_mask_dilate,
        "prob_smooth_sigma": args.prob_smooth_sigma,
        "use_boundary": args.use_boundary,
        "boundary_width": args.boundary_width,
        "boundary_weight": args.boundary_weight,
        "best_metric_used": "val_aupro_normalized",
        "image_size": args.img_size,
        "threshold": args.threshold,
        "iou": test_metrics["iou"],
        "dice": test_metrics["dice"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "pixel_auroc": test_metrics["pixel_auroc"],
        "pixel_ap": test_metrics["pixel_ap"],
        "aupro_raw": test_metrics["aupro_raw"],
        "aupro_normalized": test_metrics["aupro_normalized"],
        "aupro_percent": test_metrics["aupro_percent"],
        "num_train_images": len(train_dataset),
        "num_val_images": len(val_dataset),
        "num_test_images": len(test_dataset),
        "save_dir": str(save_dir),
    }
    append_metrics_csv(Path(args.comparison_csv), comparison_row)
    save_visualizations(model, test_dataset, device, args.threshold, save_dir / "vis_test", max_images=20, use_boundary=args.use_boundary)

    log("Final test metrics:")
    log(f"IoU={test_metrics['iou']:.6f}")
    log(f"Dice={test_metrics['dice']:.6f}")
    log(f"Precision={test_metrics['precision']:.6f}")
    log(f"Recall={test_metrics['recall']:.6f}")
    log(f"Pixel-AUROC={test_metrics['pixel_auroc']}")
    log(f"Pixel-AP={test_metrics['pixel_ap']}")
    log(f"AUPRO_raw={test_metrics['aupro_raw']}")
    log(f"AUPRO_normalized={test_metrics['aupro_normalized']}")
    log(f"AUPRO_percent={test_metrics['aupro_percent']}")
    log(f"prob_variant_used={test_metrics['prob_variant_used']}")
    for required in [
        "best_model.pth",
        "best_dice_model.pth",
        "best_aupro_model.pth",
        "best_pixel_ap_model.pth",
        "last_model.pth",
        "log.csv",
        "test_metrics.csv",
        "vis_test",
    ]:
        path = save_dir / required
        log(f"[CHECK] {path}: {'exists' if path.exists() else 'missing'}")


if __name__ == "__main__":
    main()
