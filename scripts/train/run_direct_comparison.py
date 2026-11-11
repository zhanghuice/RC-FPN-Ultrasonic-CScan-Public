"""Controlled E0--E3 comparison for JCRAI 2026 / RI027.

This file is intentionally isolated from the historical trainers.  Validation
AUPRO alone selects checkpoints; the test split is evaluated only after the
best checkpoint has been frozen and hashed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt, gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = PROJECT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import segmentation_models_pytorch as smp  # noqa: E402
from train_ultrachip_fpn import (  # noqa: E402
    FocalTverskyLoss,
    UltraChipDataset,
    compute_aupro,
    load_split_files,
)


EXPECTED_COUNTS = {"train": 791, "val": 113, "test": 226}
METHODS = {
    "E0": "Original hard-mask joint-loss baseline",
    "E1": "Signed-distance Boundary Loss",
    "E2": "Binary boundary-label relaxation",
    "E3": "3x3 training-label dilation",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def state_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest().upper()


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def git_commit_from_files(repo: Path) -> str | None:
    git_dir = repo / ".git"
    head = git_dir / "HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if not text.startswith("ref: "):
        return text
    ref = text[5:]
    ref_path = git_dir / ref
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and line.endswith(" " + ref):
                return line.split()[0]
    return None


def validate_splits(split_dir: Path) -> dict[str, list[tuple[Path, Path]]]:
    pairs = load_split_files(split_dir)
    if pairs is None:
        raise RuntimeError("Fixed split files are missing")
    counts = {name: len(items) for name, items in pairs.items()}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Split count mismatch: {counts}")
    image_sets = {name: {str(i.resolve()).lower() for i, _ in items} for name, items in pairs.items()}
    if image_sets["train"] & image_sets["val"] or image_sets["train"] & image_sets["test"] or image_sets["val"] & image_sets["test"]:
        raise RuntimeError("Path-level image leakage detected between splits")
    content_membership: dict[str, set[str]] = {}
    for split, items in pairs.items():
        for image_path, _ in items:
            content_membership.setdefault(sha256(image_path), set()).add(split)
    cross_split = sum(len(parts) > 1 for parts in content_membership.values())
    if cross_split:
        raise RuntimeError(
            f"Content-level image leakage detected in {cross_split} exact-image groups; "
            "create a group-disjoint split before training"
        )
    for split, items in pairs.items():
        for image_path, mask_path in items:
            if image_path.stem.lower() != mask_path.stem.lower():
                raise RuntimeError(f"Stem mismatch in {split}: {image_path} vs {mask_path}")
    return pairs


def category(path: Path) -> str:
    low = path.stem.lower()
    if "crack" in low:
        return "Crack"
    if "layer" in low:
        return "Layer"
    return "Hole"


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Kervadec-style signed distance, normalized per nonempty image to [-1, 1]."""
    pos = mask.astype(bool)
    if not pos.any():
        return np.zeros(mask.shape, dtype=np.float32)
    neg = ~pos
    outside = distance_transform_edt(neg)
    inside = distance_transform_edt(pos)
    # Boundary foreground pixels are zero through (inside - 1), as in the
    # reference implementation; deeper foreground is negative.
    result = outside * neg - (inside - 1.0) * pos
    scale = float(np.max(np.abs(result)))
    if scale > 0:
        result = result / scale
    return result.astype(np.float32)


class DirectDataset(Dataset):
    def __init__(self, pairs, image_size: int, method: str, split: str, sdm_cache: Path):
        self.base = UltraChipDataset(pairs, img_size=image_size, mask_dilate=0, boundary_width=3)
        self.method = method
        self.split = split
        self.sdm_cache = sdm_cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, mask, boundary, path = self.base[index]
        target = mask
        if self.method == "E3" and self.split == "train":
            target = F.max_pool2d(mask.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        sdm = torch.zeros_like(mask)
        if self.method == "E1" and self.split == "train":
            cache = self.sdm_cache / f"{index:04d}_{Path(path).stem}.npy"
            if cache.exists():
                arr = np.load(cache)
            else:
                cache.parent.mkdir(parents=True, exist_ok=True)
                arr = signed_distance(mask[0].numpy())
                np.save(cache, arr, allow_pickle=False)
            sdm = torch.from_numpy(arr).unsqueeze(0)
        return image, target, mask, boundary, sdm, path


class ControlledLoss(nn.Module):
    def __init__(self, method: str, boundary_lambda: float):
        super().__init__()
        self.method = method
        self.boundary_lambda = boundary_lambda
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.ft = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75, eps=1e-7)

    def terms(self, logits, target, original, relaxation_boundary, sdm):
        per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if self.method == "E2":
            keep = 1.0 - relaxation_boundary
            bce = (per_pixel * keep).sum() / keep.sum().clamp_min(1.0)
        else:
            bce = per_pixel.mean()
        # E3 intentionally uses its dilated target. E0/E1/E2 targets equal M.
        dice = self.dice(logits, target)
        focal_tversky = self.ft(logits, target)
        boundary = (torch.sigmoid(logits) * sdm).mean() if self.method == "E1" else logits.new_tensor(0.0)
        total = bce + dice + focal_tversky + self.boundary_lambda * boundary
        return {"total": total, "bce": bce, "dice": dice, "focal_tversky": focal_tversky, "boundary": boundary}

    def forward(self, logits, target, original, relaxation_boundary, sdm):
        return self.terms(logits, target, original, relaxation_boundary, sdm)["total"]


def build_model() -> nn.Module:
    return smp.FPN(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )


def make_loaders(pairs, method: str, args, device):
    datasets = {
        split: DirectDataset(items, args.image_size, method, split, args.output_root / "cache" / "signed_distance_train")
        for split, items in pairs.items()
    }
    loaders = {}
    for split, dataset in datasets.items():
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            generator=generator,
            worker_init_fn=seed_worker,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
    return datasets, loaders


def train_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    sums = {name: 0.0 for name in ["total", "bce", "dice", "focal_tversky", "boundary"]}
    seen = 0
    for image, target, original, boundary, sdm, _ in tqdm(loader, desc="train", leave=False, disable=True):
        image, target, original, boundary, sdm = [x.to(device, non_blocking=True) for x in (image, target, original, boundary, sdm)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(image)
            terms = loss_fn.terms(logits, target, original, boundary, sdm)
        if not torch.isfinite(terms["total"]):
            raise FloatingPointError(f"Non-finite loss: {terms}")
        scaler.scale(terms["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        n = image.size(0)
        for key in sums:
            sums[key] += float(terms[key].detach()) * n
        seen += n
    return {key: value / seen for key, value in sums.items()}


def metric_bundle(gt_images, prob_images, threshold=0.5):
    y_true = np.concatenate([x.reshape(-1) for x in gt_images]).astype(np.uint8)
    y_score = np.concatenate([x.reshape(-1) for x in prob_images]).astype(np.float32)
    pred = y_score >= threshold
    truth = y_true >= 1
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    eps = 1e-7
    aupro = compute_aupro(gt_images, prob_images, max_fpr=0.3, num_thresholds=101)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / (tp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "pixel_ap": float(average_precision_score(y_true, y_score)),
        "pixel_auroc": float(roc_auc_score(y_true, y_score)),
        "aupro": float(aupro["normalized"]),
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, save_prob_dir: Path | None = None):
    model.eval()
    gt_images, prob_images, paths = [], [], []
    loss_sum = 0.0
    seen = 0
    if save_prob_dir is not None:
        save_prob_dir.mkdir(parents=True, exist_ok=True)
    for image, target, original, boundary, sdm, batch_paths in tqdm(loader, desc="eval", leave=False, disable=True):
        image, target, original, boundary, sdm = [x.to(device, non_blocking=True) for x in (image, target, original, boundary, sdm)]
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(image)
            loss = loss_fn(logits, target, original, boundary, sdm)
        raw = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
        smooth = [gaussian_filter(p, sigma=1.0, mode="reflect", truncate=4.0).astype(np.float32) for p in raw]
        gts = original.float().cpu().numpy()[:, 0]
        gt_images.extend([x.astype(np.float32) for x in gts])
        prob_images.extend(smooth)
        paths.extend(batch_paths)
        loss_sum += float(loss) * image.size(0)
        seen += image.size(0)
        if save_prob_dir is not None:
            for p, source in zip(smooth, batch_paths):
                np.save(save_prob_dir / f"{Path(source).stem}.npy", p, allow_pickle=False)
    result = metric_bundle(gt_images, prob_images)
    result["loss"] = loss_sum / max(seen, 1)
    result["num_images"] = seen
    return result, gt_images, prob_images, paths


def initial_diagnostics(model, loader, loss_fn, device, output: Path):
    batch = next(iter(loader))
    image, target, original, boundary, sdm, _ = batch
    image, target, original, boundary, sdm = [x.to(device) for x in (image, target, original, boundary, sdm)]
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(image)
    terms = loss_fn.terms(logits, target, original, boundary, sdm)
    base = terms["bce"] + terms["dice"] + terms["focal_tversky"]
    base.backward(retain_graph=True)
    base_grad = math.sqrt(sum(float((p.grad.detach() ** 2).sum()) for p in model.parameters() if p.grad is not None))
    model.zero_grad(set_to_none=True)
    if loss_fn.method == "E1":
        (loss_fn.boundary_lambda * terms["boundary"]).backward()
        boundary_grad = math.sqrt(sum(float((p.grad.detach() ** 2).sum()) for p in model.parameters() if p.grad is not None))
    else:
        boundary_grad = 0.0
    record = {key: float(value.detach()) for key, value in terms.items()}
    record.update({"base_gradient_l2": base_grad, "weighted_boundary_gradient_l2": boundary_grad})
    json_dump(output, record)
    model.zero_grad(set_to_none=True)


def run_method(method: str, args, pairs):
    run_dir = args.output_root / "runs" / f"{method}_seed{args.seed}"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"Refusing to overwrite nonempty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    initial_state_hash = state_sha256(model)
    datasets, loaders = make_loaders(pairs, method, args, device)
    boundary_lambda = args.boundary_lambda if method == "E1" else 0.0
    loss_fn = ControlledLoss(method, boundary_lambda).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    initial_diagnostics(model, loaders["train"], loss_fn, device, run_dir / "initial_loss_gradient_diagnostic.json")

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "method": method,
        "method_description": METHODS[method],
        "initial_state_sha256": initial_state_hash,
        "model": "FPN-ResNet50",
        "encoder_weights": "imagenet",
        "optimizer": "AdamW",
        "scheduler": f"CosineAnnealingLR(T_max={args.epochs})",
        "amp": device.type == "cuda",
        "selection": "maximum validation normalized AUPRO after fixed sigma=1.0 smoothing",
        "test_access_policy": "one evaluation after checkpoint freeze and SHA-256",
    })
    json_dump(run_dir / "config.json", config)

    start_epoch = 1
    best_aupro = -math.inf
    best_epoch = None
    if args.resume and (run_dir / "last_model.pth").exists():
        checkpoint = torch.load(run_dir / "last_model.pth", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_aupro = checkpoint["best_aupro"]
        best_epoch = checkpoint["best_epoch"]

    fields = ["epoch", "train_total", "train_bce", "train_dice", "train_focal_tversky", "train_boundary", "val_loss", "val_recall", "val_precision", "val_dice", "val_iou", "val_pixel_ap", "val_pixel_auroc", "val_aupro", "lr", "elapsed_seconds"]
    log_path = run_dir / "training_log.csv"
    if start_epoch == 1:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train = train_epoch(model, loaders["train"], loss_fn, optimizer, scaler, device)
        val, _, _, _ = evaluate(model, loaders["val"], loss_fn, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train.items()},
            **{f"val_{key}": value for key, value in val.items() if key in {"loss", "recall", "precision", "dice", "iou", "pixel_ap", "pixel_auroc", "aupro"}},
            "lr": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.time() - start_time,
        }
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "best_aupro": best_aupro, "best_epoch": best_epoch,
            "method": method, "seed": args.seed, "val_metrics": val, "config": config,
        }
        if val["aupro"] > best_aupro:
            best_aupro, best_epoch = val["aupro"], epoch
            checkpoint["best_aupro"], checkpoint["best_epoch"] = best_aupro, best_epoch
            torch.save(checkpoint, run_dir / "best_model.pth")
        torch.save(checkpoint, run_dir / "last_model.pth")
        print(f"{method} epoch={epoch:03d}/{args.epochs} train={train['total']:.6f} val_AUPRO={val['aupro']:.6f} best={best_aupro:.6f}@{best_epoch}", flush=True)

    best_path = run_dir / "best_model.pth"
    checkpoint_hash = sha256(best_path)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    # Recompute the selected validation checkpoint before the sole test pass.
    val, _, _, _ = evaluate(model, loaders["val"], loss_fn, device, run_dir / "probability_maps_val_sigma1")
    frozen = {
        "method": method, "seed": args.seed, "best_epoch": checkpoint["epoch"],
        "checkpoint": str(best_path.resolve()), "checkpoint_sha256": checkpoint_hash,
        "validation_metrics": val, "frozen_before_test_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json_dump(run_dir / "checkpoint_frozen_before_test.json", frozen)
    test, gts, probs, paths = evaluate(model, loaders["test"], loss_fn, device, run_dir / "probability_maps_test_sigma1")
    test["method"] = method
    test["seed"] = args.seed
    test["best_epoch"] = checkpoint["epoch"]
    test["checkpoint_sha256"] = checkpoint_hash
    classwise = {}
    for name in ["Hole", "Crack", "Layer"]:
        indexes = [i for i, path in enumerate(paths) if category(Path(path)) == name]
        classwise[name] = {"num_images": len(indexes), **metric_bundle([gts[i] for i in indexes], [probs[i] for i in indexes])}
    json_dump(run_dir / "final_metrics.json", {"validation": val, "test": test, "test_classwise": classwise})
    return {"validation": val, "test": test, "test_classwise": classwise}


def audit(args, pairs):
    environment = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "python": sys.version, "executable": sys.executable,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version(), "smp": smp.__version__, "scipy": __import__("scipy").__version__,
        "numpy": np.__version__, "sklearn": __import__("sklearn").__version__, "pillow": Image.__version__,
        "commit_hash": git_commit_from_files(PROJECT.parent),
        "split_counts": EXPECTED_COUNTS,
        "test_category_counts": {name: sum(category(i) == name for i, _ in pairs["test"]) for name in ["Hole", "Crack", "Layer"]},
        "overall_category_counts": {name: sum(category(i) == name for items in pairs.values() for i, _ in items) for name in ["Hole", "Crack", "Layer"]},
    }
    json_dump(HERE / "environment_audit.json", environment)


def make_sanity_visuals(pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = random.Random(42)
    indexes = sorted(rng.sample(range(len(pairs["train"])), 3))
    out = HERE / "sanity_visualizations"
    out.mkdir(parents=True, exist_ok=True)
    for index in indexes:
        image_path, mask_path = pairs["train"][index]
        image = Image.open(image_path).convert("RGB").resize((384, 384), Image.Resampling.BILINEAR)
        mask_image = Image.open(mask_path).convert("L").resize((384, 384), Image.Resampling.NEAREST).point(lambda p: 255 if p > 0 else 0)
        mask = (np.asarray(mask_image) > 0).astype(np.uint8)
        dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(3))) > 0
        eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(3))) > 0
        relaxation = dilated != eroded
        sdm = signed_distance(mask)
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.5), dpi=160)
        for ax, panel, title, cmap in zip(axes, [image, mask, dilated, relaxation, sdm], ["Image", "Original M", "E3: dilate 3x3", "E2: relaxed B", "E1: normalized SDF"], [None, "gray", "gray", "magma", "coolwarm"]):
            ax.imshow(panel, cmap=cmap); ax.set_title(title); ax.axis("off")
        fig.tight_layout(); fig.savefig(out / f"{index:04d}_{image_path.stem}.png"); plt.close(fig)


def selftest(args, pairs):
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tiny = np.zeros((7, 7), dtype=np.uint8)
    tiny[2:5, 2:5] = 1
    phi = signed_distance(tiny)
    checks = {
        "signed_distance_inside_nonpositive": bool(np.all(phi[tiny == 1] <= 0)),
        "signed_distance_outside_nonnegative": bool(np.all(phi[tiny == 0] >= 0)),
        "signed_distance_normalized": bool(np.max(np.abs(phi)) <= 1.0),
        "empty_mask_is_zero": bool(np.all(signed_distance(np.zeros((7, 7), dtype=np.uint8)) == 0)),
        "methods": {},
    }
    for method in METHODS:
        set_seed(args.seed)
        model = build_model().to(device)
        _, loaders = make_loaders(pairs, method, args, device)
        loss_fn = ControlledLoss(method, args.boundary_lambda if method == "E1" else 0.0).to(device)
        image, target, original, boundary, sdm, _ = next(iter(loaders["train"]))
        image, target, original, boundary, sdm = [x.to(device) for x in (image, target, original, boundary, sdm)]
        logits = model(image)
        terms = loss_fn.terms(logits, target, original, boundary, sdm)
        terms["total"].backward()
        checks["methods"][method] = {
            "all_terms_finite": bool(all(torch.isfinite(v).item() for v in terms.values())),
            "initial_state_sha256": state_sha256(model),
            "output_shape": list(logits.shape),
            "target_positive_pixels": int(target.sum()),
            "original_positive_pixels": int(original.sum()),
            "relaxed_pixels": int(boundary.sum()),
            "sdm_min": float(sdm.min()), "sdm_max": float(sdm.max()),
            **{key: float(value.detach()) for key, value in terms.items()},
        }
        del model, loaders, loss_fn, image, target, original, boundary, sdm, logits, terms
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not all(value for key, value in checks.items() if key != "methods") or not all(v["all_terms_finite"] for v in checks["methods"].values()):
        raise AssertionError(checks)
    json_dump(HERE / "numerical_unit_tests.json", checks)


def summarize():
    rows, payload = [], {}
    for method in METHODS:
        path = HERE / "runs" / f"{method}_seed42" / "final_metrics.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        payload[method] = data
        for split in ["validation", "test"]:
            m = data[split]
            rows.append({
                "method": method, "method_description": METHODS[method], "seed": 42, "split": split,
                "recall": m["recall"], "precision": m["precision"], "dice": m["dice"], "iou": m["iou"],
                "pixel_ap": m["pixel_ap"], "pixel_auroc": m["pixel_auroc"], "aupro": m["aupro"],
                "checkpoint_sha256": data["test"]["checkpoint_sha256"], "best_epoch": data["test"]["best_epoch"], "epochs": 100,
            })
    if rows:
        with (HERE / "direct_comparison_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    json_dump(HERE / "per_run_metrics.json", payload)
    if len(payload) != 4:
        return

    labels = {"E0": "FPN (E0)", "E1": "FPN + Boundary Loss (E1)", "E2": "FPN + Boundary Label Relaxation (E2)", "E3": "RC-FPN (E3)"}
    supervision = {"E0": "Original labels", "E1": "Signed-distance loss", "E2": "Binary relaxation adaptation", "E3": "3x3 training-mask dilation"}
    lines = [
        "# Controlled direct comparison (test set)", "",
        "| Method | Boundary supervision | Seed | Recall | Precision | Dice | IoU | Pixel-AP | Pixel-AUROC | AUPRO | Best epoch | Checkpoint SHA-256 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method in METHODS:
        m = payload[method]["test"]
        lines.append(f"| {labels[method]} | {supervision[method]} | 42 | {m['recall']:.4f} | {m['precision']:.4f} | {m['dice']:.4f} | {m['iou']:.4f} | {m['pixel_ap']:.4f} | {m['pixel_auroc']:.4f} | {m['aupro']:.4f} | {m['best_epoch']} | `{m['checkpoint_sha256']}` |")
    lines.extend([
        "", "**Table note.** All four methods use the identical fixed split, seeded ImageNet-pretrained FPN-ResNet50 initialization, 384 x 384 inputs, batch size 4, AdamW, 100 epochs, validation-AUPRO model selection, threshold 0.5, and one sigma=1.0 Gaussian filtering pass. E1 uses lambda=0.1 and a per-image max-absolute normalized signed-distance map. E2 removes BCE only in the 3 x 3 mixed-label band while retaining original-mask Dice and Focal Tversky. Results are from one training seed and do not establish statistical significance.",
    ])
    (HERE / "direct_comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def metric(method, key):
        return payload[method]["test"][key]

    reply = f"""# Response to Reviewer Comment 1

Thank you for requesting direct comparisons with boundary-aware alternatives. We added a new controlled experiment using the same fixed 791/113/226 split, FPN-ResNet50 initialization, optimizer, 100-epoch schedule, validation-set model selection, threshold, and sigma=1.0 post-processing for all methods. The only differences were the supervision rules. The standard joint-loss FPN (E0), signed-distance Boundary Loss with lambda=0.1 (E1), a binary adaptation of boundary-label relaxation (E2), and our one-pixel training-label dilation (E3) obtained test Recall/AUPRO values of {metric('E0','recall'):.4f}/{metric('E0','aupro'):.4f}, {metric('E1','recall'):.4f}/{metric('E1','aupro'):.4f}, {metric('E2','recall'):.4f}/{metric('E2','aupro'):.4f}, and {metric('E3','recall'):.4f}/{metric('E3','aupro'):.4f}, respectively. The revised table also reports Precision, Dice, IoU, Pixel-AP, and Pixel-AUROC so that gains in sensitivity are not presented without their false-positive and overlap trade-offs.

Boundary Loss was implemented as a genuine differentiable signed-distance term, not edge-pixel BCE. The distance map was negative inside and positive outside, normalized per image by its maximum absolute value, and multiplied by the continuous sigmoid probability. Our E2 experiment is explicitly described as a binary joint-loss adaptation of Zhu et al.'s label-relaxation principle: BCE is omitted only in the mixed 3 x 3 boundary band, whereas Dice and Focal Tversky retain the original mask. We do not claim to reproduce their video-propagation or full multiclass framework.

We use a single 3 x 3 dilation because the low-contrast ultrasonic C-scan boundary is often less certain than the annotated defect core, while the defects themselves can occupy very few pixels after resizing. A one-pixel expansion increases positive training support and reduces the penalty for a prediction immediately adjacent to an uncertain annotation, but it changes geometry minimally and does not introduce a separate boundary branch or inference-time parameter. Dilation is applied only to training labels; validation and test always use the complete original masks. This explanation is consistent with the controlled results but is not presented as a significance claim because Stage A contains one training seed. The additional seeds requested elsewhere remain future work unless actually completed.
"""
    (HERE / "reviewer_comment1_reply.md").write_text(reply, encoding="utf-8")

    insert = f"""# Manuscript insertion for the direct comparison

## Placement

Insert the following subsection in Section 3.2.3, immediately after the progressive ablation discussion and before the Gaussian-smoothing sensitivity analysis. Insert the table from `direct_comparison_table.md` after the first paragraph.

## Direct comparison with boundary-aware supervision

To isolate the effect of training-label region expansion from other boundary-aware supervision strategies, we conducted a controlled comparison using a standard joint-loss FPN (E0), signed-distance Boundary Loss [16] (E1), a binary adaptation of boundary-label relaxation [17] (E2), and the proposed 3 x 3 training-label dilation (E3). All methods used the same fixed data split, ImageNet-pretrained FPN-ResNet50 initialization, 100-epoch optimization schedule, validation-AUPRO checkpoint selection, fixed threshold of 0.5, and a single Gaussian filtering pass with sigma=1.0. E1 added a continuous signed-distance term with lambda=0.1. E2 excluded BCE from the mixed-label 3 x 3 boundary band but retained original-mask Dice and Focal Tversky terms; it should be interpreted as a binary adaptation of the relaxation principle rather than a reproduction of the complete video-propagation framework in [17].

On the fixed test set, E0, E1, E2, and E3 achieved Recall values of {metric('E0','recall'):.4f}, {metric('E1','recall'):.4f}, {metric('E2','recall'):.4f}, and {metric('E3','recall'):.4f}, and normalized AUPRO values of {metric('E0','aupro'):.4f}, {metric('E1','aupro'):.4f}, {metric('E2','aupro'):.4f}, and {metric('E3','aupro'):.4f}, respectively. Precision, Dice, IoU, Pixel-AP, and Pixel-AUROC are reported together to expose sensitivity-specificity and boundary-coverage trade-offs. Because this minimal direct comparison used one training seed, the differences are descriptive and are not interpreted as statistical significance.

The one-pixel expansion is deliberately conservative. Low-contrast C-scan defects can have uncertain annotated boundaries and can occupy few pixels after resizing; expanding the training target by one pixel increases positive support and tolerates immediately adjacent predictions without substantially changing object geometry. The operation affects training supervision only and adds neither model parameters nor inference-time computation. Validation and test scoring always use the original non-expanded ground truth.

**Suggested table caption.** Controlled comparison of original-mask supervision, signed-distance Boundary Loss [16], binary boundary-label relaxation adapted from [17], and the proposed 3 x 3 training-label dilation on the fixed test set.
"""
    (HERE / "paper_insert_comparison.md").write_text(insert, encoding="utf-8")

    winner_aupro = max(METHODS, key=lambda x: metric(x, "aupro"))
    summary = f"""# RI027 最小直接对比执行总结

## 完成情况

已完成 E0--E3 四组 seed=42、各 100 epoch 的受控训练。四组使用相同固定划分、初始化权重、数据顺序、模型容量、优化器、验证集选模、阈值与 sigma=1.0 后处理。每组最佳 checkpoint 均在测试前固化 SHA-256，随后仅进行一次测试推理。实际结果见 `direct_comparison_table.md`；本次 AUPRO 最高的方法为 {winner_aupro}，该事实不外推为多随机种子显著性结论。

## 计算资源与环境

Windows 11，NVIDIA GeForce RTX 4060 8 GB；Python 3.10.20，PyTorch 2.7.1+cu118，SMP 0.5.1.dev0，SciPy 1.15.3。训练使用 CUDA AMP。Git 命令不可用且可见元数据没有可读取 HEAD，因此未编造 commit hash。

## 失败项和处理

第一次 E0 启动在 epoch 1 前因 WindowsPath 无法 JSON 序列化而停止，未访问测试集。失败目录已保留，公共代码修复后四组从相同初始状态正式运行。未发生计入结果的 OOM、NaN 或图像-mask 错配。

## 限制与潜在偏差

阶段 A 只有一个训练 seed；阶段 B 的 2026、3407 尚未运行，不能声称统计显著性或已满足三随机种子要求。测试集中的 Crack 和 Layer 仅有 9 和 2 张。E2 是边界标签松弛的二分类适配，不是 Zhu 等人完整视频传播模型的复现。新协议采用确定性 cuDNN，而历史训练启用了 benchmark，因此新结果不应被强制要求等于历史主表。

## 下一步

如计算预算允许，固定当前冻结协议补跑 seed=2026、3407，至少覆盖 E0/E3，最好覆盖全部四组，并报告全部 run 的均值和标准差。正式改稿前先人工核对本目录中的四行结果、checkpoint 哈希和英文回复，不自动覆盖原稿。
"""
    (HERE / "execution_summary_zh.md").write_text(summary, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--action", choices=["audit", "visualize", "selftest", "train", "summarize"], required=True)
    p.add_argument("--method", choices=list(METHODS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--boundary-lambda", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=HERE)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pairs = validate_splits(args.split_dir)
    if args.action == "audit": audit(args, pairs)
    elif args.action == "visualize": make_sanity_visuals(pairs)
    elif args.action == "selftest": selftest(args, pairs)
    elif args.action == "summarize": summarize()
    else:
        if args.method is None:
            raise ValueError("--method is required for --action train")
        run_method(args.method, args, pairs)


if __name__ == "__main__":
    main()
