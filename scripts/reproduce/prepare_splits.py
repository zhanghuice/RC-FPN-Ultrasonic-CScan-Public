"""Rebase a privately held fixed split onto an authorized local dataset.

The manuscript split list is intentionally not distributed because it contains
restricted dataset filenames. This script never generates or changes membership.
"""
from __future__ import annotations
import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--archived",
        type=Path,
        required=True,
        help="Private directory containing train.txt, val.txt, and test.txt",
    )
    parser.add_argument("--output", type=Path, default=Path("data/splits/portable"))
    args = parser.parse_args()
    image_dir, mask_dir = args.data_root / "images", args.data_root / "masks"
    if not args.archived.is_dir():
        raise FileNotFoundError(f"Private split directory not found: {args.archived}")
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = []
        for line in (args.archived / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            image, mask = line.split("\t")
            new_image, new_mask = image_dir / Path(image).name, mask_dir / Path(mask).name
            if not new_image.is_file() or not new_mask.is_file():
                raise FileNotFoundError(f"Missing rebased pair: {new_image} / {new_mask}")
            if new_image.stem.lower() != new_mask.stem.lower():
                raise ValueError(f"Stem mismatch: {new_image} / {new_mask}")
            rows.append(f"{new_image.resolve()}\t{new_mask.resolve()}")
        (args.output / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"{split}: {len(rows)}")

if __name__ == "__main__":
    main()
