"""Check local split files without publishing filenames or data hashes."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_rows(path: Path) -> list[tuple[Path, Path]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        image = Path(fields[0])
        mask = Path(fields[1]) if len(fields) > 1 else image.parent.parent / "masks" / image.name
        rows.append((image, mask))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True, type=Path)
    args = parser.parse_args()
    membership: dict[str, set[str]] = defaultdict(set)
    pair_membership: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts = {}
    for split in ("train", "val", "test"):
        rows = load_rows(args.split_dir / f"{split}.txt")
        counts[split] = len(rows)
        for image, mask in rows:
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"Missing local input referenced by {split}.txt")
            image_hash, mask_hash = digest(image), digest(mask)
            membership[image_hash].add(split)
            pair_membership[(image_hash, mask_hash)].add(split)
    image_groups = sum(len(parts) > 1 for parts in membership.values())
    pair_groups = sum(len(parts) > 1 for parts in pair_membership.values())
    print({"counts": counts, "cross_split_exact_image_groups": image_groups,
           "cross_split_exact_image_and_mask_groups": pair_groups})
    if image_groups:
        raise SystemExit("FAILED: content-level leakage exists; do not train/evaluate this split")
    print("PASS: no byte-exact image content is shared across splits")


if __name__ == "__main__":
    main()
