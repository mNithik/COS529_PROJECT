from __future__ import annotations

import argparse
import os
from pathlib import Path


def clip_box(xc: float, yc: float, w: float, h: float) -> tuple[float, float, float, float] | None:
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0

    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)

    new_w = x2 - x1
    new_h = y2 - y1
    if new_w <= 0.0 or new_h <= 0.0:
        return None

    new_xc = (x1 + x2) / 2.0
    new_yc = (y1 + y2) / 2.0
    return new_xc, new_yc, new_w, new_h


def ensure_image_link(source_images: Path, target_images: Path) -> None:
    if target_images.exists():
        return
    target_images.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source_images, target_images, target_is_directory=True)


def sanitize_labels(source_labels: Path, target_labels: Path) -> tuple[int, int, int]:
    target_labels.mkdir(parents=True, exist_ok=True)
    files_written = 0
    boxes_kept = 0
    boxes_dropped = 0

    for src_file in sorted(source_labels.glob("*.txt")):
        out_lines = []
        for raw_line in src_file.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) != 5:
                boxes_dropped += 1
                continue
            try:
                _, xc, yc, w, h = map(float, parts)
            except ValueError:
                boxes_dropped += 1
                continue

            clipped = clip_box(xc, yc, w, h)
            if clipped is None:
                boxes_dropped += 1
                continue

            nxc, nyc, nw, nh = clipped
            out_lines.append(f"0 {nxc:.6f} {nyc:.6f} {nw:.6f} {nh:.6f}")
            boxes_kept += 1

        (target_labels / src_file.name).write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        files_written += 1

    return files_written, boxes_kept, boxes_dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sanitized single-class Phase 1 VEDAI dataset view.")
    parser.add_argument("--source-root", type=Path, required=True, help="Source VEDAI dataset root with images/ and labels/.")
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/phase1_vedai"),
        help="Target root for the derived sanitized Phase 1 dataset.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    source_images = source_root / "images"
    source_labels = source_root / "labels"
    target_images = target_root / "images"
    target_labels = target_root / "labels"

    if not source_images.exists() or not source_labels.exists():
        raise FileNotFoundError(f"Source dataset root is incomplete: {source_root}")

    ensure_image_link(source_images, target_images)
    files_written, boxes_kept, boxes_dropped = sanitize_labels(source_labels, target_labels)

    print(f"Prepared sanitized Phase 1 dataset at {target_root}")
    print(f"Label files written: {files_written}")
    print(f"Boxes kept: {boxes_kept}")
    print(f"Boxes dropped: {boxes_dropped}")


if __name__ == "__main__":
    main()
