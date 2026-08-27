"""Convert simple pixel-box CSV annotations into YOLO .txt label files.

If you don't already have an annotation tool, this lets you annotate in
whatever's fastest for you (a spreadsheet, a quick script, Roboflow/CVAT/
LabelImg exported to CSV, etc.) and convert once at the end, instead of
hand-writing YOLO's normalized format.

Expected CSV columns (header row required):
    filename,x1,y1,x2,y2
        filename  - image file name, must exist in the images folder you pass in
        x1,y1     - top-left corner of the stem/crown box, in pixels
        x2,y2     - bottom-right corner of the box, in pixels

One row per stem. If a photo has no visible stem, either omit it from the
CSV entirely, or include it with an empty box (x1=x2=y1=y2=0) - both are
treated as "no label", which is fine for YOLO (it just won't be forced to
detect anything in that image, i.e. it's a useful negative example).

Usage:
    python csv_to_yolo_labels.py annotations.csv dataset/images/train dataset/labels/train
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


def image_size(path: Path) -> tuple[int, int]:
    if cv2 is not None:
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    from PIL import Image  # fallback if opencv isn't available

    with Image.open(path) as im:
        return im.size  # (w, h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("labels_dir", type=Path)
    args = parser.parse_args()

    args.labels_dir.mkdir(parents=True, exist_ok=True)

    rows_by_image: dict[str, list[tuple[float, float, float, float]]] = {}
    with args.csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = {"filename", "x1", "y1", "x2", "y2"} - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            if x2 <= x1 or y2 <= y1:
                continue  # empty/placeholder box -> no label for this row
            rows_by_image.setdefault(row["filename"], []).append((x1, y1, x2, y2))

    written, skipped = 0, 0
    for image_path in sorted(args.images_dir.iterdir()):
        if not image_path.is_file():
            continue
        label_path = args.labels_dir / (image_path.stem + ".txt")
        boxes = rows_by_image.get(image_path.name, [])
        if not boxes:
            # No stem labeled for this image: write an empty label file so
            # YOLO treats it as a valid negative example rather than
            # silently skipping it.
            label_path.write_text("")
            skipped += 1
            continue

        w, h = image_size(image_path)
        lines = []
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label_path.write_text("\n".join(lines) + "\n")
        written += 1

    print(f"Wrote {written} label file(s) with boxes, {skipped} empty/negative label file(s).")
    print(f"Labels written to: {args.labels_dir}")


if __name__ == "__main__":
    main()
