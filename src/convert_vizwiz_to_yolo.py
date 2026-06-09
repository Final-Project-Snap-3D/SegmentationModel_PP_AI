"""
One-shot script: convert VizWiz salient-object JSON annotations to YOLO
segmentation format, then copy images into a YOLO-compatible layout.

Output layout (sibling to data/):
    data_yolo/
    ├── images/
    │   ├── train/   (copied .jpg)
    │   └── val/
    └── labels/
        ├── train/   (.txt, one line per polygon)
        └── val/

YOLO segmentation .txt format, one polygon per line:
    class_id x1 y1 x2 y2 ... xn yn      (all coords normalized to [0, 1])
"""
import argparse
import json
import shutil
from pathlib import Path


def convert_split(images_dir: Path, annotations_path: Path,
                  out_images_dir: Path, out_labels_dir: Path) -> tuple[int, int]:
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for img_path in images_dir.iterdir():
        if not img_path.is_file():
            continue
        name = img_path.name
        if name not in annotations:
            skipped += 1
            continue

        ann = annotations[name]
        if "Salient Object" not in ann or "Ground Truth Dimensions" not in ann:
            skipped += 1
            continue

        h, w = ann["Ground Truth Dimensions"]
        polygons = ann["Salient Object"]

        lines = []
        for poly in polygons:
            if len(poly) < 3:
                continue
            coords = []
            for x, y in poly:
                coords.append(f"{x / w:.6f}")
                coords.append(f"{y / h:.6f}")
            lines.append("0 " + " ".join(coords))

        if not lines:
            skipped += 1
            continue

        shutil.copy2(img_path, out_images_dir / name)
        label_path = out_labels_dir / (img_path.stem + ".txt")
        label_path.write_text("\n".join(lines))
        converted += 1

    return converted, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".", type=str)
    parser.add_argument("--train_images", default="data/train")
    parser.add_argument("--val_images", default="data/val")
    parser.add_argument("--train_ann", default="data/annotations/VizWiz_SOD_train_challenge.json")
    parser.add_argument("--val_ann", default="data/annotations/VizWiz_SOD_val_challenge.json")
    parser.add_argument("--out_dir", default="data_yolo")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out = root / args.out_dir

    print(f"Converting train split...")
    n, s = convert_split(
        root / args.train_images,
        root / args.train_ann,
        out / "images" / "train",
        out / "labels" / "train",
    )
    print(f"  Train: {n} converted, {s} skipped")

    print(f"Converting val split...")
    n, s = convert_split(
        root / args.val_images,
        root / args.val_ann,
        out / "images" / "val",
        out / "labels" / "val",
    )
    print(f"  Val: {n} converted, {s} skipped")

    yaml_path = out / "vizwiz.yaml"
    yaml_path.write_text(
        f"path: {out.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: salient_object\n"
    )
    print(f"\nDataset config written to: {yaml_path}")


if __name__ == "__main__":
    main()
