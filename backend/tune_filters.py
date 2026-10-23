"""Mede recall/precision do SAM + filtros contra as anotações do dataset (por IoU de caixa).

Gera as máscaras uma vez e testa várias combinações de parâmetros dos filtros.
Uso: python -m backend.tune_filters [--n 24]
"""
import argparse
import itertools
from pathlib import Path

import cv2
import torch

from backend.holds import filter_masks
from backend.try_sam import build_generator

IMAGES = Path("data/train/images")
LABELS = Path("data/train/labels")
VOLUME_CLASS = 9


def load_gt(label_path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    """Caixas [x, y, w, h] em pixels das agarras (volumes ignorados)."""
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts or int(parts[0]) == VOLUME_CLASS:
            continue
        v = list(map(float, parts[1:]))
        if len(v) == 4:  # caixa: cx cy w h
            cx, cy, bw, bh = v
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h))
        else:  # polígono
            xs, ys = v[0::2], v[1::2]
            boxes.append((min(xs) * w, min(ys) * h, (max(xs) - min(xs)) * w, (max(ys) - min(ys)) * h))
    return boxes


def box_iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def score(pred_boxes, gt_boxes, iou_min=0.5) -> tuple[int, int, int]:
    used, tp = set(), 0
    for p in pred_boxes:
        best = max((g for g in range(len(gt_boxes)) if g not in used),
                   key=lambda g: box_iou(p, gt_boxes[g]), default=None)
        if best is not None and box_iou(p, gt_boxes[best]) >= iou_min:
            used.add(best)
            tp += 1
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=24)
    args = parser.parse_args()

    files = sorted(IMAGES.glob("*.jpg"))
    files = files[:: max(1, len(files) // args.n)][: args.n]

    generator = build_generator()
    data = []
    for f in files:
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = generator.generate(img)
        label = LABELS / (f.stem + ".txt")
        data.append((img.shape[:2], masks, load_gt(label, img.shape[1], img.shape[0])))
    print(f"{len(data)} imagens, {sum(len(d[2]) for d in data)} agarras anotadas\n")

    grid = itertools.product(
        [0.0001, 0.0003, 0.001],      # min_area_frac
        [0.02, 0.05, 0.1],            # max_area_frac
        [0.5, 0.7, 0.9],              # dup_iou
    )
    rows = []
    for min_a, max_a, dup in grid:
        tp = fp = fn = 0
        for shape, masks, gt in data:
            kept, _ = filter_masks(masks, shape, min_area_frac=min_a, max_area_frac=max_a, dup_iou=dup)
            a, b, c = score([m["bbox"] for m in kept], gt)
            tp, fp, fn = tp + a, fp + b, fn + c
        p, r = tp / (tp + fp), tp / (tp + fn)
        rows.append((2 * p * r / (p + r), p, r, min_a, max_a, dup))
    print("   F1  prec  rec | min_area max_area dup_iou")
    for f1, p, r, *params in sorted(rows, reverse=True)[:8]:
        print(f"{f1:.3f} {p:.3f} {r:.3f} | {params[0]:<8} {params[1]:<8} {params[2]}")
    base = next(r for r in rows if r[3:] == (0.0003, 0.05, 0.7))
    print(f"\nreferência (0.0003, 0.05, 0.7): F1 {base[0]:.3f} prec {base[1]:.3f} rec {base[2]:.3f}")


if __name__ == "__main__":
    main()
