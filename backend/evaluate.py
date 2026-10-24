"""Avaliação da Fase 4: compara as agarras de um detector com o gabarito de data/test (IoU de máscara).

Critério de casamento:
- Previsões em ordem de score; cada uma casa com a anotação ainda livre de maior IoU (>= --iou).
- Casou com `hold` → TP. Casou com `tape`/`volume` → ignorada (nem acerto nem erro, só contada).
- Não casou com nada → FP. Agarra anotada que sobrou → FN.

Uso: python -m backend.evaluate [--data data/test] [--iou 0.5] [--out data/out/eval]
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from backend.detectors import Detector, SamDetector

HOLD = "hold"
COLORS = {"tp": (0, 200, 0), "fp": (0, 220, 255), "fn": (0, 0, 255), "ignored": (255, 120, 0)}  # BGR


@dataclass
class Region:
    """Máscara recortada no próprio bbox (máscaras na resolução cheia não caberiam na memória)."""
    polygon: np.ndarray  # (N, 2) em pixels
    x: int
    y: int
    mask: np.ndarray  # (h, w) bool
    label: str = HOLD
    score: float = 1.0


def make_region(polygon: np.ndarray, label: str = HOLD, score: float = 1.0) -> Region:
    pts = np.round(polygon).astype(np.int32)
    x, y = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    mask = np.zeros((y2 - y + 1, x2 - x + 1), np.uint8)
    cv2.fillPoly(mask, [pts - [x, y]], 1)
    return Region(pts, int(x), int(y), mask.astype(bool), label, score)


def region_iou(a: Region, b: Region) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2 = min(a.x + a.mask.shape[1], b.x + b.mask.shape[1])
    y2 = min(a.y + a.mask.shape[0], b.y + b.mask.shape[0])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = np.logical_and(
        a.mask[y1 - a.y:y2 - a.y, x1 - a.x:x2 - a.x],
        b.mask[y1 - b.y:y2 - b.y, x1 - b.x:x2 - b.x],
    ).sum()
    union = a.mask.sum() + b.mask.sum() - inter
    return float(inter / union) if union else 0.0


def load_gt(label_path: Path, names: list[str], w: int, h: int) -> list[Region]:
    regions = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        pts = np.array(parts[1:], float).reshape(-1, 2) * [w, h]
        regions.append(make_region(pts, names[int(parts[0])]))
    return regions


def match(preds: list[Region], gt: list[Region], iou_min: float) -> dict:
    """Casa previsões com anotações; devolve as listas de cada categoria e os IoUs dos acertos."""
    used: set[int] = set()
    res = {"tp": [], "fp": [], "ignored": [], "ious": []}
    for p in sorted(preds, key=lambda r: -r.score):
        best, best_iou = None, 0.0
        for i, g in enumerate(gt):
            if i not in used and (iou := region_iou(p, g)) > best_iou:
                best, best_iou = i, iou
        if best is None or best_iou < iou_min:
            res["fp"].append(p)
            continue
        used.add(best)
        if gt[best].label == HOLD:
            res["tp"].append(gt[best])
            res["ious"].append(best_iou)
        else:
            res["ignored"].append(gt[best])
    res["fn"] = [g for i, g in enumerate(gt) if i not in used and g.label == HOLD]
    return res


def draw(img_bgr: np.ndarray, res: dict) -> np.ndarray:
    """Verde = acerto, vermelho = agarra não achada, amarelo = falso positivo, azul = fita/volume."""
    out = img_bgr.copy()
    thickness = max(2, max(img_bgr.shape[:2]) // 800)
    for kind in ("ignored", "tp", "fp", "fn"):
        cv2.polylines(out, [r.polygon for r in res[kind]], True, COLORS[kind], thickness)
    return out


def evaluate(detector: Detector, data_dir: Path, iou_min: float, out_dir: Path | None) -> None:
    names = yaml.safe_load((data_dir / "data.yaml").read_text())["names"]
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    totals = {"tp": 0, "fp": 0, "fn": 0, "ious": [], "ignored": {}}
    print(f"{'imagem':<24} {'gt':>4} {'prev':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'ign':>4} {'prec':>5} {'rec':>5} {'tempo':>6}")
    for img_path in sorted((data_dir / "images").glob("*.jpg")):
        img_bgr = cv2.imread(str(img_path))
        h, w = img_bgr.shape[:2]
        gt = load_gt(data_dir / "labels" / f"{img_path.stem}.txt", names, w, h)

        start = time.perf_counter()
        holds = detector.detect(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        elapsed = time.perf_counter() - start
        preds = [make_region(np.array(hd.polygon), score=hd.score) for hd in holds]

        res = match(preds, gt, iou_min)
        tp, fp, fn = len(res["tp"]), len(res["fp"]), len(res["fn"])
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["ious"] += res["ious"]
        for g in res["ignored"]:
            totals["ignored"][g.label] = totals["ignored"].get(g.label, 0) + 1

        n_gt = tp + fn
        print(f"{img_path.stem[:24]:<24} {n_gt:>4} {len(preds):>5} {tp:>4} {fp:>4} {fn:>4} {len(res['ignored']):>4} "
              f"{tp / max(1, tp + fp):>5.2f} {tp / max(1, n_gt):>5.2f} {elapsed:>5.1f}s")
        if out_dir:
            cv2.imwrite(str(out_dir / f"{img_path.stem[:24]}.jpg"), draw(img_bgr, res))

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = float(np.mean(totals["ious"])) if totals["ious"] else 0.0
    print(f"\nTotal: TP {tp}  FP {fp}  FN {fn}")
    print(f"precision {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}  IoU médio (acertos) {mean_iou:.3f}")
    print(f"ignoradas (casaram com outra classe): {totals['ignored'] or 'nenhuma'}")
    if out_dir:
        print(f"imagens em {out_dir}/ (verde=TP, vermelho=FN, amarelo=FP, azul=ignorada)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/test")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", default="data/out/eval", help="'' para não salvar imagens")
    args = parser.parse_args()
    evaluate(SamDetector(), Path(args.data), args.iou, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
