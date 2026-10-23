"""Filtragem das máscaras do SAM e conversão para polígonos (Fase 1)."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Hold:
    polygon: list[list[int]]
    bbox: list[int]  # [x, y, largura, altura]
    score: float


def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.005) -> list[list[int]] | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return approx.tolist() if len(approx) >= 3 else None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def filter_masks(
    masks: list[dict],
    image_shape: tuple[int, int],
    min_area_frac: float = 0.0001,
    max_area_frac: float = 0.05,
    max_aspect: float = 6.0,
    border_margin: int = 2,
    dup_iou: float = 0.5,
) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Devolve (mantidas, [(descartada, motivo)])."""
    h, w = image_shape
    total = h * w
    kept, dropped = [], []

    for m in masks:
        x, y, bw, bh = m["bbox"]
        touches_border = x <= border_margin or y <= border_margin or x + bw >= w - border_margin or y + bh >= h - border_margin
        if m["area"] < min_area_frac * total:
            dropped.append((m, "pequena"))
        elif m["area"] > max_area_frac * total:
            dropped.append((m, "grande"))
        elif max(bw, bh) / max(1, min(bw, bh)) > max_aspect:
            dropped.append((m, "formato"))
        elif touches_border and m["area"] > 0.01 * total:
            dropped.append((m, "borda"))
        else:
            kept.append(m)

    # remoção de duplicatas: maior score primeiro (estilo NMS)
    kept.sort(key=lambda m: -m["predicted_iou"])
    unique: list[dict] = []
    for m in kept:
        if any(_iou(m["segmentation"], u["segmentation"]) > dup_iou for u in unique):
            dropped.append((m, "duplicada"))
        else:
            unique.append(m)
    return unique, dropped


def to_holds(masks: list[dict]) -> list[Hold]:
    holds = []
    for m in masks:
        polygon = mask_to_polygon(m["segmentation"])
        if polygon:
            holds.append(Hold(polygon, [int(v) for v in m["bbox"]], float(m["predicted_iou"])))
    return holds
