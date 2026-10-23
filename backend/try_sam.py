"""Experimento da Fase 1: gera máscaras automáticas do SAM 2 e salva uma imagem de debug.

Uso:
    python -m backend.try_sam data/train/images/foto.jpg [--out data/out]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from backend.holds import filter_masks, to_holds

CHECKPOINT = "models/sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


GENERATOR_DEFAULTS = dict(
    points_per_side=64,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.85,
    min_mask_region_area=50,
)


def build_generator(**overrides) -> SAM2AutomaticMaskGenerator:
    model = build_sam2(CONFIG, CHECKPOINT, device="cuda", apply_postprocessing=False)
    return SAM2AutomaticMaskGenerator(model, **{**GENERATOR_DEFAULTS, **overrides})


def draw_debug(img_rgb: np.ndarray, kept: list[dict], dropped: list[tuple[dict, str]]) -> np.ndarray:
    """Mantidas em verde, descartadas em vermelho (com o motivo)."""
    overlay = img_rgb.copy()
    for m, _ in dropped:
        overlay[m["segmentation"]] = (255, 0, 0)
    for m in kept:
        overlay[m["segmentation"]] = (0, 255, 0)
    out = cv2.addWeighted(img_rgb, 0.55, overlay, 0.45, 0)
    for group, color in ((kept, (255, 255, 255)), ([m for m, _ in dropped], (255, 255, 0))):
        for m in group:
            contours, _ = cv2.findContours(
                m["segmentation"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, color, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default="data/out")
    args = parser.parse_args()

    img_rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    generator = build_generator()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks = generator.generate(img_rgb)
    kept, dropped = filter_masks(masks, img_rgb.shape[:2])
    reasons = {}
    for _, why in dropped:
        reasons[why] = reasons.get(why, 0) + 1
    holds = to_holds(kept)
    print(f"{len(masks)} máscaras → {len(holds)} agarras (descartadas: {reasons})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.image).stem[:30]}_sam.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(draw_debug(img_rgb, kept, dropped), cv2.COLOR_RGB2BGR))
    print("salvo em", out_path)


if __name__ == "__main__":
    main()
