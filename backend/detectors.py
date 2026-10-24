"""Detectores de agarras. A API depende só da interface `Detector`, então SAM e YOLO são intercambiáveis."""
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

from backend.holds import Hold, filter_masks, to_holds
from backend.try_sam import build_generator


class Detector(ABC):
    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> list[Hold]:
        """Recebe uma imagem RGB (H, W, 3) e devolve as agarras encontradas."""


class SamDetector(Detector):
    def __init__(self, max_side: int = 1024) -> None:
        self._generator = build_generator()  # carrega o modelo na GPU (lento: fazer uma vez)
        # O SAM roda internamente em 1024 px, mas devolve cada máscara na resolução da entrada:
        # numa foto de celular (4000x1848) isso leva ~9 min e enche a VRAM. Reduzimos antes.
        self.max_side = max_side

    def detect(self, image_rgb: np.ndarray) -> list[Hold]:
        scale = min(1.0, self.max_side / max(image_rgb.shape[:2]))
        small = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image_rgb
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = self._generator.generate(small)
        kept, _ = filter_masks(masks, small.shape[:2])
        holds = to_holds(kept)
        if scale < 1:  # polígonos e bbox de volta para as coordenadas da imagem original
            for h in holds:
                h.polygon = [[round(x / scale), round(y / scale)] for x, y in h.polygon]
                h.bbox = [round(v / scale) for v in h.bbox]
        return holds
