"""Detectores de agarras. A API depende só da interface `Detector`, então SAM e YOLO são intercambiáveis."""
from abc import ABC, abstractmethod

import numpy as np
import torch

from backend.holds import Hold, filter_masks, to_holds
from backend.try_sam import build_generator


class Detector(ABC):
    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> list[Hold]:
        """Recebe uma imagem RGB (H, W, 3) e devolve as agarras encontradas."""


class SamDetector(Detector):
    def __init__(self) -> None:
        self._generator = build_generator()  # carrega o modelo na GPU (lento: fazer uma vez)

    def detect(self, image_rgb: np.ndarray) -> list[Hold]:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = self._generator.generate(image_rgb)
        kept, _ = filter_masks(masks, image_rgb.shape[:2])
        return to_holds(kept)
