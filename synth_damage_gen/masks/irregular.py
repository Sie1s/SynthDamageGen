"""Генератор нерегулярных повреждений (штрихи, пятна, контакт с границей)"""

from __future__ import annotations
from typing import Any, Tuple
import cv2
import numpy as np
import torch

from .base import BaseMaskGenerator, Mask


class IrregularMaskGenerator(BaseMaskGenerator):
    """
    Случайные штрихи и отверстия произвольной формы
    """

    def __init__(self, hyperparams: dict[str, Any]):
        super().__init__("irregular_artifact", hyperparams)

    def sample_mask(self, size: Tuple[int, int], rng: np.random.Generator) -> Mask:
        h, w = size
        area_lo, area_hi = self.hyperparams.get("area_range", (0.01, 0.30))
        n_strokes = int(self.hyperparams.get("n_strokes", 12))
        dil = int(self.hyperparams.get("dilation", 2))
        p_bound = float(self.hyperparams.get("boundary_contact_prob", 0.4))

        mask = np.zeros((h, w), dtype=np.uint8)
        target_area = rng.uniform(area_lo, area_hi) * h * w

        # Отверстия (пятна)
        n_circles = rng.integers(1, 4)
        area_per_circle = target_area / n_circles
        for _ in range(n_circles):
            cx, cy = rng.integers(0, w), rng.integers(0, h)
            r = int(np.sqrt(area_per_circle / np.pi) * rng.uniform(0.5, 1.0))
            cv2.circle(mask, (cx, cy), max(1, r), 1, -1)

        # Штрихи (линии)
        for _ in range(n_strokes):
            if rng.random() < p_bound:
                # Привязка к границе
                side = rng.integers(0, 4)
                if side == 0:
                    x, y = 0, rng.integers(0, h)
                elif side == 1:
                    x, y = w - 1, rng.integers(0, h)
                elif side == 2:
                    x, y = rng.integers(0, w), 0
                else:
                    x, y = rng.integers(0, w), h - 1
            else:
                x, y = rng.integers(0, w), rng.integers(0, h)

            pts = [(x, y)]
            for _ in range(rng.integers(2, 5)):
                x = int(np.clip(x + rng.integers(-w // 5, w // 5), 0, w - 1))
                y = int(np.clip(y + rng.integers(-h // 5, h // 5), 0, h - 1))
                pts.append((x, y))
            cv2.polylines(mask, [np.array(pts, np.int32)], False, 1, thickness=int(rng.integers(1, 4)))

        # Дилатация (утолщение) и поворот
        if dil > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)

        if rng.random() < 0.5:
            angle = rng.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

        area, thick_avg, tort = self._shape_stats(mask)

        return Mask(
            raster=torch.from_numpy(mask).float().unsqueeze(0),
            degradation_code=self.degradation_code,
            area_share=area,
            thickness_avg=thick_avg,
            tortuosity=tort,
            hyperparams={"n_strokes": n_strokes, "dilation": dil, "area_range": (area_lo, area_hi)},
        )