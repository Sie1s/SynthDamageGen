"""Классический генератор трещин на основе морфологического закрытия"""

from __future__ import annotations
from typing import Any, Tuple
import cv2
import numpy as np
import torch

from .base import BaseMaskGenerator, Mask


class DeterministicCrackGenerator(BaseMaskGenerator):
    """
    Синтез трещин через морфологическое закрытие.
    Формула: y(x) = f_closed(x) - f(x), где f_closed - closing исходного поля
    """

    def __init__(self, hyperparams: dict[str, Any]):
        super().__init__("crack_morphological", hyperparams)

    def sample_mask(self, size: Tuple[int, int], rng: np.random.Generator) -> Mask:
        h, w = size
        thickness = int(self.hyperparams.get("crack_thickness", 3))
        iters = int(self.hyperparams.get("crack_iterations", 4))
        thr = float(self.hyperparams.get("threshold", 0.15))

        # Генерируем "гладкое" фоновое поле (имитация текстуры)
        base = rng.normal(0.6, 0.05, size=(h, w)).astype(np.float32)

        # Рисуем случайные темные линии (кандидаты в трещины)
        n_cracks = rng.integers(1, 4)
        for _ in range(n_cracks):
            pts = self._random_polyline(h, w, rng, n_segments=6)
            cv2.polylines(base, [pts], False, color=float(rng.uniform(0.0, 0.2)), thickness=thickness)

        # Морфологическое закрытие (nB)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=iters)

        # Вычитаем исходное и бинаризуем
        diff = np.clip(closed - base, 0.0, 1.0)
        mask = (diff > thr).astype(np.uint8)

        area, thick_avg, tort = self._shape_stats(mask)

        return Mask(
            raster=torch.from_numpy(mask).float().unsqueeze(0),
            degradation_code=self.degradation_code,
            area_share=area,
            thickness_avg=thick_avg,
            tortuosity=tort,
            hyperparams={"thickness": thickness, "iterations": iters, "threshold": thr},
        )

    @staticmethod
    def _random_polyline(h: int, w: int, rng: np.random.Generator, n_segments: int) -> np.ndarray:
        """Генерирует случайную ломаную линию"""
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        pts = [(x0, y0)]
        x, y = x0, y0
        for _ in range(n_segments):
            x = int(np.clip(x + rng.integers(-w // 5, w // 5), 0, w - 1))
            y = int(np.clip(y + rng.integers(-h // 5, h // 5), 0, h - 1))
            pts.append((x, y))
        return np.array(pts, dtype=np.int32)