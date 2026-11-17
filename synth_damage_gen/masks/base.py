"""Базовые классы и вычисление статистик формы"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple
import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize


@dataclass
class Mask:
    """Бинарная маска дефекта + геометрические параметры"""
    raster: torch.Tensor            # [1, H, W] в {0.0, 1.0}
    degradation_code: str
    area_share: float
    thickness_avg: float
    tortuosity: float
    hyperparams: dict[str, Any]

    def rasterize(self) -> np.ndarray:
        """Возвращает numpy массив [H, W] с значениями {0, 1}"""
        return self.raster.squeeze(0).cpu().numpy().astype(np.uint8)

    def compare_with(self, other: "Mask") -> dict[str, float]:
        """Сравнение двух масок по геометрическим параметрам"""
        return {
            "d_area": abs(self.area_share - other.area_share),
            "d_thickness": abs(self.thickness_avg - other.thickness_avg),
            "d_tortuosity": abs(self.tortuosity - other.tortuosity),
            "iou": self._iou(other),
        }

    def _iou(self, other: "Mask") -> float:
        """Intersection over Union"""
        a = self.raster.bool()
        b = other.raster.bool()
        inter = (a & b).sum().item()
        union = (a | b).sum().item() + 1e-8
        return inter / union


class BaseMaskGenerator(ABC):
    """Базовый интерфейс генератора масок"""

    def __init__(self, degradation_code: str, hyperparams: dict[str, Any]):
        self.degradation_code = degradation_code
        self.hyperparams = dict(hyperparams)
        self._is_fitted = False

    def fit(self, reference_masks: list[Mask] | None = None) -> "BaseMaskGenerator":
        """Оценка статистических параметров по эталонным маскам (опционально)"""
        self._is_fitted = True
        return self

    def reset(self) -> None:
        self._is_fitted = False

    @abstractmethod
    def sample_mask(self, size: Tuple[int, int], rng: np.random.Generator) -> Mask:
        """Сгенерировать маску заданного размера"""
        ...

    @staticmethod
    def _shape_stats(raster: np.ndarray) -> Tuple[float, float, float]:
        return calculate_shape_stats(raster)

def calculate_shape_stats(raster: np.ndarray) -> tuple[float, float, float]:
    """Публичная функция для расчета геометрических параметров маски."""
    area_share = float(raster.mean())
    if raster.sum() == 0:
        return area_share, 0.0, 1.0

    dt = cv2.distanceTransform(raster.astype(np.uint8), cv2.DIST_L2, 5)
    thickness_avg = float(dt[raster > 0].mean()) * 2.0

    skel = skeletonize(raster > 0)
    ys, xs = np.where(skel)
    if len(xs) < 2:
        tortuosity = 1.0
    else:
        p1 = np.array([ys[0], xs[0]])
        p2 = np.array([ys[-1], xs[-1]])
        euclid_dist = float(np.linalg.norm(p1 - p2)) + 1e-8
        tortuosity = float(len(xs) / euclid_dist)

    return area_share, thickness_avg, tortuosity