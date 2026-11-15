"""Фасад для управления стратегиями"""

from __future__ import annotations
from typing import Literal, Tuple
import numpy as np

from .base import BaseMaskGenerator, Mask
from .deterministic import DeterministicCrackGenerator
from .irregular import IrregularMaskGenerator
from ..config import MaskConfig

Approach = Literal["deterministic", "irregular"]


class MaskGenerator:
    """
    Фасад над конкретными стратегиями синтеза масок
    """

    def __init__(self, cfg: MaskConfig, approach: Approach | None = None):
        self.cfg = cfg
        self.approach: Approach = approach or cfg.default_generator  # type: ignore
        self._impl: BaseMaskGenerator = self._build(self.approach)

    def _build(self, approach: Approach) -> BaseMaskGenerator:
        if approach == "deterministic":
            return DeterministicCrackGenerator(vars(self.cfg.deterministic))
        if approach == "irregular":
            irr = self.cfg.irregular
            return IrregularMaskGenerator({
                "area_range": irr.area_range,
                "n_strokes": irr.n_strokes,
                "dilation": irr.dilation,
                "boundary_contact_prob": irr.boundary_contact_prob,
            })
        raise ValueError(f"Неизвестный подход к генерации масок: {approach}")

    def fit(self, reference_masks: list[Mask] | None = None) -> "MaskGenerator":
        self._impl.fit(reference_masks)
        return self

    def reset(self) -> None:
        self._impl.reset()

    def sample_mask(self, size: Tuple[int, int], seed: int | None = None) -> Mask:
        rng = np.random.default_rng(seed)
        return self._impl.sample_mask(size, rng)