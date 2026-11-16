"""
Класс для хранения отдельного измерения.
"""
from dataclasses import dataclass
from typing import Literal

Aspect = Literal["realism", "diversity", "preservation"]


@dataclass
class Quality:
    measurement_id: int
    metric_code: str
    aspect: Aspect
    numeric_value: float
    threshold: float
    verdict: bool

    @classmethod
    def create(cls, measurement_id: int, metric_code: str, aspect: Aspect,
               numeric_value: float, threshold: float,
               higher_is_better: bool = False) -> "Quality":
        """
        Фабричный метод для создания метрики с автоматическим вердиктом
        """
        if higher_is_better:
            verdict = numeric_value >= threshold
        else:
            verdict = numeric_value <= threshold

        return cls(
            measurement_id=measurement_id,
            metric_code=metric_code,
            aspect=aspect,
            numeric_value=float(numeric_value),
            threshold=float(threshold),
            verdict=verdict
        )