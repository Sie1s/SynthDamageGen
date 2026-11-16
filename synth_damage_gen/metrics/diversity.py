"""
Метрики разнообразия (Diversity Metrics).
Оценивают, насколько разнообразны сгенерированные маски по сравнению с целевым распределением (или между собой)
"""
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from typing import List


def calculate_wasserstein(dist1: List[float], dist2: List[float]) -> float:
    """
    Расстояние Вассерштейна (Earth Mover's Distance) между двумя распределениями.
    Показывает, насколько "дорого" переделать одно распределение в другое.
    Чем меньше значение, тем ближе распределения.
    """
    return float(wasserstein_distance(dist1, dist2))


def calculate_ks_test(dist1: List[float], dist2: List[float]) -> tuple[float, float]:
    """
    Двухвыборочный тест Колмогорова-Смирнова.
    Возвращает (statistic, p_value).
    Если p_value > 0.05, то распределения статистически неразличимы.
    """
    res = ks_2samp(dist1, dist2)
    return float(res.statistic), float(res.pvalue)


def evaluate_mask_diversity(synthetic_masks: list, real_masks: list | None = None) -> dict:
    """
    Оценивает разнообразие масок по их геометрическим параметрам.
    synthetic_masks: список объектов Mask.
    real_masks: опционально, список реальных масок для сравнения.
    """
    # Собираем параметры
    synth_areas = [m.area_share for m in synthetic_masks]
    synth_thickness = [m.thickness_avg for m in synthetic_masks]
    synth_tortuosity = [m.tortuosity for m in synthetic_masks]

    results = {
        "synth_area_mean": float(np.mean(synth_areas)),
        "synth_area_std": float(np.std(synth_areas)),
        "synth_thickness_mean": float(np.mean(synth_thickness)),
        "synth_tortuosity_mean": float(np.mean(synth_tortuosity)),
    }

    # Если есть реальные маски, сравниваем распределения
    if real_masks:
        real_areas = [m.area_share for m in real_masks]

        w1_area = calculate_wasserstein(synth_areas, real_areas)
        ks_stat, ks_pval = calculate_ks_test(synth_areas, real_areas)

        results["w1_area"] = w1_area
        results["ks_stat"] = ks_stat
        results["ks_pvalue"] = ks_pval

    return results