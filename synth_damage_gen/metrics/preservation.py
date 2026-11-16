"""
Метрики сохранности сигнала (Preservation Metrics).
Оценивают, насколько хорошо сохранились участки изображения ВНЕ маски.
"""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_masked_psnr(clean: np.ndarray, damaged: np.ndarray, mask: np.ndarray) -> float:
    """
    PSNR только для пикселей вне маски (mask == 0).
    clean, damaged: [H, W, 3] uint8
    mask: [H, W] uint8 {0, 255}
    """
    # Инвертируем маску
    bg_mask = (mask == 0)

    if bg_mask.sum() == 0:
        return 100.0

    # Считаем PSNR только по фоновым пикселям
    diff = (clean[bg_mask].astype(np.float64) - damaged[bg_mask].astype(np.float64)) ** 2
    mse = np.mean(diff)

    if mse == 0:
        return 100.0
    return float(10 * np.log10((255.0 ** 2) / mse))


def calculate_masked_ssim(clean: np.ndarray, damaged: np.ndarray, mask: np.ndarray) -> float:
    """
    SSIM только для пикселей вне маски.
    Использует маску для взвешивания локальных карт SSIM
    """
    from skimage.metrics import structural_similarity as ssim

    # Считаем полную карту SSIM
    _, ssim_map = ssim(clean, damaged, channel_axis=2, full=True, data_range=255)

    # Усредняем только по фоновым пикселям
    bg_mask = (mask == 0)
    if bg_mask.sum() == 0:
        return 1.0

    # ssim_map имеет размер [H, W], усредняем по фону
    return float(ssim_map[bg_mask].mean())