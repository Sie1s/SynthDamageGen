"""
Метрики реалистичности (Realism Metrics).
Оценивают, насколько сгенерированные изображения похожи на реальные и насколько они естественны с точки зрения человеческого восприятия.
"""
import torch
import lpips
import pyiqa
from cleanfid import fid
from pathlib import Path


class RealismMetrics:
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Инициализация LPIPS (использует веса AlexNet по умолчанию)
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()

        # Инициализация NIQE (No-Reference)
        self.niqe_fn = pyiqa.create_metric('niqe', device=device)

    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Считает LPIPS между двумя тензорами [1, 3, H, W] в диапазоне [-1, 1].
        Чем меньше значение, тем более похожи изображения
        """
        with torch.no_grad():
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            dist = self.lpips_fn(img1, img2)
        return dist.item()

    def calculate_niqe(self, img: torch.Tensor) -> float:
        """
        Считает NIQE для тензора [1, 3, H, W] в диапазоне [0, 1].
        Чем меньше значение, тем естественнее выглядит изображение
        """
        with torch.no_grad():
            img = img.to(self.device)
            score = self.niqe_fn(img)
        return score.item()

    @staticmethod
    def calculate_fid(real_dir: str, fake_dir: str, device: str = "cpu") -> float:
        """
        Считает FID между двумя папками с изображениями.
        real_dir: путь к папке с оригинальными фото
        fake_dir: путь к папке со сгенерированными фото
        """
        score = fid.compute_fid(
            real_dir,
            fake_dir,
            device=torch.device(device),
            use_dataparallel=(device == "cuda")
        )
        return score