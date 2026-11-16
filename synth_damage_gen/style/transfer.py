"""
Модуль применения обученной CycleGAN.
Загружает веса генератора G_AB и накладывает дефекты на изображения
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from .networks import ResnetGenerator


class StyleTransfer:
    def __init__(self, weights_path: str, device: str):
        self.device = device
        self.G_AB = ResnetGenerator(input_nc=4, output_nc=3).to(device)

        checkpoint_path = Path(weights_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Веса не найдены по пути: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.G_AB.load_state_dict(checkpoint['G_AB'])
        self.G_AB.eval()  # Переводим в режим тестирования (отключаем Dropout)
        print(f"[INFO] Веса успешно загружены из {checkpoint_path}")

    @torch.no_grad()
    def apply_transfer(self, clean_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Накладывает дефект на чистое изображение.
        clean_img: [1, 3, H, W] в диапазоне [-1, 1]
        mask: [1, 1, H, W] в диапазоне [0, 1]
        Возвращает: [1, 3, H, W] в диапазоне [-1, 1]
        """
        clean_img = clean_img.to(self.device)
        mask = mask.to(self.device)

        # --- Размываем маску для плавных границ ---
        # 1. Убираем все единичные размерности, получаем [H, W]
        mask_np = mask.cpu().numpy().squeeze()

        # 2. Размываем (OpenCV вернет [H, W])
        import cv2
        mask_np = cv2.GaussianBlur(mask_np, (9, 9), 0)

        # 3. Превращаем обратно в тензор [1, 1, H, W]
        mask_blurred = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Генерируем поврежденное изображение (передаем размытую маску)
        damaged_img = self.G_AB(clean_img, mask_blurred)

        # На всякий случай жестко применяем маску:
        # вне маски оставляем оригинал, внутри - сгенерированное
        final_damaged = clean_img * (1.0 - mask_blurred) + damaged_img * mask_blurred

        return final_damaged