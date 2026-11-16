"""
Модуль применения обученной CycleGAN.
Загружает веса генератора G_AB и накладывает дефекты на изображения
"""
import torch
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

        # Генерируем поврежденное изображение
        damaged_img = self.G_AB(clean_img, mask)

        # Вне маски оставляем оригинал, внутри - сгенерированное
        final_damaged = clean_img * (1.0 - mask) + damaged_img * mask

        return final_damaged