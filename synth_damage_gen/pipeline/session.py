"""
Класс Session (Сессия).
Оркестратор конвейера: объединяет загрузку данных, генерацию, применение стиля и валидацию.
"""
from pathlib import Path
from dataclasses import dataclass
import time
import cv2
import numpy as np
import torch

from ..config import AppConfig
from ..masks import MaskGenerator
from ..style.transfer import StyleTransfer
from ..metrics import Quality
from ..validation.passport import DatasetPassport
from ..validation.analysis import Analysis


@dataclass
class SessionConfig:
    session_id: int
    input_dir: str
    output_dir: str
    max_images: int | None = None
    generate_report: bool = True


class Session:
    def __init__(self, cfg: AppConfig, session_cfg: SessionConfig):
        self.cfg = cfg
        self.session_cfg = session_cfg
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")

        self.mask_gen = MaskGenerator(cfg.masks)
        self.transfer = StyleTransfer(cfg.style.weights_path, cfg.device)

    def run(self):
        print(f"=== Запуск сессии #{self.session_cfg.session_id} ===")
        print(f"Время старта: {self.started_at}")
        print(f"Устройство: {self.cfg.device}")

        input_path = Path(self.session_cfg.input_dir)
        out_path = Path(self.session_cfg.output_dir)

        # Создаем папки
        (out_path / "clean").mkdir(parents=True, exist_ok=True)
        (out_path / "damaged").mkdir(parents=True, exist_ok=True)
        (out_path / "masks").mkdir(parents=True, exist_ok=True)

        images = list(input_path.glob("*.*"))
        images = [p for p in images if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]

        if self.session_cfg.max_images:
            images = images[:self.session_cfg.max_images]

        print(f"Найдено изображений для обработки: {len(images)}")

        for idx, img_path in enumerate(images):
            # 1. Загрузка
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.cfg.data.image_size, self.cfg.data.image_size))

            # 2. Тензоры
            clean_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            mask = self.mask_gen.sample_mask((self.cfg.data.image_size, self.cfg.data.image_size), seed=idx)
            mask_tensor = mask.raster.unsqueeze(0)

            # 3. Инференс (наложение дефекта)
            damaged_tensor = self.transfer.apply_transfer(clean_tensor, mask_tensor)

            # 4. Сохранение
            name = img_path.name
            self._save_tensor(clean_tensor, out_path / "clean" / name)
            self._save_tensor(damaged_tensor, out_path / "damaged" / name)
            cv2.imwrite(str(out_path / "masks" / name), mask.rasterize() * 255)

            if (idx + 1) % 10 == 0:
                print(f"Обработано {idx + 1}/{len(images)}")

        print("Генерация датасета завершена.")

        # Валидация и отчет
        if self.session_cfg.generate_report:
            print("Запуск валидации (это может занять время)...")
            self._run_validation(out_path)

    def _save_tensor(self, tensor: torch.Tensor, path: Path):
        """Вспомогательная функция сохранения тензора [-1, 1] в файл."""
        img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _run_validation(self, out_path: Path):
        import subprocess
        import sys
        subprocess.run([
            sys.executable,
            "scripts/validate_dataset.py",
            "--dataset_dir", str(out_path),
            "--output_pdf", "reports/session_report.pdf"
        ])
        print(f"[INFO] Протокол анализа сохранен в reports/session_report.pdf")