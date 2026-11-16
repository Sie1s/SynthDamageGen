"""
Скрипт для генерации синтетического датасета.
Создает папки: clean/ (чистые), damaged/ (поврежденные), masks/ (маски)
"""
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from synth_damage_gen.config import load_config, resolve_device
from synth_damage_gen.masks import MaskGenerator
from synth_damage_gen.style.transfer import StyleTransfer


def tensor_to_numpy(tensor):
    """Конвертирует тензор [-1, 1] в numpy [0, 255] (RGB)"""
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = (img + 1.0) * 127.5
    return np.clip(img, 0, 255).astype(np.uint8)


def main(args):
    device = resolve_device("auto")
    print(f"[INFO] Генерация на устройстве: {device}")

    cfg = load_config("configs/default.yaml")

    # 1. Инициализируем маски и модель
    mask_gen = MaskGenerator(cfg.masks)
    transfer = StyleTransfer(cfg.style.weights_path, device)

    # 2. Создаем папки для сохранения
    out_dir = Path(args.output_dir)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)
    (out_dir / "damaged").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    # 3. Обрабатываем изображения
    image_paths = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[INFO] Найдено {len(image_paths)} изображений.")

    for idx, img_name in enumerate(image_paths):
        if args.max_images and idx >= args.max_images:
            break

        img_path = os.path.join(args.input_dir, img_name)

        # Загрузка и препроцессинг
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (cfg.data.image_size, cfg.data.image_size))

        # Нормализация в [-1, 1] для нейросети
        clean_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

        # Генерируем маску
        mask = mask_gen.sample_mask((cfg.data.image_size, cfg.data.image_size), seed=idx)
        mask_tensor = mask.raster.unsqueeze(0)  # [1, 1, H, W]

        # Применяем модель
        damaged_tensor = transfer.apply_transfer(clean_tensor, mask_tensor)

        # Сохраняем результаты
        clean_np = tensor_to_numpy(clean_tensor)
        damaged_np = tensor_to_numpy(damaged_tensor)
        mask_np = mask.rasterize() * 255

        # В OpenCV формат BGR
        cv2.imwrite(str(out_dir / "clean" / img_name), cv2.cvtColor(clean_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "damaged" / img_name), cv2.cvtColor(damaged_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "masks" / img_name), mask_np)

        if (idx + 1) % 10 == 0:
            print(f"Обработано {idx + 1}/{len(image_paths)}")

    print(f"[DONE] Датасет успешно сгенерирован и сохранен в {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Папка с чистыми фото")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_dataset")
    parser.add_argument("--max_images", type=int, default=None, help="Ограничить количество")
    args = parser.parse_args()
    main(args)