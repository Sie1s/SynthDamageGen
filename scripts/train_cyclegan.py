import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from synth_damage_gen.config import load_config, resolve_device
from synth_damage_gen.masks import MaskGenerator
from synth_damage_gen.style.cyclegan import ConditionalCycleGAN


class UnpairedDataset(Dataset):
    """Датасет для непарного обучения (Domain A и Domain B)."""
    def __init__(self, clean_dir: str, damaged_dir: str, image_size: int, mask_generator):
        self.clean_paths = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.damaged_paths = [os.path.join(damaged_dir, f) for f in os.listdir(damaged_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_size = image_size
        self.mask_gen = mask_generator
        self.length = max(len(self.clean_paths), len(self.damaged_paths))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Загружаем чистое фото (циклически, если их больше/меньше)
        clean_path = self.clean_paths[idx % len(self.clean_paths)]
        img_clean = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        img_clean = cv2.resize(img_clean, (self.image_size, self.image_size))
        tensor_clean = torch.from_numpy(img_clean).float().permute(2, 0, 1) / 127.5 - 1.0

        # Загружаем реальное поврежденное фото
        damaged_path = self.damaged_paths[idx % len(self.damaged_paths)]
        img_damaged = cv2.imread(damaged_path, cv2.IMREAD_COLOR)
        img_damaged = cv2.cvtColor(img_damaged, cv2.COLOR_BGR2RGB)
        img_damaged = cv2.resize(img_damaged, (self.image_size, self.image_size))
        tensor_damaged = torch.from_numpy(img_damaged).float().permute(2, 0, 1) / 127.5 - 1.0

        # Генерируем маску
        mask = self.mask_gen.sample_mask((self.image_size, self.image_size), seed=idx)
        tensor_mask = mask.raster

        return tensor_clean, tensor_damaged, tensor_mask


def main(args):
    device = resolve_device("auto")
    print(f"[INFO] Обучение на устройстве: {device}")

    cfg = load_config("configs/default.yaml")
    mask_gen = MaskGenerator(cfg.masks)

    dataset = UnpairedDataset(args.data_dir, args.damaged_dir, cfg.data.image_size, mask_gen)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(f"[INFO] Загружено {len(dataset)} изображений.")

    model = ConditionalCycleGAN(
        device=device,
        lambda_cycle=cfg.style.lambda_cycle,
        lambda_identity=cfg.style.lambda_identity,
        lambda_mask=cfg.style.lambda_mask
    )

    start_epoch = 0
    checkpoint_path = Path(args.checkpoint_dir) / "latest.pth"
    if checkpoint_path.exists() and args.resume:
        start_epoch = model.load_checkpoint(str(checkpoint_path))
        print(f"[INFO] Возобновление с эпохи {start_epoch}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.G_AB.train()
        model.G_BA.train()
        model.D_A.train()
        model.D_B.train()

        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_loss_mask = 0.0

        loop = tqdm(dataloader, desc=f"Эпоха [{epoch + 1}/{args.epochs}]")
        for clean_imgs, damaged_imgs, masks in loop:
            clean_imgs = clean_imgs.to(device)
            damaged_imgs = damaged_imgs.to(device)
            masks = masks.to(device)

            # Передаем оба домена
            stats = model.train_step(clean_imgs, damaged_imgs, masks)

            epoch_loss_G += stats["loss_G"]
            epoch_loss_D += stats["loss_D"]
            epoch_loss_mask += stats["loss_mask"]

            loop.set_postfix(loss_G=f"{stats['loss_G']:.3f}")

        # Выводим СРЕДНИЕ значения за эпоху
        avg_G = epoch_loss_G / len(dataloader)
        avg_D = epoch_loss_D / len(dataloader)
        avg_mask = epoch_loss_mask / len(dataloader)
        print(f"Эпоха {epoch + 1} завершена. Средние: loss_G={avg_G:.3f}, loss_D={avg_D:.3f}, loss_mask={avg_mask:.3f}")

        # Сохраняем чекпоинт каждые 5 эпох
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            model.save_checkpoint(str(checkpoint_path), epoch + 1)
            print(f"[SAVE] Чекпоинт сохранен: {checkpoint_path}")

            # Дублируем в папку Google Drive (если путь указан)
            if args.drive_dir:
                drive_dir = Path(args.drive_dir)
                drive_dir.mkdir(parents=True, exist_ok=True)
                drive_path = drive_dir / f"epoch_{epoch + 1}.pth"
                model.save_checkpoint(str(drive_path), epoch + 1)
                print(f"[SAVE] Копия сохранена в Drive: {drive_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Путь к папке с чистыми фото")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="weights/")
    parser.add_argument("--drive_dir", type=str, default=None,
                        help="Путь в Google Drive для сохранения (только для Colab)")
    parser.add_argument("--resume", action="store_true", help="Продолжить с последнего чекпоинта")
    args = parser.parse_args()
    main(args)