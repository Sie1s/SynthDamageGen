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


class SyntheticDamageDataset(Dataset):
    """Датасет, который на лету генерирует маски и подает чистые картинки"""

    def __init__(self, image_dir: str, image_size: int, mask_generator: MaskGenerator):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_size = image_size
        self.mask_gen = mask_generator

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))

        # [3, H, W], нормализация в [-1, 1] (так как на выходе генератора Tanh)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 127.5 - 1.0

        # Генерируем маску
        mask = self.mask_gen.sample_mask((self.image_size, self.image_size), seed=idx)
        mask_tensor = mask.raster  # уже [1, H, W] в {0, 1}

        return img_tensor, mask_tensor


def main(args):
    device = resolve_device("auto")
    print(f"[INFO] Обучение на устройстве: {device}")

    cfg = load_config("configs/default.yaml")
    mask_gen = MaskGenerator(cfg.masks)

    dataset = SyntheticDamageDataset(args.data_dir, cfg.data.image_size, mask_gen)
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
        for clean_imgs, masks in loop:
            clean_imgs = clean_imgs.to(device)
            masks = masks.to(device)

            stats = model.train_step(clean_imgs, masks)

            epoch_loss_G += stats["loss_G"]
            epoch_loss_D += stats["loss_D"]
            epoch_loss_mask += stats["loss_mask"]

            loop.set_postfix(loss_G=f"{stats['loss_G']:.3f}", loss_mask=f"{stats['loss_mask']:.3f}")

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