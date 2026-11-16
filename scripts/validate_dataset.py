"""
Скрипт для расчета метрик и генерации паспорта качества датасета
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
from synth_damage_gen.metrics import (
    calculate_masked_psnr, calculate_masked_ssim,
    RealismMetrics, Quality, evaluate_mask_diversity
)
from synth_damage_gen.validation.passport import DatasetPassport
from synth_damage_gen.validation.analysis import Analysis


def main(args):
    device = resolve_device("auto")
    print(f"[INFO] Валидация на устройстве: {device}")

    cfg = load_config("configs/default.yaml")
    dataset_dir = Path(args.dataset_dir)

    # Пути к папкам
    clean_dir = dataset_dir / "clean"
    damaged_dir = dataset_dir / "damaged"
    masks_dir = dataset_dir / "masks"

    image_names = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg'))]
    if args.max_images:
        image_names = image_names[:args.max_images]

    print(f"[INFO] Валидация {len(image_names)} изображений...")

    # Инициализация метрик
    realism_metrics = RealismMetrics(device=device)
    qualities = []
    q_id = 1

    psnr_vals, ssim_vals, lpips_vals, niqe_vals = [], [], [], []
    masks_for_diversity = []
    mask_gen = MaskGenerator(cfg.masks)

    for idx, name in enumerate(image_names):
        # Загрузка
        clean = cv2.imread(str(clean_dir / name))
        damaged = cv2.imread(str(damaged_dir / name))
        mask = cv2.imread(str(masks_dir / name), cv2.IMREAD_GRAYSCALE)

        # Для метрик переводим в RGB и правильный формат
        clean_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        damaged_rgb = cv2.cvtColor(damaged, cv2.COLOR_BGR2RGB)

        # 1. Preservation (PSNR, SSIM)
        psnr_vals.append(calculate_masked_psnr(clean_rgb, damaged_rgb, mask))
        ssim_vals.append(calculate_masked_ssim(clean_rgb, damaged_rgb, mask))

        # 2. Realism (LPIPS, NIQE)
        t_clean = torch.from_numpy(clean_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        t_damaged = torch.from_numpy(damaged_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        t_damaged_0_1 = (t_damaged + 1.0) / 2.0  # NIQE ожидает [0, 1]

        lpips_vals.append(realism_metrics.calculate_lpips(t_clean, t_damaged))
        niqe_vals.append(realism_metrics.calculate_niqe(t_damaged_0_1))

        # 3. Diversity (собираем статистику масок)
        m = mask_gen.sample_mask((cfg.data.image_size, cfg.data.image_size), seed=idx)
        masks_for_diversity.append(m)

    # Усредняем метрики
    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)
    avg_lpips = np.mean(lpips_vals)
    avg_niqe = np.mean(niqe_vals)

    print(f"Средний PSNR: {avg_psnr:.2f}")
    print(f"Средний SSIM: {avg_ssim:.4f}")
    print(f"Средний LPIPS: {avg_lpips:.4f}")
    print(f"Средний NIQE: {avg_niqe:.4f}")

    # Формируем объекты Quality
    qualities.append(Quality.create(q_id, "PSNR", "preservation", avg_psnr, cfg.validation.preservation.psnr_threshold,
                                    higher_is_better=True));
    q_id += 1
    qualities.append(Quality.create(q_id, "SSIM", "preservation", avg_ssim, cfg.validation.preservation.ssim_threshold,
                                    higher_is_better=True));
    q_id += 1
    qualities.append(Quality.create(q_id, "LPIPS", "realism", avg_lpips, cfg.validation.realism.lpips_threshold,
                                    higher_is_better=False));
    q_id += 1
    qualities.append(Quality.create(q_id, "NIQE", "realism", avg_niqe, cfg.validation.realism.niqe_threshold,
                                    higher_is_better=False));
    q_id += 1

    # 4. Diversity (Оценка масок)
    div_res = evaluate_mask_diversity(masks_for_diversity)
    qualities.append(
        Quality.create(q_id, "W1_Area_Var", "diversity", div_res["synth_area_std"], 0.01, higher_is_better=True));
    q_id += 1

    # Строим паспорт и анализ
    passport = DatasetPassport(passport_id=1).build(qualities)
    print(passport.summarize())

    analysis = Analysis(protocol_id=1)
    analysis.conclude(passport)
    analysis.generate_plots(qualities)

    pdf_path = args.output_pdf
    analysis.render_pdf(pdf_path, passport, qualities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/synthetic_dataset")
    parser.add_argument("--output_pdf", type=str, default="reports/dataset_passport.pdf")
    parser.add_argument("--max_images", type=int, default=100, help="Сколько картинок оценивать (для FID нужно больше)")
    args = parser.parse_args()
    main(args)