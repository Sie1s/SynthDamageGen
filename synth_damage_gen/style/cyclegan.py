from __future__ import annotations
import torch
import torch.nn as nn
from .networks import ResnetGenerator, PatchGANDiscriminator


class ConditionalCycleGAN:
    """
    Условный CycleGAN для наложения дефектов.
    Генератор A->B накладывает дефект по маске.
    Генератор B->A пытается восстановить изображение (нужен для cycle consistency)
    """

    def __init__(self, device: str, lambda_cycle: float = 10.0,
                 lambda_identity: float = 5.0, lambda_mask: float = 2.0):
        self.device = device
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_mask = lambda_mask

        # Генераторы (вход 4 канала: RGB + маска)
        self.G_AB = ResnetGenerator(input_nc=4, output_nc=3).to(device)  # Clean -> Damaged
        self.G_BA = ResnetGenerator(input_nc=4, output_nc=3).to(device)  # Damaged -> Clean

        # Дискриминаторы (вход 4 канала: RGB + маска)
        self.D_A = PatchGANDiscriminator(input_nc=4).to(device)  # Оценивает чистые фото
        self.D_B = PatchGANDiscriminator(input_nc=4).to(device)  # Оценивает поврежденные фото

        # Оптимизаторы
        self.opt_G = torch.optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.opt_D = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )

        # Функции потерь
        self.criterion_GAN = nn.BCEWithLogitsLoss()  # Adversarial loss
        self.criterion_cycle = nn.L1Loss()  # Cycle consistency loss
        self.criterion_identity = nn.L1Loss()  # Identity loss
        self.criterion_mask = nn.L1Loss()  # Mask loss (штраф за изменение вне маски)

    def train_step(self, real_clean: torch.Tensor, mask: torch.Tensor) -> dict:
        """
        Один шаг обучения.
        real_clean: [B, 3, H, W] - чистые фото
        mask: [B, 1, H, W] - бинарные маски (1 - дефект, 0 - фон)
        """
        # =======================================================
        # 1. Обучение Генераторов (G_AB и G_BA)
        # =======================================================
        self.opt_G.zero_grad()

        # --- Цикл А->Б->А (Clean -> Damaged -> Clean) ---
        fake_damaged = self.G_AB(real_clean, mask)

        # Восстанавливаем его обратно
        rec_clean = self.G_BA(fake_damaged, mask)

        # --- Цикл Б->А->Б (Damaged -> Clean -> Damaged) ---
        fake_clean = self.G_BA(fake_damaged, mask)
        rec_damaged = self.G_AB(fake_clean, mask)

        # --- Identity Loss (Сохранение цвета) ---
        # Если подать на вход G_AB уже поврежденное фото с нулевой маской, он должен вернуть его же
        zero_mask = torch.zeros_like(mask)
        idt_damaged = self.G_AB(real_clean, zero_mask)

        # --- Подсчет лоссов ---

        # Прогоняем через дискриминаторы, чтобы узнать реальный размер выходного тензора
        pred_fake_damaged = self.D_B(fake_damaged, mask)
        pred_fake_clean = self.D_A(fake_clean, mask)

        # Создаем тензоры меток (valid=1, fake=0) динамически, на основе выхода D
        valid_AB = torch.ones_like(pred_fake_damaged).to(self.device)
        valid_BA = torch.ones_like(pred_fake_clean).to(self.device)

        # 1. Adversarial loss (Генератор пытается обмануть D_B и D_A)
        loss_GAN_AB = self.criterion_GAN(pred_fake_damaged, valid_AB)
        loss_GAN_BA = self.criterion_GAN(pred_fake_clean, valid_BA)

        # 2. Cycle Consistency loss
        loss_cycle_A = self.criterion_cycle(rec_clean, real_clean) * self.lambda_cycle
        loss_cycle_B = self.criterion_cycle(rec_damaged, fake_damaged) * self.lambda_cycle

        # 3. Identity loss
        loss_identity = self.criterion_identity(idt_damaged, real_clean) * self.lambda_identity

        # 4. Mask Loss
        # Умножаем разницу на (1 - mask), чтобы игнорировать область дефекта
        diff = torch.abs(fake_damaged - real_clean)
        loss_mask = torch.mean(diff * (1.0 - mask)) * self.lambda_mask

        # Итоговый лосс Генератора
        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_identity + loss_mask
        loss_G.backward()
        self.opt_G.step()

        # =======================================================
        # 2. Обучение Дискриминаторов (D_A и D_B)
        # =======================================================
        self.opt_D.zero_grad()

        # Выходы дискриминаторов для реальных и фейковых данных
        pred_real_clean = self.D_A(real_clean, mask)
        pred_fake_clean_detach = self.D_A(fake_clean.detach(), mask)

        pred_real_damaged = self.D_B(fake_damaged.detach(), mask)
        pred_fake_damaged_detach = self.D_B(rec_damaged.detach(), mask)

        # Создаем метки динамически
        valid_real = torch.ones_like(pred_real_clean).to(self.device)
        fake_real = torch.zeros_like(pred_fake_clean_detach).to(self.device)

        valid_damaged = torch.ones_like(pred_real_damaged).to(self.device)
        fake_damaged_labels = torch.zeros_like(pred_fake_damaged_detach).to(self.device)

        # D_A учится отличать реальные чистые фото от восстановленных
        loss_D_A_real = self.criterion_GAN(pred_real_clean, valid_real)
        loss_D_A_fake = self.criterion_GAN(pred_fake_clean_detach, fake_real)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        # D_B учится отличать реальные поврежденные от сгенерированных
        loss_D_B_real = self.criterion_GAN(pred_real_damaged, valid_damaged)
        loss_D_B_fake = self.criterion_GAN(pred_fake_damaged_detach, fake_damaged_labels)
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        self.opt_D.step()

        # Возвращаем статистику для логирования
        return {
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
            "loss_cycle": (loss_cycle_A + loss_cycle_B).item(),
            "loss_identity": loss_identity.item(),
            "loss_mask": loss_mask.item()
        }

    def save_checkpoint(self, filepath: str, epoch: int):
        """Сохраняет веса всех сетей и оптимизаторов"""
        torch.save({
            'epoch': epoch,
            'G_AB': self.G_AB.state_dict(),
            'G_BA': self.G_BA.state_dict(),
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """Загружает веса. Возвращает эпоху, на которой остановились"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.G_AB.load_state_dict(checkpoint['G_AB'])
        self.G_BA.load_state_dict(checkpoint['G_BA'])
        self.D_A.load_state_dict(checkpoint['D_A'])
        self.D_B.load_state_dict(checkpoint['D_B'])
        self.opt_G.load_state_dict(checkpoint['opt_G'])
        self.opt_D.load_state_dict(checkpoint['opt_D'])
        return checkpoint['epoch']