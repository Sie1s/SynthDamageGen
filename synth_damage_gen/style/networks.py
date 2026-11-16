import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """Остаточный блок, помогающий сети сохранять контекст изображения"""

    def __init__(self, dim, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        block = [nn.ReflectionPad2d(1),
                 nn.Conv2d(dim, dim, 3),
                 norm_layer(dim),
                 nn.ReLU(True)]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [nn.ReflectionPad2d(1),
                  nn.Conv2d(dim, dim, 3),
                  norm_layer(dim)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """
    Генератор CycleGAN.
    Вход: clean_image (3 канала) + mask (1 канал) = 4 канала.
    Выход: damaged_image (3 канала).
    """

    def __init__(self, input_nc=4, output_nc=3, ngf=32, n_blocks=4):
        super().__init__()
        norm_layer = nn.InstanceNorm2d

        # Начальная свертка
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, 7),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Понижение размерности (Downsampling)
        n_downsampling = 2
        mult = 1
        for i in range(n_downsampling):
            mult_prev = mult
            mult = 2 ** (i + 1)
            model += [nn.Conv2d(ngf * mult_prev, ngf * mult, 3, stride=2, padding=1),
                      norm_layer(ngf * mult),
                      nn.ReLU(True)]

        # Остаточные блоки (ResNet)
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer)]

        # Повышение размерности (Upsampling)
        for i in range(n_downsampling):
            mult_prev = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                      nn.Conv2d(ngf * mult_prev, ngf * mult_prev // 2, 3, stride=1, padding=1),
                      norm_layer(ngf * mult_prev // 2),
                      nn.ReLU(True)]

        # Выходная свертка
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, 7),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, clean_img, mask):
        # Склеиваем чистое изображение и маску по каналам
        x = torch.cat([clean_img, mask], dim=1)
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """
    Условный дискриминатор PatchGAN (оценивает патчи 70x70).
    Вход: damaged_image (3) + mask (1) = 4 канала.
    Выход: карта вероятностей (реалистично/фейк).
    """

    def __init__(self, input_nc=4, ndf=32, n_layers=3):
        super().__init__()
        model = [nn.Conv2d(input_nc, ndf, 4, 2, 1),
                 nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False),
                      nn.InstanceNorm2d(ndf * nf_mult),
                      nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, bias=False),
                  nn.InstanceNorm2d(ndf * nf_mult),
                  nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        return self.model(x)