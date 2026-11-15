import os
import sys
import matplotlib.pyplot as plt

# Добавляем корень проекта в sys.path чтобы работали импорты
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth_damage_gen.config import load_config
from synth_damage_gen.masks import MaskGenerator


def main():
    print("Загрузка конфигурации...")
    cfg = load_config("configs/default.yaml")

    print(f"Инициализация генератора: {cfg.masks.default_generator}")
    generator = MaskGenerator(cfg.masks)

    # Генерируем 4 маски разных размеров
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(4):
        seed = 100 + i
        mask = generator.sample_mask(size=(256, 256), seed=seed)

        print(f"Маска {i + 1}: Площадь={mask.area_share:.3f}, "
              f"Толщина={mask.thickness_avg:.2f}, "
              f"Извилистость={mask.tortuosity:.2f}")

        axes[i].imshow(mask.rasterize(), cmap='gray')
        axes[i].set_title(f"Маска (seed={seed})")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()