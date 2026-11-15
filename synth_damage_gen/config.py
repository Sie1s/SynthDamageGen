"""Модуль для загрузки YAML и предоставления типизированного доступа к настройкам"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple
import yaml
import torch


def resolve_device(device_str: str) -> str:
    """Автоматически определяет доступное устройство: cuda, mps или cpu."""
    if device_str == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device_str == "mps" and torch.backends.mps.is_available():
        return "mps"
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


@dataclass
class DataConfig:
    image_size: int = 256
    photobank_root: str = "data/photobanks"
    cache_dir: str = "data/cache"


@dataclass
class DeterministicMaskConfig:
    crack_thickness: int = 3
    crack_iterations: int = 4
    threshold: float = 0.15


@dataclass
class IrregularMaskConfig:
    area_range: Tuple[float, float] = (0.01, 0.30)
    n_strokes: int = 12
    dilation: int = 2
    boundary_contact_prob: float = 0.4


@dataclass
class MaskConfig:
    default_generator: str = "irregular"
    deterministic: DeterministicMaskConfig = field(default_factory=DeterministicMaskConfig)
    irregular: IrregularMaskConfig = field(default_factory=IrregularMaskConfig)


@dataclass
class StyleConfig:
    weights_path: str = "weights/cyclegan_mask.pth"
    image_size: int = 256
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_adv: float = 1.0
    lambda_mask: float = 2.0


@dataclass
class RealismThresholds:
    fid_threshold: float = 45.0
    niqe_threshold: float = 6.5
    lpips_threshold: float = 0.35


@dataclass
class DiversityThresholds:
    w1_threshold: float = 0.05
    ks_pvalue_threshold: float = 0.05


@dataclass
class PreservationThresholds:
    psnr_threshold: float = 28.0
    ssim_threshold: float = 0.92


@dataclass
class ValidationConfig:
    realism: RealismThresholds = field(default_factory=RealismThresholds)
    diversity: DiversityThresholds = field(default_factory=DiversityThresholds)
    preservation: PreservationThresholds = field(default_factory=PreservationThresholds)


@dataclass
class AppConfig:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    masks: MaskConfig = field(default_factory=MaskConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def _update_dataclass_from_dict(dc: Any, raw: dict) -> Any:
    """Рекурсивно обновляет dataclass значениями из словаря YAML."""
    for key, value in raw.items():
        if not hasattr(dc, key):
            continue
        current_value = getattr(dc, key)

        if isinstance(value, dict) and hasattr(current_value, "__dataclass_fields__"):
            _update_dataclass_from_dict(current_value, value)
        elif isinstance(value, list) and key == "area_range":
            setattr(dc, key, tuple(value))
        else:
            setattr(dc, key, value)
    return dc


def load_config(config_path: str | Path = "configs/default.yaml") -> AppConfig:
    """Загружает конфигурацию из YAML и возвращает объект AppConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = _update_dataclass_from_dict(AppConfig(), raw)

    # Сразу разрешаем устройство
    config.device = resolve_device(config.device)
    return config