"""
Класс DatasetPassport (Паспорт датасета).
Собирает все метрики, группирует их по аспектам (realism, diversity, preservation) и выносит итоговый вердикт о пригодности синтетического датасета
"""
from dataclasses import dataclass, field
from typing import List, Dict
from ..metrics.quality import Quality


@dataclass
class DatasetPassport:
    passport_id: int
    aspect_summary: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    overall_verdict: bool = True

    def build(self, qualities: List[Quality]) -> "DatasetPassport":
        """
        Формирует сводку по аспектам на основе списка измерений.
        Если хотя бы одна метрика в аспекте провалена, общий вердикт становится False.
        """
        # Очищаем предыдущие результаты (если вызываем повторно)
        self.aspect_summary = {}
        self.overall_verdict = True

        for q in qualities:
            if q.aspect not in self.aspect_summary:
                self.aspect_summary[q.aspect] = {}

            # Записываем вердикт по конкретной метрике
            self.aspect_summary[q.aspect][q.metric_code] = q.verdict

            # Если метрика не прошла порог, общий вердикт - False
            if not q.verdict:
                self.overall_verdict = False

        return self

    def summarize(self) -> str:
        """Возвращает текстовое резюме для логов."""
        status = "ПРИГОДЕН" if self.overall_verdict else "НЕ ПРИГОДЕН"
        lines = [f"=== Паспорт датасета №{self.passport_id}: {status} ==="]
        for aspect, metrics in self.aspect_summary.items():
            lines.append(f"[{aspect.upper()}]")
            for metric, passed in metrics.items():
                mark = "✓" if passed else "✗"
                lines.append(f"  {mark} {metric}")
        return "\n".join(lines)