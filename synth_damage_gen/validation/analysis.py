"""
Модуль формирования итогового отчета.
Строит графики на основе метрик и генерирует PDF-документ.
"""
import os
import urllib.request
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from typing import List
from ..metrics.quality import Quality
from .passport import DatasetPassport


def _ensure_font() -> str:
    """Скачивает шрифт Roboto для поддержки кириллицы в PDF, если его нет"""
    font_dir = Path("assets")
    font_dir.mkdir(exist_ok=True)
    font_path = font_dir / "Roboto-Regular.ttf"

    if not font_path.exists():
        print("[INFO] Скачивание шрифта для PDF...")
        url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
        urllib.request.urlretrieve(url, font_path)
    return str(font_path)


class Analysis:
    def __init__(self, protocol_id: int):
        self.protocol_id = protocol_id
        self.figures: List[str] = []
        self.textual_conclusion: str = ""

    def attach_figure(self, fig_path: str):
        """Добавляет путь к графику в список для отчета"""
        self.figures.append(fig_path)

    def conclude(self, passport: DatasetPassport):
        """Формирует текстовый вывод на основе паспорта"""
        if passport.overall_verdict:
            self.textual_conclusion = (
                "Синтетический датасет успешно прошел все проверки. "
                "Метрики реалистичности, разнообразия и сохранности сигнала "
                "находятся в допустимых пределах. Датасет рекомендуется для обучения моделей реставрации."
            )
        else:
            self.textual_conclusion = (
                "Внимание! Синтетический датасет не прошел валидацию. "
                "Одна или несколько метрик вышли за допустимые пороги. "
                "Рекомендуется увеличить количество эпох обучения, изменить гиперпараметры масок или проверить качество входных данных."
            )

    def generate_plots(self, qualities: List[Quality]):
        """Строит столбчатые диаграммы для каждой метрики"""
        sns.set_theme(style="whitegrid")

        # Группируем метрики по аспектам
        aspects = {}
        for q in qualities:
            aspects.setdefault(q.aspect, []).append(q)

        for aspect, q_list in aspects.items():
            fig, ax = plt.subplots(figsize=(8, 4))

            names = [q.metric_code for q in q_list]
            values = [q.numeric_value for q in q_list]
            thresholds = [q.threshold for q in q_list]

            # Рисуем значения и пороги
            x = range(len(names))
            bars = ax.bar(x, values, width=0.4, label='Значение', color='skyblue')

            # Линия порога
            for i, (val, thr) in enumerate(zip(values, thresholds)):
                ax.plot([i - 0.2, i + 0.2], [thr, thr], color='red', linewidth=2, label='Порог' if i == 0 else "")

            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_title(f"Метрики: {aspect.upper()}")
            ax.legend()

            # Сохраняем
            fig_path = f"reports/plot_{aspect}.png"
            Path("reports").mkdir(exist_ok=True)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()

            self.attach_figure(fig_path)

    def render_pdf(self, pdf_path: str, passport: DatasetPassport, qualities: List[Quality]):
        """Собирает все в единый PDF-файл"""
        font_path = _ensure_font()

        pdf = FPDF()
        pdf.add_page()

        # Подключаем шрифт
        pdf.add_font("Roboto", "", font_path)
        pdf.set_font("Roboto", size=16)

        # Заголовок
        pdf.cell(0, 10, "Протокол оценки качества синтетического датасета", ln=True, align='C')
        pdf.ln(10)

        # Статус
        pdf.set_font("Roboto", size=12)
        status = "ПРИГОДЕН" if passport.overall_verdict else "НЕ ПРИГОДЕН"
        pdf.set_text_color(0, 128, 0) if passport.overall_verdict else pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"Итоговый вердикт: {status}", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # Текст
        pdf.multi_cell(0, 8, self.textual_conclusion)
        pdf.ln(10)

        # Таблица метрик
        pdf.set_font("Roboto", size=10)
        for q in qualities:
            mark = "OK" if q.verdict else "FAIL"
            text = f"{q.metric_code} ({q.aspect}): {q.numeric_value:.4f} (порог: {q.threshold:.4f}) -> {mark}"
            pdf.cell(0, 8, text, ln=True)

        pdf.ln(10)

        # Вставка графиков
        for fig_path in self.figures:
            if os.path.exists(fig_path):
                pdf.image(fig_path, w=160)
                pdf.ln(5)

        pdf.output(pdf_path)
        print(f"[INFO] PDF-отчет сохранен: {pdf_path}")