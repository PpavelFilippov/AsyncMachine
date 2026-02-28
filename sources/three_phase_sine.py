"""
    Модуль sources/three_phase_sine.py.
    Состав:
    Классы: ThreePhaseSineSource.
    Функции: нет.
"""
from __future__ import annotations

import numpy as np

from .base import VoltageSource


class ThreePhaseSineSource(VoltageSource):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: electrical_frequency_hz, describe.
    """

    def __init__(
        self,
        amplitude: float,
        frequency: float = 50.0,
        phase_shift: float = 0.0,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_shift = phase_shift
        self._omega = 2 * np.pi * frequency

    def __call__(self, t: float) -> np.ndarray:
        """Возвращает значение функции источника или нагрузки в момент времени t."""

        wt = self._omega * t + self.phase_shift
        return np.array([
            self.amplitude * np.sin(wt),
            self.amplitude * np.sin(wt - 2 * np.pi / 3),
            self.amplitude * np.sin(wt + 2 * np.pi / 3),
        ])

    def electrical_frequency_hz(self) -> float | None:
        """Возвращает электрическую частоту источника в герцах."""
        return float(self.frequency)

    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""
        return (
            f"3-фазная синусоида: Um={self.amplitude:.1f} В, "
            f"f={self.frequency:.1f} Гц, phi_0={np.degrees(self.phase_shift):.1f}degrees"
        )
