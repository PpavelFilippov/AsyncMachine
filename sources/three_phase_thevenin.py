"""
    Модуль sources/three_phase_thevenin.py.
    Состав:
    Классы: ThreePhaseSineTheveninSource.
    Функции: нет.
"""
from __future__ import annotations

import numpy as np

from .base import VoltageSource


class ThreePhaseSineTheveninSource(VoltageSource):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: series_resistance_matrix, series_inductance_matrix, electrical_frequency_hz, describe.
    """

    def __init__(
        self,
        amplitude: float,
        frequency: float = 50.0,
        phase_shift: float = 0.0,
        r_series: float = 0.0,
        l_series: float = 0.0,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_shift = phase_shift
        self.r_series = r_series
        self.l_series = l_series
        self._omega = 2 * np.pi * frequency

    def __call__(self, t: float) -> np.ndarray:
        """Возвращает значение функции источника или нагрузки в момент времени t."""

        wt = self._omega * t + self.phase_shift
        return np.array([
            self.amplitude * np.sin(wt),
            self.amplitude * np.sin(wt - 2 * np.pi / 3),
            self.amplitude * np.sin(wt + 2 * np.pi / 3),
        ])

    def series_resistance_matrix(self) -> np.ndarray:
        """Возвращает матрицу последовательных сопротивлений источника."""
        return np.eye(3, dtype=float) * float(self.r_series)

    def series_inductance_matrix(self) -> np.ndarray:
        """Возвращает матрицу последовательных индуктивностей источника."""
        return np.eye(3, dtype=float) * float(self.l_series)

    def electrical_frequency_hz(self) -> float | None:
        """Возвращает электрическую частоту источника в герцах."""
        return float(self.frequency)

    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""
        return (
            f"3-phase Thevenin: Um={self.amplitude:.1f} V, "
            f"f={self.frequency:.1f} Hz, Rs={self.r_series:.5f} Ohm, "
            f"Ls={self.l_series * 1e3:.5f} mH"
        )
