"""
    Модуль sources/base.py.
    Состав:
    Классы: VoltageSource.
    Функции: нет.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VoltageSource(ABC):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: series_resistance_matrix, series_inductance_matrix, electrical_frequency_hz, describe.
    """

    @abstractmethod
    def __call__(self, t: float) -> np.ndarray:
        """Возвращает значение функции источника или нагрузки в момент времени t."""
        ...

    def series_resistance_matrix(self) -> np.ndarray:
        """Возвращает матрицу последовательных сопротивлений источника."""
        return np.zeros((3, 3), dtype=float)

    def series_inductance_matrix(self) -> np.ndarray:
        """Возвращает матрицу последовательных индуктивностей источника."""
        return np.zeros((3, 3), dtype=float)

    def electrical_frequency_hz(self) -> float | None:
        """Возвращает электрическую частоту источника в герцах."""
        return None

    @abstractmethod
    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""
        ...
