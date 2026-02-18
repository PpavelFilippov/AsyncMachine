"""
Абстрактный источник напряжения.

Источник формирует вектор [U1A, U1B, U1C] в каждый момент времени
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VoltageSource(ABC):
    """Базовый класс источника напряжения"""

    @abstractmethod
    def __call__(self, t: float) -> np.ndarray:
        """
        Вернуть вектор напряжений [U1A, U1B, U1C] в момент времени t
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Описание источника для логов"""
        ...
