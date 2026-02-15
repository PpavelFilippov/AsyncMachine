"""
Абстрактный момент нагрузки на валу
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LoadTorque(ABC):
    """Базовый класс момента нагрузки"""

    @abstractmethod
    def __call__(self, t: float, omega_r: float) -> float:
        """
        Момент сопротивления Mc(t, omega_r).

        Положительный момент тормозит ротор (моторный режим).
        Отрицательный момент ускоряет ротор (генераторный режим)
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Описание нагрузки для логов"""
        ...