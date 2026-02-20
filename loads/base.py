"""
    Модуль loads/base.py.
    Состав:
    Классы: LoadTorque.
    Функции: нет.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LoadTorque(ABC):
    """
        Методы:
        Основные публичные методы: describe.
    """

    @abstractmethod
    def __call__(self, t: float, omega_r: float) -> float:
        """Возвращает значение функции источника или нагрузки в момент времени t."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""
        ...
