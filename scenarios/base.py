"""
    Модуль scenarios/base.py.
    Состав:
    Классы: Scenario.
    Функции: нет.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from core.parameters import MachineParameters
from sources.base import VoltageSource
from loads.base import LoadTorque


class Scenario(ABC):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: name, initial_state, voltage_source, load_torque, t_span, describe.
    """

    @abstractmethod
    def name(self) -> str:
        """Определяет обязательный интерфейс метода."""
        ...

    @abstractmethod
    def initial_state(self, params: MachineParameters) -> np.ndarray:
        """Возвращает начальное состояние системы."""
        ...

    @abstractmethod
    def voltage_source(self, params: MachineParameters) -> VoltageSource:
        """Формирует источник напряжения для сценария."""
        ...

    @abstractmethod
    def load_torque(self, params: MachineParameters) -> LoadTorque:
        """Формирует закон момента нагрузки для сценария."""
        ...

    @abstractmethod
    def t_span(self) -> tuple[float, float]:
        """Возвращает интервал моделирования."""
        ...

    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""
        return self.name()
