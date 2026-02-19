"""
Абстрактный сценарий моделирования.

Сценарий определяет:
  - начальные условия (y0)
  - источник напряжения
  - нагрузку
  - временной диапазон
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from core.parameters import MachineParameters
from sources.base import VoltageSource
from loads.base import LoadTorque


class Scenario(ABC):
    """Базовый класс сценария моделирования"""

    @abstractmethod
    def name(self) -> str:
        """Имя сценария для логов и графиков"""
        ...

    @abstractmethod
    def initial_state(self, params: MachineParameters) -> np.ndarray:
        """Начальный вектор состояния y0"""
        ...

    @abstractmethod
    def voltage_source(self, params: MachineParameters) -> VoltageSource:
        """Источник напряжения для данного сценария"""
        ...

    @abstractmethod
    def load_torque(self, params: MachineParameters) -> LoadTorque:
        """Момент нагрузки для данного сценария"""
        ...

    @abstractmethod
    def t_span(self) -> tuple[float, float]:
        """Временной диапазон (t_start, t_end)"""
        ...

    def describe(self) -> str:
        """Подробное описание сценария"""
        return self.name()
