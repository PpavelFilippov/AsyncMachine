"""
    Модуль scenarios/motor_no_load.py.
    Состав:
    Классы: MotorNoLoadScenario.
    Функции: нет.
"""

from __future__ import annotations

import numpy as np

from core.parameters import MachineParameters
from core.state import make_initial_state
from sources.base import VoltageSource
from sources.three_phase_sine import ThreePhaseSineSource
from loads.base import LoadTorque
from loads.constant import ConstantTorque
from .base import Scenario


class MotorNoLoadScenario(Scenario):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: name, initial_state, voltage_source, load_torque, t_span, describe.
    """

    def __init__(
        self,
        t_end: float = 4.0,
        Mc_idle: float = 0.0,
        Mc_friction: float = 50.0,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self.t_end = t_end
        self.Mc_idle = Mc_idle
        self.Mc_friction = Mc_friction

    def name(self) -> str:
        """Возвращает имя сценария моделирования."""
        return "ПУСК АД НА ХОЛОСТОМ ХОДУ"

    def initial_state(self, params: MachineParameters) -> np.ndarray:
        """Возвращает начальное состояние системы."""
        return make_initial_state(omega_r=0.0)

    def voltage_source(self, params: MachineParameters) -> VoltageSource:
        """Формирует источник напряжения для сценария."""

        return ThreePhaseSineSource(
            amplitude=params.Um,
            frequency=params.fn,
        )

    def load_torque(self, params: MachineParameters) -> LoadTorque:
        """Формирует закон момента нагрузки для сценария."""
        return ConstantTorque(
            Mc=self.Mc_idle + self.Mc_friction,
        )

    def t_span(self) -> tuple[float, float]:
        """Возвращает интервал моделирования."""
        return (0.0, self.t_end)

    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""
        return (
            f"{self.name()}: Mc={self.Mc_idle} Нм, t_end={self.t_end:.1f} с"
        )
