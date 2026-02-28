"""
    Модуль scenarios/motor_start.py.
    Состав:
    Классы: MotorStartScenario.
    Функции: нет.
"""
from __future__ import annotations

import numpy as np

from core.parameters import MachineParameters
from core.state import make_initial_state
from sources.base import VoltageSource
from sources.three_phase_sine import ThreePhaseSineSource
from loads.base import LoadTorque
from loads.constant import MotorStartTorque
from .base import Scenario


class MotorStartScenario(Scenario):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: name, initial_state, voltage_source, load_torque, t_span, describe.
    """

    def __init__(
        self,
        t_end: float = 2.5,
        Mc_load: float = 1105.0,
        Mc_friction: float = 50.0,
        t_load_start: float = 0.8,
        t_load_ramp: float = 0.3,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self.t_end = t_end
        self.Mc_load = Mc_load
        self.Mc_friction = Mc_friction
        self.t_load_start = t_load_start
        self.t_load_ramp = t_load_ramp

    def name(self) -> str:
        """Возвращает имя сценария моделирования."""
        return "ПРЯМОЙ ПУСК АД"

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

        return MotorStartTorque(
            Mc_load=self.Mc_load,
            Mc_friction=self.Mc_friction,
            t_load_start=self.t_load_start,
            t_load_ramp=self.t_load_ramp,
        )

    def t_span(self) -> tuple[float, float]:
        """Возвращает интервал моделирования."""
        return (0.0, self.t_end)

    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""

        return (
            f"{self.name()}: Mc_нагр={self.Mc_load:.0f} Нм, "
            f"t_end={self.t_end:.1f} с"
        )
