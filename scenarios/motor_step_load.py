"""
    Модуль scenarios/motor_step_load.py.
    Состав:
    Классы: MotorStepLoadScenario.
    Функции: нет.
"""
from __future__ import annotations

import numpy as np

from core.parameters import MachineParameters
from core.state import make_initial_state
from sources.base import VoltageSource
from sources.three_phase_sine import ThreePhaseSineSource
from loads.base import LoadTorque
from loads.constant import StepTorque
from .base import Scenario


class MotorStepLoadScenario(Scenario):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: name, initial_state, voltage_source, load_torque, t_span, describe.
    """

    def __init__(
        self,
        t_end: float = 4.0,
        t_step: float = 2.0,
        Mc_load: float | None = None,
        Mc_idle: float = 0.0,
        Mc_friction: float = 50.0,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self.t_end = t_end
        self.t_step = t_step
        self._Mc_load = Mc_load
        self.Mc_idle = Mc_idle
        self.Mc_friction = Mc_friction

    def name(self) -> str:
        """Возвращает имя сценария моделирования."""
        return "ПУСК АД НА ХХ + НАБРОС НАГРУЗКИ"

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

        Mc_load = self._Mc_load if self._Mc_load is not None else params.M_nom
        return StepTorque(
            Mc_load=Mc_load + self.Mc_friction,
            t_step=self.t_step,
            Mc_idle=self.Mc_idle + self.Mc_friction,
        )

    def t_span(self) -> tuple[float, float]:
        """Возвращает интервал моделирования."""
        return (0.0, self.t_end)

    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""

        return (
            f"{self.name()}: ХХ до t={self.t_step:.1f} с, "
            f"затем Mc={self._Mc_load} Нм, t_end={self.t_end:.1f} с"
        )
