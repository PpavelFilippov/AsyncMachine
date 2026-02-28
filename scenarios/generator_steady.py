"""
    Модуль scenarios/generator_steady.py.
    Состав:
    Классы: GeneratorSteadyScenario.
    Функции: нет.
"""
from __future__ import annotations

import numpy as np

from core.parameters import MachineParameters
from core.state import make_initial_state
from sources.base import VoltageSource
from sources.three_phase_sine import ThreePhaseSineSource
from loads.base import LoadTorque
from loads.constant import RampTorque
from .base import Scenario


class GeneratorSteadyScenario(Scenario):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: name, initial_state, voltage_source, load_torque, t_span.
    """

    def __init__(
        self,
        t_end: float = 2.0,
        Mc_drive: float = -1200.0,
        t_ramp: float = 0.5,
        Mc_friction: float = 50.0,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self.t_end = t_end
        self.Mc_drive = Mc_drive
        self.t_ramp = t_ramp
        self.Mc_friction = Mc_friction

    def name(self) -> str:
        """Возвращает имя сценария моделирования."""
        return "ГЕНЕРАТОРНЫЙ РЕЖИМ"

    def initial_state(self, params: MachineParameters) -> np.ndarray:
        """Возвращает начальное состояние системы."""
        return make_initial_state(omega_r=params.omega_sync)

    def voltage_source(self, params: MachineParameters) -> VoltageSource:
        """Формирует источник напряжения для сценария."""
        return ThreePhaseSineSource(
            amplitude=params.Um,
            frequency=params.fn,
        )

    def load_torque(self, params: MachineParameters) -> LoadTorque:
        """Формирует закон момента нагрузки для сценария."""
        return RampTorque(
            Mc_target=self.Mc_drive + self.Mc_friction,
            t_ramp=self.t_ramp,
            Mc_initial=self.Mc_friction,
        )

    def t_span(self) -> tuple[float, float]:
        """Возвращает интервал моделирования."""
        return (0.0, self.t_end)
