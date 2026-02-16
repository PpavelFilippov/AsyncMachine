"""
Сценарий: пуск АД на холостом ходу + ступенчатый наброс нагрузки.

Двигатель стартует с omega = 0, разгоняется на холостом ходу (Mc approx 0),
затем в момент t_p подключается номинальная нагрузка ступенчато
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
    Пуск АД на ХХ -> ступенчатый наброс нагрузки в t = t_p.

    Фазы работы:
      0 .. t_p:  холостой ход (Mc = Mc_idle, обычно approx 0)
      t_p .. :   Mc = Mc_load (номинальный момент)
    """

    def __init__(
        self,
        t_end: float = 4.0,
        t_step: float = 2.0,
        Mc_load: float | None = None,
        Mc_idle: float = 0.0,
    ):
        """
        Args:
            t_end: конец моделирования, с
            t_step: момент наброса нагрузки (t_p), с
            Mc_load: момент после наброса, Нм (None -> M_nom из параметров)
            Mc_idle: момент холостого хода, Нм
        """
        self.t_end = t_end
        self.t_step = t_step
        self._Mc_load = Mc_load
        self.Mc_idle = Mc_idle

    def name(self) -> str:
        return "ПУСК АД НА ХХ + НАБРОС НАГРУЗКИ"

    def initial_state(self, params: MachineParameters) -> np.ndarray:
        return make_initial_state(omega_r=0.0)

    def voltage_source(self, params: MachineParameters) -> VoltageSource:
        return ThreePhaseSineSource(
            amplitude=params.Um,
            frequency=params.fn,
        )

    def load_torque(self, params: MachineParameters) -> LoadTorque:
        Mc_load = self._Mc_load if self._Mc_load is not None else params.M_nom
        return StepTorque(
            Mc_load=Mc_load,
            t_step=self.t_step,
            Mc_idle=self.Mc_idle,
        )

    def t_span(self) -> tuple[float, float]:
        return (0.0, self.t_end)

    def describe(self) -> str:
        return (
            f"{self.name()}: ХХ до t={self.t_step:.1f} с, "
            f"затем Mc={self._Mc_load} Нм, t_end={self.t_end:.1f} с"
        )