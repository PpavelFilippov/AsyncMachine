"""
Сценарий: пуск АД на холостом ходу.

Двигатель стартует с omega = 0 и разгоняется на холостом ходу (Mc approx 0)
до установившегося режима
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
    Пуск АД на холостом ходу.

    Двигатель работает всё время с моментом сопротивления Mc = Mc_idle
    """

    def __init__(
        self,
        t_end: float = 4.0,
        Mc_idle: float = 0.0,
        Mc_friction: float = 50.0,
    ):
        """
        Args:
            t_end: конец моделирования, с
            Mc_idle: момент холостого хода, Нм (обычно 0 или очень маленький)
        """
        self.t_end = t_end
        self.Mc_idle = Mc_idle
        self.Mc_friction = Mc_friction

    def name(self) -> str:
        return "ПУСК АД НА ХОЛОСТОМ ХОДУ"

    def initial_state(self, params: MachineParameters) -> np.ndarray:
        return make_initial_state(omega_r=0.0)

    def voltage_source(self, params: MachineParameters) -> VoltageSource:
        return ThreePhaseSineSource(
            amplitude=params.Um,
            frequency=params.fn,
        )

    def load_torque(self, params: MachineParameters) -> LoadTorque:
        return ConstantTorque(
            Mc=self.Mc_idle + self.Mc_friction,
        )

    def t_span(self) -> tuple[float, float]:
        return (0.0, self.t_end)

    def describe(self) -> str:
        return (
            f"{self.name()}: Mc={self.Mc_idle} Нм, t_end={self.t_end:.1f} с"
        )
