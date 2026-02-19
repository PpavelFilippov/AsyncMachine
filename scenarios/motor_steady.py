"""
Сценарий: установившийся моторный режим
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


class MotorSteadyScenario(Scenario):
    """
    Установившийся моторный режим.

    Ротор стартует на синхронной скорости, момент нагрузки нарастает плавно
    """

    def __init__(
        self,
        t_end: float = 2.0,
        t_ramp: float = 0.5,
        Mc_friction: float = 50.0,
    ):
        self.t_end = t_end
        self.t_ramp = t_ramp
        self.Mc_friction = Mc_friction

    def name(self) -> str:
        return "МОТОРНЫЙ РЕЖИМ"

    def initial_state(self, params: MachineParameters) -> np.ndarray:
        return make_initial_state(omega_r=params.omega_sync)

    def voltage_source(self, params: MachineParameters) -> VoltageSource:
        return ThreePhaseSineSource(
            amplitude=params.Um,
            frequency=params.fn,
        )

    def load_torque(self, params: MachineParameters) -> LoadTorque:
        return RampTorque(
            Mc_target=params.M_nom + self.Mc_friction,
            t_ramp=self.t_ramp,
            Mc_initial=self.Mc_friction,
        )

    def t_span(self) -> tuple[float, float]:
        return (0.0, self.t_end)
