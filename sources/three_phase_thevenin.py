"""
Неидеальный трёхфазный источник (эквивалент Теvenin).

Модель источника на фазу:
    u_terminal = e_phase(t) - R_s * i_phase - L_s * di_phase/dt
"""
from __future__ import annotations

import numpy as np

from .base import VoltageSource


class ThreePhaseSineTheveninSource(VoltageSource):
    """
    Трёхфазная синусоидальная ЭДС за последовательным RL-импедансом.

    __call__(t) возвращает внутреннюю ЭДС e(t), а не клеммное напряжение.
    """

    def __init__(
        self,
        amplitude: float,
        frequency: float = 50.0,
        phase_shift: float = 0.0,
        r_series: float = 0.0,
        l_series: float = 0.0,
    ):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_shift = phase_shift
        self.r_series = r_series
        self.l_series = l_series
        self._omega = 2 * np.pi * frequency

    def __call__(self, t: float) -> np.ndarray:
        wt = self._omega * t + self.phase_shift
        return np.array([
            self.amplitude * np.sin(wt),
            self.amplitude * np.sin(wt - 2 * np.pi / 3),
            self.amplitude * np.sin(wt + 2 * np.pi / 3),
        ])

    def describe(self) -> str:
        return (
            f"3-фазный Thevenin: Um={self.amplitude:.1f} В, "
            f"f={self.frequency:.1f} Гц, Rs={self.r_series:.5f} Ом, "
            f"Ls={self.l_series * 1e3:.5f} мГн"
        )

