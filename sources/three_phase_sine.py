"""
Трёхфазный синусоидальный источник напряжения
"""
from __future__ import annotations

import numpy as np

from .base import VoltageSource


class ThreePhaseSineSource(VoltageSource):
    """
    U_k = Um * sin(omega*t + phi_k), k in {A, B, C}

    где phi_A = phase_shift, phi_B = phase_shift − 2pi/3, phi_C = phase_shift + 2pi/3
    """

    def __init__(
        self,
        amplitude: float,
        frequency: float = 50.0,
        phase_shift: float = 0.0,
    ):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_shift = phase_shift
        self._omega = 2 * np.pi * frequency

    def __call__(self, t: float) -> np.ndarray:
        wt = self._omega * t + self.phase_shift
        return np.array([
            self.amplitude * np.sin(wt),
            self.amplitude * np.sin(wt - 2 * np.pi / 3),
            self.amplitude * np.sin(wt + 2 * np.pi / 3),
        ])

    def electrical_frequency_hz(self) -> float | None:
        return float(self.frequency)

    def describe(self) -> str:
        return (
            f"3-фазная синусоида: Um={self.amplitude:.1f} В, "
            f"f={self.frequency:.1f} Гц, phi_0={np.degrees(self.phase_shift):.1f}degrees"
        )
