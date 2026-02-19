"""
Non-ideal three-phase source (Thevenin equivalent).

Per-phase model:
    u_terminal = e_phase(t) - R_s * i_phase - L_s * di_phase/dt

__call__(t) returns internal EMF e(t).
"""
from __future__ import annotations

import numpy as np

from .base import VoltageSource


class ThreePhaseSineTheveninSource(VoltageSource):
    """Three-phase sinusoidal EMF with series RL impedance."""

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

    def series_resistance_matrix(self) -> np.ndarray:
        return np.eye(3, dtype=float) * float(self.r_series)

    def series_inductance_matrix(self) -> np.ndarray:
        return np.eye(3, dtype=float) * float(self.l_series)

    def electrical_frequency_hz(self) -> float | None:
        return float(self.frequency)

    def describe(self) -> str:
        return (
            f"3-phase Thevenin: Um={self.amplitude:.1f} V, "
            f"f={self.frequency:.1f} Hz, Rs={self.r_series:.5f} Ohm, "
            f"Ls={self.l_series * 1e3:.5f} mH"
        )
