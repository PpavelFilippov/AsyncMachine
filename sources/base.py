"""
Abstract voltage source interface.

Source provides three-phase EMF vector e(t) in phase coordinates [A, B, C].
Optional series impedance is represented by 3x3 R/L matrices.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VoltageSource(ABC):
    """Base class for voltage sources."""

    @abstractmethod
    def __call__(self, t: float) -> np.ndarray:
        """
        Return source EMF vector [eA, eB, eC] at time t.
        """
        ...

    def series_resistance_matrix(self) -> np.ndarray:
        """
        Per-phase series resistance matrix (3x3), Ohm.
        Ideal source -> zeros.
        """
        return np.zeros((3, 3), dtype=float)

    def series_inductance_matrix(self) -> np.ndarray:
        """
        Per-phase series inductance matrix (3x3), Henry.
        Ideal source -> zeros.
        """
        return np.zeros((3, 3), dtype=float)

    def electrical_frequency_hz(self) -> float | None:
        """Electrical source frequency in Hz (if defined)."""
        return None

    @abstractmethod
    def describe(self) -> str:
        """Text description for logs."""
        ...
