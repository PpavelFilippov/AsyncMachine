"""
Неисправность: обрыв фазы.

Обнуляет напряжение соответствующей фазы статора
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Fault


class OpenPhaseFault(Fault):
    """
    Обрыв одной или нескольких фаз статора.

    Обнуляет U фазы начиная с t_start
    """

    PHASE_MAP = {"A": 0, "B": 1, "C": 2}

    def __init__(
        self,
        phases: str | list[str],
        t_start: float = 0.0,
        t_end: Optional[float] = None,
    ):
        """
        Args:
            phases: фаза(ы) для обрыва, напр. "A", ["A", "C"]
            t_start: момент обрыва
            t_end: момент восстановления (None = навсегда)
        """
        super().__init__(t_start=t_start, t_end=t_end)
        if isinstance(phases, str):
            phases = [phases]
        self.phases = [p.upper() for p in phases]
        self._indices = [self.PHASE_MAP[p] for p in self.phases]

    def modify_voltages(self, t: float, U: np.ndarray) -> np.ndarray:
        if not self.is_active(t):
            return U
        U_mod = U.copy()
        for idx in self._indices:
            U_mod[idx] = 0.0
        return U_mod

    def describe(self) -> str:
        ph_str = ", ".join(self.phases)
        timing = f"t={self.t_start:.3f} с"
        if self.t_end is not None:
            timing += f" .. {self.t_end:.3f} с"
        return f"Обрыв фаз(ы) {ph_str} ({timing})"