"""
Неисправности: обрыв фазы и замыкание фазы на землю
"""
from __future__ import annotations

from typing import Optional, Set

from .base import Fault, FaultType


class OpenPhaseFault(Fault):
    """
    Обрыв фазы статора — физический разрыв проводника.

    Физика: ток через оборванную фазу = 0 (di/dt = 0, i = 0).
    Фаза исключается из контура, суммарный ток источника
    уменьшается, что через импеданс линии влияет на
    напряжение шины для всех машин
    """

    PHASE_MAP = {"A": 0, "B": 1, "C": 2}

    def __init__(
        self,
        phases: str | list[str],
        t_start: float = 0.0,
        t_end: Optional[float] = None,
    ):
        super().__init__(t_start=t_start, t_end=t_end)
        if isinstance(phases, str):
            phases = [phases]
        self.phases = [p.upper() for p in phases]
        self._indices = frozenset(self.PHASE_MAP[p] for p in self.phases)

    def fault_type(self) -> FaultType:
        return FaultType.OPEN_CIRCUIT

    def affected_phases(self) -> Set[int]:
        return set(self._indices)

    def describe(self) -> str:
        ph_str = ", ".join(self.phases)
        timing = f"t={self.t_start:.3f} с"
        if self.t_end is not None:
            timing += f" .. {self.t_end:.3f} с"
        return f"Обрыв фаз(ы) {ph_str} ({timing})"


class GroundFault(Fault):
    """
    Замыкание фазы статора на землю.

    Физика: напряжение на зажиме фазы = 0, но ток через
    фазу течёт свободно (определяется из уравнений).
    Ток КЗ нагружает источник через импеданс линии,
    просаживая напряжение шины для всех машин
    """

    PHASE_MAP = {"A": 0, "B": 1, "C": 2}

    def __init__(
        self,
        phases: str | list[str],
        t_start: float = 0.0,
        t_end: Optional[float] = None,
    ):
        super().__init__(t_start=t_start, t_end=t_end)
        if isinstance(phases, str):
            phases = [phases]
        self.phases = [p.upper() for p in phases]
        self._indices = frozenset(self.PHASE_MAP[p] for p in self.phases)

    def fault_type(self) -> FaultType:
        return FaultType.GROUND_FAULT

    def affected_phases(self) -> Set[int]:
        return set(self._indices)

    def describe(self) -> str:
        ph_str = ", ".join(self.phases)
        timing = f"t={self.t_start:.3f} с"
        if self.t_end is not None:
            timing += f" .. {self.t_end:.3f} с"
        return f"Замыкание на землю фаз(ы) {ph_str} ({timing})"