"""
Контейнер результатов мульти-машинного моделирования.

Хранит результаты каждой машины как отдельный SimulationResults,
плюс общие данные (суммарные токи, напряжения источника)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np

from .results import SimulationResults


@dataclass
class MultiMachineResults:
    """Результаты моделирования нескольких машин на общей шине"""

    # Время (общее)
    t: np.ndarray

    # Результаты по каждой машине (SimulationResults с post-process)
    machines: List[SimulationResults]

    # Напряжения общей шины (до применения индивидуальных faults)
    U_bus_A: Optional[np.ndarray] = None
    U_bus_B: Optional[np.ndarray] = None
    U_bus_C: Optional[np.ndarray] = None

    # Суммарные токи от источника
    I_total_A: Optional[np.ndarray] = None
    I_total_B: Optional[np.ndarray] = None
    I_total_C: Optional[np.ndarray] = None

    # Метаданные
    scenario_name: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_machines(self) -> int:
        return len(self.machines)

    @property
    def N(self) -> int:
        return len(self.t)

    def summary(self) -> str:
        lines = [
            f"  Сценарий: {self.scenario_name}",
            f"  Машин: {self.n_machines}",
            f"  Точек: {self.N}, t = [{self.t[0]:.3f} .. {self.t[-1]:.3f}] с",
            "",
        ]
        for i, res in enumerate(self.machines):
            lines.append(f"  --- Машина #{i + 1} ---")
            lines.append(res.summary())
            lines.append("")
        return "\n".join(lines)