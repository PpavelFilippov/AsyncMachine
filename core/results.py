"""
Контейнер результатов моделирования.

Хранит временные ряды всех переменных и метаданные запуска
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np

from .parameters import MachineParameters


@dataclass
class SimulationResults:
    """Результаты одного прогона моделирования"""

    #Время
    t: np.ndarray

    #Токи статора
    i1A: np.ndarray
    i1B: np.ndarray
    i1C: np.ndarray

    #Токи ротора
    i2a: np.ndarray
    i2b: np.ndarray
    i2c: np.ndarray

    #Механика
    omega_r: np.ndarray

    #Метаданные
    params: MachineParameters
    scenario_name: str = ""
    solver_name: str = ""

    #Вычисляемые (заполняются post_process)
    Mem: Optional[np.ndarray] = None
    I1_mod: Optional[np.ndarray] = None
    I2_mod: Optional[np.ndarray] = None
    Im_mod: Optional[np.ndarray] = None
    n_rpm: Optional[np.ndarray] = None
    slip: Optional[np.ndarray] = None
    Psi1A: Optional[np.ndarray] = None
    U1A: Optional[np.ndarray] = None
    U1B: Optional[np.ndarray] = None
    U1C: Optional[np.ndarray] = None
    U_mod: Optional[np.ndarray] = None
    imA: Optional[np.ndarray] = None
    imB: Optional[np.ndarray] = None
    imC: Optional[np.ndarray] = None
    P_elec: Optional[np.ndarray] = None
    P_mech: Optional[np.ndarray] = None

    # Дополнительные данные (для расширения)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_solver_output(
        cls,
        t: np.ndarray,
        y: np.ndarray,
        params: MachineParameters,
        scenario_name: str = "",
        solver_name: str = "",
    ) -> SimulationResults:
        """Создать из массива solve_ivp-стиля (y shape = [7, N])"""
        return cls(
            t=t,
            i1A=y[0], i1B=y[1], i1C=y[2],
            i2a=y[3], i2b=y[4], i2c=y[5],
            omega_r=y[6],
            params=params,
            scenario_name=scenario_name,
            solver_name=solver_name,
        )

    @property
    def N(self) -> int:
        return len(self.t)

    def steady_state_slice(self, fraction: float = 0.75) -> slice:
        """Срез для анализа установившегося режима (последние 25% данных)"""
        idx = int(fraction * self.N)
        return slice(idx, None)

    def summary(self) -> str:
        """Краткая сводка результатов установившегося режима"""
        ss = self.steady_state_slice()
        lines = [
            f"  Сценарий: {self.scenario_name}",
            f"  Солвер: {self.solver_name}",
            f"  Точек: {self.N}, t = [{self.t[0]:.3f} .. {self.t[-1]:.3f}] с",
        ]
        if self.n_rpm is not None:
            lines.append(f"  Скорость (уст.): {np.mean(self.n_rpm[ss]):.1f} об/мин")
        if self.slip is not None:
            lines.append(f"  Скольжение (уст.): {np.mean(self.slip[ss]):.5f}")
        if self.Mem is not None:
            lines.append(f"  Mэм (уст.): {np.mean(self.Mem[ss]):.1f} Нм")
        if self.I1_mod is not None:
            lines.append(f"  |I1| (уст.): {np.mean(self.I1_mod[ss]):.1f} А")
        if self.P_elec is not None:
            lines.append(f"  P_элек (уст.): {np.mean(self.P_elec[ss]) / 1e3:.1f} кВт")
        if self.P_mech is not None:
            lines.append(f"  P_мех (уст.): {np.mean(self.P_mech[ss]) / 1e3:.1f} кВт")
        return "\n".join(lines)