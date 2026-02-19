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

    # Механика
    omega_r: np.ndarray

    #Метаданные
    params: MachineParameters
    scenario_name: str = ""
    solver_name: str = ""

    # Опциональные производные ряды.
    # По умолчанию не заполняются в SimulationBuilder.run(), чтобы хранить
    # только сырой выход солвера.
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
    Mc: Optional[np.ndarray] = None  # момент нагрузки на валу

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

        # Speed/slip from raw omega_r.
        n_rpm = self.omega_r * 60.0 / (2.0 * np.pi)
        lines.append(f"  Скорость (уст.): {np.mean(n_rpm[ss]):.1f} об/мин")

        source = self.extra.get("source")
        omega_sync = self.params.omega_sync
        if source is not None:
            f_src = source.electrical_frequency_hz()
            if f_src is not None:
                omega_sync = 2.0 * np.pi * float(f_src) / float(self.params.p)
        if abs(omega_sync) > 1e-10:
            slip = (omega_sync - self.omega_r) / omega_sync
            lines.append(f"  Скольжение (уст.): {np.mean(slip[ss]):.5f}")

        # Resultant stator current module from raw phase currents.
        i1_mod = np.sqrt((self.i1B - self.i1C) ** 2 / 3.0 + self.i1A ** 2)
        i1_p50 = float(np.median(i1_mod[ss]))
        i1_p95 = float(np.percentile(i1_mod[ss], 95))
        lines.append(f"  |I1| (уст., P50): {i1_p50:.1f} А")
        lines.append(f"  |I1| (уст., P95): {i1_p95:.1f} А")

        i1a_rms = np.sqrt(np.mean(self.i1A[ss] ** 2))
        i1b_rms = np.sqrt(np.mean(self.i1B[ss] ** 2))
        lines.append(f"  I1A (уст., фазн., RMS): {i1a_rms:.1f} А")
        lines.append(f"  I1B (уст., фазн., RMS): {i1b_rms:.1f} А")

        # Optional model/source-based stats from extra metadata.
        machine = self.extra.get("machine")
        if machine is not None:
            mem = np.fromiter(
                (
                    machine.electromagnetic_torque(
                        self.i1A[k], self.i1B[k], self.i1C[k],
                        self.i2a[k], self.i2b[k], self.i2c[k],
                    )
                    for k in range(self.N)
                ),
                dtype=float,
                count=self.N,
            )
            lines.append(f"  Mэм (уст.): {np.mean(mem[ss]):.1f} Нм")
            p_mech = mem * self.omega_r
            lines.append(f"  P_мех (уст.): {np.mean(p_mech[ss]) / 1e3:.1f} кВт")

        rhs = self.extra.get("rhs")
        if source is not None:
            R_src = self.extra.get("source_r_matrix")
            L_src = self.extra.get("source_l_matrix")
            if R_src is None:
                R_src = np.asarray(source.series_resistance_matrix(), dtype=float)
            if L_src is None:
                L_src = np.asarray(source.series_inductance_matrix(), dtype=float)
            R_src = np.asarray(R_src, dtype=float)
            L_src = np.asarray(L_src, dtype=float)
            if R_src.shape != (3, 3):
                raise ValueError(
                    f"source_r_matrix has shape {R_src.shape}, expected (3, 3)."
                )
            if L_src.shape != (3, 3):
                raise ValueError(
                    f"source_l_matrix has shape {L_src.shape}, expected (3, 3)."
                )

            has_source_drop = bool(np.any(np.abs(R_src) > 0.0) or np.any(np.abs(L_src) > 0.0))
            if has_source_drop and rhs is None:
                lines.append("  P_элек (уст.): n/a (нет rhs для учета падения в источнике)")
            else:
                p_elec = np.zeros(self.N)
                for k in range(self.N):
                    u = np.asarray(source(self.t[k]), dtype=float)
                    if u.shape != (3,):
                        raise ValueError(
                            f"{source.__class__.__name__} returned shape {u.shape}, expected (3,)."
                        )
                    if has_source_drop:
                        yk = np.array([
                            self.i1A[k], self.i1B[k], self.i1C[k],
                            self.i2a[k], self.i2b[k], self.i2c[k],
                            self.omega_r[k],
                        ])
                        di_stator = rhs(self.t[k], yk)[0:3]
                        i_stator = np.array([self.i1A[k], self.i1B[k], self.i1C[k]])
                        u = u - R_src @ i_stator - L_src @ di_stator

                    p_elec[k] = (
                        u[0] * self.i1A[k]
                        + u[1] * self.i1B[k]
                        + u[2] * self.i1C[k]
                    )
                lines.append(f"  P_элек (уст.): {np.mean(p_elec[ss]) / 1e3:.1f} кВт")

        return "\n".join(lines)
