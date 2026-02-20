"""
Схемный ассемблер: собирает расширенную систему ОДУ из топологии.

Ключевая идея: двигатель предоставляет (L_m, b0_m) через
electrical_matrices(). Ассемблер алгебраически исключает напряжение
на зажимах u_term и строит расширенную систему L_ext * di/dt = rhs_ext,
включающую источник и контуры КЗ. Двигатель не модифицируется.

Математика:
    L_ext = | L_m + L_src_embed       L_src * D_embed^T    |
            | D_embed * L_src          D * L_src * D^T      |

    rhs_ext = | b0_m[0:3] + e_src - R_src*(i_st + D^T*i_f)  |  статор
              | b0_m[3:6]                                      |  ротор
              | D*e_src - D*R_src*(i_st + D^T*i_f) - R_f*i_f  |  КЗ
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.linalg import block_diag

from .topology import CircuitTopology
from .components import FaultBranchComponent


class CircuitAssembler:
    """
    Собирает ODE RHS из CircuitTopology.

    Для линейной модели L_m = const, поэтому L_ext кэшируется
    отдельно для каждой конфигурации активных КЗ
    """

    def __init__(self, topology: CircuitTopology):
        if not topology.is_finalized:
            raise RuntimeError("Topology must be finalized before assembly")
        self._topo = topology
        self._L_src = topology.source.L_src
        self._R_src = topology.source.R_src
        self._L_ext_cache: dict[frozenset, np.ndarray] = {}

    def _active_key(self, faults: list[FaultBranchComponent]) -> frozenset:
        """Хэшируемый ключ для набора активных контуров КЗ"""
        return frozenset(id(f) for f in faults)

    def build_L_ext(
        self,
        L_m: np.ndarray,
        active_faults: list[FaultBranchComponent],
    ) -> np.ndarray:
        """
        Расширенная матрица индуктивностей.

        Без активных КЗ: (6x6) = L_m + L_src_embed.
        С КЗ: (6+n_extra) x (6+n_extra)
        """
        L_src = self._L_src

        if not active_faults:
            L_ext = L_m.copy()
            L_ext[0:3, 0:3] += L_src
            return L_ext

        D_stacked = np.vstack([f.D for f in active_faults])  # [n_extra x 3]
        n_extra = D_stacked.shape[0]
        n_elec = 6 + n_extra

        L_ext = np.zeros((n_elec, n_elec), dtype=float)

        # Верхний-левый блок: L_m + L_src_embed
        L_ext[0:6, 0:6] = L_m
        L_ext[0:3, 0:3] += L_src

        # Верхний-правый: L_src @ D^T (статорные строки 0:3)
        L_ext[0:3, 6:n_elec] = L_src @ D_stacked.T

        # Нижний-левый: D @ L_src (статорные столбцы 0:3)
        L_ext[6:n_elec, 0:3] = D_stacked @ L_src

        # Нижний-правый: D @ L_src @ D^T
        L_ext[6:n_elec, 6:n_elec] = D_stacked @ L_src @ D_stacked.T

        return L_ext

    def build_rhs_ext(
        self,
        b0_m: np.ndarray,
        e_src: np.ndarray,
        i_stator: np.ndarray,
        active_faults: list[FaultBranchComponent],
        y_global: np.ndarray,
    ) -> np.ndarray:
        """
        Правая часть расширенной электрической системы
        """
        R_src = self._R_src

        if not active_faults:
            rhs = b0_m.copy()
            rhs[0:3] += e_src - R_src @ i_stator
            return rhs

        D_stacked = np.vstack([f.D for f in active_faults])
        n_extra = D_stacked.shape[0]

        # Извлечь токи КЗ активных контуров
        i_fault = self._extract_fault_currents(active_faults, y_global)

        # Блочно-диагональная R_fault
        R_f = self._build_R_fault(active_faults)

        # KCL: эффективный ток источника
        i_src_eff = i_stator + D_stacked.T @ i_fault

        n_elec = 6 + n_extra
        rhs = np.zeros(n_elec, dtype=float)

        # Статор (0:3)
        rhs[0:3] = b0_m[0:3] + e_src - R_src @ i_src_eff

        # Ротор (3:6) — без изменений
        rhs[3:6] = b0_m[3:6]

        # Контуры КЗ (6:)
        rhs[6:n_elec] = (
            D_stacked @ e_src
            - (D_stacked @ R_src) @ i_src_eff
            - R_f @ i_fault
        )

        return rhs

    def _extract_fault_currents(
        self,
        active_faults: list[FaultBranchComponent],
        y_global: np.ndarray,
    ) -> np.ndarray:
        """Собрать токи КЗ всех активных контуров в один вектор"""
        parts = [f.extract_currents(y_global) for f in active_faults]
        return np.concatenate(parts) if parts else np.array([], dtype=float)

    @staticmethod
    def _build_R_fault(
        active_faults: list[FaultBranchComponent],
    ) -> np.ndarray:
        """Блочно-диагональная матрица сопротивлений КЗ"""
        blocks = [f.R_fault for f in active_faults]
        if not blocks:
            return np.zeros((0, 0), dtype=float)
        if len(blocks) == 1:
            return blocks[0].copy()
        return block_diag(*blocks)

    def build_ode_rhs(
        self,
        active_faults: list[FaultBranchComponent],
        state_size: int,
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """
        Построить функцию rhs(t, y) -> dydt для solve_ivp.

        Параметры:
            active_faults: список активных контуров КЗ, определяет размерность
            state_size:    полная размерность вектора состояния
        """
        topo = self._topo
        motor_comp = topo.motor
        source_comp = topo.source
        motor_offset = motor_comp.state_offset

        # Кэшируем L_ext: для линейной модели L_m = const
        key = self._active_key(active_faults)
        if key not in self._L_ext_cache:
            L_m, _ = motor_comp._motor.electrical_matrices(0.0, np.zeros(7))
            L_m = np.asarray(L_m, dtype=float)
            self._L_ext_cache[key] = self.build_L_ext(L_m, active_faults)
        L_ext_cached = self._L_ext_cache[key]

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            # 1. Электрические матрицы двигателя (неизменный интерфейс)
            _, b0_m = motor_comp.electrical_matrices(t, y)

            # 2. ЭДС ист
            e_src = source_comp.emf(t)

            # 3. Токи статора
            i_stator = y[motor_offset: motor_offset + 3]

            # 4. Правая часть расширенной системы
            rhs_ext = self.build_rhs_ext(
                b0_m, e_src, i_stator, active_faults, y,
            )

            # 5. Решение электрической подсистемы
            di_ext = np.linalg.solve(L_ext_cached, rhs_ext)

            # 6. Механическое уравнение
            domega_dt = motor_comp.mechanical_rhs(t, y)

            # 7. Сборка dydt
            dydt = np.zeros(state_size, dtype=float)

            # Электрические производные двигателя (6 из di_ext)
            dydt[motor_offset: motor_offset + 6] = di_ext[0:6]
            # Механическая производная
            dydt[motor_offset + 6] = domega_dt

            # Производные токов КЗ
            if active_faults:
                fault_di = di_ext[6:]
                idx = 0
                for f in active_faults:
                    n = f.n_diff_vars
                    dydt[f.state_offset: f.state_offset + n] = fault_di[idx: idx + n]
                    idx += n

            return dydt

        return rhs

    def validate_source_impedance(
        self,
        active_faults: list[FaultBranchComponent],
    ) -> None:
        """
        Проверка: контур КЗ не вырожден, то есть источник имеет ненулевую индуктивность в контуре КЗ
        (источник имеет ненулевую индуктивность в контуре КЗ).
        """
        if not active_faults:
            return
        D_stacked = np.vstack([f.D for f in active_faults])
        L_ff = D_stacked @ self._L_src @ D_stacked.T
        eigs = np.linalg.eigvalsh(L_ff)
        if np.min(eigs) < 1e-15:
            raise ValueError(
                "Контур КЗ вырожден: источник имеет нулевой импеданс (L_src approx 0). "
                "Для моделирования КЗ необходим источник с ненулевой индуктивностью "
                f"(Thevenin source). min(eig(L_fault_loop)) = {np.min(eigs):.3e}"
            )
