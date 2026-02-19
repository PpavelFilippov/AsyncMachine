"""
Линейная модель АМ с поддержкой межфазного КЗ на клеммах статора.

Расширяет LinearInductionMachine, добавляя:
  - Учёт межфазного КЗ в правой части ОДУ
  - Предвычисление обратных матриц для нормального и аварийного режимов
  - Корректный расчёт клеммных напряжений при КЗ

Физика:
  При КЗ между n_f фазами (множество F) на клеммах образуется
  общий электрический узел с напряжением:

    U_f = (1/n_f) * Σ E_k - (Z_src/n_f) * Σ i_1k,   k ∈ F

  Это модифицирует эффективную матрицу индуктивностей:
    L_eff[i, j] += L_src / n_f   для i, j ∈ F

  И вектор правой части:
    b[k] = E_avg - (R1 + R_src/n_f)*i_1k - (R_src/n_f)*Σ(i_1j, j≠k),  k ∈ F
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from core.parameters import MachineParameters
from faults.base import Fault
from faults.short_circuit import InterPhaseShortCircuit
from .linear import LinearInductionMachine


class LinearInductionMachineSC(LinearInductionMachine):
    """
    Линейная модель АМ с межфазным КЗ на клеммах статора.

    Использование:
        machine = LinearInductionMachineSC(params, faults=faults,
                                           r_src=0.01, l_src=1e-4)

    Модель сама управляет переключением между нормальным режимом
    и режимом КЗ на основе is_active(t) каждого фолта.
    """

    def __init__(
        self,
        params: MachineParameters,
        faults: Optional[List[Fault]] = None,
        r_src: float = 0.0,
        l_src: float = 0.0,
    ):
        super().__init__(params)

        self.r_src = r_src
        self.l_src = l_src

        faults = faults or []
        self._sc_faults: List[InterPhaseShortCircuit] = [
            f for f in faults if isinstance(f, InterPhaseShortCircuit)
        ]
        self._non_sc_faults: List[Fault] = [
            f for f in faults if not isinstance(f, InterPhaseShortCircuit)
        ]

        # Предвычислить обратные матрицы индуктивностей
        # Нормальный режим (с Z_src)
        L_eff_normal = self._L_matrix.copy()
        for ph in range(3):
            L_eff_normal[ph, ph] += l_src
        self._L_inv_normal = np.linalg.inv(L_eff_normal)

        # Режимы КЗ (для каждого InterPhaseShortCircuit)
        self._L_inv_sc = {}
        for i, sc in enumerate(self._sc_faults):
            self._L_inv_sc[i] = sc.precompute_L_eff_inv(
                self._L_matrix, l_src
            )

    # ------------------------------------------------------------------
    # Правая часть ОДУ
    # ------------------------------------------------------------------

    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc_func: Callable[[float, float], float],
        U_func: Callable[[float], np.ndarray],
    ) -> np.ndarray:
        """
        Правая часть ОДУ с учётом межфазного КЗ.

        U_func должна возвращать «сырую» ЭДС источника (без КЗ-модификаций).
        Не-КЗ фолты (обрыв, замыкание на землю) применяются через modify_voltages.

        Args:
            t: текущее время
            y: вектор состояния [i1A, i1B, i1C, i2a, i2b, i2c, omega_r]
            Mc_func: функция момента сопротивления Mc(t, omega_r)
            U_func: функция ЭДС источника E(t) -> [E_A, E_B, E_C]
        """
        i1A, i1B, i1C = y[0], y[1], y[2]
        i2a, i2b, i2c = y[3], y[4], y[5]
        omega_r = y[6]

        imA = i1A + i2a
        imB = i1B + i2b
        imC = i1C + i2c
        E_rot = self._rotor_emf(i2a, i2b, i2c, imA, imB, imC, omega_r)

        E_src_raw = U_func(t)
        y_stator = np.array([i1A, i1B, i1C])
        y_rotor = np.array([i2a, i2b, i2c])

        # Определить активный КЗ-фолт
        active_sc_idx = None
        for i, sc in enumerate(self._sc_faults):
            if sc.is_active(t):
                active_sc_idx = i
                break

        if active_sc_idx is not None:
            # Режим межфазного КЗ
            sc = self._sc_faults[active_sc_idx]
            L_inv = self._L_inv_sc[active_sc_idx]

            # Применить не-КЗ фолты к ЭДС источника
            E_src = E_src_raw.copy()
            for fault in self._non_sc_faults:
                E_src = fault.modify_voltages(t, E_src)

            b = sc.build_b_vector(
                E_src, y_stator,
                self.R1, self.R2, E_rot, y_rotor,
                self.r_src,
            )
        else:
            # Нормальный режим (с неидеальным источником)
            L_inv = self._L_inv_normal

            # Применить все фолты к ЭДС
            E_src = E_src_raw.copy()
            for fault in self._non_sc_faults:
                E_src = fault.modify_voltages(t, E_src)
            for fault in self._sc_faults:
                E_src = fault.modify_voltages(t, E_src)

            b = np.array([
                E_src[0] - (self.R1 + self.r_src) * i1A,
                E_src[1] - (self.R1 + self.r_src) * i1B,
                E_src[2] - (self.R1 + self.r_src) * i1C,
                -self.R2 * i2a - E_rot[0],
                -self.R2 * i2b - E_rot[1],
                -self.R2 * i2c - E_rot[2],
            ])

        di_dt = L_inv @ b

        Mem = self.electromagnetic_torque(i1A, i1B, i1C, i2a, i2b, i2c)
        Mc = Mc_func(t, omega_r)
        domega_dt = (Mem - Mc) / self.J

        dydt = np.empty(7)
        dydt[0:6] = di_dt
        dydt[6] = domega_dt
        return dydt

    # ------------------------------------------------------------------
    # Пост-обработка: клеммные напряжения
    # ------------------------------------------------------------------

    def compute_terminal_voltages(
        self,
        t: float,
        E_src_raw: np.ndarray,
        y_stator: np.ndarray,
        di_stator: np.ndarray,
    ) -> np.ndarray:
        """
        Вычислить клеммные напряжения статора с учётом КЗ.

        Args:
            t: текущее время
            E_src_raw: «сырая» ЭДС источника [E_A, E_B, E_C]
            y_stator: токи статора [i_1A, i_1B, i_1C]
            di_stator: производные токов [di_1A/dt, di_1B/dt, di_1C/dt]

        Returns:
            U_terminal: [U_A, U_B, U_C]
        """
        # Применить не-КЗ фолты к ЭДС
        E_src = E_src_raw.copy()
        for fault in self._non_sc_faults:
            E_src = fault.modify_voltages(t, E_src)

        # Найти активный КЗ-фолт
        active_sc = None
        for sc in self._sc_faults:
            if sc.is_active(t):
                active_sc = sc
                break

        if active_sc is not None:
            return active_sc.terminal_voltage(
                E_src, y_stator, di_stator,
                self.r_src, self.l_src,
            )

        # Нормальный режим: все фолты через modify_voltages
        for fault in self._sc_faults:
            E_src = fault.modify_voltages(t, E_src)

        r, l = self.r_src, self.l_src
        if abs(r) > 0.0 or abs(l) > 0.0:
            return np.array([
                E_src[0] - r * y_stator[0] - l * di_stator[0],
                E_src[1] - r * y_stator[1] - l * di_stator[1],
                E_src[2] - r * y_stator[2] - l * di_stator[2],
            ])
        return E_src
