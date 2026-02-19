"""
Линейная модель асинхронной машины в фазных координатах ABC.

Модель НЕ учитывает:
  - насыщение магнитопровода
  - вытеснение тока в роторе
  - потери в стали

Это базовая физическая модель для дальнейшего расширения.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from core.parameters import MachineParameters
from .base import MachineModel


class LinearInductionMachine(MachineModel):
    """
    Линейная модель АМ в фазных координатах ABC.

    Вектор состояния: y = [i1A, i1B, i1C, i2a, i2b, i2c, omega_r]
    """

    def __init__(self, params: MachineParameters):
        super().__init__(params)

        self.R1 = params.R1
        self.R2 = params.R2
        self.L1s = params.L1sigma
        self.L2s = params.L2sigma
        self.Lm = params.Lm
        self.J = params.J
        self.p = params.p

        self.L1 = self.L1s + self.Lm
        self.L2 = self.L2s + self.Lm

        # Матрица индуктивностей постоянна для линейной модели → кэшируем
        self._L_matrix = self._build_inductance_matrix()

    # ------------------------------------------------------------------
    # Внутренние расчёты
    # ------------------------------------------------------------------

    def _build_inductance_matrix(self) -> np.ndarray:
        """
        Полная матрица индуктивностей 6x6 в фазных координатах ABC.

        Используется симметричная линейная модель с учётом взаимных связей
        между всеми фазами статора и ротора:
          - диагональные элементы: Lsigma + 2/3*Lm
          - взаимные межфазные:   -1/3*Lm
          - статер-ротор:
              co-phase   +2/3*Lm
              crossphase -1/3*Lm
        """
        Mm = 2.0 * self.Lm / 3.0
        m_off = -0.5 * Mm

        L1_diag = self.L1s + Mm
        L2_diag = self.L2s + Mm

        Lss = np.array([
            [L1_diag, m_off,   m_off],
            [m_off,   L1_diag, m_off],
            [m_off,   m_off,   L1_diag],
        ])
        Lrr = np.array([
            [L2_diag, m_off,   m_off],
            [m_off,   L2_diag, m_off],
            [m_off,   m_off,   L2_diag],
        ])
        Lsr = np.array([
            [Mm,    m_off, m_off],
            [m_off, Mm,    m_off],
            [m_off, m_off, Mm],
        ])

        return np.block([
            [Lss, Lsr],
            [Lsr, Lrr],
        ])

    def _rotor_emf(
        self,
        i2a: float, i2b: float, i2c: float,
        imA: float, imB: float, imC: float,
        omega_mech: float,
    ) -> np.ndarray:
        """Противо-ЭДС ротора от вращения"""
        omega_e = omega_mech * self.p
        inv_sqrt3 = 1.0 / np.sqrt(3)

        Ea = omega_e * inv_sqrt3 * (self.L2s * (i2b - i2c) + self.Lm * (imB - imC))
        Eb = omega_e * inv_sqrt3 * (self.L2s * (i2c - i2a) + self.Lm * (imC - imA))
        Ec = omega_e * inv_sqrt3 * (self.L2s * (i2a - i2b) + self.Lm * (imA - imB))
        return np.array([Ea, Eb, Ec])


    def electromagnetic_torque(
        self,
        i1A: float, i1B: float, i1C: float,
        i2a: float, i2b: float, i2c: float,
    ) -> float:
        """Электромагнитный момент"""
        return (self.p * self.Lm / np.sqrt(3)) * (
            (i1A * i2c + i1B * i2a + i1C * i2b)
            - (i1A * i2b + i1B * i2c + i1C * i2a)
        )

    def flux_linkage_phaseA(
        self,
        i1A: float, i1B: float, i1C: float,
        i2a: float, i2b: float, i2c: float,
    ) -> float:
        """Потокосцепление фазы A статора"""
        Mm = 2.0 * self.Lm / 3.0
        return (
            self.L1s * i1A
            + Mm * (i1A - (i1B + i1C) / 2.0)
            + Mm * (i2a - (i2b + i2c) / 2.0)
        )

    def electrical_matrices(
        self,
        t: float,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Электрическая подсистема (без напряжения статора источника):
            L * di/dt = b0 + [UsA, UsB, UsC, 0, 0, 0]
        """
        i1A, i1B, i1C = y[0], y[1], y[2]
        i2a, i2b, i2c = y[3], y[4], y[5]
        omega_r = y[6]

        imA = i1A + i2a
        imB = i1B + i2b
        imC = i1C + i2c
        E_rot = self._rotor_emf(i2a, i2b, i2c, imA, imB, imC, omega_r)

        b0 = np.array([
            -self.R1 * i1A,
            -self.R1 * i1B,
            -self.R1 * i1C,
            -self.R2 * i2a - E_rot[0],
            -self.R2 * i2b - E_rot[1],
            -self.R2 * i2c - E_rot[2],
        ])
        return self._L_matrix, b0

    def mechanical_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc: float,
    ) -> float:
        i1A, i1B, i1C = y[0], y[1], y[2]
        i2a, i2b, i2c = y[3], y[4], y[5]
        Mem = self.electromagnetic_torque(i1A, i1B, i1C, i2a, i2b, i2c)
        return (Mem - Mc) / self.J

    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc_func: Callable[[float, float], float],
        U_func: Callable[[float], np.ndarray],
    ) -> np.ndarray:
        """
        Правая часть ОДУ:
            [L]*[di/dt] = [U] − [R*i] − [E_rot]
            J*d(omega)/dt = Mэм − Mc
        """
        Us = U_func(t)
        L, b0 = self.electrical_matrices(t, y)
        b = b0.copy()
        b[0:3] += Us
        di_dt = np.linalg.solve(L, b)

        Mc = Mc_func(t, y[6])
        domega_dt = self.mechanical_rhs(t, y, Mc)

        dydt = np.empty(7)
        dydt[0:6] = di_dt
        dydt[6] = domega_dt
        return dydt
