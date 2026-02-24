"""
    Модуль models/nonlinear.py.
    Состав:
    Классы: NonlinearInductionMachine.
    Функции: нет.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from core.parameters import MachineParameters
from .base import MachineModel
from .parameter_laws import ElectricalLawContext, ElectricalParameterLaws


class NonlinearInductionMachine(MachineModel):
    """
    Модель асинхронной машины с динамическими параметрами Rs, Rr, Lm.
    Законы параметров передаются через ElectricalParameterLaws.
    """

    def __init__(
        self,
        params: MachineParameters,
        laws: ElectricalParameterLaws | None = None,
    ):
        super().__init__(params)

        self._rs0 = params.R1
        self._rr0 = params.R2
        self._lm0 = params.Lm

        self.L1s = params.L1sigma
        self.L2s = params.L2sigma
        self.J = params.J
        self.p = params.p

        self._laws = laws or ElectricalParameterLaws()

    @property
    def laws(self) -> ElectricalParameterLaws:
        return self._laws

    def set_laws(self, laws: ElectricalParameterLaws) -> None:
        self._laws = laws

    def _electrical_params(
        self,
        t: float,
        y: np.ndarray,
    ) -> tuple[float, float, float]:
        context = ElectricalLawContext(
            t=float(t),
            i1A=float(y[0]),
            i1B=float(y[1]),
            i1C=float(y[2]),
            i2a=float(y[3]),
            i2b=float(y[4]),
            i2c=float(y[5]),
            omega_r=float(y[6]),
        )
        Rs, Rr, Lm = self._laws.evaluate(
            context=context,
            rs0=self._rs0,
            rr0=self._rr0,
            lm0=self._lm0,
        )
        return Rs, Rr, Lm

    def _build_inductance_matrix(self, Lm: float) -> np.ndarray:
        """Строит матрицу индуктивностей статора/ротора для текущего Lm."""

        Mm = 2.0 * Lm / 3.0
        m_off = -0.5 * Mm

        L1_diag = self.L1s + Mm
        L2_diag = self.L2s + Mm

        Lss = np.array([
            [L1_diag, m_off, m_off],
            [m_off, L1_diag, m_off],
            [m_off, m_off, L1_diag],
        ])
        Lrr = np.array([
            [L2_diag, m_off, m_off],
            [m_off, L2_diag, m_off],
            [m_off, m_off, L2_diag],
        ])
        Lsr = np.array([
            [Mm, m_off, m_off],
            [m_off, Mm, m_off],
            [m_off, m_off, Mm],
        ])

        return np.block([
            [Lss, Lsr],
            [Lsr, Lrr],
        ])

    def _rotor_emf(
        self,
        i2a: float,
        i2b: float,
        i2c: float,
        imA: float,
        imB: float,
        imC: float,
        omega_mech: float,
        Lm: float,
    ) -> np.ndarray:
        """Вычисляет ЭДС ротора."""

        omega_e = omega_mech * self.p
        inv_sqrt3 = 1.0 / np.sqrt(3)

        Ea = omega_e * inv_sqrt3 * (self.L2s * (i2b - i2c) + Lm * (imB - imC))
        Eb = omega_e * inv_sqrt3 * (self.L2s * (i2c - i2a) + Lm * (imC - imA))
        Ec = omega_e * inv_sqrt3 * (self.L2s * (i2a - i2b) + Lm * (imA - imB))
        return np.array([Ea, Eb, Ec])

    @staticmethod
    def _electromagnetic_torque_with_lm(
        p: int,
        Lm: float,
        i1A: float,
        i1B: float,
        i1C: float,
        i2a: float,
        i2b: float,
        i2c: float,
    ) -> float:
        return (p * Lm / np.sqrt(3)) * (
            (i1A * i2c + i1B * i2a + i1C * i2b)
            - (i1A * i2b + i1B * i2c + i1C * i2a)
        )

    def electromagnetic_torque(
        self,
        i1A: float,
        i1B: float,
        i1C: float,
        i2a: float,
        i2b: float,
        i2c: float,
    ) -> float:
        """Возвращает электромагнитный момент для интерфейса MachineModel."""
        return self._electromagnetic_torque_with_lm(
            self.p, self._lm0, i1A, i1B, i1C, i2a, i2b, i2c
        )

    def flux_linkage_phaseA(
        self,
        i1A: float,
        i1B: float,
        i1C: float,
        i2a: float,
        i2b: float,
        i2c: float,
    ) -> float:
        """Возвращает потокосцепление фазы A для интерфейса MachineModel."""

        Mm = 2.0 * self._lm0 / 3.0
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
        """Формирует матрицу и правую часть электрической подсистемы."""

        Rs, Rr, Lm = self._electrical_params(t, y)
        L = self._build_inductance_matrix(Lm)

        i1A, i1B, i1C = y[0], y[1], y[2]
        i2a, i2b, i2c = y[3], y[4], y[5]
        omega_r = y[6]

        imA = i1A + i2a
        imB = i1B + i2b
        imC = i1C + i2c
        E_rot = self._rotor_emf(i2a, i2b, i2c, imA, imB, imC, omega_r, Lm)

        b0 = np.array([
            -Rs * i1A,
            -Rs * i1B,
            -Rs * i1C,
            -Rr * i2a - E_rot[0],
            -Rr * i2b - E_rot[1],
            -Rr * i2c - E_rot[2],
        ])
        return L, b0

    def mechanical_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc: float,
    ) -> float:
        """Вычисляет производную механической скорости."""

        _, _, Lm = self._electrical_params(t, y)
        i1A, i1B, i1C = y[0], y[1], y[2]
        i2a, i2b, i2c = y[3], y[4], y[5]
        Mem = self._electromagnetic_torque_with_lm(
            self.p, Lm, i1A, i1B, i1C, i2a, i2b, i2c
        )
        return (Mem - Mc) / self.J

    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc_func: Callable[[float, float], float],
        U_func: Callable[[float], np.ndarray],
    ) -> np.ndarray:
        """Вычисляет правую часть системы ОДУ."""

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
