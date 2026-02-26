"""Nonlinear induction machine model with variable stator/rotor resistances."""
from __future__ import annotations

from typing import Callable

import numpy as np

from core.parameters import MachineParameters
from .base import MachineModel
from .parameter_laws import ElectricalLawContext, ElectricalParameterLaws


class NonlinearInductionMachine(MachineModel):
    """Asynchronous machine model with dynamic Rs/Rr and constant Lm."""

    def __init__(
        self,
        params: MachineParameters,
        laws: ElectricalParameterLaws | None = None,
    ):
        super().__init__(params)

        self._rs0 = params.R1
        self._rr0 = params.R2

        self.Lm = params.Lm
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
    ) -> tuple[float, float]:
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
        return self._laws.evaluate(
            context=context,
            rs0=self._rs0,
            rr0=self._rr0,
        )

    def _build_inductance_matrix(self) -> np.ndarray:
        mm = 2.0 * self.Lm / 3.0
        m_off = -0.5 * mm

        l1_diag = self.L1s + mm
        l2_diag = self.L2s + mm

        lss = np.array([
            [l1_diag, m_off, m_off],
            [m_off, l1_diag, m_off],
            [m_off, m_off, l1_diag],
        ])
        lrr = np.array([
            [l2_diag, m_off, m_off],
            [m_off, l2_diag, m_off],
            [m_off, m_off, l2_diag],
        ])
        lsr = np.array([
            [mm, m_off, m_off],
            [m_off, mm, m_off],
            [m_off, m_off, mm],
        ])

        return np.block([
            [lss, lsr],
            [lsr, lrr],
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
    ) -> np.ndarray:
        omega_e = omega_mech * self.p
        inv_sqrt3 = 1.0 / np.sqrt(3)

        ea = omega_e * inv_sqrt3 * (self.L2s * (i2b - i2c) + self.Lm * (imB - imC))
        eb = omega_e * inv_sqrt3 * (self.L2s * (i2c - i2a) + self.Lm * (imC - imA))
        ec = omega_e * inv_sqrt3 * (self.L2s * (i2a - i2b) + self.Lm * (imA - imB))
        return np.array([ea, eb, ec])

    @staticmethod
    def _electromagnetic_torque_with_lm(
        p: int,
        lm: float,
        i1A: float,
        i1B: float,
        i1C: float,
        i2a: float,
        i2b: float,
        i2c: float,
    ) -> float:
        return (p * lm / np.sqrt(3.0)) * (
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
        return self._electromagnetic_torque_with_lm(
            self.p, self.Lm, i1A, i1B, i1C, i2a, i2b, i2c
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
        mm = 2.0 * self.Lm / 3.0
        return (
            self.L1s * i1A
            + mm * (i1A - (i1B + i1C) / 2.0)
            + mm * (i2a - (i2b + i2c) / 2.0)
        )

    def electrical_matrices(
        self,
        t: float,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rs, rr = self._electrical_params(t, y)
        l = self._build_inductance_matrix()

        i1A, i1B, i1C = y[0], y[1], y[2]
        i2a, i2b, i2c = y[3], y[4], y[5]
        omega_r = y[6]

        imA = i1A + i2a
        imB = i1B + i2b
        imC = i1C + i2c
        e_rot = self._rotor_emf(i2a, i2b, i2c, imA, imB, imC, omega_r)

        b0 = np.array([
            -rs * i1A,
            -rs * i1B,
            -rs * i1C,
            -rr * i2a - e_rot[0],
            -rr * i2b - e_rot[1],
            -rr * i2c - e_rot[2],
        ])
        return l, b0

    def mechanical_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc: float,
    ) -> float:
        _ = t
        i1A, i1B, i1C = y[0], y[1], y[2]
        i2a, i2b, i2c = y[3], y[4], y[5]
        mem = self._electromagnetic_torque_with_lm(
            self.p, self.Lm, i1A, i1B, i1C, i2a, i2b, i2c
        )
        return (mem - Mc) / self.J

    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc_func: Callable[[float, float], float],
        U_func: Callable[[float], np.ndarray],
    ) -> np.ndarray:
        us = U_func(t)
        l, b0 = self.electrical_matrices(t, y)
        b = b0.copy()
        b[0:3] += us
        di_dt = np.linalg.solve(l, b)

        mc = Mc_func(t, y[6])
        domega_dt = self.mechanical_rhs(t, y, mc)

        dydt = np.empty(7)
        dydt[0:6] = di_dt
        dydt[6] = domega_dt
        return dydt
