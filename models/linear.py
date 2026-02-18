"""
Линейная модель асинхронной машины в фазных координатах ABC.

Модель НЕ учитывает:
  - насыщение магнитопровода
  - вытеснение тока в роторе
  - потери в стали

Это базовая физическая модель для дальнейшего расширения.
"""
from __future__ import annotations

from typing import Callable, Optional

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
        self.fn = params.fn

        self.L1 = self.L1s + self.Lm
        self.L2 = self.L2s + self.Lm
        self.Um = params.Um

        # Матрица индуктивностей постоянна для линейной модели → кэшируем
        self._L_matrix = self._build_inductance_matrix()
        self._L_inv = np.linalg.inv(self._L_matrix)

    # ------------------------------------------------------------------
    # Внутренние расчёты
    # ------------------------------------------------------------------

    def _build_inductance_matrix(self) -> np.ndarray:
        """
        Матрица индуктивностей 6x6
        """
        L1 = self.L1
        L2 = self.L2
        Lm = self.Lm
        return np.array([
            [L1,  0,  0, Lm,  0,  0],
            [ 0, L1,  0,  0, Lm,  0],
            [ 0,  0, L1,  0,  0, Lm],
            [Lm,  0,  0, L2,  0,  0],
            [ 0, Lm,  0,  0, L2,  0],
            [ 0,  0, Lm,  0,  0, L2],
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


    # Публичный интерфейс MachineModel
    def stator_voltages(
        self, t: float,
        freq: Optional[float] = None,
        amplitude: Optional[float] = None,
    ) -> np.ndarray:
        """Трёхфазное синусоидальное напряжение статора"""
        f = freq if freq is not None else self.fn
        Um = amplitude if amplitude is not None else self.Um
        omega = 2 * np.pi * f

        U1A = Um * np.sin(omega * t)
        U1B = Um * np.sin(omega * t - 2 * np.pi / 3)
        U1C = Um * np.sin(omega * t + 2 * np.pi / 3)
        return np.array([U1A, U1B, U1C])

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

    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc_func: Callable[[float, float], float],
        U_func: Optional[Callable[[float], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Правая часть ОДУ:
            [L]*[di/dt] = [U] − [R*i] − [E_rot]
            J*d(omega)/dt = Mэм − Mc
        """
        i1A, i1B, i1C = y[0], y[1], y[2]
        i2a, i2b, i2c = y[3], y[4], y[5]
        omega_r = y[6]

        # Токи намагничивания
        imA = i1A + i2a
        imB = i1B + i2b
        imC = i1C + i2c

        # Напряжения статора
        if U_func is not None:
            Us = U_func(t)
        else:
            Us = self.stator_voltages(t)

        # Короткозамкнутый ротор: Ur = 0
        E_rot = self._rotor_emf(i2a, i2b, i2c, imA, imB, imC, omega_r)

        # Правая часть электрических уравнений
        b = np.array([
            Us[0] - self.R1 * i1A,
            Us[1] - self.R1 * i1B,
            Us[2] - self.R1 * i1C,
            -self.R2 * i2a - E_rot[0],
            -self.R2 * i2b - E_rot[1],
            -self.R2 * i2c - E_rot[2],
        ])

        # di/dt = invL * b  (L постоянна -> используем кэш)
        di_dt = self._L_inv @ b

        # Механическое уравнение
        Mem = self.electromagnetic_torque(i1A, i1B, i1C, i2a, i2b, i2c)
        Mc = Mc_func(t, omega_r)
        domega_dt = (Mem - Mc) / self.J

        dydt = np.empty(7)
        dydt[0:6] = di_dt
        dydt[6] = domega_dt
        return dydt