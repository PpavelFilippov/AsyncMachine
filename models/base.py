"""
Абстрактный интерфейс электрической машины.

Любая модель (линейная, нелинейная, с насыщением, etc.)
должна реализовать этот интерфейс
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from core.parameters import MachineParameters


class MachineModel(ABC):
    """
    Базовый класс модели асинхронной машины.

    Контракт:
      - electrical_matrices(t, y)         -> (L, b0)         - эл. подсистема без источника
      - mechanical_rhs(t, y, Mc)          -> domega_dt       - мех. уравнение
      - ode_rhs(t, y, Mc_func, U_func)    -> dydt            - совместимость
      - electromagnetic_torque(y)         -> float           - момент по токам
      - result_current_module(iA, iB, iC) -> float           - модуль тока
      - flux_linkage_phaseA(y)            -> float           - потокосцепление фазы A
    """

    def __init__(self, params: MachineParameters):
        self.params = params

    @abstractmethod
    def electrical_matrices(
        self,
        t: float,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Электрическая часть модели в виде:
            L(y, t) * di/dt = b0(y, t) + [UsA, UsB, UsC, 0, 0, 0]

        где:
            L  - матрица [6x6]
            b0 - вектор [6], не включает напряжение источника статора.
        """
        ...

    @abstractmethod
    def mechanical_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc: float,
    ) -> float:
        """Правая часть механического уравнения d(omega_r)/dt"""
        ...

    @abstractmethod
    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc_func: Callable[[float, float], float],
        U_func: Callable[[float], np.ndarray],
    ) -> np.ndarray:
        """
        Правая часть системы ОДУ.

        Args:
            t: текущее время
            y: вектор состояния [i1A, i1B, i1C, i2a, i2b, i2c, omega_r]
            Mc_func: функция момента сопротивления Mc(t, omega_r)
            U_func: функция напряжения U(t) -> [U1A, U1B, U1C]

        Returns:
            dydt: производные вектора состояния
        """
        ...

    @abstractmethod
    def electromagnetic_torque(
        self, i1A: float, i1B: float, i1C: float,
        i2a: float, i2b: float, i2c: float,
    ) -> float:
        """Электромагнитный момент по мгновенным значениям токов"""
        ...

    def result_current_module(self, iA: float, iB: float, iC: float) -> float:
        """Модуль результирующего вектора тока (общая формула)"""
        return np.sqrt((iB - iC) ** 2 / 3.0 + iA ** 2)

    def result_voltage_module(self, uA: float, uB: float, uC: float) -> float:
        """Модуль результирующего вектора напряжения"""
        return np.sqrt((uB - uC) ** 2 / 3.0 + uA ** 2)

    @abstractmethod
    def flux_linkage_phaseA(
        self, i1A: float, i1B: float, i1C: float,
        i2a: float, i2b: float, i2c: float,
    ) -> float:
        """Потокосцепление фазы A статора"""
        ...
