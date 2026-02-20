"""
    Модуль models/base.py.
    Состав:
    Классы: MachineModel.
    Функции: нет.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from core.parameters import MachineParameters


class MachineModel(ABC):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: electrical_matrices, mechanical_rhs, ode_rhs, electromagnetic_torque, result_current_module, result_voltage_module, flux_linkage_phaseA.
    """

    def __init__(self, params: MachineParameters):
        self.params = params

    @abstractmethod
    def electrical_matrices(
        self,
        t: float,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Формирует матрицу и правую часть электрической подсистемы."""
        ...

    @abstractmethod
    def mechanical_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc: float,
    ) -> float:
        """Вычисляет производную механической скорости."""
        ...

    @abstractmethod
    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc_func: Callable[[float, float], float],
        U_func: Callable[[float], np.ndarray],
    ) -> np.ndarray:
        """Вычисляет правую часть системы дифференциальных уравнений."""
        ...

    @abstractmethod
    def electromagnetic_torque(
        self, i1A: float, i1B: float, i1C: float,
        i2a: float, i2b: float, i2c: float,
    ) -> float:
        """Вычисляет электромагнитный момент по токам."""
        ...

    def result_current_module(self, iA: float, iB: float, iC: float) -> float:
        """Вычисляет модуль трехфазного тока по результатам моделирования."""
        return np.sqrt((iB - iC) ** 2 / 3.0 + iA ** 2)

    def result_voltage_module(self, uA: float, uB: float, uC: float) -> float:
        """Вычисляет модуль трехфазного напряжения по результатам моделирования."""
        return np.sqrt((uB - uC) ** 2 / 3.0 + uA ** 2)

    @abstractmethod
    def flux_linkage_phaseA(
        self, i1A: float, i1B: float, i1C: float,
        i2a: float, i2b: float, i2c: float,
    ) -> float:
        """Вычисляет потокосцепление фазы A."""
        ...
