"""
    Модуль circuit/components.py.
    Состав:
    Классы: MotorComponent, SourceComponent, FaultBranchComponent.
    Функции: нет.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from models.base import MachineModel
from sources.base import VoltageSource
from faults.descriptors import FaultDescriptor


class MotorComponent:
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: n_diff_vars, motor, extract_state, electrical_matrices, mechanical_rhs.
    """

    def __init__(
        self,
        motor: MachineModel,
        Mc_func: Callable[[float, float], float],
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self._motor = motor
        self._Mc_func = Mc_func
        self.state_offset: int = 0

    @property
    def n_diff_vars(self) -> int:
        """Возвращает число дифференциальных электрических переменных."""

        return 7

    @property
    def motor(self) -> MachineModel:
        return self._motor

    def extract_state(self, y_global: np.ndarray) -> np.ndarray:
        """Извлекает данные из вектора состояния."""

        return y_global[self.state_offset: self.state_offset + 7]

    def electrical_matrices(
        self, t: float, y_global: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Формирует матрицу и правую часть электрической подсистемы."""

        y_motor = self.extract_state(y_global)
        return self._motor.electrical_matrices(t, y_motor)

    def mechanical_rhs(self, t: float, y_global: np.ndarray) -> float:
        """Вычисляет производную механической скорости."""

        y_motor = self.extract_state(y_global)
        omega_r = y_motor[6]
        Mc = self._Mc_func(t, omega_r)
        return self._motor.mechanical_rhs(t, y_motor, Mc)


class SourceComponent:
    """
        Поля:
        Явные поля уровня класса отсутствуют.

        Методы:
        Основные публичные методы: n_diff_vars, source, emf, R_src, L_src.
    """

    def __init__(self, source: VoltageSource):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self._source = source
        self._R_src = np.asarray(source.series_resistance_matrix(), dtype=float)
        self._L_src = np.asarray(source.series_inductance_matrix(), dtype=float)

    @property
    def n_diff_vars(self) -> int:
        """Возвращает число дифференциальных электрических переменных."""
        return 0

    @property
    def source(self) -> VoltageSource:
        return self._source

    def emf(self, t: float) -> np.ndarray:
        """Вычисляет ЭДС ротора по токам и механической скорости."""

        return np.asarray(self._source(t), dtype=float)

    @property
    def R_src(self) -> np.ndarray:
        return self._R_src

    @property
    def L_src(self) -> np.ndarray:
        return self._L_src


class FaultBranchComponent:
    """
        Поля:
        Явные поля уровня класса отсутствуют.

        Методы:
        Основные публичные методы: n_diff_vars, fault, D, R_fault, t_fault, is_active, extract_currents.
    """

    def __init__(self, fault: FaultDescriptor):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self._fault = fault
        self._D = fault.D_2d                                   
        self._R_f = fault.R_fault_2d                                 
        self._t_fault = fault.t_fault
        self._n_extra = fault.n_extra
        self.state_offset: int = 0

    @property
    def n_diff_vars(self) -> int:
        return self._n_extra

    @property
    def fault(self) -> FaultDescriptor:
        return self._fault

    @property
    def D(self) -> np.ndarray:
        return self._D

    @property
    def R_fault(self) -> np.ndarray:
        return self._R_f

    @property
    def t_fault(self) -> float:
        return self._t_fault

    def is_active(self, t: float) -> bool:
        """Проверяет активность объекта в заданный момент времени."""

        return t >= self._t_fault

    def extract_currents(self, y_global: np.ndarray) -> np.ndarray:
        """Извлекает данные из вектора состояния."""

        return y_global[self.state_offset: self.state_offset + self._n_extra]
