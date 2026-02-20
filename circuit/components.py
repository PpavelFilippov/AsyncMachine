"""
Компоненты схемы: двигатель, источник, контур КЗ.

Каждый компонент — независимый элемент, который подключается
к общей схеме через CircuitTopology. Двигатель не знает о КЗ.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from models.base import MachineModel
from sources.base import VoltageSource
from faults.descriptors import FaultDescriptor


class MotorComponent:
    """
    Обёртка над MachineModel для включения в схему.

    Не модифицирует модель двигателя. Использует только:
      - motor.electrical_matrices(t, y_motor) -> (L_6x6, b0_6)
      - motor.mechanical_rhs(t, y_motor, Mc)

    Владеет 7 переменными состояния:
        [i1A, i1B, i1C, i2a, i2b, i2c, omega_r]
    """

    def __init__(
        self,
        motor: MachineModel,
        Mc_func: Callable[[float, float], float],
    ):
        self._motor = motor
        self._Mc_func = Mc_func
        self.state_offset: int = 0

    @property
    def n_diff_vars(self) -> int:
        return 7

    @property
    def motor(self) -> MachineModel:
        return self._motor

    def extract_state(self, y_global: np.ndarray) -> np.ndarray:
        """Извлечь 7 переменных двигателя из глобального вектора."""
        return y_global[self.state_offset: self.state_offset + 7]

    def electrical_matrices(
        self, t: float, y_global: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Делегирует в motor.electrical_matrices (L_6x6, b0_6)."""
        y_motor = self.extract_state(y_global)
        return self._motor.electrical_matrices(t, y_motor)

    def mechanical_rhs(self, t: float, y_global: np.ndarray) -> float:
        """Делегирует в motor.mechanical_rhs."""
        y_motor = self.extract_state(y_global)
        omega_r = y_motor[6]
        Mc = self._Mc_func(t, omega_r)
        return self._motor.mechanical_rhs(t, y_motor, Mc)


class SourceComponent:
    """
    Источник напряжения (Тевенен).

    Предоставляет e_src(t), R_src (3x3), L_src (3x3).
    Не имеет собственных переменных состояния — ток источника
    определяется через KCL: i_src = i_stator + D^T * i_fault.
    """

    def __init__(self, source: VoltageSource):
        self._source = source
        self._R_src = np.asarray(source.series_resistance_matrix(), dtype=float)
        self._L_src = np.asarray(source.series_inductance_matrix(), dtype=float)

    @property
    def n_diff_vars(self) -> int:
        return 0

    @property
    def source(self) -> VoltageSource:
        return self._source

    def emf(self, t: float) -> np.ndarray:
        """ЭДС источника [eA, eB, eC]."""
        return np.asarray(self._source(t), dtype=float)

    @property
    def R_src(self) -> np.ndarray:
        """Матрица последовательного сопротивления [3x3]."""
        return self._R_src

    @property
    def L_src(self) -> np.ndarray:
        """Матрица последовательной индуктивности [3x3]."""
        return self._L_src


class FaultBranchComponent:
    """
    Контур короткого замыкания на зажимах статора.

    Активируется при t >= t_fault (замыкание ключа).
    До этого момента контур разомкнут и не участвует в схеме.

    Переменные состояния: [i_f1, ..., i_fn] — токи КЗ.
    """

    def __init__(self, fault: FaultDescriptor):
        self._fault = fault
        self._D = fault.D_2d                    # [n_extra x 3]
        self._R_f = fault.R_fault_2d            # [n_extra x n_extra]
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
        """Матрица инцидентности [n_extra x 3]."""
        return self._D

    @property
    def R_fault(self) -> np.ndarray:
        """Матрица сопротивлений КЗ [n_extra x n_extra]."""
        return self._R_f

    @property
    def t_fault(self) -> float:
        return self._t_fault

    def is_active(self, t: float) -> bool:
        """Контур КЗ активен при t >= t_fault."""
        return t >= self._t_fault

    def extract_currents(self, y_global: np.ndarray) -> np.ndarray:
        """Извлечь токи КЗ из глобального вектора состояния."""
        return y_global[self.state_offset: self.state_offset + self._n_extra]
