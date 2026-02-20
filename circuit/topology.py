"""
Топология схемы: управляет набором компонентов и раскладкой
глобального вектора состояния.
"""
from __future__ import annotations

from typing import Optional

from .components import MotorComponent, SourceComponent, FaultBranchComponent


class CircuitTopology:
    """
    Управляет компонентами схемы и назначает каждому компоненту
    диапазон индексов в глобальном state vector.

    Раскладка:
        [motor_vars (7)] [fault_1_vars (n1)] [fault_2_vars (n2)] ...

    Источник не имеет собственных переменных.
    """

    def __init__(self):
        self._motor: Optional[MotorComponent] = None
        self._source: Optional[SourceComponent] = None
        self._faults: list[FaultBranchComponent] = []
        self._is_finalized = False
        self._total_state_size = 0

    def set_motor(self, motor: MotorComponent) -> CircuitTopology:
        self._motor = motor
        self._is_finalized = False
        return self

    def set_source(self, source: SourceComponent) -> CircuitTopology:
        self._source = source
        self._is_finalized = False
        return self

    def add_fault(self, fault: FaultBranchComponent) -> CircuitTopology:
        self._faults.append(fault)
        self._is_finalized = False
        return self

    def finalize(self) -> None:
        """
        Назначить offset каждому компоненту.

        Должен вызываться после добавления всех компонентов
        и перед созданием CircuitAssembler.
        """
        if self._motor is None:
            raise ValueError("Motor component is required")
        if self._source is None:
            raise ValueError("Source component is required")

        offset = 0

        # Двигатель всегда первый
        self._motor.state_offset = offset
        offset += self._motor.n_diff_vars  # 7

        # Контуры КЗ идут после двигателя
        for fc in self._faults:
            fc.state_offset = offset
            offset += fc.n_diff_vars

        self._total_state_size = offset
        self._is_finalized = True

    @property
    def is_finalized(self) -> bool:
        return self._is_finalized

    @property
    def state_size(self) -> int:
        if not self._is_finalized:
            raise RuntimeError("Topology not finalized. Call finalize() first.")
        return self._total_state_size

    @property
    def motor(self) -> MotorComponent:
        if self._motor is None:
            raise ValueError("Motor component not set")
        return self._motor

    @property
    def source(self) -> SourceComponent:
        if self._source is None:
            raise ValueError("Source component not set")
        return self._source

    @property
    def faults(self) -> list[FaultBranchComponent]:
        return list(self._faults)

    @property
    def n_faults(self) -> int:
        return len(self._faults)

    @property
    def total_fault_vars(self) -> int:
        """Суммарное число переменных всех контуров КЗ."""
        return sum(f.n_diff_vars for f in self._faults)

    def active_faults(self, t: float) -> list[FaultBranchComponent]:
        """Контуры КЗ, активные в момент времени t."""
        return [f for f in self._faults if f.is_active(t)]

    def fault_switch_times(self) -> list[float]:
        """Отсортированный список моментов включения КЗ."""
        return sorted(set(f.t_fault for f in self._faults))
