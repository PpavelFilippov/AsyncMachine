"""
    Модуль circuit/topology.py.
    Состав:
    Классы: CircuitTopology.
    Функции: нет.
"""
from __future__ import annotations

from typing import Optional

from .components import MotorComponent, SourceComponent, FaultBranchComponent


class CircuitTopology:
    """
        Поля:
        Явные поля уровня класса отсутствуют.

        Методы:
        Основные публичные методы: set_motor, set_source, add_fault, finalize, is_finalized, state_size, motor, source,
        faults, n_faults, total_fault_vars, active_faults, fault_switch_times.
    """

    def __init__(self):
        """Создает объект и сохраняет параметры для последующих вычислений."""

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
        """Добавляет короткое замыкание в топологию расчета."""

        self._faults.append(fault)
        self._is_finalized = False
        return self

    def finalize(self) -> None:
        """Фиксирует топологию и пересчитывает индексы дополнительных переменных."""

        if self._motor is None:
            raise ValueError("Motor component is required")
        if self._source is None:
            raise ValueError("Source component is required")

        offset = 0
                                 
        self._motor.state_offset = offset
        offset += self._motor.n_diff_vars
                                         
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
        """Возвращает размер вектора состояния с учетом топологии."""

        if not self._is_finalized:
            raise RuntimeError("Topology not finalized. Call finalize() first.")
        return self._total_state_size

    @property
    def motor(self) -> MotorComponent:
        """Возвращает объект модели двигателя из топологии."""

        if self._motor is None:
            raise ValueError("Motor component not set")
        return self._motor

    @property
    def source(self) -> SourceComponent:
        """Возвращает объект источника питания из топологии."""

        if self._source is None:
            raise ValueError("Source component not set")
        return self._source

    @property
    def faults(self) -> list[FaultBranchComponent]:
        """Возвращает список зарегистрированных коротких замыканий."""
        return list(self._faults)

    @property
    def n_faults(self) -> int:
        """Возвращает количество коротких замыканий в топологии."""
        return len(self._faults)

    @property
    def total_fault_vars(self) -> int:
        """Возвращает суммарное число дополнительных токов ветвей КЗ."""
        return sum(f.n_diff_vars for f in self._faults)

    def active_faults(self, t: float) -> list[FaultBranchComponent]:
        """Возвращает список коротких замыканий, активных в момент времени t."""
        return [f for f in self._faults if f.is_active(t)]

    def fault_switch_times(self) -> list[float]:
        """Возвращает моменты активации короткого замыкания."""
        return sorted(set(f.t_fault for f in self._faults))
