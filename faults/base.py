"""
Абстрактный интерфейс неисправности.

Неисправности описывают физическое изменение в цепи машины:
  - обрыв фазы (open circuit) -> ток через фазу = 0
  - замыкание на землю (ground fault) -> напряжение фазы = 0
  - межвитковое КЗ -> модификация параметров (будущее)

Для корректной работы в связанной мульти-машинной системе
fault возвращает информацию об ограничениях, а не просто
модифицирует напряжение. Сборку системы уравнений с учётом
ограничений выполняет оркестратор (simulation_multi).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional, Set

import numpy as np


class FaultType(Enum):
    """Тип физического воздействия неисправности"""
    OPEN_CIRCUIT = auto()    # Обрыв: i_phase = 0 (разрыв цепи)
    GROUND_FAULT = auto()    # Замыкание на землю: U_phase = 0 (ток свободен)


class Fault(ABC):
    """Базовый класс неисправности"""

    def __init__(
        self,
        t_start: float = 0.0,
        t_end: Optional[float] = None,
    ):
        self.t_start = t_start
        self.t_end = t_end

    def is_active(self, t: float) -> bool:
        if t < self.t_start:
            return False
        if self.t_end is not None and t > self.t_end:
            return False
        return True

    def severity(self, t: float) -> float:
        return 1.0 if self.is_active(t) else 0.0

    @abstractmethod
    def fault_type(self) -> FaultType:
        """Тип неисправности"""
        ...

    @abstractmethod
    def affected_phases(self) -> Set[int]:
        """
        Множество индексов поражённых фаз статора {0, 1, 2}.
        0=A, 1=B, 2=C
        """
        ...

    def get_open_phases(self, t: float) -> Set[int]:
        """Фазы с обрывом (ток = 0) в момент t"""
        if self.is_active(t) and self.fault_type() == FaultType.OPEN_CIRCUIT:
            return self.affected_phases()
        return set()

    def get_grounded_phases(self, t: float) -> Set[int]:
        """Фазы с замыканием на землю (U = 0) в момент t"""
        if self.is_active(t) and self.fault_type() == FaultType.GROUND_FAULT:
            return self.affected_phases()
        return set()

    # Обратная совместимость для одиночной симуляции (SimulationBuilder)
    def modify_voltages(self, t: float, U: np.ndarray) -> np.ndarray:
        if not self.is_active(t):
            return U
        U_mod = U.copy()
        for idx in self.affected_phases():
            U_mod[idx] = 0.0
        return U_mod

    @abstractmethod
    def describe(self) -> str:
        ...