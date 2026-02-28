"""
    Модуль faults/interturn.py.
    Состав:
    Классы: InterTurnFaultDescriptor.
    Функции: interturn_fault.

    Описание межвитковых замыканий (МВЗ) в обмотке статора.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_PHASE_INDEX = {"A": 0, "B": 1, "C": 2}


@dataclass(frozen=True)
class InterTurnFaultDescriptor:
    """Описание межвиткового замыкания.

    Поля:
        name: текстовое описание.
        phase: фаза замыкания ("A", "B" или "C").
        mu: доля замкнутых витков (0 < mu < 1).
        R_f: сопротивление контура замыкания, Ом.
        t_fault: момент замыкания, с.
    """

    name: str
    phase: str
    mu: float
    R_f: float
    t_fault: float

    def __post_init__(self):
        """Проверяет корректность параметров."""

        phase = self.phase.upper()
        if phase not in _PHASE_INDEX:
            raise ValueError(
                f"Unknown phase '{self.phase}', expected A, B, or C"
            )
        if not (0.0 < self.mu < 1.0):
            raise ValueError(
                f"mu={self.mu} must be in (0, 1) exclusive"
            )
        if self.R_f < 0.0:
            raise ValueError(f"R_f={self.R_f} must be non-negative")
        if self.t_fault < 0.0:
            raise ValueError(f"t_fault={self.t_fault} must be non-negative")

        # Нормализуем фазу к верхнему регистру
        object.__setattr__(self, 'phase', phase)

    @property
    def n_extra(self) -> int:
        """Число дополнительных переменных состояния (i_X2)."""
        return 1

    @property
    def phase_index(self) -> int:
        """Числовой индекс фазы: A=0, B=1, C=2."""
        return _PHASE_INDEX[self.phase]


def interturn_fault(
    phase: str,
    mu: float,
    R_f: float,
    t_fault: float,
) -> InterTurnFaultDescriptor:
    """Создает описание межвиткового замыкания.

    Args:
        phase: фаза замыкания ("A", "B" или "C").
        mu: доля замкнутых витков (0 < mu < 1).
        R_f: сопротивление контура замыкания, Ом.
        t_fault: момент замыкания, с.

    Returns:
        InterTurnFaultDescriptor.
    """
    return InterTurnFaultDescriptor(
        name=(
            f"Inter-turn {phase.upper()}, "
            f"mu={mu:.3f}, Rf={R_f:.4g} Ohm"
        ),
        phase=phase,
        mu=mu,
        R_f=R_f,
        t_fault=t_fault,
    )
