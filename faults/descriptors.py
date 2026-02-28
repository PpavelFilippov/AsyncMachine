"""
    Модуль faults/descriptors.py.
    Состав:
    Классы: FaultDescriptor.
    Функции: phase_to_phase_fault, phase_to_ground_fault, two_phase_to_ground_fault.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_PHASE_INDEX = {"A": 0, "B": 1, "C": 2}


@dataclass(frozen=True)
class FaultDescriptor:
    """
        Поля:
        name: Поле класса. Тип: str.
        D: Поле класса. Тип: np.ndarray.
        R_fault: Поле класса. Тип: np.ndarray.
        t_fault: Поле класса. Тип: float.

        Методы:
        Основные публичные методы: n_extra, D_2d, R_fault_2d.
    """

    name: str
    D: np.ndarray
    R_fault: np.ndarray
    t_fault: float

    def __post_init__(self):
        """Вычисляет производные параметры после инициализации."""

        D = np.atleast_2d(self.D)
        if D.ndim != 2 or D.shape[1] != 3:
            raise ValueError(
                f"D must have shape (n_extra, 3), got {self.D.shape}"
            )
        R = np.atleast_2d(self.R_fault)
        if R.shape != (D.shape[0], D.shape[0]):
            raise ValueError(
                f"R_fault must have shape ({D.shape[0]}, {D.shape[0]}), "
                f"got {self.R_fault.shape}"
            )

    @property
    def n_extra(self) -> int:
        """Возвращает количество дополнительных токов ветвей короткого замыкания."""
        return np.atleast_2d(self.D).shape[0]

    @property
    def D_2d(self) -> np.ndarray:
        """Возвращает диагональную матрицу проводимостей ветвей короткого замыкания."""
        return np.atleast_2d(self.D)

    @property
    def R_fault_2d(self) -> np.ndarray:
        """Возвращает диагональную матрицу сопротивлений ветвей короткого замыкания."""
        return np.atleast_2d(self.R_fault)


def _phase_idx(phase: str) -> int:
    """Выполняет шаг расчета для операции phase idx."""

    phase = phase.upper()
    if phase not in _PHASE_INDEX:
        raise ValueError(f"Unknown phase '{phase}', expected A, B, or C")
    return _PHASE_INDEX[phase]


def phase_to_phase_fault(
    phase_from: str,
    phase_to: str,
    R_f: float,
    t_fault: float,
) -> FaultDescriptor:
    """Создает описание режима короткого замыкания."""

    idx_from = _phase_idx(phase_from)
    idx_to = _phase_idx(phase_to)
    if idx_from == idx_to:
        raise ValueError("phase_from and phase_to must be different")

    d = np.zeros(3, dtype=float)
    d[idx_from] = +1.0
    d[idx_to] = -1.0
    D = d.reshape(1, 3)
    R_fault = np.array([[R_f]], dtype=float)

    return FaultDescriptor(
        name=f"Phase-to-phase {phase_from.upper()}-{phase_to.upper()}, "
             f"Rf={R_f:.4g} Ohm",
        D=D,
        R_fault=R_fault,
        t_fault=t_fault,
    )


def phase_to_ground_fault(
    phase: str,
    R_f: float,
    t_fault: float,
    R_ground: float = 0.0,
) -> FaultDescriptor:
    """Создает описание режима короткого замыкания."""

    idx = _phase_idx(phase)
    d = np.zeros(3, dtype=float)
    d[idx] = 1.0
    D = d.reshape(1, 3)
    R_total = R_f + R_ground
    R_fault = np.array([[R_total]], dtype=float)

    return FaultDescriptor(
        name=f"Phase-to-ground {phase.upper()}, "
             f"Rf={R_f:.4g}+Rg={R_ground:.4g} Ohm",
        D=D,
        R_fault=R_fault,
        t_fault=t_fault,
    )


def two_phase_to_ground_fault(
    phase_a: str,
    phase_b: str,
    R_f: float,
    t_fault: float,
    R_ground: float = 0.0,
) -> FaultDescriptor:
    """Создает описание режима короткого замыкания."""

    idx_a = _phase_idx(phase_a)
    idx_b = _phase_idx(phase_b)
    if idx_a == idx_b:
        raise ValueError("phase_a and phase_b must be different")

    D = np.zeros((2, 3), dtype=float)
    D[0, idx_a] = 1.0
    D[1, idx_b] = 1.0

    R_total = R_f + R_ground
    R_fault = np.diag([R_total, R_total]).astype(float)

    return FaultDescriptor(
        name=f"Two-phase-to-ground {phase_a.upper()},{phase_b.upper()}, "
             f"Rf={R_f:.4g}+Rg={R_ground:.4g} Ohm",
        D=D,
        R_fault=R_fault,
        t_fault=t_fault,
    )
