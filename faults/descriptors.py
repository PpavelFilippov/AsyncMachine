"""
Дескрипторы повреждений на зажимах статора асинхронной машины.

Каждый тип КЗ описывается матрицей инцидентности D,
связывающей дополнительные токи КЗ с фазами статора,
и матрицей сопротивлений КЗ R_fault.

Два класса повреждений:
  1) Межфазное КЗ без земли (phase-to-phase)
  2) КЗ фазы на землю (phase-to-ground)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_PHASE_INDEX = {"A": 0, "B": 1, "C": 2}


@dataclass(frozen=True)
class FaultDescriptor:
    """
    Описание повреждения на зажимах статора.

    Attributes:
        name:    человеко-читаемое имя повреждения
        D:       матрица инцидентности [n_extra x 3].
                 Связывает доп. токи КЗ с фазными токами статора.
                 Для межфазного КЗ (B-C): D = [[0, +1, -1]]
                 Для КЗ на землю (фаза A): D = [[1, 0, 0]]
        R_fault: матрица сопротивлений КЗ [n_extra x n_extra].
                 Диагональная для независимых путей КЗ.
        t_fault: момент наступления КЗ, с
    """

    name: str
    D: np.ndarray
    R_fault: np.ndarray
    t_fault: float

    def __post_init__(self):
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
        """Количество дополнительных переменных состояния (токов КЗ)."""
        return np.atleast_2d(self.D).shape[0]

    @property
    def D_2d(self) -> np.ndarray:
        """D гарантированно как 2D массив."""
        return np.atleast_2d(self.D)

    @property
    def R_fault_2d(self) -> np.ndarray:
        """R_fault гарантированно как 2D массив."""
        return np.atleast_2d(self.R_fault)


def _phase_idx(phase: str) -> int:
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
    """
    Межфазное КЗ без земли.

    Ток КЗ i_f течёт из phase_from в phase_to через R_f.
    Вектор инцидентности d = [+1 на phase_from, -1 на phase_to].

    Args:
        phase_from: фаза-источник тока КЗ ("A", "B", или "C")
        phase_to:   фаза-приёмник тока КЗ
        R_f:        сопротивление КЗ, Ом (>0, болтовое ≈ 1e-4)
        t_fault:    момент КЗ, с
    """
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
    """
    КЗ одной фазы на землю.

    Ток КЗ i_fg течёт из фазы phase в землю через R_f + R_ground.

    Args:
        phase:    замыкаемая фаза ("A", "B", или "C")
        R_f:      сопротивление перехода КЗ, Ом
        t_fault:  момент КЗ, с
        R_ground: сопротивление заземления, Ом (по умолчанию 0)
    """
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
    """
    Двухфазное КЗ на землю.

    Каждая из двух фаз замыкается на землю через свой путь R_f + R_ground.
    Это вводит 2 дополнительных переменных (ток КЗ каждой фазы).

    Args:
        phase_a:  первая замыкаемая фаза
        phase_b:  вторая замыкаемая фаза
        R_f:      сопротивление КЗ, Ом (одинаковое для обоих путей)
        t_fault:  момент КЗ, с
        R_ground: сопротивление заземления, Ом
    """
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
