"""
Вектор состояния и вспомогательные обёртки.

State vector layout:
    y = [i1A, i1B, i1C, i2a, i2b, i2c, omega_r]
         0     1     2     3     4     5     6
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# Индексы в state vector
I1A, I1B, I1C = 0, 1, 2
I2A, I2B, I2C = 3, 4, 5
OMEGA_R = 6

STATE_SIZE = 7


@dataclass(frozen=True)
class StateView:
    """Именованный read-only доступ к текущему вектору состояния"""

    i1A: float
    i1B: float
    i1C: float
    i2a: float
    i2b: float
    i2c: float
    omega_r: float

    @classmethod
    def from_array(cls, y: np.ndarray) -> StateView:
        return cls(
            i1A=y[I1A], i1B=y[I1B], i1C=y[I1C],
            i2a=y[I2A], i2b=y[I2B], i2c=y[I2C],
            omega_r=y[OMEGA_R],
        )

    @property
    def imA(self) -> float:
        return self.i1A + self.i2a

    @property
    def imB(self) -> float:
        return self.i1B + self.i2b

    @property
    def imC(self) -> float:
        return self.i1C + self.i2c

    @property
    def n_rpm(self) -> float:
        """Частота вращения в об/мин"""
        return self.omega_r * 60 / (2 * np.pi)


def make_initial_state(omega_r: float = 0.0) -> np.ndarray:
    """Создать начальный вектор состояния с заданной скоростью"""
    y0 = np.zeros(STATE_SIZE)
    y0[OMEGA_R] = omega_r
    return y0