"""
    Модуль core/state.py.
    Состав:
    Классы: StateView.
    Функции: make_initial_state, extend_state_for_fault.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


I1A, I1B, I1C = 0, 1, 2
I2A, I2B, I2C = 3, 4, 5
OMEGA_R = 6

STATE_SIZE = 7


@dataclass(frozen=True)
class StateView:
    """Класс состояния
        Методы:
        Основные публичные методы: from_array, imA, imB, imC, n_rpm.
    """

    i1A: float
    i1B: float
    i1C: float
    i2a: float
    i2b: float
    i2c: float
    omega_r: float

    @classmethod
    def from_array(cls, y: np.ndarray) -> StateView:
        """Создает объект состояния машины из вектора значений."""

        return cls(
            i1A=y[I1A], i1B=y[I1B], i1C=y[I1C],
            i2a=y[I2A], i2b=y[I2B], i2c=y[I2C],
            omega_r=y[OMEGA_R],
        )

    @property
    def imA(self) -> float:
        """Возвращает суммарный ток намагничивания по фазе A."""
        return self.i1A + self.i2a

    @property
    def imB(self) -> float:
        """Возвращает суммарный ток намагничивания по фазе B."""
        return self.i1B + self.i2b

    @property
    def imC(self) -> float:
        """Возвращает суммарный ток намагничивания по фазе C."""
        return self.i1C + self.i2c

    @property
    def n_rpm(self) -> float:
        """Переводит механическую скорость из радиан в секунду в обороты в минуту."""
        return self.omega_r * 60 / (2 * np.pi)


def make_initial_state(omega_r: float = 0.0) -> np.ndarray:
    """Формирует начальный вектор состояния для интегратора."""

    y0 = np.zeros(STATE_SIZE)
    y0[OMEGA_R] = omega_r
    return y0


def extend_state_for_fault(y_normal: np.ndarray, n_extra: int) -> np.ndarray:
    """Добавляет в состояние токи ветвей короткого замыкания."""

    y_ext = np.zeros(STATE_SIZE + n_extra, dtype=float)
    y_ext[0:STATE_SIZE] = y_normal
    return y_ext
