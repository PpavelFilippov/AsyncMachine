"""
Абстрактный интерфейс численного решателя ОДУ
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class SolverConfig:
    """Параметры решателя"""
    t_end: float = 2.0
    dt_out: float = 1e-4       # Шаг вывода
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step: float = 1e-4


class Solver(ABC):
    """Базовый класс решателя ОДУ"""

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()

    @abstractmethod
    def solve(
        self,
        rhs: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, bool, str]:
        """
        Решить систему ОДУ dy/dt = rhs(t, y).

        Returns:
            (t, y, success, message)
            t: массив времён [N]
            y: массив решений [n_vars, N]
            success: True если решение получено
            message: сообщение о статусе
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Название/описание метода"""
        ...