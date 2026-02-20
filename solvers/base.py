"""
    Модуль solvers/base.py.
    Состав:
    Классы: SolverConfig, Solver.
    Функции: нет.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class SolverConfig:
    """
        Поля:
        t_end: Поле класса. Тип: float.
        dt_out: Поле класса. Тип: float.
        rtol: Поле класса. Тип: float.
        atol: Поле класса. Тип: float.
        max_step: Поле класса. Тип: float.

        Методы:
        Публичные методы не выделены.
    """
    t_end: float = 2.0
    dt_out: float = 1e-4                   
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step: float = 1e-4


class Solver(ABC):
    """
        Поля:
        Явные поля уровня класса отсутствуют.

        Методы:
        Основные публичные методы: solve, describe.
    """

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
        """Определяет обязательный интерфейс метода."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""
        ...
