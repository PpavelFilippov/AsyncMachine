"""
Модуль solvers/scipy_solver.py.
Состав:
Классы: ScipySolver.
Функции: нет."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .base import Solver, SolverConfig


class ScipySolver(Solver):
    """
        Поля:
        METHODS: Поле класса. Тип: tuple.
        Методы:
        Основные публичные методы: solve, describe.
    """

    METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")

    def __init__(
        self,
        method: str = "RK45",
        config: Optional[SolverConfig] = None,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        super().__init__(config)
        if method not in self.METHODS:
            raise ValueError(
                f"Неизвестный метод '{method}'. "
                f"Доступные: {', '.join(self.METHODS)}"
            )
        self.method = method

    def solve(
        self,
        rhs: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, bool, str]:

        """Численно решает систему и возвращает объект решения."""

        if t_eval is None:
            t0, t1 = float(t_span[0]), float(t_span[1])
            dt = float(self.config.dt_out)
            if dt <= 0.0:
                raise ValueError(f"dt_out must be > 0, got {dt}.")
            if t1 < t0:
                raise ValueError(f"t_span must satisfy t1 >= t0, got {t_span}.")
            if abs(t1 - t0) <= 1e-15:
                t_eval = np.array([t0], dtype=float)
            else:
                n = int(np.floor((t1 - t0) / dt))
                t_eval = t0 + np.arange(n + 1, dtype=float) * dt
                if t_eval[-1] < t1:
                    t_eval = np.append(t_eval, t1)
                else:
                    t_eval[-1] = t1

        sol = solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method=self.method,
            t_eval=t_eval,
            rtol=self.config.rtol,
            atol=self.config.atol,
            max_step=self.config.max_step,
        )

        return sol.t, sol.y, sol.success, sol.message

    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""

        return (
            f"SciPy solve_ivp ({self.method}), "
            f"rtol={self.config.rtol}, atol={self.config.atol}, "
            f"max_step={self.config.max_step}"
        )
