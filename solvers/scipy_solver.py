"""
Решатель на базе scipy.integrate.solve_ivp
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .base import Solver, SolverConfig


class ScipySolver(Solver):
    """
    Обёртка над solve_ivp с выбором метода.

    Поддерживаемые методы: RK45, RK23, DOP853, Radau, BDF, LSODA
    """

    METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")

    def __init__(
        self,
        method: str = "RK45",
        config: Optional[SolverConfig] = None,
    ):
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

        if t_eval is None:
            t_eval = np.arange(t_span[0], t_span[1], self.config.dt_out)

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
        return (
            f"SciPy solve_ivp ({self.method}), "
            f"rtol={self.config.rtol}, atol={self.config.atol}, "
            f"max_step={self.config.max_step}"
        )