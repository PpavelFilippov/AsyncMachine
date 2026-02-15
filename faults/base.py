"""
Абстрактный интерфейс неисправности.

Неисправности модифицируют поведение модели:
  - обрыв фазы -> модификация напряжений
  - межвитковое КЗ -> модификация параметров
  - эксцентриситет -> модификация индуктивностей

Неисправности могут быть:
  - постоянными (с момента t_start)
  - временными (от t_start до t_end)
  - прогрессирующими (нарастающими со временем)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Fault(ABC):
    """Базовый класс неисправности"""

    def __init__(
        self,
        t_start: float = 0.0,
        t_end: Optional[float] = None,
    ):
        """
        Args:
            t_start: момент возникновения неисправности
            t_end: момент окончания (None = постоянная неисправность)
        """
        self.t_start = t_start
        self.t_end = t_end

    def is_active(self, t: float) -> bool:
        """Активна ли неисправность в момент времени t"""
        if t < self.t_start:
            return False
        if self.t_end is not None and t > self.t_end:
            return False
        return True

    def severity(self, t: float) -> float:
        """
        Степень неисправности [0..1]
        По умолчанию: 0 до t_start, 1 после t_start (ступенчатая)
        Переопределить для прогрессирующих неисправностей
        """
        return 1.0 if self.is_active(t) else 0.0

    def modify_voltages(self, t: float, U: np.ndarray) -> np.ndarray:
        """
        Модификация напряжений статора
        По умолчанию — без изменений
        """
        return U

    def modify_parameters(self, t: float, params_dict: dict) -> dict:
        """
        Модификация параметров модели (R, L, etc.)
        По умолчанию — без изменений
        """
        return params_dict

    @abstractmethod
    def describe(self) -> str:
        """Описание неисправности"""
        ...