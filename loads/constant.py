"""
    Модуль loads/constant.py.
    Состав:
    Классы: ConstantTorque, RampTorque, MotorStartTorque, StepTorque.
    Функции: нет.
"""

from __future__ import annotations
from .base import LoadTorque


class ConstantTorque(LoadTorque):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: describe.
    """

    def __init__(self, Mc: float):
        self.Mc = Mc

    def __call__(self, t: float, omega_r: float) -> float:
        """Возвращает значение функции источника или нагрузки в момент времени t."""
        return self.Mc

    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""
        return f"Постоянный момент: Mc = {self.Mc:.1f} Нм"


class RampTorque(LoadTorque):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: describe.
    """

    def __init__(self, Mc_target: float, t_ramp: float, Mc_initial: float = 0.0):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self.Mc_target = Mc_target
        self.t_ramp = t_ramp
        self.Mc_initial = Mc_initial

    def __call__(self, t: float, omega_r: float) -> float:
        """Возвращает значение функции источника или нагрузки в момент времени t."""

        if t < self.t_ramp:
            frac = t / self.t_ramp
            return self.Mc_initial + (self.Mc_target - self.Mc_initial) * frac
        return self.Mc_target

    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""

        return (
            f"Рамп: {self.Mc_initial:.0f} → {self.Mc_target:.0f} Нм "
            f"за {self.t_ramp:.2f} с"
        )


class MotorStartTorque(LoadTorque):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: describe.
    """

    def __init__(
        self,
        Mc_load: float,
        Mc_friction: float = 50.0,
        t_load_start: float = 0.8,
        t_load_ramp: float = 0.3,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self.Mc_load = Mc_load
        self.Mc_friction = Mc_friction
        self.t_load_start = t_load_start
        self.t_load_ramp = t_load_ramp

    def __call__(self, t: float, omega_r: float) -> float:
        """Возвращает значение функции источника или нагрузки в момент времени t."""

        if t < self.t_load_start:
            return self.Mc_friction
        frac = min(1.0, (t - self.t_load_start) / self.t_load_ramp)
        return self.Mc_load * frac + self.Mc_friction

    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""

        return (
            f"Пуск: трение {self.Mc_friction:.0f} Нм, "
            f"нагрузка {self.Mc_load:.0f} Нм с t={self.t_load_start:.1f} с"
        )


class StepTorque(LoadTorque):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: describe.
    """

    def __init__(
        self,
        Mc_load: float,
        t_step: float = 2.0,
        Mc_idle: float = 0.0,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self.Mc_load = Mc_load
        self.t_step = t_step
        self.Mc_idle = Mc_idle

    def __call__(self, t: float, omega_r: float) -> float:
        """Возвращает значение функции источника или нагрузки в момент времени t."""

        if t < self.t_step:
            return self.Mc_idle
        return self.Mc_load

    def describe(self) -> str:
        """Возвращает текстовое описание объекта."""

        return (
            f"Ступенька: {self.Mc_idle:.0f} Нм → {self.Mc_load:.0f} Нм "
            f"в t={self.t_step:.2f} с"
        )
