"""
Типовые нагрузочные характеристики.
"""
from __future__ import annotations

from .base import LoadTorque


class ConstantTorque(LoadTorque):
    """Постоянный момент нагрузки."""

    def __init__(self, Mc: float):
        self.Mc = Mc

    def __call__(self, t: float, omega_r: float) -> float:
        return self.Mc

    def describe(self) -> str:
        return f"Постоянный момент: Mc = {self.Mc:.1f} Нм"


class RampTorque(LoadTorque):
    """Момент с линейным нарастанием до целевого значения."""

    def __init__(self, Mc_target: float, t_ramp: float, Mc_initial: float = 0.0):
        """
        Args:
            Mc_target: целевой момент (конечное значение)
            t_ramp: время нарастания от Mc_initial до Mc_target
            Mc_initial: начальный момент
        """
        self.Mc_target = Mc_target
        self.t_ramp = t_ramp
        self.Mc_initial = Mc_initial

    def __call__(self, t: float, omega_r: float) -> float:
        if t < self.t_ramp:
            frac = t / self.t_ramp
            return self.Mc_initial + (self.Mc_target - self.Mc_initial) * frac
        return self.Mc_target

    def describe(self) -> str:
        return (
            f"Рамп: {self.Mc_initial:.0f} → {self.Mc_target:.0f} Нм "
            f"за {self.t_ramp:.2f} с"
        )


class MotorStartTorque(LoadTorque):
    """
    Нагрузка при пуске двигателя:
    минимальный момент трения при разгоне, затем нарастание до номинала.
    """

    def __init__(
        self,
        Mc_load: float,
        Mc_friction: float = 50.0,
        t_load_start: float = 0.8,
        t_load_ramp: float = 0.3,
    ):
        self.Mc_load = Mc_load
        self.Mc_friction = Mc_friction
        self.t_load_start = t_load_start
        self.t_load_ramp = t_load_ramp

    def __call__(self, t: float, omega_r: float) -> float:
        if t < self.t_load_start:
            return self.Mc_friction
        frac = min(1.0, (t - self.t_load_start) / self.t_load_ramp)
        return self.Mc_load * frac + self.Mc_friction

    def describe(self) -> str:
        return (
            f"Пуск: трение {self.Mc_friction:.0f} Нм, "
            f"нагрузка {self.Mc_load:.0f} Нм с t={self.t_load_start:.1f} с"
        )