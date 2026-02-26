"""Infrastructure for dynamic stator/rotor resistance laws."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ElectricalLawContext:
    """Instant model state passed into law callables."""

    t: float
    i1A: float
    i1B: float
    i1C: float
    i2a: float
    i2b: float
    i2c: float
    omega_r: float


@dataclass(frozen=True)
class ElectricalLawInputs:
    """Inputs for stator and rotor resistance laws."""

    context: ElectricalLawContext
    rs0: float
    rr0: float
    temperature: float

    @property
    def t(self) -> float:
        return self.context.t

    @property
    def i1A(self) -> float:
        return self.context.i1A

    @property
    def i1B(self) -> float:
        return self.context.i1B

    @property
    def i1C(self) -> float:
        return self.context.i1C

    @property
    def i2a(self) -> float:
        return self.context.i2a

    @property
    def i2b(self) -> float:
        return self.context.i2b

    @property
    def i2c(self) -> float:
        return self.context.i2c

    @property
    def omega_r(self) -> float:
        return self.context.omega_r


ProfileFunction = Callable[[ElectricalLawContext], float]
ParameterFunction = Callable[[ElectricalLawInputs], float]


def _constant_temperature(_: ElectricalLawContext) -> float:
    return 25.0


def _rs_constant(inp: ElectricalLawInputs) -> float:
    return inp.rs0


def _rr_constant(inp: ElectricalLawInputs) -> float:
    return inp.rr0


@dataclass
class PiecewiseConstantTemperatureProfile:
    """Piecewise-constant temperature profile on the provided time span."""

    t: np.ndarray
    temperature_values: np.ndarray

    def __post_init__(self) -> None:
        t_arr = np.asarray(self.t, dtype=float).reshape(-1)
        temp_arr = np.asarray(self.temperature_values, dtype=float).reshape(-1)

        if t_arr.size < 2:
            raise ValueError("t must contain at least 2 points.")
        if temp_arr.size < 1:
            raise ValueError("temperature_values must contain at least 1 value.")
        if not np.all(np.diff(t_arr) >= 0.0):
            raise ValueError("t must be sorted in ascending order.")

        self.t = t_arr
        self.temperature_values = temp_arr
        self._bounds = np.linspace(t_arr[0], t_arr[-1], temp_arr.size + 1, dtype=float)

    def __call__(self, context: ElectricalLawContext) -> float:
        t_cur = float(context.t)
        idx = int(np.searchsorted(self._bounds, t_cur, side="right") - 1)
        idx = max(0, min(idx, self.temperature_values.size - 1))
        return float(self.temperature_values[idx])


@dataclass
class ElectricalParameterLaws:
    """Container for pluggable stator/rotor resistance laws."""

    temperature: ProfileFunction = _constant_temperature
    rs: ParameterFunction = _rs_constant
    rr: ParameterFunction = _rr_constant

    def evaluate(
        self,
        *,
        context: ElectricalLawContext,
        rs0: float,
        rr0: float,
    ) -> tuple[float, float]:
        temperature = float(self.temperature(context))
        inputs = ElectricalLawInputs(
            context=context,
            rs0=float(rs0),
            rr0=float(rr0),
            temperature=temperature,
        )
        rs = float(self.rs(inputs))
        rr = float(self.rr(inputs))
        return rs, rr
