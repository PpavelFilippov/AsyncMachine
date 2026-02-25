"""
    Модуль models/saturation.py.
    Состав:
    Классы: SaturationCharacteristic.
    Функции: нет.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.parameters import MachineParameters

# Таблица насыщения из nonlin.pdf (2-я таблица, проект OHM).
# I_oe — относительный ток намагничивания |Im| / Im_nominal.
# K_SAT_TABLE — коэффициент насыщения Кн = Lm_unsat / Lm(I_oe).
I_OE_TABLE = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
], dtype=float)

K_SAT_TABLE = np.array([
    1.0, 1.0, 1.0, 1.006, 1.0114, 1.020, 1.0305,
    1.0489, 1.0991, 1.192, 1.335, 1.5507, 1.8857, 2.3932,
], dtype=float)


@dataclass
class SaturationCharacteristic:
    """
    Откалиброванная характеристика насыщения Lm(Im).

    Хранит таблицу Кн(I_oe) и вычисляет насыщенное значение Lm
    по мгновенным токам намагничивания трёх фаз.
    """

    Lm_nominal: float
    Im_nominal: float
    K_sat_nominal: float
    i_oe_table: np.ndarray
    k_sat_table: np.ndarray

    def __post_init__(self) -> None:
        k_at_1 = float(np.interp(1.0, self.i_oe_table, self.k_sat_table))
        delta = k_at_1 - 1.0
        if abs(delta) < 1e-12:
            # Таблица без насыщения (Кн ≡ 1) — калибровка не требуется.
            self._cal_factor = 0.0
            self._k_sat_calibrated = self.k_sat_table.copy()
        else:
            self._cal_factor = (self.K_sat_nominal - 1.0) / delta
            self._k_sat_calibrated = 1.0 + (self.k_sat_table - 1.0) * self._cal_factor
        self._Lm_unsat = self.Lm_nominal * self.K_sat_nominal
        self._Im_nominal_peak = self.Im_nominal * np.sqrt(2.0)

    @classmethod
    def from_params(
        cls,
        params: MachineParameters,
        i_oe_table: np.ndarray | None = None,
        k_sat_table: np.ndarray | None = None,
    ) -> SaturationCharacteristic:
        """Создаёт характеристику из MachineParameters с таблицей по умолчанию."""
        return cls(
            Lm_nominal=params.Lm,
            Im_nominal=params.Im_nominal,
            K_sat_nominal=params.K_sat_nominal,
            i_oe_table=i_oe_table if i_oe_table is not None else I_OE_TABLE.copy(),
            k_sat_table=k_sat_table if k_sat_table is not None else K_SAT_TABLE.copy(),
        )

    @staticmethod
    def magnetizing_current_amplitude(
        imA: float, imB: float, imC: float,
    ) -> float:
        """Амплитуда пространственного вектора тока намагничивания."""
        return np.sqrt(2.0 / 3.0 * (imA * imA + imB * imB + imC * imC))

    def compute_lm(self, imA: float, imB: float, imC: float) -> float:
        """
        Вычисляет насыщенное значение Lm по мгновенным токам намагничивания.

        1. Im_amp = sqrt(2/3 * (imA² + imB² + imC²))
        2. I_oe = Im_amp / (Im_nominal * sqrt(2))
        3. Кн = interp(I_oe, table)
        4. Lm = Lm_unsat / Кн
        """
        Im_amp = self.magnetizing_current_amplitude(imA, imB, imC)
        I_oe = Im_amp / self._Im_nominal_peak
        K_sat = float(np.interp(I_oe, self.i_oe_table, self._k_sat_calibrated))
        return self._Lm_unsat / K_sat

    @property
    def Lm_unsaturated(self) -> float:
        """Ненасыщенное значение Lm (при Im → 0)."""
        return self._Lm_unsat


