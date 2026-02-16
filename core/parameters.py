"""
Параметры асинхронной машины.

Все электрические параметры приведены к статору
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MachineParameters:
    """Параметры трёхфазной асинхронной машины"""

    #Номинальные данные
    P2n: float = 170e3          # Номинальная мощность на валу, Вт
    Unom_line: float = 570.0    # Номинальное линейное напряжение, В
    In: float = 204.42          # Номинальный ток статора, А
    fn: float = 50.0            # Номинальная частота, Гц
    cos_phi: float = 0.8993     # Коэффициент мощности
    eta: float = 0.93664        # КПД
    n_nom: float = 1481.0       # Номинальная частота вращения, об/мин
    n_sync: float = 1500.0      # Синхронная частота вращения, об/мин
    p: int = 2                  # Число пар полюсов
    m: int = 3                  # Число фаз
    s_nom: float = 0.0123      # Номинальное скольжение

    # Параметры схемы замещения
    R1: float = 0.0291          # Активное сопротивление статора, Ом
    L1sigma: float = 0.000544   # Индуктивность рассеяния статора, Гн (Xs=0.171 Ом)
    R2: float = 0.02017         # Приведённое акт. сопротивление ротора, Ом
    L2sigma: float = 0.000476   # Приведённая индуктивность рассеяния ротора, Гн (Xr=0.171 Ом)
    Lm: float = 0.017911  #0.000228      # Индуктивность намагничивания, Гн (Xm=5.627 Ом)

    #Механика
    J: float = 2.875            # Момент инерции ротора, кг·м^2

    #Кратности
    Kmm: float = 2.89           # Кратность максимального момента
    Kmp: float = 1.22           # Кратность пускового момента
    Kip: float = 6.02           # Кратность пускового тока

    #Вычисляемые величины
    Unom_phase: float = field(init=False)
    omega_n: float = field(init=False)

    def __post_init__(self):
        self.Unom_phase = self.Unom_line / np.sqrt(3)
        self.omega_n = 2 * np.pi * self.fn

    @property
    def omega_sync(self) -> float:
        """Синхронная механическая угловая скорость, рад/с"""
        return 2 * np.pi * self.fn / self.p

    @property
    def Um(self) -> float:
        """Амплитуда фазного напряжения, В"""
        return self.Unom_phase * np.sqrt(2)

    @property
    def L1(self) -> float:
        """Полная индуктивность статора, Гн"""
        return self.L1sigma + self.Lm

    @property
    def L2(self) -> float:
        """Полная индуктивность ротора, Гн"""
        return self.L2sigma + self.Lm

    @property
    def M_nom(self) -> float:
        """Номинальный момент на валу, Нм"""
        return self.P2n / (2 * np.pi * self.n_nom / 60.0)

    def info(self) -> str:
        """Форматированная строка с основными параметрами"""
        lines = [
            "\n",
            "  ПАРАМЕТРЫ МАШИНЫ",
            "\n",
            f"  P2н = {self.P2n / 1e3:.0f} кВт",
            f"  Uн(лин) = {self.Unom_line:.0f} В, Uн(фаз) = {self.Unom_phase:.1f} В",
            f"  Iн = {self.In:.2f} А",
            f"  fн = {self.fn:.0f} Гц, p = {self.p}",
            f"  cos_phi = {self.cos_phi:.4f}, η = {self.eta * 100:.2f}%",
            f"  nном = {self.n_nom:.0f} об/мин, sн = {self.s_nom:.4f}",
            "\n",
            f"  R1 = {self.R1:.4f} Ом",
            f"  L1s = {self.L1sigma * 1e3:.4f} мГн",
            f"  R2' = {self.R2:.4f} Ом",
            f"  L2s' = {self.L2sigma * 1e3:.4f} мГн",
            f"  Lm = {self.Lm * 1e3:.3f} мГн",
            f"  J = {self.J:.3f} кг·м²",
            "\n" ,
        ]
        return "\n".join(lines)