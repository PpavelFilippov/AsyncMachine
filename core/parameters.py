"""
    Модуль core/parameters.py.
    Состав:
    Классы: MachineParameters.
    Функции: нет.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MachineParameters:
    """Класс параметров модели."""
                       
    P2n: float = 170e3                                            
    Unom_line: float = 570.0                                        
    In: float = 204.42                                      
    fn: float = 50.0                                     
    cos_phi: float = 0.8993                           
    eta: float = 0.93664             
    n_nom: float = 1481.0                                             
    n_sync: float = 1500.0                                           
    p: int = 2                                     
    m: int = 3                             
    s_nom: float = 0.0123                              

                               
    R1: float = 0.0291                                                                                                            
    L1sigma: float = 0.000544                                                      
    R2: float = 0.02017                                                    
    L2sigma: float = 0.000476                                                                 
    Lm: float = 0.017911


    J: float = 2.875                                           

              
    Kmm: float = 2.89                                            
    Kmp: float = 1.22                                        
    Kip: float = 6.02                                     

                         
    Unom_phase: float = field(init=False)
    omega_n: float = field(init=False)

    def __post_init__(self):
        """Вычисляет производные параметры после инициализации."""

        self.Unom_phase = self.Unom_line / np.sqrt(3)
        self.omega_n = 2 * np.pi * self.fn

    @property
    def omega_sync(self) -> float:
        """Возвращает синхронную механическую угловую скорость двигателя."""
        return 2 * np.pi * self.fn / self.p

    @property
    def Um(self) -> float:
        """Возвращает амплитуду фазного напряжения источника."""
        return self.Unom_phase * np.sqrt(2)

    @property
    def L1(self) -> float:
        """Возвращает полную индуктивность статора."""
        return self.L1sigma + self.Lm

    @property
    def L2(self) -> float:
        """Возвращает полную индуктивность ротора."""
        return self.L2sigma + self.Lm

    @property
    def M_nom(self) -> float:
        """Возвращает номинальный момент на валу."""
        return self.P2n / (2 * np.pi * self.n_nom / 60.0)

    def info(self) -> str:
        """Возвращает словарь с основными параметрами машины."""

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
