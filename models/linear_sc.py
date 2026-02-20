"""
    Модуль models/linear_sc.py.
    Состав:
    Классы: LinearInductionMachineSC.
    Функции: нет.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from core.parameters import MachineParameters
from faults.descriptors import FaultDescriptor
from .linear import LinearInductionMachine


class LinearInductionMachineSC:
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: state_size, n_extra, base_model, fault, L_ext, ode_rhs, electromagnetic_torque,
        flux_linkage_phaseA, extract_fault_currents, source_currents, fault_power, terminal_voltage.
    """

    def __init__(
        self,
        params: MachineParameters,
        fault: FaultDescriptor,
        r_source: np.ndarray,
        l_source: np.ndarray,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self._base = LinearInductionMachine(params)
        self._fault = fault
        self._params = params
        self._r_src = np.asarray(r_source, dtype=float)
        self._l_src = np.asarray(l_source, dtype=float)

        self._D = fault.D_2d                                   
        self._R_f = fault.R_fault_2d                                  
        self._n_extra = fault.n_extra
        self._n_elec = 6 + self._n_extra                                            
        self._state_size = 7 + self._n_extra                                           

        self._validate_source_impedance()
        self._L_ext = self._build_extended_inductance()

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def n_extra(self) -> int:
        return self._n_extra

    @property
    def base_model(self) -> LinearInductionMachine:
        return self._base

    @property
    def fault(self) -> FaultDescriptor:
        return self._fault

    @property
    def L_ext(self) -> np.ndarray:
        """Возвращает расширенную матрицу индуктивностей модели с коротким замыканием."""
        return self._L_ext.copy()


    def _validate_source_impedance(self) -> None:
        """Проверяет корректность параметров перед расчетом."""

        D = self._D                         
        l_src = self._l_src           
                                                                                         
        L_ff_full = D @ l_src @ D.T
        eigs = np.linalg.eigvalsh(L_ff_full)
        if np.min(eigs) < 1e-15:
            raise ValueError(
                "Контур КЗ вырожден: источник имеет нулевой или слишком малый "
                "импеданс (L_src ≈ 0). Для моделирования КЗ необходим источник "
                "с ненулевой индуктивностью (Thevenin source). "
                f"Минимальное собственное значение L_fault_loop = {np.min(eigs):.3e}"
            )

    def _build_extended_inductance(self) -> np.ndarray:
        """Собирает расширенную матрицу индуктивностей для ветвей КЗ."""

        n_e = self._n_elec
        n_x = self._n_extra
        D = self._D                           
        l_src = self._l_src                 

        L_machine = self._base._L_matrix.copy()
                                                   
        L_src_embed = np.zeros((6, 6), dtype=float)
        L_src_embed[0:3, 0:3] = l_src
        L_base = L_machine + L_src_embed
                                                                               
        D_embed = np.zeros((n_x, 6), dtype=float)
        D_embed[:, 0:3] = D
                                                                                 
        coupling_col = np.zeros((6, n_x), dtype=float)
        coupling_col[0:3, :] = (l_src @ D.T)                     

                                                 
        coupling_row = np.zeros((n_x, 6), dtype=float)
        coupling_row[:, 0:3] = (D @ l_src)                        

        L_ff = D @ l_src @ D.T

        L_ext = np.zeros((n_e, n_e), dtype=float)
        L_ext[0:6, 0:6] = L_base
        L_ext[0:6, 6:n_e] = coupling_col
        L_ext[6:n_e, 0:6] = coupling_row
        L_ext[6:n_e, 6:n_e] = L_ff

        return L_ext


    def _build_extended_rhs(
        self,
        b0_machine: np.ndarray,
        E_src: np.ndarray,
        i_stator: np.ndarray,
        i_fault: np.ndarray,
    ) -> np.ndarray:
        """Собирает расширенную правую часть с учетом ветвей КЗ."""

        n_e = self._n_elec
        D = self._D                             
        R_src = self._r_src                   
        R_f = self._R_f                           

        i_src_eff = i_stator + D.T @ i_fault        

        rhs = np.zeros(n_e, dtype=float)
                                   
        rhs[0:3] = b0_machine[0:3] + E_src - R_src @ i_src_eff
                                                  
        rhs[3:6] = b0_machine[3:6]
                                    
        rhs[6:n_e] = (
            D @ E_src
            - (D @ R_src) @ i_src_eff
            - R_f @ i_fault
        )

        return rhs

    def ode_rhs(
        self,
        t: float,
        y_ext: np.ndarray,
        Mc_func: Callable[[float, float], float],
        E_src_func: Callable[[float], np.ndarray],
    ) -> np.ndarray:
        """Вычисляет правую часть системы дифференциальных уравнений."""
                                                      
        y_base = y_ext[0:7]                                              
        i_stator = y_ext[0:3]                                      
        i_fault = y_ext[7:]                                     
        omega_r = y_ext[6]

        _, b0_machine = self._base.electrical_matrices(t, y_base)

        E_src = np.asarray(E_src_func(t), dtype=float)
                                  
        rhs_ext = self._build_extended_rhs(b0_machine, E_src, i_stator, i_fault)
                                          
        di_ext = np.linalg.solve(self._L_ext, rhs_ext)
                                                                    
        Mc = Mc_func(t, omega_r)
        domega_dt = self._base.mechanical_rhs(t, y_base, Mc)
                                            
        dydt = np.empty(self._state_size, dtype=float)
        dydt[0:6] = di_ext[0:6]                                  
        dydt[6] = domega_dt                       
        dydt[7:] = di_ext[6:]                       

        return dydt
                                                                        

    def electromagnetic_torque(self, y_ext: np.ndarray) -> float:
        """Вычисляет электромагнитный момент по токам."""

        return self._base.electromagnetic_torque(
            y_ext[0], y_ext[1], y_ext[2],
            y_ext[3], y_ext[4], y_ext[5],
        )

    def flux_linkage_phaseA(self, y_ext: np.ndarray) -> float:
        """Вычисляет потокосцепление фазы A."""

        return self._base.flux_linkage_phaseA(
            y_ext[0], y_ext[1], y_ext[2],
            y_ext[3], y_ext[4], y_ext[5],
        )

    def extract_fault_currents(self, y_ext: np.ndarray) -> np.ndarray:
        """Извлекает данные из вектора состояния."""
        return y_ext[7:7 + self._n_extra]

    def source_currents(
        self,
        i_stator: np.ndarray,
        i_fault: np.ndarray,
    ) -> np.ndarray:
        """Вычисляет токи источника с учетом ветвей КЗ."""
        return i_stator + self._D.T @ i_fault

    def fault_power(self, i_fault: np.ndarray) -> float:
        """Вычисляет мощность потерь в ветви КЗ."""
        return float(i_fault @ self._R_f @ i_fault)

    def terminal_voltage(
        self,
        E_src: np.ndarray,
        i_stator: np.ndarray,
        i_fault: np.ndarray,
        di_stator_dt: np.ndarray,
        di_fault_dt: np.ndarray,
    ) -> np.ndarray:
        """Вычисляет напряжения на зажимах машины."""

        i_src = i_stator + self._D.T @ i_fault
        di_src_dt = di_stator_dt + self._D.T @ di_fault_dt
        return E_src - self._r_src @ i_src - self._l_src @ di_src_dt
