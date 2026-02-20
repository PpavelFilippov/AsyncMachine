"""
Расширенная линейная модель асинхронной машины с КЗ на зажимах статора.

Модель строится через композицию: оборачивает базовую LinearInductionMachine
и расширяет систему уравнений дополнительными контурами КЗ.

Математическая модель
---------------------
Базовая система (6 электрических + 1 механическое уравнение):

    (L_machine + L_src_embed) * di/dt = b0 + E_src - R_src * i_stator
    J * dω/dt = M_em - M_c

При КЗ на зажимах статора ток от источника ответвляется в контур КЗ.
Для каждого контура КЗ k с вектором инцидентности d_k ∈ R^3:

    i_source = i_machine + D^T * i_fault

где D — матрица инцидентности [n_extra x 3], i_fault — вектор токов КЗ.

Расширенная электрическая система (6 + n_extra) уравнений:

    L_ext * [di/dt; di_fault/dt] = rhs_ext

где:
    L_ext = | L_machine + L_src_embed     -L_src * D_e^T  |
            | -L_src * D_e                 L_src * D*D^T   |

    D_e = [D | 0_{n_extra x 3}]  — D дополненная нулями для роторных индексов

    rhs_ext = | b0[0:3] + E_src - R_src*(i_st + D^T*i_f)  |  статор
              | b0[3:6]                                     |  ротор
              | D*E_src - R_src*D*(i_st+D^T*i_f) - R_f*i_f |  контуры КЗ

Вектор состояния: y_ext = [i1A, i1B, i1C, i2a, i2b, i2c, omega_r, i_f1, ..., i_fn]
Размерность: 7 + n_extra
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from core.parameters import MachineParameters
from faults.descriptors import FaultDescriptor
from .linear import LinearInductionMachine


class LinearInductionMachineSC:
    """
    Расширенная модель АМ с КЗ на зажимах статора.

    Использует композицию: внутри содержит базовую LinearInductionMachine
    и добавляет контуры КЗ согласно FaultDescriptor.

    Параметры источника (R_src, L_src) необходимы для формирования
    расширенной матрицы индуктивностей — они определяют импеданс контура КЗ.
    """

    def __init__(
        self,
        params: MachineParameters,
        fault: FaultDescriptor,
        r_source: np.ndarray,
        l_source: np.ndarray,
    ):
        """
        Args:
            params:   параметры машины
            fault:    дескриптор повреждения
            r_source: матрица последовательного сопротивления источника [3x3]
            l_source: матрица последовательной индуктивности источника [3x3]
        """
        self._base = LinearInductionMachine(params)
        self._fault = fault
        self._params = params
        self._r_src = np.asarray(r_source, dtype=float)
        self._l_src = np.asarray(l_source, dtype=float)

        self._D = fault.D_2d                    # [n_extra x 3]
        self._R_f = fault.R_fault_2d             # [n_extra x n_extra]
        self._n_extra = fault.n_extra
        self._n_elec = 6 + self._n_extra         # размерность электрической системы
        self._state_size = 7 + self._n_extra     # полная размерность вектора состояния

        self._validate_source_impedance()
        self._L_ext = self._build_extended_inductance()

    @property
    def state_size(self) -> int:
        """Размерность расширенного вектора состояния."""
        return self._state_size

    @property
    def n_extra(self) -> int:
        """Количество дополнительных переменных (токов КЗ)."""
        return self._n_extra

    @property
    def base_model(self) -> LinearInductionMachine:
        """Базовая модель машины."""
        return self._base

    @property
    def fault(self) -> FaultDescriptor:
        """Дескриптор повреждения."""
        return self._fault

    @property
    def L_ext(self) -> np.ndarray:
        """Расширенная матрица индуктивностей (read-only копия)."""
        return self._L_ext.copy()

    # ------------------------------------------------------------------
    # Построение расширенных матриц
    # ------------------------------------------------------------------

    def _validate_source_impedance(self) -> None:
        """Проверка, что источник имеет ненулевой импеданс для контура КЗ."""
        D = self._D          # [n_extra x 3]
        l_src = self._l_src  # [3 x 3]
        # L_ff = D @ L_src @ D^T  — матрица индуктивности контуров КЗ [n_extra x n_extra]
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
        """
        Строит расширенную матрицу индуктивностей L_ext.

        L_ext = | L_machine + L_src_embed     -L_src * D_embed^T  |
                | -L_src * D_embed             L_src * D * D^T     |

        Размерность: (6+n_extra) x (6+n_extra)
        """
        n_e = self._n_elec
        n_x = self._n_extra
        D = self._D                # [n_x x 3]
        l_src = self._l_src        # [3 x 3]

        # Базовая матрица машины [6x6]
        L_machine = self._base._L_matrix.copy()

        # L_src встраивается в статорный блок [6x6]
        L_src_embed = np.zeros((6, 6), dtype=float)
        L_src_embed[0:3, 0:3] = l_src
        L_base = L_machine + L_src_embed

        # D_embed = [D | 0] — расширение D до 6 столбцов (доп. нули для ротора)
        D_embed = np.zeros((n_x, 6), dtype=float)
        D_embed[:, 0:3] = D

        # Блок связи: -L_src * D_embed^T  [6 x n_x]
        # Для каждого контура j: столбец = -l_src @ D[j, :]  (только стат. часть)
        coupling_col = np.zeros((6, n_x), dtype=float)
        coupling_col[0:3, :] = (l_src @ D.T)  # [3 x n_x] ##TODO:

        # Блок связи: -L_src * D_embed  [n_x x 6]
        coupling_row = np.zeros((n_x, 6), dtype=float)
        coupling_row[:, 0:3] = (D @ l_src)    # [n_x x 3]  ##TODO:

        # Диагональный блок контуров КЗ: L_src * D * D^T  [n_x x n_x]
        L_ff = D @ l_src @ D.T

        # Собираем расширенную матрицу
        L_ext = np.zeros((n_e, n_e), dtype=float)
        L_ext[0:6, 0:6] = L_base
        L_ext[0:6, 6:n_e] = coupling_col
        L_ext[6:n_e, 0:6] = coupling_row
        L_ext[6:n_e, 6:n_e] = L_ff

        return L_ext

    # ------------------------------------------------------------------
    # Правая часть ОДУ
    # ------------------------------------------------------------------

    def _build_extended_rhs(
        self,
        b0_machine: np.ndarray,
        E_src: np.ndarray,
        i_stator: np.ndarray,
        i_fault: np.ndarray,
    ) -> np.ndarray:
        """
        Строит правую часть расширенной электрической системы.

        rhs[0:3] = b0[0:3] + E_src - R_src * (i_stator + D^T * i_fault)
        rhs[3:6] = b0[3:6]
        rhs[6:]  = D @ E_src - R_src * D @ (i_stator + D^T*i_fault) - R_f @ i_fault
        """
        n_e = self._n_elec
        D = self._D                  # [n_x x 3]
        R_src = self._r_src          # [3 x 3]
        R_f = self._R_f              # [n_x x n_x]

        # Эффективный ток от источника (для каждой фазы)
        i_src_eff = i_stator + D.T @ i_fault   # [3]

        rhs = np.zeros(n_e, dtype=float)

        # Статорные уравнения (0:3)
        rhs[0:3] = b0_machine[0:3] + E_src - R_src @ i_src_eff

        # Роторные уравнения (3:6) — без изменений
        rhs[3:6] = b0_machine[3:6]

        # Уравнения контуров КЗ (6:)
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
        """
        Правая часть расширенной системы ОДУ.

        Args:
            t:          текущее время
            y_ext:      расширенный вектор состояния
                        [i1A, i1B, i1C, i2a, i2b, i2c, omega_r, i_f1, ...]
            Mc_func:    функция момента нагрузки Mc(t, omega_r)
            E_src_func: функция ЭДС источника E(t) -> [eA, eB, eC]

        Returns:
            dydt: производные расширенного вектора состояния
        """
        # Извлечение компонент из расширенного вектора
        y_base = y_ext[0:7]          # базовый вектор [i1A..i2c, omega_r]
        i_stator = y_ext[0:3]        # токи статора [i1A, i1B, i1C]
        i_fault = y_ext[7:]          # токи КЗ [i_f1, ..., i_fn]
        omega_r = y_ext[6]

        # Базовая модель: матрицы без источника
        _, b0_machine = self._base.electrical_matrices(t, y_base)

        # ЭДС источника
        E_src = np.asarray(E_src_func(t), dtype=float)

        # Расширенная правая часть
        rhs_ext = self._build_extended_rhs(b0_machine, E_src, i_stator, i_fault)

        # Решение электрической подсистемы
        di_ext = np.linalg.solve(self._L_ext, rhs_ext)

        # Механическое уравнение (использует только базовые 6 токов)
        Mc = Mc_func(t, omega_r)
        domega_dt = self._base.mechanical_rhs(t, y_base, Mc)

        # Сборка полного вектора производных
        dydt = np.empty(self._state_size, dtype=float)
        dydt[0:6] = di_ext[0:6]       # di_stator/dt, di_rotor/dt
        dydt[6] = domega_dt            # domega/dt
        dydt[7:] = di_ext[6:]          # di_fault/dt

        return dydt

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def electromagnetic_torque(self, y_ext: np.ndarray) -> float:
        """Электромагнитный момент (делегирует в базовую модель)."""
        return self._base.electromagnetic_torque(
            y_ext[0], y_ext[1], y_ext[2],
            y_ext[3], y_ext[4], y_ext[5],
        )

    def flux_linkage_phaseA(self, y_ext: np.ndarray) -> float:
        """Потокосцепление фазы A (делегирует в базовую модель)."""
        return self._base.flux_linkage_phaseA(
            y_ext[0], y_ext[1], y_ext[2],
            y_ext[3], y_ext[4], y_ext[5],
        )

    def extract_fault_currents(self, y_ext: np.ndarray) -> np.ndarray:
        """Извлечение токов КЗ из расширенного вектора состояния."""
        return y_ext[7:7 + self._n_extra]

    def source_currents(
        self,
        i_stator: np.ndarray,
        i_fault: np.ndarray,
    ) -> np.ndarray:
        """
        Токи от источника: i_source = i_machine + D^T * i_fault

        Полезно для расчёта мощности источника и анализа.
        """
        return i_stator + self._D.T @ i_fault

    def fault_power(self, i_fault: np.ndarray) -> float:
        """Мощность, рассеиваемая на сопротивлениях КЗ: P_f = i_f^T * R_f * i_f"""
        return float(i_fault @ self._R_f @ i_fault)

    def terminal_voltage(
        self,
        E_src: np.ndarray,
        i_stator: np.ndarray,
        i_fault: np.ndarray,
        di_stator_dt: np.ndarray,
        di_fault_dt: np.ndarray,
    ) -> np.ndarray:
        """
        Напряжение на зажимах машины:
        u_term = E_src - R_src*(i_st + D^T*i_f) - L_src*(di_st/dt + D^T*di_f/dt)
        """
        i_src = i_stator + self._D.T @ i_fault
        di_src_dt = di_stator_dt + self._D.T @ di_fault_dt
        return E_src - self._r_src @ i_src - self._l_src @ di_src_dt
