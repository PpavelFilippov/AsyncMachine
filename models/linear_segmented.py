"""
    Модуль models/linear_segmented.py.
    Состав:
    Классы: SegmentedLinearInductionMachine.
    Функции: нет.

    Модель асинхронной машины с сегментацией обмотки статора
    для моделирования межвитковых замыканий.

    Каждая фазная обмотка статора (A, B, C) разделена на два сегмента:
    - основной (доля 1-mu витков) и выделенный (доля mu витков).
    Внутренний вектор состояния содержит 10 переменных:
    y = [i_A1, i_A2, i_B1, i_B2, i_C1, i_C2, i_2a, i_2b, i_2c, omega_r]

    Все индуктивности масштабируются как произведение долей витков:
    L(Xi, Yj) = xi * yj * L_full(X, Y)
    Сопротивления пропорциональны числу витков:
    R(Xi) = xi * R1
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from core.parameters import MachineParameters
from .base import MachineModel


_PHASE_INDEX = {"A": 0, "B": 1, "C": 2}


class SegmentedLinearInductionMachine(MachineModel):
    """Линейная модель асинхронной машины с сегментацией обмотки статора.

    Внутренний вектор: 9 электрических + 1 механическая = 10 переменных.
    Интерфейс electrical_matrices возвращает (L_9x9, b0_9).
    Для подключения к внешней схеме используется constraint matrix.
    """

    # Индексы в 10-элементном внутреннем векторе состояния
    _I_A1 = 0
    _I_A2 = 1
    _I_B1 = 2
    _I_B2 = 3
    _I_C1 = 4
    _I_C2 = 5
    _I_2A = 6
    _I_2B = 7
    _I_2C = 8
    _OMEGA = 9

    def __init__(
        self,
        params: MachineParameters,
        mu_A: float = 0.1,
        mu_B: float = 0.1,
        mu_C: float = 0.1,
    ):
        """Создает модель с сегментацией обмоток статора.

        Args:
            params: параметры машины.
            mu_A: доля витков выделенного сегмента фазы A (0 < mu < 1).
            mu_B: доля витков выделенного сегмента фазы B.
            mu_C: доля витков выделенного сегмента фазы C.
        """
        super().__init__(params)

        for name, val in [("mu_A", mu_A), ("mu_B", mu_B), ("mu_C", mu_C)]:
            if not (0.0 < val < 1.0):
                raise ValueError(
                    f"{name}={val} must be in (0, 1) exclusive. "
                    f"Use LinearInductionMachine for unsegmented model."
                )

        self.R1 = params.R1
        self.R2 = params.R2
        self.L1s = params.L1sigma
        self.L2s = params.L2sigma
        self.Lm = params.Lm
        self.J = params.J
        self.p = params.p

        self.mu_A = mu_A
        self.mu_B = mu_B
        self.mu_C = mu_C

        # Доли витков
        self.a1 = 1.0 - mu_A
        self.a2 = mu_A
        self.b1 = 1.0 - mu_B
        self.b2 = mu_B
        self.c1 = 1.0 - mu_C
        self.c2 = mu_C

        # Матрица расширения T_s (3x6): i_phase_eq = T_s @ i_stator_seg
        self._T_s = self._build_expansion_matrix()

        # Исходные матрицы (несегментированные) 3x3
        self._Lss_orig, self._Lrr_orig, self._Lsr_orig = (
            self._build_original_submatrices()
        )

        # Сегментированная матрица индуктивностей 9x9
        self._L_seg = self._build_segmented_inductance_matrix()

        # Вектор сопротивлений сегментов (9 элементов)
        self._R_seg = self._build_segmented_resistances()


    # Построение матриц
    def _build_expansion_matrix(self) -> np.ndarray:
        """Строит матрицу расширения T_s (3x6).

        T_s @ i_seg_6 = [i_A_eq, i_B_eq, i_C_eq]
        где i_X_eq = x1*i_X1 + x2*i_X2 — эквивалентный ток фазы.
        """
        T = np.zeros((3, 6), dtype=float)
        T[0, 0] = self.a1
        T[0, 1] = self.a2
        T[1, 2] = self.b1
        T[1, 3] = self.b2
        T[2, 4] = self.c1
        T[2, 5] = self.c2
        return T

    def _build_original_submatrices(self):
        """Строит исходные подматрицы Lss, Lrr, Lsr (3x3 каждая)."""

        Mm = 2.0 * self.Lm / 3.0
        m_off = -0.5 * Mm

        L1_diag = self.L1s + Mm
        L2_diag = self.L2s + Mm

        Lss = np.array([
            [L1_diag, m_off, m_off],
            [m_off, L1_diag, m_off],
            [m_off, m_off, L1_diag],
        ])
        Lrr = np.array([
            [L2_diag, m_off, m_off],
            [m_off, L2_diag, m_off],
            [m_off, m_off, L2_diag],
        ])
        Lsr = np.array([
            [Mm, m_off, m_off],
            [m_off, Mm, m_off],
            [m_off, m_off, Mm],
        ])
        return Lss, Lrr, Lsr

    def _build_segmented_inductance_matrix(self) -> np.ndarray:
        """Строит сегментированную матрицу индуктивностей L_seg (9x9).

        L_seg = | T_s^T * Lss * T_s    T_s^T * Lsr |
                | Lsr^T * T_s          Lrr          |
        """

        T = self._T_s

        Lss_seg = T.T @ self._Lss_orig @ T        # 6x6
        Lsr_seg = T.T @ self._Lsr_orig             # 6x3
        Lrr_seg = self._Lrr_orig.copy()            # 3x3

        L = np.zeros((9, 9), dtype=float)
        L[0:6, 0:6] = Lss_seg
        L[0:6, 6:9] = Lsr_seg
        L[6:9, 0:6] = Lsr_seg.T
        L[6:9, 6:9] = Lrr_seg

        return L

    def _build_segmented_resistances(self) -> np.ndarray:
        """Строит вектор сопротивлений сегментов (9 элементов)."""

        return np.array([
            self.a1 * self.R1,     # R_A1
            self.a2 * self.R1,     # R_A2
            self.b1 * self.R1,     # R_B1
            self.b2 * self.R1,     # R_B2
            self.c1 * self.R1,     # R_C1
            self.c2 * self.R1,     # R_C2
            self.R2,               # R_2a
            self.R2,               # R_2b
            self.R2,               # R_2c
        ], dtype=float)


    # Свойства модели
    @property
    def n_electrical_vars(self) -> int:
        """Число электрических переменных внутренней модели."""
        return 9

    @property
    def n_state_vars(self) -> int:
        """Полное число переменных состояния (электр. + мех.)."""
        return 10

    @property
    def segment_fractions(self) -> tuple:
        """Возвращает (a1, a2, b1, b2, c1, c2)."""
        return (self.a1, self.a2, self.b1, self.b2, self.c1, self.c2)

    @property
    def T_s(self) -> np.ndarray:
        """Матрица расширения (3x6)."""
        return self._T_s.copy()

    @property
    def L_segmented(self) -> np.ndarray:
        """Сегментированная матрица индуктивностей (9x9)."""
        return self._L_seg.copy()


    # Эквивалентные токи
    def equivalent_stator_currents(
        self, i_seg: np.ndarray
    ) -> tuple[float, float, float]:
        """Вычисляет эквивалентные фазные токи статора из сегментных.

        Args:
            i_seg: вектор из 6 сегментных токов [i_A1, i_A2, i_B1, i_B2, i_C1, i_C2].

        Returns:
            (i_A_eq, i_B_eq, i_C_eq): эквивалентные фазные токи.
        """
        i_eq = self._T_s @ i_seg
        return float(i_eq[0]), float(i_eq[1]), float(i_eq[2])


    # Электромагнитный момент
    def electromagnetic_torque(
        self,
        i1A: float, i1B: float, i1C: float,
        i2a: float, i2b: float, i2c: float,
    ) -> float:
        """Вычисляет электромагнитный момент по эквивалентным токам.

        Для совместимости с MachineModel принимает 6 аргументов.
        Здесь i1A, i1B, i1C — это уже эквивалентные статорные токи.
        """
        return (self.p * self.Lm / np.sqrt(3)) * (
            (i1A * i2c + i1B * i2a + i1C * i2b)
            - (i1A * i2b + i1B * i2c + i1C * i2a)
        )

    def electromagnetic_torque_from_state(self, y: np.ndarray) -> float:
        """Вычисляет электромагнитный момент из полного вектора состояния (10 перем.)."""
        i_seg = y[0:6]
        i_A_eq, i_B_eq, i_C_eq = self.equivalent_stator_currents(i_seg)
        i2a, i2b, i2c = y[6], y[7], y[8]
        return self.electromagnetic_torque(i_A_eq, i_B_eq, i_C_eq, i2a, i2b, i2c)


    # Потокосцепление
    def flux_linkage_phaseA(
        self,
        i1A: float, i1B: float, i1C: float,
        i2a: float, i2b: float, i2c: float,
    ) -> float:
        """Вычисляет потокосцепление фазы A по эквивалентным токам."""
        Mm = 2.0 * self.Lm / 3.0
        return (
            self.L1s * i1A
            + Mm * (i1A - (i1B + i1C) / 2.0)
            + Mm * (i2a - (i2b + i2c) / 2.0)
        )


    # ЭДС ротора
    def _rotor_emf(
        self,
        i2a: float, i2b: float, i2c: float,
        imA: float, imB: float, imC: float,
        omega_mech: float,
    ) -> np.ndarray:
        """Вычисляет ЭДС ротора от вращения.

        Использует те же формулы, что и LinearInductionMachine,
        но с эквивалентными намагничивающими токами.
        """
        omega_e = omega_mech * self.p
        inv_sqrt3 = 1.0 / np.sqrt(3)

        Ea = omega_e * inv_sqrt3 * (self.L2s * (i2b - i2c) + self.Lm * (imB - imC))
        Eb = omega_e * inv_sqrt3 * (self.L2s * (i2c - i2a) + self.Lm * (imC - imA))
        Ec = omega_e * inv_sqrt3 * (self.L2s * (i2a - i2b) + self.Lm * (imA - imB))
        return np.array([Ea, Eb, Ec])


    # Основной интерфейс: electrical_matrices
    def electrical_matrices(
        self,
        t: float,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Формирует матрицу L(9x9) и вектор b0(9) электрической подсистемы.

        Args:
            t: время.
            y: вектор состояния (10 элементов):
               [i_A1, i_A2, i_B1, i_B2, i_C1, i_C2, i_2a, i_2b, i_2c, omega_r]

        Returns:
            (L_9x9, b0_9): матрица индуктивностей и правая часть.
        """
        i_seg = y[0:6]
        i2a, i2b, i2c = y[6], y[7], y[8]
        omega_r = y[9]

        # Эквивалентные статорные токи
        i_A_eq, i_B_eq, i_C_eq = self.equivalent_stator_currents(i_seg)

        # Намагничивающие токи
        imA = i_A_eq + i2a
        imB = i_B_eq + i2b
        imC = i_C_eq + i2c

        # ЭДС ротора
        E_rot = self._rotor_emf(i2a, i2b, i2c, imA, imB, imC, omega_r)

        # Правая часть: -R_i * i_i для каждого сегмента/обмотки
        R = self._R_seg
        b0 = np.array([
            -R[0] * y[0],          # -R_A1 * i_A1
            -R[1] * y[1],          # -R_A2 * i_A2
            -R[2] * y[2],          # -R_B1 * i_B1
            -R[3] * y[3],          # -R_B2 * i_B2
            -R[4] * y[4],          # -R_C1 * i_C1
            -R[5] * y[5],          # -R_C2 * i_C2
            -R[6] * i2a - E_rot[0],
            -R[7] * i2b - E_rot[1],
            -R[8] * i2c - E_rot[2],
        ])

        return self._L_seg, b0


    # Механическая подсистема
    def mechanical_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc: float,
    ) -> float:
        """Вычисляет производную механической скорости."""
        Mem = self.electromagnetic_torque_from_state(y)
        return (Mem - Mc) / self.J


    # ODE RHS (для автономного использования без ассемблера)
    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        Mc_func: Callable[[float, float], float],
        U_func: Callable[[float], np.ndarray],
    ) -> np.ndarray:
        """Вычисляет правую часть системы дифференциальных уравнений.

        Для автономного использования. Напряжение U_func возвращает
        9-элементный вектор [u_A1, u_A2, u_B1, u_B2, u_C1, u_C2, 0, 0, 0].
        """
        Us = U_func(t)
        L, b0 = self.electrical_matrices(t, y)
        b = b0.copy()
        b += Us
        di_dt = np.linalg.solve(L, b)

        Mc = Mc_func(t, y[9])
        domega_dt = self.mechanical_rhs(t, y, Mc)

        dydt = np.empty(10)
        dydt[0:9] = di_dt
        dydt[9] = domega_dt
        return dydt


    # Constraint matrices для DAE-ассемблера
    def constraint_matrix_healthy(self) -> np.ndarray:
        """Возвращает constraint matrix для здорового режима C_h (9x6).

        Все сегменты каждой фазы соединены последовательно:
        i_X1 = i_X2 = i_X для всех фаз.

        Независимые переменные: [i_A, i_B, i_C, i_2a, i_2b, i_2c].
        """
        C = np.zeros((9, 6), dtype=float)
        # Фаза A: i_A1 = i_A, i_A2 = i_A
        C[0, 0] = 1.0   # i_A1
        C[1, 0] = 1.0   # i_A2
        # Фаза B: i_B1 = i_B, i_B2 = i_B
        C[2, 1] = 1.0   # i_B1
        C[3, 1] = 1.0   # i_B2
        # Фаза C: i_C1 = i_C, i_C2 = i_C
        C[4, 2] = 1.0   # i_C1
        C[5, 2] = 1.0   # i_C2
        # Ротор (без изменений)
        C[6, 3] = 1.0   # i_2a
        C[7, 4] = 1.0   # i_2b
        C[8, 5] = 1.0   # i_2c
        return C

    def constraint_matrix_fault(self, phase: str) -> np.ndarray:
        """Возвращает constraint matrix для МВЗ в указанной фазе C_f (9x7).

        Сегменты замкнутой фазы имеют независимые токи (i_X1, i_X2).
        Остальные фазы — последовательное соединение.

        Независимые переменные при МВЗ фазы A:
        [i_A1, i_A2, i_B, i_C, i_2a, i_2b, i_2c]
        """
        phase = phase.upper()
        if phase not in _PHASE_INDEX:
            raise ValueError(f"Unknown phase '{phase}', expected A, B, or C")

        phase_idx = _PHASE_INDEX[phase]

        C = np.zeros((9, 7), dtype=float)

        col = 0
        for ph in range(3):  # 0=A, 1=B, 2=C
            seg1_row = 2 * ph       # индекс основного сегмента в 9-векторе
            seg2_row = 2 * ph + 1   # индекс выделенного сегмента

            if ph == phase_idx:
                # Замкнутая фаза: два независимых тока
                C[seg1_row, col] = 1.0    # i_X1
                col += 1
                C[seg2_row, col] = 1.0    # i_X2
                col += 1
            else:
                # Здоровая фаза: один ток
                C[seg1_row, col] = 1.0    # i_X1 = i_X
                C[seg2_row, col] = 1.0    # i_X2 = i_X
                col += 1

        # Ротор (всегда 3 независимых тока)
        C[6, col] = 1.0     # i_2a
        C[7, col + 1] = 1.0  # i_2b
        C[8, col + 2] = 1.0  # i_2c

        return C

    def line_current_indices_healthy(self) -> list[int]:
        """Индексы линейных токов [i_A, i_B, i_C] в здоровом i_ind (6-эл.)."""
        return [0, 1, 2]

    def line_current_indices_fault(self, phase: str) -> list[int]:
        """Индексы линейных токов [i_X1, ...] в i_ind при МВЗ (7-эл.).

        Линейный ток фазы = ток основного сегмента (i_X1).
        """
        phase = phase.upper()
        if phase not in _PHASE_INDEX:
            raise ValueError(f"Unknown phase '{phase}', expected A, B, or C")

        phase_idx = _PHASE_INDEX[phase]

        # Строим индексы: замкнутая фаза занимает 2 столбца, остальные по 1
        indices = []
        col = 0
        for ph in range(3):
            if ph == phase_idx:
                indices.append(col)       # i_X1 = ток линии
                col += 2                  # пропускаем i_X2
            else:
                indices.append(col)       # i_X = ток линии
                col += 1

        return indices

    def faulted_segment_index_in_ind(self, phase: str) -> int:
        """Индекс тока замкнутого сегмента (i_X2) в i_ind при МВЗ (7-эл.)."""

        phase = phase.upper()
        if phase not in _PHASE_INDEX:
            raise ValueError(f"Unknown phase '{phase}', expected A, B, or C")

        phase_idx = _PHASE_INDEX[phase]

        col = 0
        for ph in range(3):
            if ph == phase_idx:
                return col + 1   # i_X2 — второй столбец замкнутой фазы
            elif ph < phase_idx:
                col += 1
            else:
                col += 1

        raise RuntimeError("Should not reach here")

    def line_current_index_of_faulted_phase(self, phase: str) -> int:
        """Индекс линейного тока (i_X1) замкнутой фазы в i_ind при МВЗ."""

        phase = phase.upper()
        phase_idx = _PHASE_INDEX[phase]

        col = 0
        for ph in range(3):
            if ph == phase_idx:
                return col
            elif ph < phase_idx:
                col += 1
            else:
                col += 1

        raise RuntimeError("Should not reach here")
