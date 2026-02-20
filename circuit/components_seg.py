"""
    Модуль circuit/components_seg.py.
    Состав:
    Классы: SegmentedMotorComponent.
    Функции: нет.

    Обёртка сегментированной модели двигателя для DAE-ассемблера.
    Использует constraint matrix для приведения внутреннего 9-электрического
    вектора к независимым переменным (6 в здоровом или 7 при МВЗ).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from models.linear_segmented import SegmentedLinearInductionMachine


class SegmentedMotorComponent:
    """Обёртка сегментированной модели для DAE-ассемблера.

    Хранит модель и текущую constraint matrix C.
    Приводит внутреннюю 9-электрическую систему к системе
    с независимыми переменными через C^T * L * C и C^T * b0.
    """

    def __init__(
        self,
        motor: SegmentedLinearInductionMachine,
        Mc_func: Callable[[float, float], float],
        constraint_matrix: np.ndarray,
        line_current_indices: list[int],
    ):
        """Создает компонент сегментированного двигателя.

        Args:
            motor: сегментированная модель.
            Mc_func: функция нагрузки (t, omega_r) -> Mc.
            constraint_matrix: C (9 x n_ind), связывает полный и приведённый вектор.
            line_current_indices: индексы линейных токов [i_A, i_B, i_C] в i_ind.
        """
        self._motor = motor
        self._Mc_func = Mc_func
        self._C = constraint_matrix
        self._line_indices = line_current_indices
        self.state_offset: int = 0

        # n_ind - число независимых электрических переменных
        self._n_ind = constraint_matrix.shape[1]
        # +1 для omega_r
        self._n_diff_vars = self._n_ind + 1

        # Кэш приведённой матрицы индуктивностей (линейная модель -> L_seg = const)
        self._L_reduced: Optional[np.ndarray] = None
        self._build_L_reduced_cache()

    def _build_L_reduced_cache(self) -> None:
        """Строит и кэширует приведённую матрицу L_red = C^T * L_seg * C."""
        C = self._C
        L_seg = self._motor.L_segmented
        self._L_reduced = C.T @ L_seg @ C

    @property
    def n_diff_vars(self) -> int:
        """Число переменных состояния (n_ind электр. + 1 мех.)."""
        return self._n_diff_vars

    @property
    def n_electrical_ind(self) -> int:
        """Число независимых электрических переменных."""
        return self._n_ind

    @property
    def motor(self) -> SegmentedLinearInductionMachine:
        """Возвращает модель двигателя."""
        return self._motor

    @property
    def constraint_matrix(self) -> np.ndarray:
        """Текущая constraint matrix C (9 x n_ind)."""
        return self._C

    @property
    def line_current_indices(self) -> list[int]:
        """Индексы линейных токов в приведённом векторе i_ind."""
        return list(self._line_indices)

    def extract_state(self, y_global: np.ndarray) -> np.ndarray:
        """Извлекает переменные состояния двигателя из глобального вектора.

        Возвращает n_diff_vars элементов: [i_ind..., omega_r].
        """
        return y_global[self.state_offset: self.state_offset + self._n_diff_vars]

    def expand_to_full(self, y_motor: np.ndarray) -> np.ndarray:
        """Разворачивает приведённый вектор в полный 10-элементный.

        Args:
            y_motor: [i_ind(n_ind), omega_r] — приведённый вектор.

        Returns:
            y_full(10): [i_A1, i_A2, ..., i_2c, omega_r].
        """
        i_ind = y_motor[0:self._n_ind]
        omega_r = y_motor[self._n_ind]

        i_full = self._C @ i_ind   # 9-элементный

        y_full = np.empty(10, dtype=float)
        y_full[0:9] = i_full
        y_full[9] = omega_r
        return y_full

    def electrical_matrices_reduced(
        self, t: float, y_global: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Вычисляет приведённые (L_red, b0_red).

        L_red = C^T * L_9x9 * C  (кэшированное)
        b0_red = C^T * b0_9

        Returns:
            (L_red(n_ind x n_ind), b0_red(n_ind))
        """
        y_motor = self.extract_state(y_global)
        y_full = self.expand_to_full(y_motor)

        _, b0_full = self._motor.electrical_matrices(t, y_full)

        b0_red = self._C.T @ b0_full

        return self._L_reduced, b0_red

    def mechanical_rhs(self, t: float, y_global: np.ndarray) -> float:
        """Вычисляет производную механической скорости."""

        y_motor = self.extract_state(y_global)
        y_full = self.expand_to_full(y_motor)
        omega_r = y_full[9]
        Mc = self._Mc_func(t, omega_r)
        return self._motor.mechanical_rhs(t, y_full, Mc)
