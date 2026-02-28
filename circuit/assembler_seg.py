"""
    Модуль circuit/assembler_seg.py.
    Состав:
    Классы: SegmentedCircuitAssembler.
    Функции: нет.

    Ассемблер для сегментированной модели двигателя с поддержкой
    constraint matrix и межвитковых замыканий.

    Здоровый режим:
        L_ext(6x6) = C_h^T * L_9x9 * C_h + L_src_embed
        rhs_ext(6) = C_h^T * b0_9 + e_src - R_src * i_line

    Режим МВЗ (фаза A):
        L_ext(7x7) = C_f^T * L_9x9 * C_f + L_src_embed
        rhs_ext(7) = C_f^T * b0_9 + e_src(для линейных) + R_f * (перераспределение)
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .components import SourceComponent
from .components_seg import SegmentedMotorComponent
from faults.interturn import InterTurnFaultDescriptor


class SegmentedCircuitAssembler:
    """Ассемблер для сегментированной модели с constraint matrix.

    Строит L_ext и rhs_ext, подключая источник к линейным индексам
    и (при МВЗ) добавляя R_f к уравнениям замкнутого сегмента.
    """

    def __init__(
        self,
        motor_comp: SegmentedMotorComponent,
        source_comp: SourceComponent,
        fault: Optional[InterTurnFaultDescriptor] = None,
    ):
        """Создает ассемблер.

        Args:
            motor_comp: компонент сегментированного двигателя.
            source_comp: компонент источника.
            fault: описание МВЗ (None для здорового режима).
        """
        self._motor = motor_comp
        self._source = source_comp
        self._fault = fault

        self._R_src = source_comp.R_src
        self._L_src = source_comp.L_src

        self._line_indices = motor_comp.line_current_indices
        self._n_ind = motor_comp.n_electrical_ind

        # Кэш L_ext
        self._L_ext_cached: Optional[np.ndarray] = None
        self._build_L_ext_cache()

    def _build_L_ext_cache(self) -> None:
        """Строит и кэширует расширенную матрицу индуктивностей."""

        L_red = self._motor._L_reduced   # C^T * L_9x9 * C
        n = self._n_ind
        li = self._line_indices

        L_ext = L_red.copy()

        # Добавляем L_src к блоку линейных индексов
        for i_loc, i_glob in enumerate(li):
            for j_loc, j_glob in enumerate(li):
                L_ext[i_glob, j_glob] += self._L_src[i_loc, j_loc]

        self._L_ext_cached = L_ext

    def build_rhs_ext(
        self,
        t: float,
        y_global: np.ndarray,
    ) -> np.ndarray:
        """Строит правую часть расширенной системы.

        rhs[line_idx] += e_src - R_src * i_line
        rhs[fault_seg_idx] += R_f * (i_line_X - i_X2)   (при МВЗ)
        rhs[line_fault_idx] -= R_f * (i_line_X - i_X2)  (при МВЗ)
        """
        _, b0_red = self._motor.electrical_matrices_reduced(t, y_global)
        rhs = b0_red.copy()

        # Источник: e_src - R_src * i_line
        e_src = self._source.emf(t)

        y_motor = self._motor.extract_state(y_global)
        li = self._line_indices
        i_line = np.array([y_motor[idx] for idx in li])

        src_contribution = e_src - self._R_src @ i_line

        for i_loc, i_glob in enumerate(li):
            rhs[i_glob] += src_contribution[i_loc]

        # МВЗ: добавляем R_f
        if self._fault is not None:
            motor = self._motor.motor
            phase = self._fault.phase
            R_f = self._fault.R_f

            # Индекс линейного тока замкнутой фазы в i_ind
            idx_line = motor.line_current_index_of_faulted_phase(phase)
            # Индекс тока замкнутого сегмента в i_ind
            idx_seg2 = motor.faulted_segment_index_in_ind(phase)

            i_line_X = y_motor[idx_line]
            i_seg2 = y_motor[idx_seg2]
            i_f = i_line_X - i_seg2    # ток утечки через R_f

            # u_A2 = R_f * i_f -> добавляем к rhs[idx_seg2]
            rhs[idx_seg2] += R_f * i_f

            # u_A1 = u_source_A - u_A2 -> вычитаем R_f*i_f из rhs[idx_line]
            rhs[idx_line] -= R_f * i_f

        return rhs

    def build_ode_rhs(
        self,
        state_size: int,
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """Создает функцию правой части для solve_ivp.

        Args:
            state_size: размер глобального вектора состояния.

        Returns:
            rhs(t, y) -> dydt
        """
        motor_comp = self._motor
        motor_offset = motor_comp.state_offset
        n_ind = self._n_ind
        L_ext = self._L_ext_cached

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            """Вычисляет производные для интегратора."""
            rhs_ext = self.build_rhs_ext(t, y)

            # Решаем L_ext * di/dt = rhs_ext
            di_dt = np.linalg.solve(L_ext, rhs_ext)

            # Механическая подсистема
            domega_dt = motor_comp.mechanical_rhs(t, y)

            # Собираем dydt
            dydt = np.zeros(state_size, dtype=float)
            dydt[motor_offset: motor_offset + n_ind] = di_dt
            dydt[motor_offset + n_ind] = domega_dt

            return dydt

        return rhs
