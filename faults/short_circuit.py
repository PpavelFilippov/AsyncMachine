"""
Межфазные КЗ на клеммах статора.

Физика:
  При межфазном КЗ на клеммах образуется общий электрический узел
  для замкнутых фаз. Напряжение этого узла определяется из КЗТ и КЗН.

  Для n_f замкнутых фаз (множество F) с импедансом источника Z_src:

    U_f = (1/n_f) * sum(E_k) - (Z_src / n_f) * sum(i_1k),  k in F

  Это изменяет эффективную матрицу индуктивностей и вектор правой части ОДУ.

Реализует 4 типа КЗ:
  - Двухфазное (A-B, B-C, C-A)           -- PHASE_PHASE
  - Трёхфазное (A-B-C)                   -- PHASE_PHASE
  - Двухфазное на землю (A-B-земля)      -- PHASE_PHASE_GROUND
  - Трёхфазное на землю (A-B-C-земля)    -- PHASE_PHASE_GROUND
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Set, FrozenSet, Tuple

import numpy as np

from .base import Fault, FaultType


class ShortCircuitType(Enum):
    """Подтип межфазного КЗ"""
    PHASE_PHASE = auto()         # Межфазное без земли
    PHASE_PHASE_GROUND = auto()  # Межфазное с замыканием на землю


class InterPhaseShortCircuit(Fault):
    """
    Межфазное КЗ (металлическое, Rf=0) на клеммах статора.

    При КЗ без земли:
      Клеммное напряжение замкнутых фаз выравнивается:
        U_f = (sum E_k) / n_f - (R_src / n_f) * sum(i_1k)
                               - (L_src / n_f) * d/dt[sum(i_1k)]

      Это модифицирует:
        - Матрицу индуктивностей: L_eff[i, j] += L_src / n_f для i, j in F
        - Вектор правой части: b[k] использует осреднённую ЭДС и
          перекрёстно-связанные сопротивления

    При КЗ на землю:
      Клеммное напряжение замкнутых фаз = 0:
        U_terminal_k = 0 для всех k in F
      Импеданс источника не влияет на обмотку (ток КЗ идёт на землю)
    """

    PHASE_MAP = {"A": 0, "B": 1, "C": 2}

    def __init__(
        self,
        phases: str | list[str],
        to_ground: bool = False,
        t_start: float = 0.0,
        t_end: Optional[float] = None,
    ):
        """
        Args:
            phases: Замкнутые фазы, например ["A", "B"] или ["A", "B", "C"]
            to_ground: True если КЗ на землю
            t_start: Время включения КЗ (с)
            t_end: Время снятия КЗ (с), None = до конца симуляции
        """
        super().__init__(t_start=t_start, t_end=t_end)

        if isinstance(phases, str):
            phases = list(phases)
        phases = [p.upper() for p in phases]

        if len(phases) < 2:
            raise ValueError(
                f"Межфазное КЗ требует минимум 2 фазы, получено: {phases}"
            )
        if len(phases) > 3:
            raise ValueError(
                f"Максимум 3 фазы статора, получено: {phases}"
            )
        for p in phases:
            if p not in self.PHASE_MAP:
                raise ValueError(f"Неизвестная фаза: {p}. Допустимы: A, B, C")

        self.phases = phases
        self._indices: FrozenSet[int] = frozenset(
            self.PHASE_MAP[p] for p in self.phases
        )
        self.to_ground = to_ground
        self.sc_type = (
            ShortCircuitType.PHASE_PHASE_GROUND if to_ground
            else ShortCircuitType.PHASE_PHASE
        )
        self._n_faulted = len(self._indices)
        self._sorted_indices = sorted(self._indices)

    # ------------------------------------------------------------------
    # Интерфейс Fault
    # ------------------------------------------------------------------

    def fault_type(self) -> FaultType:
        return FaultType.INTER_PHASE_SC

    def affected_phases(self) -> Set[int]:
        return set(self._indices)

    @property
    def faulted_phase_indices(self) -> FrozenSet[int]:
        """Множество индексов замкнутых фаз"""
        return self._indices

    @property
    def n_faulted(self) -> int:
        """Количество замкнутых фаз"""
        return self._n_faulted

    @property
    def is_grounded(self) -> bool:
        """КЗ на землю?"""
        return self.to_ground

    def get_grounded_phases(self, t: float) -> Set[int]:
        """Фазы с нулевым напряжением (только для КЗ на землю)"""
        if self.is_active(t) and self.to_ground:
            return set(self._indices)
        return set()

    def modify_voltages(self, t: float, U: np.ndarray) -> np.ndarray:
        """
        Модификация напряжений для обратной совместимости
        (используется при идеальном источнике без явной матричной системы).

        Без земли: осредняет ЭДС замкнутых фаз.
        На землю: обнуляет напряжения замкнутых фаз.
        """
        if not self.is_active(t):
            return U
        U_mod = U.copy()
        if self.to_ground:
            for idx in self._indices:
                U_mod[idx] = 0.0
        else:
            avg = np.mean([U[idx] for idx in self._sorted_indices])
            for idx in self._indices:
                U_mod[idx] = avg
        return U_mod

    # ------------------------------------------------------------------
    # Модификация системы ОДУ
    # ------------------------------------------------------------------

    def build_L_eff_and_b(
        self,
        L_base: np.ndarray,
        E_src: np.ndarray,
        y_stator: np.ndarray,
        R1: float,
        R2: float,
        E_rot: np.ndarray,
        y_rotor: np.ndarray,
        r_src: float,
        l_src: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Построить эффективную матрицу индуктивностей и вектор правой части
        с учётом межфазного КЗ.

        Args:
            L_base: Базовая матрица индуктивностей машины 6x6
            E_src: ЭДС источника [E_A, E_B, E_C]
            y_stator: Токи статора [i_1A, i_1B, i_1C]
            R1: Сопротивление статора
            R2: Сопротивление ротора
            E_rot: Противо-ЭДС ротора [Ea, Eb, Ec]
            y_rotor: Токи ротора [i_2a, i_2b, i_2c]
            r_src: Активное сопротивление источника (на фазу)
            l_src: Индуктивность источника (на фазу)

        Returns:
            (L_eff, b) — 6x6 матрица и 6-вектор правой части
        """
        L_eff = L_base.copy()
        b = np.empty(6)

        faulted = self._sorted_indices
        unfaulted = [ph for ph in range(3) if ph not in self._indices]
        n_f = self._n_faulted

        if self.to_ground:
            # КЗ на землю: U_terminal = 0 для замкнутых фаз
            # Импеданс источника не добавляется (ток идёт на землю)
            for ph in faulted:
                b[ph] = -R1 * y_stator[ph]

            # Незамкнутые фазы: обычный неидеальный источник
            for ph in unfaulted:
                L_eff[ph, ph] += l_src
                b[ph] = E_src[ph] - (R1 + r_src) * y_stator[ph]
        else:
            # КЗ без земли: замкнутые фазы имеют общее напряжение U_f
            E_avg = np.mean(E_src[faulted])

            for ph in faulted:
                # L_eff: L_src/n_f на все пары замкнутых фаз
                for ph2 in faulted:
                    L_eff[ph, ph2] += l_src / n_f

                # b: осреднённая ЭДС с перекрёстными сопротивлениями
                b[ph] = E_avg - (R1 + r_src / n_f) * y_stator[ph]
                for ph2 in faulted:
                    if ph2 != ph:
                        b[ph] -= (r_src / n_f) * y_stator[ph2]

            # Незамкнутые фазы: обычный неидеальный источник
            for ph in unfaulted:
                L_eff[ph, ph] += l_src
                b[ph] = E_src[ph] - (R1 + r_src) * y_stator[ph]

        # Роторные уравнения (не зависят от КЗ на статоре)
        for k in range(3):
            b[3 + k] = -R2 * y_rotor[k] - E_rot[k]

        return L_eff, b

    def precompute_L_eff_inv(
        self,
        L_base: np.ndarray,
        l_src: float,
    ) -> np.ndarray:
        """
        Предвычислить обратную матрицу L_eff для режима КЗ.

        Матрица индуктивностей постоянна (не зависит от токов),
        поэтому может быть инвертирована один раз перед симуляцией.

        Args:
            L_base: Базовая матрица индуктивностей машины 6x6
            l_src: Индуктивность источника (на фазу)

        Returns:
            L_eff_inv: Обратная 6x6 матрица
        """
        L_eff = L_base.copy()
        faulted = self._sorted_indices
        unfaulted = [ph for ph in range(3) if ph not in self._indices]
        n_f = self._n_faulted

        if self.to_ground:
            # Нет L_src для замкнутых фаз
            for ph in unfaulted:
                L_eff[ph, ph] += l_src
        else:
            # L_src/n_f на все пары замкнутых фаз
            for ph in faulted:
                for ph2 in faulted:
                    L_eff[ph, ph2] += l_src / n_f
            for ph in unfaulted:
                L_eff[ph, ph] += l_src

        return np.linalg.inv(L_eff)

    def build_b_vector(
        self,
        E_src: np.ndarray,
        y_stator: np.ndarray,
        R1: float,
        R2: float,
        E_rot: np.ndarray,
        y_rotor: np.ndarray,
        r_src: float,
    ) -> np.ndarray:
        """
        Построить только вектор правой части (b) для режима КЗ.

        Используется совместно с предвычисленной L_eff_inv
        для быстрого вычисления di/dt = L_eff_inv @ b.

        Args:
            E_src: ЭДС источника [E_A, E_B, E_C]
            y_stator: Токи статора [i_1A, i_1B, i_1C]
            R1: Сопротивление статора
            R2: Сопротивление ротора
            E_rot: Противо-ЭДС ротора [Ea, Eb, Ec]
            y_rotor: Токи ротора [i_2a, i_2b, i_2c]
            r_src: Активное сопротивление источника (на фазу)

        Returns:
            b: 6-вектор правой части
        """
        b = np.empty(6)
        faulted = self._sorted_indices
        unfaulted = [ph for ph in range(3) if ph not in self._indices]
        n_f = self._n_faulted

        if self.to_ground:
            for ph in faulted:
                b[ph] = -R1 * y_stator[ph]
            for ph in unfaulted:
                b[ph] = E_src[ph] - (R1 + r_src) * y_stator[ph]
        else:
            E_avg = np.mean(E_src[faulted])
            for ph in faulted:
                b[ph] = E_avg - (R1 + r_src / n_f) * y_stator[ph]
                for ph2 in faulted:
                    if ph2 != ph:
                        b[ph] -= (r_src / n_f) * y_stator[ph2]
            for ph in unfaulted:
                b[ph] = E_src[ph] - (R1 + r_src) * y_stator[ph]

        # Роторные уравнения
        for k in range(3):
            b[3 + k] = -R2 * y_rotor[k] - E_rot[k]

        return b

    def terminal_voltage(
        self,
        E_src: np.ndarray,
        y_stator: np.ndarray,
        di_stator: np.ndarray,
        r_src: float,
        l_src: float,
    ) -> np.ndarray:
        """
        Вычислить клеммные напряжения статора при активном КЗ.

        Args:
            E_src: ЭДС источника [E_A, E_B, E_C]
            y_stator: Токи статора [i_1A, i_1B, i_1C]
            di_stator: Производные токов статора [di_1A/dt, di_1B/dt, di_1C/dt]
            r_src: Активное сопротивление источника
            l_src: Индуктивность источника

        Returns:
            U_terminal: [U_A, U_B, U_C] — клеммные напряжения
        """
        U = np.zeros(3)
        faulted = self._sorted_indices
        unfaulted = [ph for ph in range(3) if ph not in self._indices]
        n_f = self._n_faulted

        if self.to_ground:
            # Замкнутые фазы: U = 0
            for ph in unfaulted:
                U[ph] = E_src[ph] - r_src * y_stator[ph] - l_src * di_stator[ph]
        else:
            # Замкнутые фазы: общий потенциал U_f
            E_avg = np.mean(E_src[faulted])
            i_sum = np.sum(y_stator[faulted])
            di_sum = np.sum(di_stator[faulted])
            U_f = E_avg - (r_src / n_f) * i_sum - (l_src / n_f) * di_sum
            for ph in faulted:
                U[ph] = U_f
            for ph in unfaulted:
                U[ph] = E_src[ph] - r_src * y_stator[ph] - l_src * di_stator[ph]

        return U

    # ------------------------------------------------------------------
    # Описание
    # ------------------------------------------------------------------

    def describe(self) -> str:
        ph_str = "-".join(self.phases)
        gnd_str = " на землю" if self.to_ground else ""
        timing = f"t={self.t_start:.3f} с"
        if self.t_end is not None:
            timing += f" .. {self.t_end:.3f} с"
        return f"Межфазное КЗ {ph_str}{gnd_str} ({timing})"

    # ------------------------------------------------------------------
    # Фабричные методы
    # ------------------------------------------------------------------

    @classmethod
    def two_phase(
        cls,
        phase1: str,
        phase2: str,
        **kwargs,
    ) -> InterPhaseShortCircuit:
        """Двухфазное КЗ (без земли)"""
        return cls(phases=[phase1, phase2], to_ground=False, **kwargs)

    @classmethod
    def three_phase(cls, **kwargs) -> InterPhaseShortCircuit:
        """Трёхфазное КЗ (без земли)"""
        return cls(phases=["A", "B", "C"], to_ground=False, **kwargs)

    @classmethod
    def two_phase_ground(
        cls,
        phase1: str,
        phase2: str,
        **kwargs,
    ) -> InterPhaseShortCircuit:
        """Двухфазное КЗ на землю"""
        return cls(phases=[phase1, phase2], to_ground=True, **kwargs)

    @classmethod
    def three_phase_ground(cls, **kwargs) -> InterPhaseShortCircuit:
        """Трёхфазное КЗ на землю"""
        return cls(phases=["A", "B", "C"], to_ground=True, **kwargs)
