"""
Демо-скрипт: моделирование через схемный ассемблер (DAE).

Валидация:
  Часть A: DAESimulationBuilder (без КЗ) vs SimulationBuilder
           (4 сценария: пуск, ХХ, ступенька, установившийся режим)
  Часть B: DAESimulationBuilder (с КЗ) vs FaultSimulationBuilder
           (4 сценария: ХХ+КЗ, ступенька+КЗ)

Сравнивает результаты и проверяет числовую идентичность.
"""
from __future__ import annotations

import sys

import numpy as np

from core.parameters import MachineParameters
from core.results import SimulationResults
from faults.descriptors import phase_to_phase_fault, phase_to_ground_fault
from models.linear import LinearInductionMachine
from scenarios.motor_no_load import MotorNoLoadScenario
from scenarios.motor_start import MotorStartScenario
from scenarios.motor_steady import MotorSteadyScenario
from scenarios.motor_step_load import MotorStepLoadScenario
from simulation import SimulationBuilder
from simulation_fault import FaultSimulationBuilder
from circuit import DAESimulationBuilder
from solvers import ScipySolver, SolverConfig
from sources.three_phase_thevenin import ThreePhaseSineTheveninSource


# ======================================================================
# Сценарии с Thevenin-источником (для КЗ)
# ======================================================================

class MotorNoLoadThevenin(MotorNoLoadScenario):
    def __init__(self, t_end: float = 4.0, Mc_idle: float = 0.0,
                 Mc_friction: float = 50.0,
                 r_series: float = 0.02, l_series: float = 2e-4):
        super().__init__(t_end=t_end, Mc_idle=Mc_idle, Mc_friction=Mc_friction)
        self._r_series = r_series
        self._l_series = l_series

    def voltage_source(self, params):
        return ThreePhaseSineTheveninSource(
            amplitude=params.Um, frequency=params.fn,
            r_series=self._r_series, l_series=self._l_series,
        )


class MotorStepLoadThevenin(MotorStepLoadScenario):
    def __init__(self, t_end: float = 4.0, t_step: float = 2.0,
                 Mc_load: float | None = None, Mc_idle: float = 0.0,
                 Mc_friction: float = 50.0,
                 r_series: float = 0.02, l_series: float = 2e-4):
        super().__init__(t_end=t_end, t_step=t_step,
                         Mc_load=Mc_load, Mc_idle=Mc_idle, Mc_friction=Mc_friction)
        self._r_series = r_series
        self._l_series = l_series

    def voltage_source(self, params):
        return ThreePhaseSineTheveninSource(
            amplitude=params.Um, frequency=params.fn,
            r_series=self._r_series, l_series=self._l_series,
        )


# ======================================================================
# Утилиты сравнения
# ======================================================================

def compare_base_results(
    res_old: SimulationResults,
    res_new: SimulationResults,
    label: str,
    tol: float = 1e-8,
) -> bool:
    """Сравнить базовые 7 переменных. Вернуть True если совпадают."""
    print(f"\n  --- Сравнение: {label} ---")

    if len(res_old.t) != len(res_new.t):
        print(f"  FAIL: different lengths: {len(res_old.t)} vs {len(res_new.t)}")
        return False

    errors = {
        "dt":    np.max(np.abs(res_old.t - res_new.t)),
        "di1A":  np.max(np.abs(res_old.i1A - res_new.i1A)),
        "di1B":  np.max(np.abs(res_old.i1B - res_new.i1B)),
        "di1C":  np.max(np.abs(res_old.i1C - res_new.i1C)),
        "di2a":  np.max(np.abs(res_old.i2a - res_new.i2a)),
        "di2b":  np.max(np.abs(res_old.i2b - res_new.i2b)),
        "di2c":  np.max(np.abs(res_old.i2c - res_new.i2c)),
        "domega": np.max(np.abs(res_old.omega_r - res_new.omega_r)),
    }

    for name, val in errors.items():
        print(f"  max |{name:8s}| = {val:.2e}")

    all_max = max(errors.values())
    passed = all_max < tol
    status = "PASS" if passed else "FAIL"
    print(f"  Overall max error = {all_max:.2e}  [{status}] (tol={tol:.0e})")
    return passed


def compare_fault_results(
    res_old: SimulationResults,
    res_new: SimulationResults,
    label: str,
    tol: float = 1e-8,
) -> bool:
    """Сравнить базовые 7 переменных + токи КЗ."""
    passed = compare_base_results(res_old, res_new, label, tol)

    # Дополнительно: токи КЗ
    i_f_old = res_old.extra["i_fault"]
    i_f_new = res_new.extra["i_fault"]
    if i_f_old.shape != i_f_new.shape:
        print(f"  FAIL: i_fault shapes differ: {i_f_old.shape} vs {i_f_new.shape}")
        return False

    di_f = np.max(np.abs(i_f_old - i_f_new))
    print(f"  max |di_fault | = {di_f:.2e}")

    if di_f >= tol:
        passed = False
        print(f"  i_fault check: FAIL")

    return passed


# ======================================================================
# Часть A: без КЗ — DAESimulationBuilder vs SimulationBuilder
# ======================================================================

def run_part_a(params: MachineParameters, solver_cfg: SolverConfig) -> bool:
    """Сравнение DAESimulationBuilder (без КЗ) с SimulationBuilder."""
    print("\n" + "#" * 70)
    print("  PART A: DAESimulationBuilder (no fault) vs SimulationBuilder")
    print("#" * 70)

    scenarios = [
        ("Motor Start", MotorStartScenario(t_end=2.5)),
        ("Motor Steady", MotorSteadyScenario(t_end=2.0)),
        ("Motor No-Load", MotorNoLoadScenario(t_end=4.0, Mc_idle=0.5)),
        ("Motor Step-Load", MotorStepLoadScenario(t_end=4.0, t_step=2.0)),
    ]

    all_passed = True

    for i, (label, scenario) in enumerate(scenarios, 1):
        print("\n" + "=" * 70)
        print(f"  A.{i}: {label}")
        print("=" * 70)

        # --- Старый путь: SimulationBuilder ---
        print("\n  >>> OLD path (SimulationBuilder):")
        res_old = (
            SimulationBuilder(params)
            .model(LinearInductionMachine)
            .solver(ScipySolver("RK45", config=solver_cfg))
            .scenario(scenario)
            .run()
        )

        # --- Новый путь: DAESimulationBuilder (без fault) ---
        print("\n  >>> NEW path (DAESimulationBuilder, no fault):")
        res_new = (
            DAESimulationBuilder(params)
            .model(LinearInductionMachine)
            .solver(ScipySolver("RK45", config=solver_cfg))
            .scenario(scenario)
            .run()
        )

        passed = compare_base_results(res_old, res_new, label)
        if not passed:
            all_passed = False

    return all_passed


# ======================================================================
# Часть B: с КЗ — DAESimulationBuilder vs FaultSimulationBuilder
# ======================================================================

def run_part_b(params: MachineParameters, solver_cfg: SolverConfig) -> bool:
    """Сравнение DAESimulationBuilder (с КЗ) с FaultSimulationBuilder."""
    print("\n" + "#" * 70)
    print("  PART B: DAESimulationBuilder (with fault) vs FaultSimulationBuilder")
    print("#" * 70)

    R_SRC = 0.02
    L_SRC = 2e-4
    R_F_GROUND = 0.001
    R_GROUND = 0.0
    R_F_PHASE = 0.001

    cases = [
        ("No-load + Phase-A-to-ground",
         MotorNoLoadThevenin(t_end=8.0, Mc_idle=0.0, r_series=R_SRC, l_series=L_SRC),
         phase_to_ground_fault("A", R_f=R_F_GROUND, t_fault=4.0, R_ground=R_GROUND)),
        ("No-load + Phase-A-B",
         MotorNoLoadThevenin(t_end=8.0, Mc_idle=0.0, r_series=R_SRC, l_series=L_SRC),
         phase_to_phase_fault("A", "B", R_f=R_F_PHASE, t_fault=4.0)),
        ("Step-load + Phase-A-to-ground",
         MotorStepLoadThevenin(t_end=8.0, t_step=4.0, r_series=R_SRC, l_series=L_SRC),
         phase_to_ground_fault("A", R_f=R_F_GROUND, t_fault=4.0, R_ground=R_GROUND)),
        ("Step-load + Phase-A-B",
         MotorStepLoadThevenin(t_end=8.0, t_step=4.0, r_series=R_SRC, l_series=L_SRC),
         phase_to_phase_fault("A", "B", R_f=R_F_PHASE, t_fault=4.0)),
    ]

    all_passed = True

    for i, (label, scenario, fault) in enumerate(cases, 1):
        print("\n" + "=" * 70)
        print(f"  B.{i}: {label}")
        print("=" * 70)

        # --- Старый путь: FaultSimulationBuilder ---
        print("\n  >>> OLD path (FaultSimulationBuilder):")
        res_old = (
            FaultSimulationBuilder(params)
            .model(LinearInductionMachine)
            .solver(ScipySolver("Radau", config=solver_cfg))
            .scenario(scenario)
            .fault(fault)
            .run()
        )

        # --- Новый путь: DAESimulationBuilder ---
        print("\n  >>> NEW path (DAESimulationBuilder):")
        res_new = (
            DAESimulationBuilder(params)
            .model(LinearInductionMachine)
            .solver(ScipySolver("Radau", config=solver_cfg))
            .scenario(scenario)
            .fault(fault)
            .run()
        )

        passed = compare_fault_results(res_old, res_new, label)
        if not passed:
            all_passed = False

    return all_passed


# ======================================================================
# Основной скрипт
# ======================================================================

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    params = MachineParameters()
    print(params.info())

    solver_cfg = SolverConfig(dt_out=2e-4, max_step=2e-4, rtol=1e-6, atol=1e-8)

    passed_a = run_part_a(params, solver_cfg)
    passed_b = run_part_b(params, solver_cfg)

    # Итог
    print("\n" + "=" * 70)
    print(f"  PART A (no fault): {'ALL PASSED' if passed_a else 'SOME FAILED'}")
    print(f"  PART B (with fault): {'ALL PASSED' if passed_b else 'SOME FAILED'}")
    if passed_a and passed_b:
        print("  OVERALL: ALL TESTS PASSED")
    else:
        print("  OVERALL: SOME TESTS FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
