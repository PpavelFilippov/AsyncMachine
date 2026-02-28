"""
    Модуль graphs.py.
    Состав:
    Классы: нет.
    Функции: main.

    Три сценария:
      1. Холостой ход (no load)
      2. Наброс нагрузки (step load)
      3. Заторможенный ротор (locked rotor)
"""

from __future__ import annotations

import os
from dataclasses import replace

from core.parameters import MachineParameters
from plotting import plot_motor_start, plot_steady_state
from scenarios import MotorNoLoadScenario, MotorStepLoadScenario, MotorLockedRotorScenario
from simulation import SimulationBuilder
from solvers import ScipySolver, SolverConfig


def main() -> None:
    """Запускает три сценария и сохраняет графики."""

    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    params = MachineParameters()
    print(params.info())

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    solver_cfg = SolverConfig(dt_out=2e-4)

    # ── 1. Холостой ход ──────────────────────────────────────────────
    res_no_load = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(MotorNoLoadScenario(t_end=4.0, Mc_idle=0.0, Mc_friction=50.0))
        .run()
    )
    plot_steady_state(res_no_load, save_path=os.path.join(output_dir, "graphs_no_load.png"))

    # ── 2. Наброс нагрузки ───────────────────────────────────────────
    res_step = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(MotorStepLoadScenario(t_end=4.0, t_step=2.0))
        .run()
    )
    plot_steady_state(res_step, save_path=os.path.join(output_dir, "graphs_step_load.png"))

    # ── 3. Заторможенный ротор ────────────────────────────────────────
    # J → ∞ (очень большой момент инерции) — ротор не может вращаться.
    params_locked = replace(params, J=1e10)
    res_locked = (
        SimulationBuilder(params_locked)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(MotorLockedRotorScenario(t_end=1.0))
        .run()
    )
    plot_motor_start(res_locked, save_path=os.path.join(output_dir, "graphs_locked_rotor.png"))

    # ── Итог ─────────────────────────────────────────────────────────
    print(f"\nСохранены графики в: {output_dir}/")
    for f_name in sorted(os.listdir(output_dir)):
        if f_name.startswith("graphs_"):
            print(f"  - {f_name}")


if __name__ == "__main__":
    main()
