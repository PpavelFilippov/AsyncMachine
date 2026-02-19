"""Project entry point: single-machine simulation examples without faults."""
from __future__ import annotations

import os

from core.parameters import MachineParameters
from plotting import plot_motor_start, plot_steady_state
from scenarios import MotorNoLoadScenario, MotorStartScenario, MotorSteadyScenario, MotorStepLoadScenario
from simulation import SimulationBuilder
from solvers import ScipySolver, SolverConfig


def main() -> None:
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

    res_start = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(MotorStartScenario(t_end=2.5))
        .run()
    )
    plot_motor_start(res_start, save_path=os.path.join(output_dir, "motor_start.png"))

    res_motor = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(MotorSteadyScenario(t_end=2.0))
        .run()
    )
    plot_steady_state(res_motor, save_path=os.path.join(output_dir, "motor_mode.png"))

    res_step = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(MotorStepLoadScenario(t_end=4.0, t_step=2.0))
        .run()
    )
    plot_steady_state(res_step, save_path=os.path.join(output_dir, "motor_step.png"))

    res_no_load = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(MotorNoLoadScenario(t_end=4.0, Mc_idle=0.5))
        .run()
    )
    plot_steady_state(res_no_load, save_path=os.path.join(output_dir, "motor_no_load.png"))

    print(f"\nSaved plots in: {output_dir}/")
    for f_name in sorted(os.listdir(output_dir)):
        print(f"  - {f_name}")


if __name__ == "__main__":
    main()
