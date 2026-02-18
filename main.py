"""
Точка входа - примеры использования.

Сценарии одиночной машины: пуск, моторный, генераторный, обрыв фазы.

Связанные сценарии (две машины на общей шине):
  5. Замыкание фазы A на землю у АД-1 с восстановлением
  6. Обрыв фазы A у АД-1 с восстановлением
"""
import os
import numpy as np

from core.parameters import MachineParameters
from solvers import ScipySolver, SolverConfig
from scenarios import MotorStartScenario, MotorSteadyScenario, GeneratorSteadyScenario, MotorStepLoadScenario, MotorNoLoadScenario
from faults import OpenPhaseFault, GroundFault
from sources import ThreePhaseSineSource
from loads import RampTorque
from simulation import SimulationBuilder
from simulation_multi import MultiMachineSimulation, SourceImpedance
from plotting import (
    plot_motor_start, plot_steady_state, plot_waveforms,
    plot_multi_comparison, plot_multi_phase_currents,
)


def main():
    params = MachineParameters()
    print(params.info())

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    solver_cfg = SolverConfig(dt_out=2e-4)


    # 1–4: Одиночные сценарии (как раньше)
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
    .scenario(MotorStepLoadScenario(t_end=4.0, t_step=2.0))
    .run()
    )
    plot_steady_state(res_step, save_path=os.path.join(output_dir, "motor_step.png"))

    res_step = (
        SimulationBuilder(params)
        .scenario(MotorNoLoadScenario(t_end=4.0, Mc_idle=0.5))
        .run()
    )
    plot_steady_state(res_step, save_path=os.path.join(output_dir, "motor_no_load.png"))

    # =========================================================
    print(f"\nГрафики сохранены в: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()