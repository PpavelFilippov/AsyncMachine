"""
Точка входа — примеры использования SimulationBuilder.

Воспроизводит все три сценария из оригинального main.py
плюс демонстрирует обрыв фазы.
"""
import os
import numpy as np

from core.parameters import MachineParameters
from solvers import ScipySolver, SolverConfig
from scenarios import MotorStartScenario, MotorSteadyScenario, GeneratorSteadyScenario
from faults import OpenPhaseFault
from simulation import SimulationBuilder
from plotting import plot_motor_start, plot_steady_state, plot_waveforms


def main():
    params = MachineParameters()
    print(params.info())

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    solver_cfg = SolverConfig(dt_out=2e-4)


    # 1. Пуск АД в моторном режиме
    res_start = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(MotorStartScenario(t_end=2.5))
        .run()
    )
    plot_motor_start(res_start, save_path=os.path.join(output_dir, "motor_start.png"))

    I1_max = np.max(res_start.I1_mod)
    print(f"\n  |I1|макс = {I1_max:.0f} А "
          f"(кратность: {I1_max / params.In:.2f}, расч: {params.Kip:.2f})")


    # 2. Моторный режим (установившийся)
    res_motor = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(MotorSteadyScenario(t_end=2.0))
        .run()
    )
    plot_steady_state(res_motor, save_path=os.path.join(output_dir, "motor_mode.png"))
    plot_waveforms(res_motor, save_path=os.path.join(output_dir, "motor_waveforms.png"))


    # 3. Генераторный режим
    res_gen = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(GeneratorSteadyScenario(t_end=2.0))
        .run()
    )
    plot_steady_state(res_gen, save_path=os.path.join(output_dir, "generator_mode.png"))
    plot_waveforms(res_gen, save_path=os.path.join(output_dir, "generator_waveforms.png"))


    # 4. Обрыв фазы A в моторном режиме (демонстрация faults)
    res_fault = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(MotorSteadyScenario(t_end=2.0))
        .add_fault(OpenPhaseFault("A", t_start=1.0))
        .run()
    )
    plot_steady_state(res_fault, save_path=os.path.join(output_dir, "motor_fault_openA.png"))

    # =========================================================
    print(f"\nГрафики сохранены в: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()