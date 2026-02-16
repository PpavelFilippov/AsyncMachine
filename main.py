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
from scenarios import MotorStartScenario, MotorSteadyScenario, GeneratorSteadyScenario, MotorStepLoadScenario
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

    res_gen = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(GeneratorSteadyScenario(t_end=2.0))
        .run()
    )
    plot_steady_state(res_gen, save_path=os.path.join(output_dir, "generator_mode.png"))

    res_fault = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(MotorSteadyScenario(t_end=2.0))
        .add_fault(OpenPhaseFault("A", t_start=1.0))
        .run()
    )
    plot_steady_state(res_fault, save_path=os.path.join(output_dir, "motor_fault_openA.png"))

    res_step = (
    SimulationBuilder(params)
    .scenario(MotorStepLoadScenario(t_end=4.0, t_step=2.0))
    .run()
    )
    plot_steady_state(res_step, save_path=os.path.join(output_dir, "motor_step.png"))


    # 5. ДВЕ МАШИНЫ: замыкание фазы A на землю у АД-1
    #    (ток КЗ просаживает шину -> влияет на АД-2)
    params1 = MachineParameters()
    params2 = MachineParameters()

    load1 = RampTorque(Mc_target=params1.M_nom, t_ramp=0.5)
    load2 = RampTorque(Mc_target=params2.M_nom, t_ramp=0.5)

    multi_ground = (
        MultiMachineSimulation()
        .name("2 МАШИНЫ — ЗАМЫКАНИЕ ФАЗЫ A НА ЗЕМЛЮ")
        .source(ThreePhaseSineSource(amplitude=params1.Um, frequency=params1.fn))
        .source_impedance(SourceImpedance(R=0.01, L=1e-4))
        .add_machine(params1, load=load1, label="АД-1 (КЗ на землю A)",
                     faults=[GroundFault("A", t_start=1.0, t_end=1.5)])
        .add_machine(params2, load=load2, label="АД-2 (штатный)")
        .time(0.0, 3.0)
        .solver(ScipySolver("RK45", config=SolverConfig(dt_out=2e-4)))
        .run()
    )

    faults_ground = [
        [GroundFault("A", t_start=1.0, t_end=1.5)],  # АД-1
        [],                                             # АД-2
    ]
    plot_multi_comparison(
        multi_ground,
        faults_by_machine=faults_ground,
        save_path=os.path.join(output_dir, "two_motors_ground_fault.png"),
    )
    plot_multi_phase_currents(
        multi_ground,
        save_path=os.path.join(output_dir, "two_motors_ground_currents.png"),
    )


    # 6. ДВЕ МАШИНЫ: обрыв фазы A у АД-1
    #    (ток фазы A АД-1 = 0 -> перераспределение токов)
    multi_open = (
        MultiMachineSimulation()
        .name("2 МАШИНЫ — ОБРЫВ ФАЗЫ A")
        .source(ThreePhaseSineSource(amplitude=params1.Um, frequency=params1.fn))
        .source_impedance(SourceImpedance(R=0.01, L=1e-4))
        .add_machine(params1, load=load1, label="АД-1 (обрыв A)",
                     faults=[OpenPhaseFault("A", t_start=1.0, t_end=1.5)])
        .add_machine(params2, load=load2, label="АД-2 (штатный)")
        .time(0.0, 3.0)
        .solver(ScipySolver("RK45", config=SolverConfig(dt_out=2e-4)))
        .run()
    )

    faults_open = [
        [OpenPhaseFault("A", t_start=1.0, t_end=1.5)],  # АД-1
        [],                                               # АД-2
    ]
    plot_multi_comparison(
        multi_open,
        faults_by_machine=faults_open,
        save_path=os.path.join(output_dir, "two_motors_open_phase.png"),
    )
    plot_multi_phase_currents(
        multi_open,
        save_path=os.path.join(output_dir, "two_motors_open_currents.png"),
    )

    # =========================================================
    print(f"\nГрафики сохранены в: {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()