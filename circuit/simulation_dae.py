"""
DAESimulationBuilder — схемный билдер симуляции.

Двигатель используется как чёрный ящик (только electrical_matrices).
Схема собирается внешним ассемблером.

Поддерживает два режима:
  1) Без КЗ  — односегментная симуляция (аналог SimulationBuilder)
  2) С КЗ    — двухсегментная симуляция (аналог FaultSimulationBuilder)

Fluent API:
    # Без КЗ:
    results = (
        DAESimulationBuilder(params)
        .scenario(MotorStartScenario())
        .solver(ScipySolver("RK45"))
        .run()
    )

    # С КЗ:
    results = (
        DAESimulationBuilder(params)
        .scenario(MotorNoLoadThevenin(t_end=8.0))
        .fault(phase_to_ground_fault("A", R_f=0.001, t_fault=4.0))
        .solver(ScipySolver("Radau"))
        .run()
    )
"""
from __future__ import annotations

from typing import Optional, Type

import numpy as np

from core.parameters import MachineParameters
from core.results import SimulationResults
from faults.descriptors import FaultDescriptor
from loads.base import LoadTorque
from models.base import MachineModel
from models.linear import LinearInductionMachine
from scenarios.base import Scenario
from solvers.base import Solver, SolverConfig
from solvers.scipy_solver import ScipySolver
from sources.base import VoltageSource

from .components import MotorComponent, SourceComponent, FaultBranchComponent
from .topology import CircuitTopology
from .assembler import CircuitAssembler


class DAESimulationBuilder:
    """
    Универсальный билдер симуляции через схемный ассемблер.

    ОДУ двигателя не модифицируются. Ассемблер строит систему уравнений
    снаружи двигателя. Поддерживает как нормальный режим, так и КЗ.
    """

    def __init__(self, params: Optional[MachineParameters] = None):
        self._params = params or MachineParameters()
        self._model_cls: Type[MachineModel] = LinearInductionMachine
        self._source: Optional[VoltageSource] = None
        self._faults: list[FaultDescriptor] = []
        self._load: Optional[LoadTorque] = None
        self._solver: Optional[Solver] = None
        self._solver_config: Optional[SolverConfig] = None
        self._t_span: Optional[tuple[float, float]] = None
        self._scenario: Optional[Scenario] = None

    # ------------------------------------------------------------------
    # Fluent API
    # ------------------------------------------------------------------

    def model(self, model_cls: Type[MachineModel]) -> DAESimulationBuilder:
        self._model_cls = model_cls
        return self

    def source(self, source: VoltageSource) -> DAESimulationBuilder:
        self._source = source
        return self

    def fault(self, fault: FaultDescriptor) -> DAESimulationBuilder:
        self._faults.append(fault)
        return self

    def load(self, load: LoadTorque) -> DAESimulationBuilder:
        self._load = load
        return self

    def solver(self, solver: Solver) -> DAESimulationBuilder:
        self._solver = solver
        return self

    def solver_config(self, config: SolverConfig) -> DAESimulationBuilder:
        self._solver_config = config
        return self

    def time_span(self, t_start: float, t_end: float) -> DAESimulationBuilder:
        self._t_span = (t_start, t_end)
        return self

    def scenario(self, scenario: Scenario) -> DAESimulationBuilder:
        self._scenario = scenario
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> SimulationResults:
        """Собрать схему и запустить симуляцию."""
        if self._faults:
            return self._run_with_fault()
        else:
            return self._run_normal()

    # ------------------------------------------------------------------
    # Нормальный режим (без КЗ)
    # ------------------------------------------------------------------

    def _run_normal(self) -> SimulationResults:
        """Односегментная симуляция без КЗ."""
        params = self._params
        source, load_torque, y0, t_span = self._resolve_scenario(params)
        solver = self._resolve_solver()

        machine = self._model_cls(params)

        def Mc_func(t: float, omega_r: float) -> float:
            return load_torque(t, omega_r)

        # Топология: двигатель + источник
        motor_comp = MotorComponent(machine, Mc_func)
        source_comp = SourceComponent(source)

        topo = CircuitTopology()
        topo.set_motor(motor_comp).set_source(source_comp)
        topo.finalize()

        assembler = CircuitAssembler(topo)

        rhs = assembler.build_ode_rhs(
            active_faults=[],
            state_size=topo.state_size,
        )

        # Logging
        scenario_name = self._scenario.name() if self._scenario else "Custom"
        print("\n")
        print(f"  {scenario_name} (DAE circuit assembler)")
        print(f"  Model: {machine.__class__.__name__}")
        print(f"  Solver: {solver.describe()}")
        print(f"  Source: {source.describe()}")
        print(f"  Load: {load_torque.describe()}")
        print(f"  t = [{t_span[0]:.2f}, {t_span[1]:.2f}] s")
        print("\n")

        # Solve
        t, y, ok, msg = solver.solve(rhs, y0, t_span)
        if not ok:
            raise RuntimeError(f"Solver failed: {msg}")

        print(f"  Solution obtained. Points: {len(t)}")

        results = SimulationResults.from_solver_output(
            t=t, y=y, params=params,
            scenario_name=scenario_name,
            solver_name=solver.describe(),
        )

        results.extra["machine"] = machine
        results.extra["source"] = source
        results.extra["load"] = load_torque
        results.extra["rhs"] = rhs
        results.extra["source_r_matrix"] = source_comp.R_src.copy()
        results.extra["source_l_matrix"] = source_comp.L_src.copy()

        print(f"\n{results.summary()}")
        print("\n")

        return results

    # ------------------------------------------------------------------
    # Режим с КЗ (двухсегментный)
    # ------------------------------------------------------------------

    def _run_with_fault(self) -> SimulationResults:
        """Двухсегментная симуляция: нормальный режим -> КЗ."""
        params = self._params
        source, load_torque, y0, t_span = self._resolve_scenario(params)
        solver = self._resolve_solver()

        fault = self._faults[0]
        t_start, t_end = t_span
        t_fault = fault.t_fault

        if t_fault <= t_start or t_fault >= t_end:
            raise ValueError(
                f"t_fault={t_fault} must be within t_span=({t_start}, {t_end})"
            )

        machine = self._model_cls(params)

        def Mc_func(t: float, omega_r: float) -> float:
            return load_torque(t, omega_r)

        # ----------------------------------------------------------
        # Топология для сегмента 1: двигатель + источник (без КЗ)
        # ----------------------------------------------------------
        motor_comp = MotorComponent(machine, Mc_func)
        source_comp = SourceComponent(source)

        topo_normal = CircuitTopology()
        topo_normal.set_motor(motor_comp).set_source(source_comp)
        topo_normal.finalize()

        assembler_normal = CircuitAssembler(topo_normal)

        # ----------------------------------------------------------
        # Топология для сегмента 2: двигатель + источник + КЗ
        # ----------------------------------------------------------
        fault_comp = FaultBranchComponent(fault)

        motor_comp_2 = MotorComponent(machine, Mc_func)
        source_comp_2 = SourceComponent(source)

        topo_fault = CircuitTopology()
        topo_fault.set_motor(motor_comp_2).set_source(source_comp_2)
        topo_fault.add_fault(fault_comp)
        topo_fault.finalize()

        assembler_fault = CircuitAssembler(topo_fault)
        assembler_fault.validate_source_impedance(topo_fault.faults)

        # Logging
        scenario_name = self._scenario.name() if self._scenario else "Custom"
        print("\n")
        print(f"  {scenario_name} + FAULT (DAE circuit assembler)")
        print(f"  Model: {machine.__class__.__name__}")
        print(f"  Solver: {solver.describe()}")
        print(f"  Source: {source.describe()}")
        print(f"  Load: {load_torque.describe()}")
        print(f"  Fault: {fault.name}")
        print(f"  t = [{t_start:.2f}, {t_end:.2f}] s, "
              f"t_fault = {t_fault:.4f} s")
        print("\n")

        # ==============================================================
        # Сегмент 1: нормальный режим [t_start, t_fault]
        # ==============================================================
        rhs_normal = assembler_normal.build_ode_rhs(
            active_faults=[],
            state_size=topo_normal.state_size,
        )

        print("  Segment 1: normal operation (circuit assembler)...")
        t1, y1, ok1, msg1 = solver.solve(rhs_normal, y0, (t_start, t_fault))
        if not ok1:
            raise RuntimeError(f"Segment 1 solver failed: {msg1}")
        print(f"  Segment 1 done. Points: {len(t1)}")

        # ==============================================================
        # Переход: расширяем вектор состояния
        # ==============================================================
        y_at_fault = y1[:, -1]
        n_extra = fault.n_extra
        y0_ext = np.zeros(7 + n_extra, dtype=float)
        y0_ext[0:7] = y_at_fault

        # ==============================================================
        # Сегмент 2: режим КЗ [t_fault, t_end]
        # ==============================================================
        rhs_fault = assembler_fault.build_ode_rhs(
            active_faults=topo_fault.faults,
            state_size=topo_fault.state_size,
        )

        print("  Segment 2: short-circuit (circuit assembler)...")
        t2, y2, ok2, msg2 = solver.solve(rhs_fault, y0_ext, (t_fault, t_end))
        if not ok2:
            raise RuntimeError(f"Segment 2 solver failed: {msg2}")
        print(f"  Segment 2 done. Points: {len(t2)}")

        # ==============================================================
        # Склейка результатов
        # ==============================================================
        if len(t2) > 0 and len(t1) > 0 and abs(t2[0] - t1[-1]) < 1e-14:
            t2 = t2[1:]
            y2 = y2[:, 1:]

        N1 = len(t1)
        N2 = len(t2)
        N_total = N1 + N2

        t_merged = np.concatenate([t1, t2])

        y_base_merged = np.zeros((7, N_total), dtype=float)
        y_base_merged[:, :N1] = y1[0:7, :]
        y_base_merged[:, N1:] = y2[0:7, :]

        i_fault_merged = np.zeros((n_extra, N_total), dtype=float)
        if n_extra > 0 and N2 > 0:
            i_fault_merged[:, N1:] = y2[7:7 + n_extra, :]

        results = SimulationResults.from_solver_output(
            t=t_merged,
            y=y_base_merged,
            params=params,
            scenario_name=f"{scenario_name} + {fault.name}",
            solver_name=solver.describe(),
        )

        results.extra["machine"] = machine
        results.extra["machine_base"] = machine
        results.extra["source"] = source
        results.extra["load"] = load_torque
        results.extra["fault"] = fault
        results.extra["source_r_matrix"] = source_comp.R_src.copy()
        results.extra["source_l_matrix"] = source_comp.L_src.copy()
        results.extra["i_fault"] = i_fault_merged
        results.extra["t_fault"] = t_fault
        results.extra["N_segment1"] = N1
        results.extra["N_segment2"] = N2
        results.extra["rhs_normal"] = rhs_normal
        results.extra["rhs_fault"] = rhs_fault

        print(f"\n  Total points: {N_total}")
        print(f"\n{results.summary()}")
        print("\n")

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_solver(self) -> Solver:
        """Инициализировать солвер."""
        if self._solver is None:
            cfg = self._solver_config or SolverConfig()
            self._solver = ScipySolver(method="RK45", config=cfg)
        elif self._solver_config is not None:
            self._solver.config = self._solver_config
        return self._solver

    def _resolve_scenario(self, params: MachineParameters):
        """
        Извлечь source, load, y0, t_span из сценария или прямых настроек.
        """
        if self._scenario is not None:
            source = self._source or self._scenario.voltage_source(params)
            load_torque = self._load or self._scenario.load_torque(params)
            y0 = self._scenario.initial_state(params)
            t_span = self._t_span or self._scenario.t_span()
        else:
            if self._source is None:
                raise ValueError("Source is required. Call .source(...) or .scenario(...)")
            if self._load is None:
                raise ValueError("Load is required. Call .load(...) or .scenario(...)")
            if self._t_span is None:
                raise ValueError("Time span is required. Call .time_span(...) or .scenario(...)")
            source = self._source
            load_torque = self._load
            y0 = np.zeros(7, dtype=float)
            t_span = self._t_span

        return source, load_torque, y0, t_span
