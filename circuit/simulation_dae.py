"""
    Модуль circuit/simulation_dae.py.
    Состав:
    Классы: DAESimulationBuilder.
    Функции: нет.
"""
from __future__ import annotations

from typing import Optional, Type

import numpy as np

from core.parameters import MachineParameters
from core.results import SimulationResults
from faults.descriptors import FaultDescriptor
from faults.interturn import InterTurnFaultDescriptor
from loads.base import LoadTorque
from models.base import MachineModel
from models.linear import LinearInductionMachine
from models.linear_segmented import SegmentedLinearInductionMachine
from scenarios.base import Scenario
from solvers.base import Solver, SolverConfig
from solvers.scipy_solver import ScipySolver
from sources.base import VoltageSource

from .components import MotorComponent, SourceComponent, FaultBranchComponent
from .components_seg import SegmentedMotorComponent
from .topology import CircuitTopology
from .assembler import CircuitAssembler
from .assembler_seg import SegmentedCircuitAssembler


class DAESimulationBuilder:
    """
        Поля:
        Явные поля уровня класса отсутствуют.

        Методы:
        Основные публичные методы: model, source, fault, load, solver, solver_config, time_span, scenario, run.
    """

    def __init__(self, params: Optional[MachineParameters] = None):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self._params = params or MachineParameters()
        self._model_cls: Type[MachineModel] = LinearInductionMachine
        self._source: Optional[VoltageSource] = None
        self._faults: list[FaultDescriptor] = []
        self._interturn_fault: Optional[InterTurnFaultDescriptor] = None
        self._load: Optional[LoadTorque] = None
        self._solver: Optional[Solver] = None
        self._solver_config: Optional[SolverConfig] = None
        self._t_span: Optional[tuple[float, float]] = None
        self._scenario: Optional[Scenario] = None

                                                                        

    def model(self, model_cls: Type[MachineModel]) -> DAESimulationBuilder:
        self._model_cls = model_cls
        return self

    def source(self, source: VoltageSource) -> DAESimulationBuilder:
        self._source = source
        return self

    def fault(self, fault: FaultDescriptor) -> DAESimulationBuilder:
        """Возвращает описание короткого замыкания для текущего расчета."""

        self._faults.append(fault)
        return self

    def interturn_fault(
        self, fault: InterTurnFaultDescriptor
    ) -> DAESimulationBuilder:
        """Задает межвитковое замыкание для расчета."""
        self._interturn_fault = fault
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


    def run(self) -> SimulationResults:
        """Запускает расчет и возвращает результат."""

        if self._interturn_fault is not None:
            return self._run_with_interturn_fault()
        elif self._faults:
            return self._run_with_fault()
        else:
            return self._run_normal()

                                                                        

    def _run_normal(self) -> SimulationResults:
        """Выполняет DAE расчет без короткого замыкания."""

        params = self._params
        source, load_torque, y0, t_span = self._resolve_scenario(params)
        solver = self._resolve_solver()

        machine = self._model_cls(params)

        def Mc_func(t: float, omega_r: float) -> float:
            """Возвращает момент нагрузки в момент времени t."""

            return load_torque(t, omega_r)

                                         
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

                 
        scenario_name = self._scenario.name() if self._scenario else "Custom"
        print("\n")
        print(f"  {scenario_name} (DAE circuit assembler)")
        print(f"  Model: {machine.__class__.__name__}")
        print(f"  Solver: {solver.describe()}")
        print(f"  Source: {source.describe()}")
        print(f"  Load: {load_torque.describe()}")
        print(f"  t = [{t_span[0]:.2f}, {t_span[1]:.2f}] s")
        print("\n")

               
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


    def _run_with_fault(self) -> SimulationResults:
        """Выполняет DAE расчет с включением короткого замыкания."""

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
            """Возвращает момент нагрузки в момент времени t."""
            return load_torque(t, omega_r)


        motor_comp = MotorComponent(machine, Mc_func)
        source_comp = SourceComponent(source)

        topo_normal = CircuitTopology()
        topo_normal.set_motor(motor_comp).set_source(source_comp)
        topo_normal.finalize()

        assembler_normal = CircuitAssembler(topo_normal)

                                                                    
        fault_comp = FaultBranchComponent(fault)

        motor_comp_2 = MotorComponent(machine, Mc_func)
        source_comp_2 = SourceComponent(source)

        topo_fault = CircuitTopology()
        topo_fault.set_motor(motor_comp_2).set_source(source_comp_2)
        topo_fault.add_fault(fault_comp)
        topo_fault.finalize()

        assembler_fault = CircuitAssembler(topo_fault)
        assembler_fault.validate_source_impedance(topo_fault.faults)

                 
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
                                                        
                                                                        
        rhs_normal = assembler_normal.build_ode_rhs(
            active_faults=[],
            state_size=topo_normal.state_size,
        )

        print("  Segment 1: normal operation (circuit assembler)...")
        t1, y1, ok1, msg1 = solver.solve(rhs_normal, y0, (t_start, t_fault))
        if not ok1:
            raise RuntimeError(f"Segment 1 solver failed: {msg1}")
        print(f"  Segment 1 done. Points: {len(t1)}")

                                                                        
        y_at_fault = y1[:, -1]
        n_extra = fault.n_extra
        y0_ext = np.zeros(7 + n_extra, dtype=float)
        y0_ext[0:7] = y_at_fault

                                                                        
        rhs_fault = assembler_fault.build_ode_rhs(
            active_faults=topo_fault.faults,
            state_size=topo_fault.state_size,
        )

        print("  Segment 2: short-circuit (circuit assembler)...")
        t2, y2, ok2, msg2 = solver.solve(rhs_fault, y0_ext, (t_fault, t_end))
        if not ok2:
            raise RuntimeError(f"Segment 2 solver failed: {msg2}")
        print(f"  Segment 2 done. Points: {len(t2)}")

                                                                        
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


    def _run_with_interturn_fault(self) -> SimulationResults:
        """Выполняет DAE расчет с межвитковым замыканием.

        Двухсегментный подход:
        - Сегмент 1 (0 -> t_fault): сегментированная модель в здоровом режиме (7 перем.)
        - Сегмент 2 (t_fault -> t_end): МВЗ, расширенная система (8 перем.)
        """
        params = self._params
        source, load_torque, y0_base, t_span = self._resolve_scenario(params)
        solver = self._resolve_solver()

        itf = self._interturn_fault
        t_start, t_end = t_span
        t_fault = itf.t_fault

        if t_fault <= t_start or t_fault >= t_end:
            raise ValueError(
                f"t_fault={t_fault} must be within t_span=({t_start}, {t_end})"
            )

        # Создаём сегментированную модель с mu соответствующей фазы
        mu_kw = {"mu_A": 0.1, "mu_B": 0.1, "mu_C": 0.1}
        mu_kw[f"mu_{itf.phase}"] = itf.mu
        machine = SegmentedLinearInductionMachine(params, **mu_kw)

        def Mc_func(t: float, omega_r: float) -> float:
            """Возвращает момент нагрузки."""
            return load_torque(t, omega_r)

        # ---- Сегмент 1: здоровый режим ----
        C_healthy = machine.constraint_matrix_healthy()
        line_idx_h = machine.line_current_indices_healthy()

        motor_comp_h = SegmentedMotorComponent(
            machine, Mc_func, C_healthy, line_idx_h,
        )
        motor_comp_h.state_offset = 0
        source_comp = SourceComponent(source)

        assembler_h = SegmentedCircuitAssembler(
            motor_comp_h, source_comp, fault=None,
        )

        state_size_h = motor_comp_h.n_diff_vars   # 7

        rhs_normal = assembler_h.build_ode_rhs(state_size=state_size_h)

        # Печать информации
        scenario_name = self._scenario.name() if self._scenario else "Custom"
        print("\n")
        print(f"  {scenario_name} + INTERTURN FAULT (segmented DAE)")
        print(f"  Model: {machine.__class__.__name__}")
        print(f"  Solver: {solver.describe()}")
        print(f"  Source: {source.describe()}")
        print(f"  Load: {load_torque.describe()}")
        print(f"  Fault: {itf.name}")
        print(f"  mu = {itf.mu:.4f}, R_f = {itf.R_f:.4g} Ohm")
        print(f"  t = [{t_start:.2f}, {t_end:.2f}] s, "
              f"t_fault = {t_fault:.4f} s")
        print("\n")

        # y0 для здорового режима: [i_A, i_B, i_C, i_2a, i_2b, i_2c, omega_r]
        # = то же, что у базовой модели
        y0 = y0_base.copy()

        print("  Segment 1: healthy operation (segmented model)...")
        t1, y1, ok1, msg1 = solver.solve(rhs_normal, y0, (t_start, t_fault))
        if not ok1:
            raise RuntimeError(f"Segment 1 solver failed: {msg1}")
        print(f"  Segment 1 done. Points: {len(t1)}")

        # ---- Переход к сегменту 2 ----
        y_at_fault = y1[:, -1]   # [i_A, i_B, i_C, i_2a, i_2b, i_2c, omega_r]

        # Расширяем: вставляем i_X2 = i_X (до замыкания сегменты несут одинаковый ток)
        C_fault = machine.constraint_matrix_fault(itf.phase)
        line_idx_f = machine.line_current_indices_fault(itf.phase)
        idx_seg2 = machine.faulted_segment_index_in_ind(itf.phase)
        idx_line_faulted = machine.line_current_index_of_faulted_phase(itf.phase)

        # Строим y0_ext (8 элементов)
        # В здоровом i_ind: [i_A, i_B, i_C, i_2a, i_2b, i_2c]
        # В fault i_ind: [i_A1, i_A2, i_B, i_C, i_2a, i_2b, i_2c] (для фазы A)
        # i_A1 = i_A, i_A2 = i_A (начальное условие)
        n_ind_f = C_fault.shape[1]   # 7
        state_size_f = n_ind_f + 1   # 8

        y0_ext = np.zeros(state_size_f, dtype=float)

        # Заполняем i_ind в fault-порядке
        phase_idx = itf.phase_index
        src_col = 0   # позиция в y_at_fault (healthy i_ind)
        dst_col = 0   # позиция в y0_ext (fault i_ind)

        for ph in range(3):
            if ph == phase_idx:
                # Замкнутая фаза: i_X1 = i_X, i_X2 = i_X
                y0_ext[dst_col] = y_at_fault[src_col]       # i_X1
                y0_ext[dst_col + 1] = y_at_fault[src_col]   # i_X2 = i_X
                src_col += 1
                dst_col += 2
            else:
                y0_ext[dst_col] = y_at_fault[src_col]
                src_col += 1
                dst_col += 1

        # Ротор
        for k in range(3):
            y0_ext[dst_col] = y_at_fault[src_col]
            src_col += 1
            dst_col += 1

        # omega_r
        y0_ext[n_ind_f] = y_at_fault[6]   # omega_r (последний в y_at_fault)

        # ---- Сегмент 2: МВЗ ----
        motor_comp_f = SegmentedMotorComponent(
            machine, Mc_func, C_fault, line_idx_f,
        )
        motor_comp_f.state_offset = 0
        source_comp_f = SourceComponent(source)

        assembler_f = SegmentedCircuitAssembler(
            motor_comp_f, source_comp_f, fault=itf,
        )

        rhs_fault = assembler_f.build_ode_rhs(state_size=state_size_f)

        print("  Segment 2: inter-turn fault (segmented model)...")
        t2, y2, ok2, msg2 = solver.solve(rhs_fault, y0_ext, (t_fault, t_end))
        if not ok2:
            raise RuntimeError(f"Segment 2 solver failed: {msg2}")
        print(f"  Segment 2 done. Points: {len(t2)}")

        # ---- Объединение результатов ----
        if len(t2) > 0 and len(t1) > 0 and abs(t2[0] - t1[-1]) < 1e-14:
            t2 = t2[1:]
            y2 = y2[:, 1:]

        N1 = len(t1)
        N2 = len(t2)
        N_total = N1 + N2

        t_merged = np.concatenate([t1, t2])

        # Восстанавливаем базовые 7 переменных (i_A, i_B, i_C, i_2a, i_2b, i_2c, omega_r)
        # из приведённых i_ind
        y_base_merged = np.zeros((7, N_total), dtype=float)

        # Сегмент 1: y1 уже в формате [i_A, i_B, i_C, i_2a, i_2b, i_2c, omega_r]
        y_base_merged[:, :N1] = y1[0:7, :]

        # Сегмент 2: нужно извлечь эквивалентные фазные токи
        if N2 > 0:
            for k in range(N2):
                y2_k = y2[:, k]
                i_ind_k = y2_k[0:n_ind_f]

                # Восстанавливаем полный 9-вектор
                i_full = C_fault @ i_ind_k

                # Эквивалентные фазные токи
                i_eq = machine._T_s @ i_full[0:6]

                y_base_merged[0, N1 + k] = i_eq[0]    # i_A_eq
                y_base_merged[1, N1 + k] = i_eq[1]    # i_B_eq
                y_base_merged[2, N1 + k] = i_eq[2]    # i_C_eq
                y_base_merged[3, N1 + k] = i_full[6]  # i_2a
                y_base_merged[4, N1 + k] = i_full[7]  # i_2b
                y_base_merged[5, N1 + k] = i_full[8]  # i_2c
                y_base_merged[6, N1 + k] = y2_k[n_ind_f]  # omega_r

        # Извлекаем токи сегментов для extra
        i_line_faulted = np.zeros(N_total, dtype=float)
        i_seg2_faulted = np.zeros(N_total, dtype=float)
        i_fault_loop = np.zeros(N_total, dtype=float)

        # Сегмент 1: i_line = i_seg2 (= i_фазы)
        for k in range(N1):
            i_ph = y1[phase_idx, k]
            i_line_faulted[k] = i_ph
            i_seg2_faulted[k] = i_ph
            i_fault_loop[k] = 0.0

        # Сегмент 2: извлекаем из fault i_ind
        if N2 > 0:
            for k in range(N2):
                y2_k = y2[:, k]
                i_line_X = y2_k[idx_line_faulted]
                i_X2 = y2_k[idx_seg2]
                i_line_faulted[N1 + k] = i_line_X
                i_seg2_faulted[N1 + k] = i_X2
                i_fault_loop[N1 + k] = i_line_X - i_X2

        # Результаты
        results = SimulationResults.from_solver_output(
            t=t_merged,
            y=y_base_merged,
            params=params,
            scenario_name=f"{scenario_name} + {itf.name}",
            solver_name=solver.describe(),
        )

        results.extra["machine"] = machine
        results.extra["machine_base"] = machine
        results.extra["source"] = source
        results.extra["load"] = load_torque
        results.extra["interturn_fault"] = itf
        results.extra["source_r_matrix"] = source_comp.R_src.copy()
        results.extra["source_l_matrix"] = source_comp.L_src.copy()
        results.extra["i_line_faulted"] = i_line_faulted
        results.extra["i_seg2_faulted"] = i_seg2_faulted
        results.extra["i_fault_loop"] = i_fault_loop
        results.extra["t_fault"] = t_fault
        results.extra["N_segment1"] = N1
        results.extra["N_segment2"] = N2
        results.extra["rhs_normal"] = rhs_normal
        results.extra["rhs_fault"] = rhs_fault

        print(f"\n  Total points: {N_total}")
        print(f"\n{results.summary()}")
        print("\n")

        return results

    def _resolve_solver(self) -> Solver:
        """Возвращает решатель из аргумента или использует решатель по умолчанию."""

        if self._solver is None:
            cfg = self._solver_config or SolverConfig()
            self._solver = ScipySolver(method="RK45", config=cfg)
        elif self._solver_config is not None:
            self._solver.config = self._solver_config
        return self._solver

    def _resolve_scenario(self, params: MachineParameters):
        """Возвращает сценарий из аргумента или использует сценарий по умолчанию."""

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
