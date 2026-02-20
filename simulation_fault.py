"""
    Модуль simulation_fault.py.
    Состав:
    Классы: FaultSimulationBuilder.
    Функции: нет.
"""
from __future__ import annotations

from typing import Optional, Type

import numpy as np

from core.parameters import MachineParameters
from core.results import SimulationResults
from faults.descriptors import FaultDescriptor
from models.base import MachineModel
from models.linear import LinearInductionMachine
from models.linear_sc import LinearInductionMachineSC
from solvers.base import Solver, SolverConfig
from solvers.scipy_solver import ScipySolver
from scenarios.base import Scenario


class FaultSimulationBuilder:
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: model, solver, solver_config, scenario, fault, run.
    """

    def __init__(self, params: Optional[MachineParameters] = None):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self._params = params or MachineParameters()
        self._model_cls: Type[MachineModel] = LinearInductionMachine
        self._solver: Optional[Solver] = None
        self._scenario: Optional[Scenario] = None
        self._solver_config: Optional[SolverConfig] = None
        self._fault: Optional[FaultDescriptor] = None


    def model(self, model_cls: Type[MachineModel]) -> FaultSimulationBuilder:
        self._model_cls = model_cls
        return self

    def solver(self, solver: Solver) -> FaultSimulationBuilder:
        self._solver = solver
        return self

    def solver_config(self, config: SolverConfig) -> FaultSimulationBuilder:
        self._solver_config = config
        return self

    def scenario(self, scenario: Scenario) -> FaultSimulationBuilder:
        self._scenario = scenario
        return self

    def fault(self, fault: FaultDescriptor) -> FaultSimulationBuilder:
        self._fault = fault
        return self


    def run(self) -> SimulationResults:
        """Запускает расчет и возвращает результат."""

        if self._scenario is None:
            raise ValueError("Scenario is not set. Call .scenario(...)")
        if self._fault is None:
            raise ValueError("Fault is not set. Call .fault(...)")

        if self._solver is None:
            cfg = self._solver_config or SolverConfig()
            self._solver = ScipySolver(method="Radau", config=cfg)
        elif self._solver_config is not None:
            self._solver.config = self._solver_config

        params = self._params
        scenario = self._scenario
        fault = self._fault

                             
        y0 = scenario.initial_state(params)
        source = scenario.voltage_source(params)
        load = scenario.load_torque(params)
        t_span = scenario.t_span()

                                     
        r_src_mat = np.asarray(source.series_resistance_matrix(), dtype=float)
        l_src_mat = np.asarray(source.series_inductance_matrix(), dtype=float)
        _validate_3x3(r_src_mat, "series_resistance_matrix")
        _validate_3x3(l_src_mat, "series_inductance_matrix")

        l_src_embed = np.zeros((6, 6), dtype=float)
        l_src_embed[0:3, 0:3] = l_src_mat

                               
        machine_base = self._model_cls(params)

                                                           
        machine_sc = LinearInductionMachineSC(
            params, fault, r_src_mat, l_src_mat,
        )

        t_fault = fault.t_fault
        t_start, t_end = t_span

        if t_fault <= t_start or t_fault >= t_end:
            raise ValueError(
                f"t_fault={t_fault} must be within t_span=({t_start}, {t_end})"
            )

                   
        def source_emf(t: float) -> np.ndarray:
            """Возвращает фазные ЭДС источника в момент времени t."""
            return source(t)

        def Mc_func(t: float, omega_r: float) -> float:
            """Возвращает момент нагрузки в момент времени t."""
            return load(t, omega_r)

                 
        print("\n")
        print(f"  {scenario.name()} + FAULT")
        print(f"  Model: {machine_base.__class__.__name__} → "
              f"{machine_sc.__class__.__name__}")
        print(f"  Solver: {self._solver.describe()}")
        print(f"  Source: {source.describe()}")
        print(f"  Load: {load.describe()}")
        print(f"  Fault: {fault.name}")
        print(f"  t = [{t_start:.2f}, {t_end:.2f}] s, "
              f"t_fault = {t_fault:.4f} s")
        print("\n")

                                                                        
        def rhs_normal(t: float, y: np.ndarray) -> np.ndarray:
            """Вычисляет производные до момента включения короткого замыкания."""

            L_machine, b0 = machine_base.electrical_matrices(t, y)
            L_machine = np.asarray(L_machine, dtype=float)
            b0 = np.asarray(b0, dtype=float)

            E_src = np.asarray(source_emf(t), dtype=float)
            i_stator = y[0:3]

            lhs = L_machine + l_src_embed
            rhs_e = b0.copy()
            rhs_e[0:3] += E_src - r_src_mat @ i_stator
            di_dt = np.linalg.solve(lhs, rhs_e)

            Mc = Mc_func(t, y[6])
            domega_dt = machine_base.mechanical_rhs(t, y, Mc)

            dydt = np.empty(7)
            dydt[0:6] = di_dt
            dydt[6] = domega_dt
            return dydt

        print("  Segment 1: normal operation...")
        t1, y1, ok1, msg1 = self._solver.solve(
            rhs_normal, y0, (t_start, t_fault),
        )
        if not ok1:
            raise RuntimeError(f"Segment 1 solver failed: {msg1}")
        print(f"  Segment 1 done. Points: {len(t1)}")


        y_at_fault = y1[:, -1]                                  
        n_extra = fault.n_extra
        y0_ext = np.zeros(7 + n_extra, dtype=float)
        y0_ext[0:7] = y_at_fault

                                                                        
        def rhs_fault(t: float, y_ext: np.ndarray) -> np.ndarray:
            """Вычисляет производные после включения короткого замыкания."""

            return machine_sc.ode_rhs(t, y_ext, Mc_func, source_emf)

        print("  Segment 2: short-circuit operation...")
        t2, y2, ok2, msg2 = self._solver.solve(
            rhs_fault, y0_ext, (t_fault, t_end),
        )
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
            scenario_name=f"{scenario.name()} + {fault.name}",
            solver_name=self._solver.describe(),
        )

                                         
        results.extra["machine_base"] = machine_base
        results.extra["machine_sc"] = machine_sc
        results.extra["machine"] = machine_base
        results.extra["source"] = source
        results.extra["load"] = load
        results.extra["fault"] = fault
        results.extra["source_r_matrix"] = r_src_mat.copy()
        results.extra["source_l_matrix"] = l_src_mat.copy()
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


def _validate_3x3(mat: np.ndarray, name: str) -> None:
    """Проверяет корректность параметров перед расчетом."""
    if mat.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {mat.shape}")
