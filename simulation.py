"""
    Модуль simulation.py.
    Состав:
    Классы: SimulationBuilder.
    Функции: нет.
"""
from __future__ import annotations

from typing import Optional, Type

import numpy as np

from core.parameters import MachineParameters
from core.results import SimulationResults
from models.base import MachineModel
from models.linear import LinearInductionMachine
from solvers.base import Solver, SolverConfig
from solvers.scipy_solver import ScipySolver
from scenarios.base import Scenario


class SimulationBuilder:
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: model, solver, solver_config, scenario, run.
    """

    def __init__(self, params: Optional[MachineParameters] = None):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        self._params = params or MachineParameters()
        self._model_cls: Type[MachineModel] = LinearInductionMachine
        self._solver: Optional[Solver] = None
        self._scenario: Optional[Scenario] = None
        self._solver_config: Optional[SolverConfig] = None

                
    def model(self, model_cls: Type[MachineModel]) -> SimulationBuilder:
        self._model_cls = model_cls
        return self

    def solver(self, solver: Solver) -> SimulationBuilder:
        self._solver = solver
        return self

    def solver_config(self, config: SolverConfig) -> SimulationBuilder:
        self._solver_config = config
        return self

    def scenario(self, scenario: Scenario) -> SimulationBuilder:
        self._scenario = scenario
        return self

               
    def run(self) -> SimulationResults:
        """Запускает расчет и возвращает результат."""

        if self._scenario is None:
            raise ValueError("Scenario is not set. Call .scenario(...)")

        if self._solver is None:
            cfg = self._solver_config or SolverConfig()
            self._solver = ScipySolver(method="RK45", config=cfg)
        elif self._solver_config is not None:
            self._solver.config = self._solver_config

        params = self._params
        scenario = self._scenario

        machine = self._model_cls(params)
                                     
        y0 = scenario.initial_state(params)
        source = scenario.voltage_source(params)
        load = scenario.load_torque(params)
        t_span = scenario.t_span()
        r_src_mat, l_src_mat = self._source_series_matrices(source)
        l_src_embed = np.zeros((6, 6), dtype=float)
        l_src_embed[0:3, 0:3] = l_src_mat


        def source_emf(t: float) -> np.ndarray:
            """Возвращает фазные ЭДС источника в момент времени t."""
            return source(t)


        def Mc_func(t: float, omega_r: float) -> float:
            """Возвращает момент нагрузки в момент времени t."""
            return load(t, omega_r)

                                                                           
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            """Вычисляет производные состояния для интегратора."""

            L_machine, b0_machine = machine.electrical_matrices(t, y)
            L_machine = np.asarray(L_machine, dtype=float)
            b0_machine = np.asarray(b0_machine, dtype=float)
            if L_machine.shape != (6, 6):
                raise ValueError(
                    f"{machine.__class__.__name__}.electrical_matrices returned "
                    f"L with shape {L_machine.shape}, expected (6, 6)."
                )
            if b0_machine.shape != (6,):
                raise ValueError(
                    f"{machine.__class__.__name__}.electrical_matrices returned "
                    f"b0 with shape {b0_machine.shape}, expected (6,)."
                )

            E_src = np.asarray(source_emf(t), dtype=float)
            if E_src.shape != (3,):
                raise ValueError(
                    f"{source.__class__.__name__} returned source vector with shape "
                    f"{E_src.shape}, expected (3,)."
                )

            i_stator = y[0:3]
            lhs = L_machine + l_src_embed
            rhs_e = b0_machine.copy()
            rhs_e[0:3] += E_src - r_src_mat @ i_stator
            di_dt = np.linalg.solve(lhs, rhs_e)

            Mc = Mc_func(t, y[6])
            domega_dt = machine.mechanical_rhs(t, y, Mc)

            dydt = np.empty(7)
            dydt[0:6] = di_dt
            dydt[6] = domega_dt
            return dydt

                    
        print("\n")
        print(f"  {scenario.name()}")
        print(f"  Model: {machine.__class__.__name__}")
        print(f"  Solver: {self._solver.describe()}")
        print(f"  Source: {source.describe()}")
        print(f"  Load: {load.describe()}")
        print(f"  t = [{t_span[0]:.2f}, {t_span[1]:.2f}] s")
        print("\n")

                  
        t, y, success, message = self._solver.solve(rhs, y0, t_span)

        if not success:
            raise RuntimeError(f"Solver failed: {message}")

        print(f"  Solution obtained. Points: {len(t)}")

                                   
        results = SimulationResults.from_solver_output(
            t=t,
            y=y,
            params=params,
            scenario_name=scenario.name(),
            solver_name=self._solver.describe(),
        )
                                                                             
                                                                             
        results.extra["machine"] = machine
        results.extra["source"] = source
        results.extra["load"] = load
        results.extra["rhs"] = rhs
        results.extra["source_r_matrix"] = r_src_mat.copy()
        results.extra["source_l_matrix"] = l_src_mat.copy()

        print(f"\n{results.summary()}")
        print("\n")

        return results

    @staticmethod
    def _source_series_matrices(source) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает матрицы последовательных сопротивлений и индуктивностей источника."""

        R = np.asarray(source.series_resistance_matrix(), dtype=float)
        L = np.asarray(source.series_inductance_matrix(), dtype=float)
        if R.shape != (3, 3):
            raise ValueError(
                f"{source.__class__.__name__}.series_resistance_matrix returned "
                f"shape {R.shape}, expected (3, 3)."
            )
        if L.shape != (3, 3):
            raise ValueError(
                f"{source.__class__.__name__}.series_inductance_matrix returned "
                f"shape {L.shape}, expected (3, 3)."
            )
        return R, L
