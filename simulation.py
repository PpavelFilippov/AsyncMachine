"""
SimulationBuilder - single-machine simulation orchestrator.

Fluent API for configuration and run:

    results = (
        SimulationBuilder(params)
        .model(LinearInductionMachine)
        .solver(ScipySolver(method="RK45"))
        .scenario(MotorStartScenario())
        .run()
    )
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
    Single-machine simulation builder.

    Collects configuration and runs calculation with .run().
    """

    def __init__(self, params: Optional[MachineParameters] = None):
        self._params = params or MachineParameters()
        self._model_cls: Type[MachineModel] = LinearInductionMachine
        self._solver: Optional[Solver] = None
        self._scenario: Optional[Scenario] = None
        self._solver_config: Optional[SolverConfig] = None

    # Fluent API
    def model(self, model_cls: Type[MachineModel]) -> SimulationBuilder:
        """Choose machine model"""
        self._model_cls = model_cls
        return self

    def solver(self, solver: Solver) -> SimulationBuilder:
        """Choose numerical solver"""
        self._solver = solver
        return self

    def solver_config(self, config: SolverConfig) -> SimulationBuilder:
        """Set solver configuration"""
        self._solver_config = config
        return self

    def scenario(self, scenario: Scenario) -> SimulationBuilder:
        """Choose simulation scenario"""
        self._scenario = scenario
        return self

    # Execution
    def run(self) -> SimulationResults:
        """Run simulation and return results"""
        if self._scenario is None:
            raise ValueError("Scenario is not set. Call .scenario(...)")

        if self._solver is None:
            cfg = self._solver_config or SolverConfig()
            self._solver = ScipySolver(method="RK45", config=cfg)
        elif self._solver_config is not None:
            self._solver.config = self._solver_config

        params = self._params
        scenario = self._scenario

        # 1. Create machine model
        machine = self._model_cls(params)

        # 2. Read scenario components
        y0 = scenario.initial_state(params)
        source = scenario.voltage_source(params)
        load = scenario.load_torque(params)
        t_span = scenario.t_span()
        r_src_mat, l_src_mat = self._source_series_matrices(source)
        l_src_embed = np.zeros((6, 6), dtype=float)
        l_src_embed[0:3, 0:3] = l_src_mat

        # 3. Source and load callbacks
        def source_emf(t: float) -> np.ndarray:
            return source(t)

        def Mc_func(t: float, omega_r: float) -> float:
            return load(t, omega_r)

        # 4. ODE right-hand side (single modular path for any source model)
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
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

        # 5. Logging
        print("\n")
        print(f"  {scenario.name()}")
        print(f"  Model: {machine.__class__.__name__}")
        print(f"  Solver: {self._solver.describe()}")
        print(f"  Source: {source.describe()}")
        print(f"  Load: {load.describe()}")
        print(f"  t = [{t_span[0]:.2f}, {t_span[1]:.2f}] s")
        print("\n")

        # 6. Solve
        t, y, success, message = self._solver.solve(rhs, y0, t_span)

        if not success:
            raise RuntimeError(f"Solver failed: {message}")

        print(f"  Solution obtained. Points: {len(t)}")

        # 7. Build result container
        results = SimulationResults.from_solver_output(
            t=t,
            y=y,
            params=params,
            scenario_name=scenario.name(),
            solver_name=self._solver.describe(),
        )
        # Keep only raw solver output in results. Derived values are computed
        # on demand from the same model equations when statistics are needed.
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
        """Return source series R/L matrices in phase coordinates (3x3)."""
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
