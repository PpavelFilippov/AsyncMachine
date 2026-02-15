"""
SimulationBuilder — конструктор и оркестратор моделирования.

Fluent API для конфигурации и запуска:

    results = (
        SimulationBuilder(params)
        .model(LinearInductionMachine)
        .solver(ScipySolver(method="RK45"))
        .scenario(MotorStartScenario())
        .add_fault(OpenPhaseFault("A", t_start=1.0))
        .run()
    )
"""
from __future__ import annotations

from typing import Optional, Type, List

import numpy as np

from core.parameters import MachineParameters
from core.results import SimulationResults
from models.base import MachineModel
from models.linear import LinearInductionMachine
from solvers.base import Solver, SolverConfig
from solvers.scipy_solver import ScipySolver
from scenarios.base import Scenario
from faults.base import Fault


class SimulationBuilder:
    """
    Конструктор моделирования.

    Собирает конфигурацию, затем run() запускает расчёт
    """

    def __init__(self, params: Optional[MachineParameters] = None):
        self._params = params or MachineParameters()
        self._model_cls: Type[MachineModel] = LinearInductionMachine
        self._solver: Optional[Solver] = None
        self._scenario: Optional[Scenario] = None
        self._faults: List[Fault] = []
        self._solver_config: Optional[SolverConfig] = None


    # Fluent API
    def model(self, model_cls: Type[MachineModel]) -> SimulationBuilder:
        """Выбрать модель машины"""
        self._model_cls = model_cls
        return self

    def solver(self, solver: Solver) -> SimulationBuilder:
        """Выбрать решатель"""
        self._solver = solver
        return self

    def solver_config(self, config: SolverConfig) -> SimulationBuilder:
        """Задать параметры решателя"""
        self._solver_config = config
        return self

    def scenario(self, scenario: Scenario) -> SimulationBuilder:
        """Выбрать сценарий моделирования"""
        self._scenario = scenario
        return self

    def add_fault(self, fault: Fault) -> SimulationBuilder:
        """Добавить неисправность"""
        self._faults.append(fault)
        return self

    def clear_faults(self) -> SimulationBuilder:
        """Убрать все неисправности"""
        self._faults.clear()
        return self


    # Execution
    def run(self) -> SimulationResults:
        """Запустить моделирование и вернуть результаты"""
        # Defaults
        if self._scenario is None:
            raise ValueError("Сценарий не задан. Вызовите .scenario(...)")

        if self._solver is None:
            cfg = self._solver_config or SolverConfig()
            self._solver = ScipySolver(method="RK45", config=cfg)
        elif self._solver_config is not None:
            self._solver.config = self._solver_config

        params = self._params
        scenario = self._scenario

        # 1. Создать модель
        machine = self._model_cls(params)

        # 2. Получить компоненты из сценария
        y0 = scenario.initial_state(params)
        source = scenario.voltage_source(params)
        load = scenario.load_torque(params)
        t_span = scenario.t_span()

        # 3. Обернуть source с учётом faults
        faults = self._faults

        def U_func(t: float) -> np.ndarray:
            U = source(t)
            for fault in faults:
                U = fault.modify_voltages(t, U)
            return U

        def Mc_func(t: float, omega_r: float) -> float:
            return load(t, omega_r)

        # 4. Правая часть ОДУ
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return machine.ode_rhs(t, y, Mc_func, U_func)

        # 5. Логирование
        print("\n")
        print(f"  {scenario.name()}")
        print(f"  Модель: {machine.__class__.__name__}")
        print(f"  Солвер: {self._solver.describe()}")
        print(f"  Источник: {source.describe()}")
        print(f"  Нагрузка: {load.describe()}")
        if faults:
            for f in faults:
                print(f"  Неисправность: {f.describe()}")
        print(f"  t = [{t_span[0]:.2f}, {t_span[1]:.2f}] с")
        print("\n")

        # 6. Решить
        t, y, success, message = self._solver.solve(rhs, y0, t_span)

        if not success:
            raise RuntimeError(f"Решатель не сошёлся: {message}")

        print(f"  Решение получено. Точек: {len(t)}")

        # 7. Собрать результаты
        results = SimulationResults.from_solver_output(
            t=t, y=y, params=params,
            scenario_name=scenario.name(),
            solver_name=self._solver.describe(),
        )

        # 8. Пост-обработка
        self._post_process(results, machine, source)

        print(f"\n{results.summary()}")

        return results


    # Post-processing
    @staticmethod
    def _post_process(
        res: SimulationResults,
        machine: MachineModel,
        source,
    ):
        """Вычислить производные величины (момент, модули, мощности, etc.)"""
        N = res.N
        params = res.params

        res.Mem = np.zeros(N)
        res.I1_mod = np.zeros(N)
        res.I2_mod = np.zeros(N)
        res.Im_mod = np.zeros(N)
        res.U1A = np.zeros(N)
        res.U1B = np.zeros(N)
        res.U1C = np.zeros(N)
        res.U_mod = np.zeros(N)
        res.slip = np.zeros(N)
        res.Psi1A = np.zeros(N)
        res.imA = np.zeros(N)
        res.imB = np.zeros(N)
        res.imC = np.zeros(N)

        omega_sync = params.omega_sync

        for k in range(N):
            i1A, i1B, i1C = res.i1A[k], res.i1B[k], res.i1C[k]
            i2a, i2b, i2c = res.i2a[k], res.i2b[k], res.i2c[k]

            res.Mem[k] = machine.electromagnetic_torque(i1A, i1B, i1C, i2a, i2b, i2c)
            res.I1_mod[k] = machine.result_current_module(i1A, i1B, i1C)
            res.I2_mod[k] = machine.result_current_module(i2a, i2b, i2c)

            Us = source(res.t[k])
            res.U1A[k], res.U1B[k], res.U1C[k] = Us[0], Us[1], Us[2]
            res.U_mod[k] = machine.result_voltage_module(Us[0], Us[1], Us[2])

            if abs(omega_sync) > 1e-10:
                res.slip[k] = (omega_sync - res.omega_r[k]) / omega_sync

            res.Psi1A[k] = machine.flux_linkage_phaseA(i1A, i1B, i1C, i2a, i2b, i2c)

            res.imA[k] = i1A + i2a
            res.imB[k] = i1B + i2b
            res.imC[k] = i1C + i2c
            res.Im_mod[k] = machine.result_current_module(
                res.imA[k], res.imB[k], res.imC[k]
            )

        res.n_rpm = res.omega_r * 60 / (2 * np.pi)
        res.P_elec = res.U1A * res.i1A + res.U1B * res.i1B + res.U1C * res.i1C
        res.P_mech = res.Mem * res.omega_r