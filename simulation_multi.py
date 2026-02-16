"""
MultiMachineSimulation — связанное моделирование N машин на общей шине.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type, List, Set

import numpy as np

from core.parameters import MachineParameters
from core.results import SimulationResults
from core.multi_results import MultiMachineResults
from core.state import STATE_SIZE
from models.base import MachineModel
from models.linear import LinearInductionMachine
from solvers.base import Solver, SolverConfig
from solvers.scipy_solver import ScipySolver
from sources.base import VoltageSource
from sources.three_phase_sine import ThreePhaseSineSource
from loads.base import LoadTorque
from faults.base import Fault

# Число электрических переменных на машину (без omega_r)
ELEC_SIZE = 6


@dataclass
class SourceImpedance:
    """Импеданс линии/источника (на фазу)"""
    R: float = 0.005    # Ом
    L: float = 5e-5     # Гн


@dataclass
class MachineSlot:
    """Описание одной машины в мульти-машинной симуляции"""
    params: MachineParameters
    model_cls: Type[MachineModel] = LinearInductionMachine
    load: Optional[LoadTorque] = None
    faults: List[Fault] = field(default_factory=list)
    omega0: float = 0.0
    label: str = ""


class MultiMachineSimulation:
    """
    Связанное моделирование N машин на общей шине.

    Пример:
        results = (
            MultiMachineSimulation()
            .source(ThreePhaseSineSource(Um, 50))
            .source_impedance(SourceImpedance(R=0.01, L=1e-4))
            .add_machine(params1, load=load1, label="АД-1",
                         faults=[GroundFault("A", t_start=1.0, t_end=1.5)])
            .add_machine(params2, load=load2, label="АД-2")
            .time(0, 3.0)
            .solver(ScipySolver("RK45"))
            .run()
        )
    """

    def __init__(self):
        self._machines: List[MachineSlot] = []
        self._source: Optional[VoltageSource] = None
        self._source_z: SourceImpedance = SourceImpedance()
        self._solver: Optional[Solver] = None
        self._solver_config: Optional[SolverConfig] = None
        self._t_start: float = 0.0
        self._t_end: float = 2.0
        self._scenario_name: str = "МУЛЬТИ-МАШИННАЯ СИМУЛЯЦИЯ"


    # Fluent API
    def source(self, src: VoltageSource) -> MultiMachineSimulation:
        self._source = src
        return self

    def source_impedance(self, z: SourceImpedance) -> MultiMachineSimulation:
        self._source_z = z
        return self

    def add_machine(
        self,
        params: MachineParameters,
        load: Optional[LoadTorque] = None,
        model_cls: Type[MachineModel] = LinearInductionMachine,
        faults: Optional[List[Fault]] = None,
        omega0: Optional[float] = None,
        label: str = "",
    ) -> MultiMachineSimulation:
        if omega0 is None:
            omega0 = params.omega_sync
        self._machines.append(MachineSlot(
            params=params,
            model_cls=model_cls,
            load=load,
            faults=faults or [],
            omega0=omega0,
            label=label or f"АД-{len(self._machines) + 1}",
        ))
        return self

    def time(self, t_start: float, t_end: float) -> MultiMachineSimulation:
        self._t_start = t_start
        self._t_end = t_end
        return self

    def solver(self, solver: Solver) -> MultiMachineSimulation:
        self._solver = solver
        return self

    def solver_config(self, config: SolverConfig) -> MultiMachineSimulation:
        self._solver_config = config
        return self

    def name(self, scenario_name: str) -> MultiMachineSimulation:
        self._scenario_name = scenario_name
        return self


    # Execution
    def run(self) -> MultiMachineResults:
        n_mach = len(self._machines)
        if n_mach == 0:
            raise ValueError("Не добавлено ни одной машины.")

        if self._source is None:
            p0 = self._machines[0].params
            self._source = ThreePhaseSineSource(p0.Um, p0.fn)

        if self._solver is None:
            cfg = self._solver_config or SolverConfig()
            self._solver = ScipySolver("RK45", config=cfg)
        elif self._solver_config is not None:
            self._solver.config = self._solver_config

        src = self._source
        z = self._source_z
        total_vars = STATE_SIZE * n_mach
        elec_total = ELEC_SIZE * n_mach

        # Создать экземпляры моделей
        models: List[MachineModel] = []
        for slot in self._machines:
            models.append(slot.model_cls(slot.params))

        # Начальные условия
        y0 = np.zeros(total_vars)
        for i, slot in enumerate(self._machines):
            y0[i * STATE_SIZE + 6] = slot.omega0

        # Кэш: матрицы индуктивностей машин 6x6
        L_machines = []
        for model in models:
            if hasattr(model, '_L_matrix'):
                L_machines.append(model._L_matrix.copy())
            else:
                L_machines.append(model._build_inductance_matrix())

        # Предвычислить базовую глобальную матрицу (без faults)
        # L_global_base = block_diag(L_m0, L_m1, ...) + L_src в статорных позициях
        L_global_base = np.zeros((elec_total, elec_total))
        for i in range(n_mach):
            off = i * ELEC_SIZE
            L_global_base[off:off+ELEC_SIZE, off:off+ELEC_SIZE] = L_machines[i]

        # L_src добавляется ко ВСЕМ парам (i, j) в статорных фазах:
        # L_global[6*i + ph, 6*j + ph] += L_src  для ph = 0,1,2
        for ph in range(3):
            for i in range(n_mach):
                for j in range(n_mach):
                    L_global_base[i * ELEC_SIZE + ph,
                                  j * ELEC_SIZE + ph] += z.L


        # Правая часть связанной системы ОДУ
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            E_src = src(t)
            dydt = np.zeros(total_vars)

            # 1. Собрать faults
            open_sets: List[Set[int]] = []
            ground_sets: List[Set[int]] = []
            for slot in self._machines:
                op = set()
                gr = set()
                for fault in slot.faults:
                    op |= fault.get_open_phases(t)
                    gr |= fault.get_grounded_phases(t)
                open_sets.append(op)
                ground_sets.append(gr)

            # 2. Суммарный ток для R_src (обрыв исключает фазу)
            I_total = np.zeros(3)
            for i in range(n_mach):
                off = i * STATE_SIZE
                for ph in range(3):
                    if ph not in open_sets[i]:
                        I_total[ph] += y[off + ph]

            # 3. U_bus = E_src - R_src * I_total
            U_bus = E_src - z.R * I_total

            # 4. Собрать глобальный вектор правой части b (6N)
            b_global = np.zeros(elec_total)
            for i in range(n_mach):
                state_off = i * STATE_SIZE
                elec_off = i * ELEC_SIZE
                y_i = y[state_off: state_off + STATE_SIZE]
                i1A, i1B, i1C = y_i[0], y_i[1], y_i[2]
                i2a, i2b, i2c = y_i[3], y_i[4], y_i[5]
                omega_r = y_i[6]
                model = models[i]

                imA = i1A + i2a
                imB = i1B + i2b
                imC = i1C + i2c

                E_rot = model._rotor_emf(i2a, i2b, i2c, imA, imB, imC, omega_r)

                # Напряжение на зажимах с учётом ground fault
                U_stator = U_bus.copy()
                for ph in ground_sets[i]:
                    U_stator[ph] = 0.0

                b_global[elec_off + 0] = U_stator[0] - model.R1 * i1A
                b_global[elec_off + 1] = U_stator[1] - model.R1 * i1B
                b_global[elec_off + 2] = U_stator[2] - model.R1 * i1C
                b_global[elec_off + 3] = -model.R2 * i2a - E_rot[0]
                b_global[elec_off + 4] = -model.R2 * i2b - E_rot[1]
                b_global[elec_off + 5] = -model.R2 * i2c - E_rot[2]

            # 5. Собрать глобальную L с учётом faults
            L_global = L_global_base.copy()

            for i in range(n_mach):
                elec_off_i = i * ELEC_SIZE
                for ph in open_sets[i]:
                    # Обрыв фазы ph у машины i:
                    # - обнулить строку и столбец в глобальной матрице
                    row = elec_off_i + ph
                    L_global[row, :] = 0.0
                    L_global[:, row] = 0.0
                    L_global[row, row] = 1.0
                    # - демпфирование тока к нулю
                    state_off = i * STATE_SIZE
                    b_global[row] = -y[state_off + ph] / 1e-5

                    # Также убрать L_src-связь этой фазы
                    # с другими машинами (уже обнулено выше
                    # через L_global[:, row] = 0 и L_global[row, :] = 0)

            # 6. Решить глобальную систему: L_global * di/dt = b_global
            di_dt_global = np.linalg.solve(L_global, b_global)

            # 7. Раскидать результаты + механика
            for i in range(n_mach):
                state_off = i * STATE_SIZE
                elec_off = i * ELEC_SIZE

                dydt[state_off: state_off + ELEC_SIZE] = \
                    di_dt_global[elec_off: elec_off + ELEC_SIZE]

                # Механическое уравнение
                y_i = y[state_off: state_off + STATE_SIZE]
                i1A, i1B, i1C = y_i[0], y_i[1], y_i[2]
                i2a, i2b, i2c = y_i[3], y_i[4], y_i[5]
                omega_r = y_i[6]

                model = models[i]
                slot = self._machines[i]
                Mem = model.electromagnetic_torque(i1A, i1B, i1C,
                                                   i2a, i2b, i2c)
                Mc_func = slot.load if slot.load is not None \
                    else lambda t_, w_: 0.0
                Mc = Mc_func(t, omega_r)
                dydt[state_off + 6] = (Mem - Mc) / model.J

            return dydt

        # Логирование
        self._log_config(n_mach)

        # Решить
        t_span = (self._t_start, self._t_end)
        t_arr, y_arr, success, message = self._solver.solve(rhs, y0, t_span)

        if not success:
            raise RuntimeError(f"Решатель не сошёлся: {message}")

        print(f"  Решение получено. Точек: {len(t_arr)}")

        return self._build_results(t_arr, y_arr, models, n_mach)


    # Internal
    def _log_config(self, n_mach: int):
        z = self._source_z
        print("\n")
        print(f"  {self._scenario_name}")
        print(f"  Машин: {n_mach}, связь: глобальная матрица {6*n_mach}×{6*n_mach}")
        print(f"  Солвер: {self._solver.describe()}")
        print(f"  Источник: {self._source.describe()}")
        print(f"  Импеданс линии: R={z.R:.4f} Ом, L={z.L*1e3:.4f} мГн")
        for i, slot in enumerate(self._machines):
            faults_str = ""
            if slot.faults:
                faults_str = "; ".join(f.describe() for f in slot.faults)
                faults_str = f"  faults: [{faults_str}]"
            load_str = slot.load.describe() if slot.load else "нет нагрузки"
            print(f"  [{slot.label}] {slot.model_cls.__name__}, "
                  f"нагрузка={load_str}{faults_str}")
        print(f"  t = [{self._t_start:.2f}, {self._t_end:.2f}] с")
        print("\n")

    def _build_results(
        self,
        t_arr: np.ndarray,
        y_arr: np.ndarray,
        models: List[MachineModel],
        n_mach: int,
    ) -> MultiMachineResults:
        src = self._source
        z = self._source_z
        N = len(t_arr)

        machine_results: List[SimulationResults] = []
        for i in range(n_mach):
            off = i * STATE_SIZE
            y_i = y_arr[off: off + STATE_SIZE, :]
            slot = self._machines[i]

            res_i = SimulationResults.from_solver_output(
                t=t_arr, y=y_i, params=slot.params,
                scenario_name=f"{self._scenario_name} [{slot.label}]",
                solver_name=self._solver.describe(),
            )
            self._post_process(res_i, models[i], src, z, slot.faults,
                               y_arr, n_mach, i, slot.load)
            machine_results.append(res_i)

        # Суммарные токи и напряжения шины
        I_total_A = np.zeros(N)
        I_total_B = np.zeros(N)
        I_total_C = np.zeros(N)
        U_bus_A = np.zeros(N)
        U_bus_B = np.zeros(N)
        U_bus_C = np.zeros(N)

        for k in range(N):
            E = src(t_arr[k])
            I_tot = np.zeros(3)
            for i in range(n_mach):
                off = i * STATE_SIZE
                for ph in range(3):
                    is_open = False
                    for fault in self._machines[i].faults:
                        if ph in fault.get_open_phases(t_arr[k]):
                            is_open = True
                            break
                    if not is_open:
                        I_tot[ph] += y_arr[off + ph, k]

            I_total_A[k] = I_tot[0]
            I_total_B[k] = I_tot[1]
            I_total_C[k] = I_tot[2]
            U_bus = E - z.R * I_tot
            U_bus_A[k] = U_bus[0]
            U_bus_B[k] = U_bus[1]
            U_bus_C[k] = U_bus[2]

        multi_res = MultiMachineResults(
            t=t_arr,
            machines=machine_results,
            U_bus_A=U_bus_A, U_bus_B=U_bus_B, U_bus_C=U_bus_C,
            I_total_A=I_total_A, I_total_B=I_total_B, I_total_C=I_total_C,
            scenario_name=self._scenario_name,
        )

        print(f"\n{multi_res.summary()}")
        print("\n")
        return multi_res

    @staticmethod
    def _post_process(
        res: SimulationResults,
        machine: MachineModel,
        source: VoltageSource,
        z: SourceImpedance,
        faults: List[Fault],
        y_all: np.ndarray,
        n_mach: int,
        machine_idx: int,
        load=None,
    ):
        """Вычислить производные величины для одной машины"""
        N = res.N
        params = res.params
        omega_sync = params.omega_sync

        res.Mem = np.zeros(N)
        res.Mc = np.zeros(N)
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

        for k in range(N):
            i1A, i1B, i1C = res.i1A[k], res.i1B[k], res.i1C[k]
            i2a, i2b, i2c = res.i2a[k], res.i2b[k], res.i2c[k]

            res.Mem[k] = machine.electromagnetic_torque(i1A, i1B, i1C,
                                                         i2a, i2b, i2c)
            if load is not None:
                res.Mc[k] = load(res.t[k], res.omega_r[k])
            res.I1_mod[k] = machine.result_current_module(i1A, i1B, i1C)
            res.I2_mod[k] = machine.result_current_module(i2a, i2b, i2c)

            E = source(res.t[k])
            I_tot = np.zeros(3)
            for j in range(n_mach):
                off_j = j * STATE_SIZE
                I_tot += y_all[off_j:off_j+3, k]
            U_bus = E - z.R * I_tot

            U_stator = U_bus.copy()
            for fault in faults:
                for ph in fault.get_grounded_phases(res.t[k]):
                    U_stator[ph] = 0.0

            res.U1A[k], res.U1B[k], res.U1C[k] = U_stator
            res.U_mod[k] = machine.result_voltage_module(*U_stator)

            if abs(omega_sync) > 1e-10:
                res.slip[k] = (omega_sync - res.omega_r[k]) / omega_sync

            res.Psi1A[k] = machine.flux_linkage_phaseA(i1A, i1B, i1C,
                                                        i2a, i2b, i2c)
            res.imA[k] = i1A + i2a
            res.imB[k] = i1B + i2b
            res.imC[k] = i1C + i2c
            res.Im_mod[k] = machine.result_current_module(
                res.imA[k], res.imB[k], res.imC[k])

        res.n_rpm = res.omega_r * 60 / (2 * np.pi)
        res.P_elec = res.U1A * res.i1A + res.U1B * res.i1B + res.U1C * res.i1C
        res.P_mech = res.Mem * res.omega_r