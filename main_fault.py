"""
Скрипт моделирования КЗ на зажимах статора асинхронной машины.

Сценарии:
  1) motor_no_load  + КЗ фазы A на землю
  2) motor_no_load  + КЗ фаз A-B без земли
  3) motor_step     + КЗ фазы A на землю  (КЗ после наброса нагрузки)
  4) motor_step     + КЗ фаз A-B без земли (КЗ после наброса нагрузки)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["figure.dpi"] = 150

from core.parameters import MachineParameters
from core.results import SimulationResults
from faults.descriptors import phase_to_phase_fault, phase_to_ground_fault
from models.linear import LinearInductionMachine
from scenarios.motor_no_load import MotorNoLoadScenario
from scenarios.motor_step_load import MotorStepLoadScenario
from simulation_fault import FaultSimulationBuilder
from solvers import ScipySolver, SolverConfig
from sources.three_phase_thevenin import ThreePhaseSineTheveninSource


# ======================================================================
# Сценарии с Thevenin-источником (КЗ требует ненулевой импеданс)
# ======================================================================

class MotorNoLoadThevenin(MotorNoLoadScenario):
    """Холостой ход с Thevenin-источником."""

    def __init__(self, t_end: float = 4.0, Mc_idle: float = 0.0,
                 r_series: float = 0.02, l_series: float = 2e-4):
        super().__init__(t_end=t_end, Mc_idle=Mc_idle)
        self._r_series = r_series
        self._l_series = l_series

    def voltage_source(self, params):
        return ThreePhaseSineTheveninSource(
            amplitude=params.Um, frequency=params.fn,
            r_series=self._r_series, l_series=self._l_series,
        )


class MotorStepLoadThevenin(MotorStepLoadScenario):
    """Наброс нагрузки с Thevenin-источником."""

    def __init__(self, t_end: float = 4.0, t_step: float = 2.0,
                 Mc_load: float | None = None, Mc_idle: float = 0.0,
                 r_series: float = 0.02, l_series: float = 2e-4):
        super().__init__(t_end=t_end, t_step=t_step,
                         Mc_load=Mc_load, Mc_idle=Mc_idle)
        self._r_series = r_series
        self._l_series = l_series

    def voltage_source(self, params):
        return ThreePhaseSineTheveninSource(
            amplitude=params.Um, frequency=params.fn,
            r_series=self._r_series, l_series=self._l_series,
        )


# ======================================================================
# Графики для режимов КЗ
# ======================================================================

def plot_fault_results(
    res: SimulationResults,
    save_path: str | None = None,
) -> plt.Figure:
    """Графики для симуляции с КЗ: токи, момент, скорость, ток КЗ."""
    t = res.t
    params = res.params
    t_fault = res.extra["t_fault"]
    i_fault = res.extra["i_fault"]       # [n_extra, N]
    n_extra = i_fault.shape[0]

    machine = res.extra["machine"]
    load = res.extra.get("load")

    # Расчёт производных величин
    n_rpm = res.omega_r * 60.0 / (2.0 * np.pi)
    i1_mod = np.sqrt((res.i1B - res.i1C) ** 2 / 3.0 + res.i1A ** 2)

    mem = np.fromiter(
        (
            machine.electromagnetic_torque(
                res.i1A[k], res.i1B[k], res.i1C[k],
                res.i2a[k], res.i2b[k], res.i2c[k],
            )
            for k in range(res.N)
        ),
        dtype=float, count=res.N,
    )

    n_rows = 3 + n_extra
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
    fig.suptitle(
        f"Результаты моделирования КЗ\n({res.scenario_name})",
        fontsize=13, fontweight="bold",
    )

    # --- Row 0: Фазные токи статора ---
    axes[0, 0].plot(t, res.i1A, "b-", lw=0.5, label="i1A")
    axes[0, 0].plot(t, res.i1B, "r-", lw=0.5, label="i1B")
    axes[0, 0].plot(t, res.i1C, "g-", lw=0.5, label="i1C")
    axes[0, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--",
                        label=f"t_fault={t_fault:.2f}")
    axes[0, 0].set(xlabel="Время, с", ylabel="Ток, А",
                    title="Фазные токи статора")
    axes[0, 0].legend(fontsize=8)

    # --- Row 0: Модуль тока статора ---
    axes[0, 1].plot(t, i1_mod, "b-", lw=0.6)
    axes[0, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--",
                        label=f"t_fault={t_fault:.2f}")
    axes[0, 1].set(xlabel="Время, с", ylabel="|I1|, А",
                    title="Модуль результирующего тока статора")
    axes[0, 1].legend(fontsize=8)

    # --- Row 1: Электромагнитный момент ---
    axes[1, 0].plot(t, mem, "b-", lw=0.5)
    axes[1, 0].axhline(y=0, color="k", lw=0.5, ls="--")
    axes[1, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    if load is not None:
        mc = np.fromiter(
            (load(t[k], res.omega_r[k]) for k in range(res.N)),
            dtype=float, count=res.N,
        )
        axes[1, 0].plot(t, mc, "r--", lw=0.8, label="Mc")
        axes[1, 0].legend(fontsize=8)
    axes[1, 0].set(xlabel="Время, с", ylabel="Момент, Нм",
                    title="Электромагнитный момент")

    # --- Row 1: Скорость ---
    axes[1, 1].plot(t, n_rpm, "b-", lw=0.8)
    omega_sync = params.omega_sync
    n_sync = omega_sync * 60.0 / (2.0 * np.pi)
    axes[1, 1].axhline(y=n_sync, color="r", lw=0.8, ls="--",
                        label=f"n0={n_sync:.0f}")
    axes[1, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[1, 1].set(xlabel="Время, с", ylabel="Скорость, об/мин",
                    title="Частота вращения ротора")
    axes[1, 1].legend(fontsize=8)

    # --- Row 2: Фазные токи ротора ---
    axes[2, 0].plot(t, res.i2a, "b-", lw=0.5, label="i2a")
    axes[2, 0].plot(t, res.i2b, "r-", lw=0.5, label="i2b")
    axes[2, 0].plot(t, res.i2c, "g-", lw=0.5, label="i2c")
    axes[2, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[2, 0].set(xlabel="Время, с", ylabel="Ток, А",
                    title="Фазные токи ротора")
    axes[2, 0].legend(fontsize=8)

    # --- Row 2: Скольжение ---
    if abs(omega_sync) > 1e-10:
        slip = (omega_sync - res.omega_r) / omega_sync
        axes[2, 1].plot(t, slip, "b-", lw=0.6)
        axes[2, 1].axhline(y=0, color="r", lw=0.8, ls="--")
        axes[2, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[2, 1].set(xlabel="Время, с", ylabel="Скольжение s",
                    title="Скольжение s(t)")

    # --- Rows 3+: Токи КЗ ---
    for j in range(n_extra):
        row = 3 + j
        axes[row, 0].plot(t, i_fault[j], "m-", lw=0.7)
        axes[row, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--")
        axes[row, 0].axhline(y=0, color="k", lw=0.5, ls="--")
        axes[row, 0].set(xlabel="Время, с", ylabel="Ток КЗ, А",
                          title=f"Ток короткого замыкания i_f{j+1}")

        # Мощность на R_fault
        R_f_jj = res.extra["fault"].R_fault_2d[j, j]
        p_fault_j = R_f_jj * i_fault[j] ** 2
        axes[row, 1].plot(t, p_fault_j / 1e3, "r-", lw=0.7)
        axes[row, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--")
        axes[row, 1].axhline(y=0, color="k", lw=0.5, ls="--")
        axes[row, 1].set(xlabel="Время, с", ylabel="Мощность, кВт",
                          title=f"Мощность на R_fault[{j+1}] = {R_f_jj:.4g} Ом")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Сохранено: {save_path}")
    plt.close(fig)
    return fig


# ======================================================================
# Основной скрипт
# ======================================================================

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    params = MachineParameters()
    print(params.info())

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    solver_cfg = SolverConfig(dt_out=2e-4, max_step=2e-4, rtol=1e-6, atol=1e-8)

    # Параметры источника (Thevenin)
    R_SRC = 0.02       # Ом
    L_SRC = 2e-4       # Гн

    # Параметры КЗ
    R_F_GROUND = 0.001  # сопротивление КЗ на землю, Ом
    R_GROUND = 0.0      # сопротивление заземления, Ом
    R_F_PHASE = 0.001   # сопротивление межфазного КЗ, Ом

    # ==================================================================
    # 1) motor_no_load + КЗ фазы A на землю
    # ==================================================================
    print("=" * 70)
    print("  CASE 1: No-load + Phase-A-to-ground fault")
    print("=" * 70)

    t_fault_no_load = 2.0  # КЗ после разгона

    res1 = (
        FaultSimulationBuilder(params)
        .model(LinearInductionMachine)
        .solver(ScipySolver("Radau", config=solver_cfg))
        .scenario(MotorNoLoadThevenin(
            t_end=4.0, Mc_idle=0.0,
            r_series=R_SRC, l_series=L_SRC,
        ))
        .fault(phase_to_ground_fault(
            "A", R_f=R_F_GROUND, t_fault=t_fault_no_load, R_ground=R_GROUND,
        ))
        .run()
    )
    plot_fault_results(res1, save_path=os.path.join(
        output_dir, "fault_no_load_phase_A_ground.png"))

    # ==================================================================
    # 2) motor_no_load + КЗ фаз A-B без земли
    # ==================================================================
    print("=" * 70)
    print("  CASE 2: No-load + Phase-A-B fault (no ground)")
    print("=" * 70)

    res2 = (
        FaultSimulationBuilder(params)
        .model(LinearInductionMachine)
        .solver(ScipySolver("Radau", config=solver_cfg))
        .scenario(MotorNoLoadThevenin(
            t_end=4.0, Mc_idle=0.0,
            r_series=R_SRC, l_series=L_SRC,
        ))
        .fault(phase_to_phase_fault(
            "A", "B", R_f=R_F_PHASE, t_fault=t_fault_no_load,
        ))
        .run()
    )
    plot_fault_results(res2, save_path=os.path.join(
        output_dir, "fault_no_load_phase_AB.png"))

    # ==================================================================
    # 3) motor_step + КЗ фазы A на землю (КЗ после step)
    # ==================================================================
    print("=" * 70)
    print("  CASE 3: Step-load + Phase-A-to-ground fault (after step)")
    print("=" * 70)

    t_step = 2.0
    t_fault_step = 3.0   # КЗ через 1 с после наброса нагрузки

    res3 = (
        FaultSimulationBuilder(params)
        .model(LinearInductionMachine)
        .solver(ScipySolver("Radau", config=solver_cfg))
        .scenario(MotorStepLoadThevenin(
            t_end=5.0, t_step=t_step,
            r_series=R_SRC, l_series=L_SRC,
        ))
        .fault(phase_to_ground_fault(
            "A", R_f=R_F_GROUND, t_fault=t_fault_step, R_ground=R_GROUND,
        ))
        .run()
    )
    plot_fault_results(res3, save_path=os.path.join(
        output_dir, "fault_step_phase_A_ground.png"))

    # ==================================================================
    # 4) motor_step + КЗ фаз A-B без земли (КЗ после step)
    # ==================================================================
    print("=" * 70)
    print("  CASE 4: Step-load + Phase-A-B fault (no ground, after step)")
    print("=" * 70)

    res4 = (
        FaultSimulationBuilder(params)
        .model(LinearInductionMachine)
        .solver(ScipySolver("Radau", config=solver_cfg))
        .scenario(MotorStepLoadThevenin(
            t_end=5.0, t_step=t_step,
            r_series=R_SRC, l_series=L_SRC,
        ))
        .fault(phase_to_phase_fault(
            "A", "B", R_f=R_F_PHASE, t_fault=t_fault_step,
        ))
        .run()
    )
    plot_fault_results(res4, save_path=os.path.join(
        output_dir, "fault_step_phase_AB.png"))

    # ==================================================================
    # Итог
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"  Saved fault plots in: {output_dir}/")
    for f_name in sorted(os.listdir(output_dir)):
        if f_name.startswith("fault_"):
            print(f"    - {f_name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
