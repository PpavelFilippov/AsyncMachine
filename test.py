"""
    Модуль test.py.
    Состав:
    Классы: MotorNoLoadThevenin, MotorStepLoadThevenin.
    Функции: parse_args, plot_fault_results_portable, run_no_fault_cases, run_fault_cases, main.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from core.parameters import MachineParameters
from faults.descriptors import phase_to_ground_fault, phase_to_phase_fault
from models.linear import LinearInductionMachine
from plotting import plot_steady_state
from scenarios.motor_no_load import MotorNoLoadScenario
from scenarios.motor_start import MotorStartScenario
from scenarios.motor_step_load import MotorStepLoadScenario
from simulation import SimulationBuilder
from simulation_fault import FaultSimulationBuilder
from circuit import DAESimulationBuilder
from solvers import ScipySolver, SolverConfig
from sources.three_phase_thevenin import ThreePhaseSineTheveninSource

matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["figure.dpi"] = 150


class MotorNoLoadThevenin(MotorNoLoadScenario):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: voltage_source.
    """

    def __init__(
        self,
        t_end: float,
        mc_idle: float,
        mc_friction: float,
        r_series: float,
        l_series: float,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        super().__init__(t_end=t_end, Mc_idle=mc_idle, Mc_friction=mc_friction)
        self._r_series = r_series
        self._l_series = l_series

    def voltage_source(self, params: MachineParameters):
        """Формирует источник напряжения для сценария."""
        return ThreePhaseSineTheveninSource(
            amplitude=params.Um,
            frequency=params.fn,
            r_series=self._r_series,
            l_series=self._l_series,
        )


class MotorStepLoadThevenin(MotorStepLoadScenario):
    """
        Поля:
        Явные поля уровня класса отсутствуют.
        Методы:
        Основные публичные методы: voltage_source.
    """

    def __init__(
        self,
        t_end: float,
        t_step: float,
        mc_load: float | None,
        mc_idle: float,
        mc_friction: float,
        r_series: float,
        l_series: float,
    ):
        """Создает объект и сохраняет параметры для последующих вычислений."""

        super().__init__(
            t_end=t_end,
            t_step=t_step,
            Mc_load=mc_load,
            Mc_idle=mc_idle,
            Mc_friction=mc_friction,
        )
        self._r_series = r_series
        self._l_series = l_series

    def voltage_source(self, params: MachineParameters):
        """Формирует источник напряжения для сценария."""
        return ThreePhaseSineTheveninSource(
            amplitude=params.Um,
            frequency=params.fn,
            r_series=self._r_series,
            l_series=self._l_series,
        )


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="Build plots for legacy vs DAE implementations (fault/no-fault)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "test",
        help="Directory for generated plots.",
    )
    parser.add_argument("--dt-out", type=float, default=2e-4, help="Output time step, s.")
    parser.add_argument("--max-step", type=float, default=2e-4, help="Solver max step, s.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Solver rtol.")
    parser.add_argument("--atol", type=float, default=1e-8, help="Solver atol.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use shorter simulation horizons for quicker runs.",
    )
    return parser.parse_args()


def _slug(value: str) -> str:
    """Преобразует строку в безопасный идентификатор имени файла."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _print_base_diff(label: str, res_old, res_dae) -> None:
    """Печатает отличие результатов DAE и legacy без короткого замыкания."""

    errs = {
        "dt": np.max(np.abs(res_old.t - res_dae.t)),
        "di1A": np.max(np.abs(res_old.i1A - res_dae.i1A)),
        "di1B": np.max(np.abs(res_old.i1B - res_dae.i1B)),
        "di1C": np.max(np.abs(res_old.i1C - res_dae.i1C)),
        "di2a": np.max(np.abs(res_old.i2a - res_dae.i2a)),
        "di2b": np.max(np.abs(res_old.i2b - res_dae.i2b)),
        "di2c": np.max(np.abs(res_old.i2c - res_dae.i2c)),
        "domega": np.max(np.abs(res_old.omega_r - res_dae.omega_r)),
    }
    overall = float(max(errs.values()))
    print(f"\n[DIFF] {label}:")
    for name, value in errs.items():
        print(f"  max |{name}| = {float(value):.3e}")
    print(f"  overall = {overall:.3e}")


def _print_fault_diff(label: str, res_old, res_dae) -> None:
    """Печатает отличие результатов DAE и legacy при коротком замыкании."""

    _print_base_diff(label, res_old, res_dae)
    i_f_old = res_old.extra["i_fault"]
    i_f_dae = res_dae.extra["i_fault"]
    if i_f_old.shape == i_f_dae.shape:
        di_f = float(np.max(np.abs(i_f_old - i_f_dae)))
        print(f"  max |di_fault| = {di_f:.3e}")
    else:
        print(f"  i_fault shape mismatch: {i_f_old.shape} vs {i_f_dae.shape}")


def _compute_fault_stator_phase_voltages(
    res,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вычисляет фазные напряжения статора в режиме с коротким замыканием."""

    source = res.extra["source"]
    fault = res.extra["fault"]
    rhs_normal = res.extra["rhs_normal"]
    rhs_fault = res.extra["rhs_fault"]
    i_fault = np.asarray(res.extra["i_fault"], dtype=float)
    n_seg1 = int(res.extra["N_segment1"])

    R_src = np.asarray(res.extra["source_r_matrix"], dtype=float)
    L_src = np.asarray(res.extra["source_l_matrix"], dtype=float)
    D = np.asarray(fault.D_2d, dtype=float)

    uA = np.zeros(res.N, dtype=float)
    uB = np.zeros(res.N, dtype=float)
    uC = np.zeros(res.N, dtype=float)

    for k in range(res.N):
        t_k = float(res.t[k])
        y_base = np.array(
            [
                res.i1A[k], res.i1B[k], res.i1C[k],
                res.i2a[k], res.i2b[k], res.i2c[k],
                res.omega_r[k],
            ],
            dtype=float,
        )
        i_st = y_base[0:3]
        e_src = np.asarray(source(t_k), dtype=float)

        if k < n_seg1:
            d_base = np.asarray(rhs_normal(t_k, y_base), dtype=float)
            di_st = d_base[0:3]
            i_f = np.zeros(D.shape[0], dtype=float)
            di_f = np.zeros(D.shape[0], dtype=float)
        else:
            i_f = np.asarray(i_fault[:, k], dtype=float)
            y_ext = np.concatenate([y_base, i_f])
            d_ext = np.asarray(rhs_fault(t_k, y_ext), dtype=float)
            di_st = d_ext[0:3]
            di_f = d_ext[7:]

        i_src = i_st + D.T @ i_f
        di_src = di_st + D.T @ di_f

        u = e_src - R_src @ i_src - L_src @ di_src
        uA[k], uB[k], uC[k] = float(u[0]), float(u[1]), float(u[2])

    return uA, uB, uC


def plot_fault_results_portable(
    res,
    *,
    save_fault_path: str,
    save_iv6_path: str,
) -> None:
    """Строит графики по данным моделирования."""

    t = res.t
    t_fault = float(res.extra["t_fault"])
    i_fault = np.asarray(res.extra["i_fault"], dtype=float)
    n_extra = i_fault.shape[0]

    machine = res.extra["machine"]
    load = res.extra["load"]
    fault = res.extra["fault"]
    params = res.params

    n_rpm = res.omega_r * 60.0 / (2.0 * np.pi)
    mem = np.fromiter(
        (
            machine.electromagnetic_torque(
                res.i1A[k], res.i1B[k], res.i1C[k],
                res.i2a[k], res.i2b[k], res.i2c[k],
            )
            for k in range(res.N)
        ),
        dtype=float,
        count=res.N,
    )
    mc = np.fromiter(
        (load(t[k], res.omega_r[k]) for k in range(res.N)),
        dtype=float,
        count=res.N,
    )
    u1A, u1B, u1C = _compute_fault_stator_phase_voltages(res)


    n_rows = 3 + n_extra
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
    fig.suptitle(f"Fault Results ({res.scenario_name})", fontsize=13, fontweight="bold")

    axes[0, 0].plot(t, res.i1A, "b-", lw=0.5, label="i1A")
    axes[0, 0].plot(t, res.i1B, "r-", lw=0.5, label="i1B")
    axes[0, 0].plot(t, res.i1C, "g-", lw=0.5, label="i1C")
    axes[0, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--", label=f"t_fault={t_fault:.2f}")
    axes[0, 0].set(xlabel="Time, s", ylabel="Current, A", title="Stator phase currents")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t, u1A, "b-", lw=0.5, label="U1A")
    axes[0, 1].plot(t, u1B, "r-", lw=0.5, label="U1B")
    axes[0, 1].plot(t, u1C, "g-", lw=0.5, label="U1C")
    axes[0, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--", label=f"t_fault={t_fault:.2f}")
    axes[0, 1].set(xlabel="Time, s", ylabel="Voltage, V", title="Stator phase voltages")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(t, mem, "b-", lw=0.5, label="Mem")
    axes[1, 0].plot(t, mc, "r--", lw=0.8, label="Mc")
    axes[1, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[1, 0].axhline(y=0, color="k", lw=0.5, ls="--")
    axes[1, 0].set(xlabel="Time, s", ylabel="Torque, Nm", title="Torque")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(t, n_rpm, "b-", lw=0.8)
    n_sync = params.omega_sync * 60.0 / (2.0 * np.pi)
    axes[1, 1].axhline(y=n_sync, color="r", lw=0.8, ls="--", label=f"n_sync={n_sync:.0f}")
    axes[1, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[1, 1].set(xlabel="Time, s", ylabel="Speed, rpm", title="Rotor speed")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].plot(t, res.i2a, "b-", lw=0.5, label="i2a")
    axes[2, 0].plot(t, res.i2b, "r-", lw=0.5, label="i2b")
    axes[2, 0].plot(t, res.i2c, "g-", lw=0.5, label="i2c")
    axes[2, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[2, 0].set(xlabel="Time, s", ylabel="Current, A", title="Rotor phase currents")
    axes[2, 0].legend(fontsize=8)

    omega_sync = params.omega_sync
    if abs(omega_sync) > 1e-10:
        slip = (omega_sync - res.omega_r) / omega_sync
        axes[2, 1].plot(t, slip, "b-", lw=0.6)
    axes[2, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[2, 1].axhline(y=0, color="k", lw=0.5, ls="--")
    axes[2, 1].set(xlabel="Time, s", ylabel="Slip", title="Slip")

    for j in range(n_extra):
        row = 3 + j
        axes[row, 0].plot(t, i_fault[j], "m-", lw=0.7)
        axes[row, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--")
        axes[row, 0].axhline(y=0, color="k", lw=0.5, ls="--")
        axes[row, 0].set(xlabel="Time, s", ylabel="A", title=f"Fault current i_f{j + 1}")

        r_f_jj = float(fault.R_fault_2d[j, j])
        p_fault_kw = r_f_jj * i_fault[j] ** 2 / 1e3
        axes[row, 1].plot(t, p_fault_kw, "r-", lw=0.7)
        axes[row, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--")
        axes[row, 1].axhline(y=0, color="k", lw=0.5, ls="--")
        axes[row, 1].set(xlabel="Time, s", ylabel="kW", title=f"Fault branch power {j + 1}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_fault_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_fault_path}")

                           
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig2.suptitle(f"Stator phase I/V 3x2 ({res.scenario_name})", fontsize=13, fontweight="bold")
    phase_rows = [
        ("A", res.i1A, u1A, "b"),
        ("B", res.i1B, u1B, "r"),
        ("C", res.i1C, u1C, "g"),
    ]
    for row, (name, i_phase, u_phase, color) in enumerate(phase_rows):
        axes2[row, 0].plot(t, i_phase, color=color, lw=0.7, label=f"i1{name}")
        axes2[row, 1].plot(t, u_phase, color=color, lw=0.7, label=f"U1{name}")
        axes2[row, 0].axvline(x=t_fault, color="k", lw=1.0, ls="--")
        axes2[row, 1].axvline(x=t_fault, color="k", lw=1.0, ls="--")
        axes2[row, 0].set_ylabel("A")
        axes2[row, 1].set_ylabel("V")
        axes2[row, 0].set_title(f"Phase {name} current")
        axes2[row, 1].set_title(f"Phase {name} voltage")
        axes2[row, 0].legend(fontsize=8, loc="upper right")
        axes2[row, 1].legend(fontsize=8, loc="upper right")
    axes2[2, 0].set_xlabel("Time, s")
    axes2[2, 1].set_xlabel("Time, s")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(save_iv6_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {save_iv6_path}")


def run_no_fault_cases(
    *,
    params: MachineParameters,
    cfg: SolverConfig,
    out_dir: Path,
    fast: bool,
) -> None:
    """Запускает сравнение DAE и legacy без короткого замыкания."""

    if fast:
        cases = [
            ("motor_start", MotorStartScenario(t_end=1.5)),
            ("motor_step_load", MotorStepLoadScenario(t_end=2.5, t_step=1.25)),
        ]
    else:
        cases = [
            ("motor_start", MotorStartScenario(t_end=2.5)),
            ("motor_step_load", MotorStepLoadScenario(t_end=4.0, t_step=2.0)),
        ]

    no_fault_dir = out_dir / "no_fault"
    no_fault_dir.mkdir(parents=True, exist_ok=True)

    for label, scenario in cases:
        print("\n")
        print(f"NO-FAULT CASE: {label}")
        print("\n")

        res_old = (
            SimulationBuilder(params)
            .model(LinearInductionMachine)
            .solver(ScipySolver("RK45", config=cfg))
            .scenario(scenario)
            .run()
        )
        res_dae = (
            DAESimulationBuilder(params)
            .model(LinearInductionMachine)
            .solver(ScipySolver("RK45", config=cfg))
            .scenario(scenario)
            .run()
        )

        prefix = _slug(label)
        plot_steady_state(
            res_old,
            save_path=str(no_fault_dir / f"{prefix}_legacy_steady.png"),
        )
        plot_steady_state(
            res_dae,
            save_path=str(no_fault_dir / f"{prefix}_dae_steady.png"),
        )

        _print_base_diff(label, res_old, res_dae)


def run_fault_cases(
    *,
    params: MachineParameters,
    cfg: SolverConfig,
    out_dir: Path,
    fast: bool,
) -> None:
    """Запускает сравнение DAE и legacy для режимов короткого замыкания."""

    r_src = 0.02
    l_src = 2e-4
    r_f_ground = 0.001
    r_ground = 0.0
    r_f_phase = 0.001

    if fast:
        t_end = 4.0
        t_step = 2.0
        t_fault = 2.0
    else:
        t_end = 8.0
        t_step = 4.0
        t_fault = 4.0

    cases = [
        (
            "no_load_phase_a_ground",
            MotorNoLoadThevenin(
                t_end=t_end,
                mc_idle=0.0,
                mc_friction=50.0,
                r_series=r_src,
                l_series=l_src,
            ),
            phase_to_ground_fault("A", R_f=r_f_ground, t_fault=t_fault, R_ground=r_ground),
        ),
        (
            "step_load_phase_ab",
            MotorStepLoadThevenin(
                t_end=t_end,
                t_step=t_step,
                mc_load=None,
                mc_idle=0.0,
                mc_friction=50.0,
                r_series=r_src,
                l_series=l_src,
            ),
            phase_to_phase_fault("A", "B", R_f=r_f_phase, t_fault=t_fault),
        ),
    ]

    fault_dir = out_dir / "fault"
    fault_dir.mkdir(parents=True, exist_ok=True)

    for label, scenario, fault in cases:
        print("\n")
        print(f"FAULT CASE: {label}")
        print("\n")

        res_old = (
            FaultSimulationBuilder(params)
            .model(LinearInductionMachine)
            .solver(ScipySolver("Radau", config=cfg))
            .scenario(scenario)
            .fault(fault)
            .run()
        )
        res_dae = (
            DAESimulationBuilder(params)
            .model(LinearInductionMachine)
            .solver(ScipySolver("Radau", config=cfg))
            .scenario(scenario)
            .fault(fault)
            .run()
        )

        prefix = _slug(label)
        plot_fault_results_portable(
            res_old,
            save_fault_path=str(fault_dir / f"{prefix}_legacy_fault.png"),
            save_iv6_path=str(fault_dir / f"{prefix}_legacy_stator_iv_6.png"),
        )
        plot_fault_results_portable(
            res_dae,
            save_fault_path=str(fault_dir / f"{prefix}_dae_fault.png"),
            save_iv6_path=str(fault_dir / f"{prefix}_dae_stator_iv_6.png"),
        )

        _print_fault_diff(label, res_old, res_dae)


def main() -> None:
    """Запускает сценарий расчета из этого файла."""

    args = parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    params = MachineParameters()
    print(params.info())

    cfg = SolverConfig(
        dt_out=args.dt_out,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
    )

    run_no_fault_cases(
        params=params,
        cfg=cfg,
        out_dir=out_dir,
        fast=args.fast,
    )
    run_fault_cases(
        params=params,
        cfg=cfg,
        out_dir=out_dir,
        fast=args.fast,
    )

    print("\nSaved plots in:")
    print(f"  {os.path.abspath(str(out_dir))}")


if __name__ == "__main__":
    main()

