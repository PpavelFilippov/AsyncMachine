"""
    Модуль main_interturn.py.
    Состав:
    Классы: MotorNoLoadThevenin, MotorStepLoadThevenin.
    Функции: plot_interturn_fault_results, run_interturn_cases, main.

    Запуск и визуализация сценариев межвитковых замыканий (МВЗ)
    через DAESimulationBuilder + SegmentedLinearInductionMachine.
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
from faults.interturn import interturn_fault
from scenarios.motor_no_load import MotorNoLoadScenario
from scenarios.motor_step_load import MotorStepLoadScenario
from circuit import DAESimulationBuilder
from solvers import ScipySolver, SolverConfig
from sources.three_phase_thevenin import ThreePhaseSineTheveninSource

matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["figure.dpi"] = 150


# Сценарии с Thevenin-источником
class MotorNoLoadThevenin(MotorNoLoadScenario):
    """Сценарий холостого хода с Thevenin-источником."""

    def __init__(
        self,
        t_end: float = 8.0,
        Mc_idle: float = 0.0,
        Mc_friction: float = 50.0,
        r_series: float = 0.02,
        l_series: float = 2e-4,
    ):
        """Создает объект сценария."""
        super().__init__(t_end=t_end, Mc_idle=Mc_idle, Mc_friction=Mc_friction)
        self._r_series = r_series
        self._l_series = l_series

    def voltage_source(self, params):
        """Формирует источник напряжения."""
        return ThreePhaseSineTheveninSource(
            amplitude=params.Um, frequency=params.fn,
            r_series=self._r_series, l_series=self._l_series,
        )


class MotorStepLoadThevenin(MotorStepLoadScenario):
    """Сценарий наброса нагрузки с Thevenin-источником."""

    def __init__(
        self,
        t_end: float = 8.0,
        t_step: float = 2.0,
        Mc_load: float | None = None,
        Mc_idle: float = 0.0,
        Mc_friction: float = 50.0,
        r_series: float = 0.02,
        l_series: float = 2e-4,
    ):
        """Создает объект сценария."""
        super().__init__(
            t_end=t_end, t_step=t_step,
            Mc_load=Mc_load, Mc_idle=Mc_idle, Mc_friction=Mc_friction,
        )
        self._r_series = r_series
        self._l_series = l_series

    def voltage_source(self, params):
        """Формирует источник напряжения."""
        return ThreePhaseSineTheveninSource(
            amplitude=params.Um, frequency=params.fn,
            r_series=self._r_series, l_series=self._l_series,
        )



# Вспомогательные функции
def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Inter-turn fault simulation with plots"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "interturn",
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


# Построение графиков для межвиткового замыкания
def plot_interturn_fault_results(
    res,
    *,
    save_main_path: str,
    save_segments_path: str,
) -> None:
    """Строит графики результатов межвиткового замыкания.

    Основной график (save_main_path):
      Row 0: Стат. токи (i1A, i1B, i1C) | Скорость (об/мин)
      Row 1: Момент (Mem, Mc)            | Скольжение
      Row 2: Роторные токи               | Ток контура МВЗ
      Row 3: Мощность КЗ (R_f*i_f^2)    | Токи линии и сегмента

    Дополнительный график (save_segments_path):
      3x2: фазные токи (линия vs. ток фазы) для каждой фазы.
    """
    t = res.t
    t_fault = float(res.extra["t_fault"])
    params = res.params
    itf = res.extra["interturn_fault"]

    i_line_faulted = res.extra["i_line_faulted"]
    i_seg2_faulted = res.extra["i_seg2_faulted"]
    i_fault_loop = res.extra["i_fault_loop"]

    machine = res.extra["machine"]
    load = res.extra["load"]

    # Электромагнитный момент (через эквивалентные токи, уже в res)
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

    n_rpm = res.omega_r * 60.0 / (2.0 * np.pi)
    n_sync = params.omega_sync * 60.0 / (2.0 * np.pi)
    omega_sync = params.omega_sync
    slip = (omega_sync - res.omega_r) / omega_sync if abs(omega_sync) > 1e-10 else None

    R_f = itf.R_f
    p_fault_kw = R_f * i_fault_loop ** 2 / 1e3

    # Основной график (4x2)
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(
        f"Inter-turn fault: {itf.name}\n({res.scenario_name})",
        fontsize=13, fontweight="bold",
    )

    # Row 0, col 0: Стат. токи
    axes[0, 0].plot(t, res.i1A, "b-", lw=0.5, label="i1A")
    axes[0, 0].plot(t, res.i1B, "r-", lw=0.5, label="i1B")
    axes[0, 0].plot(t, res.i1C, "g-", lw=0.5, label="i1C")
    axes[0, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--", label=f"t_fault={t_fault:.2f}")
    axes[0, 0].set(xlabel="Time, s", ylabel="Current, A",
                    title="Stator phase currents (equivalent)")
    axes[0, 0].legend(fontsize=8)

    # Row 0, col 1: Скорость
    axes[0, 1].plot(t, n_rpm, "b-", lw=0.8)
    axes[0, 1].axhline(y=n_sync, color="r", lw=0.8, ls="--", label=f"n_sync={n_sync:.0f}")
    axes[0, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[0, 1].set(xlabel="Time, s", ylabel="Speed, rpm", title="Rotor speed")
    axes[0, 1].legend(fontsize=8)

    # Row 1, col 0: Момент
    axes[1, 0].plot(t, mem, "b-", lw=0.5, label="Mem")
    axes[1, 0].plot(t, mc, "r--", lw=0.8, label="Mc")
    axes[1, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[1, 0].axhline(y=0, color="k", lw=0.5, ls="--")
    axes[1, 0].set(xlabel="Time, s", ylabel="Torque, Nm", title="Torque")
    axes[1, 0].legend(fontsize=8)

    # Row 1, col 1: Скольжение
    if slip is not None:
        axes[1, 1].plot(t, slip, "b-", lw=0.6)
    axes[1, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[1, 1].axhline(y=0, color="k", lw=0.5, ls="--")
    axes[1, 1].set(xlabel="Time, s", ylabel="Slip", title="Slip")

    # Row 2, col 0: Роторные токи
    axes[2, 0].plot(t, res.i2a, "b-", lw=0.5, label="i2a")
    axes[2, 0].plot(t, res.i2b, "r-", lw=0.5, label="i2b")
    axes[2, 0].plot(t, res.i2c, "g-", lw=0.5, label="i2c")
    axes[2, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[2, 0].set(xlabel="Time, s", ylabel="Current, A", title="Rotor phase currents")
    axes[2, 0].legend(fontsize=8)

    # Row 2, col 1: Ток контура МВЗ
    axes[2, 1].plot(t, i_fault_loop, "m-", lw=0.7, label=f"i_fault (phase {itf.phase})")
    axes[2, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[2, 1].axhline(y=0, color="k", lw=0.5, ls="--")
    axes[2, 1].set(xlabel="Time, s", ylabel="Current, A",
                    title=f"Fault loop current (i_{{A1}} - i_{{A2}})")
    axes[2, 1].legend(fontsize=8)

    # Row 3, col 0: Мощность КЗ
    axes[3, 0].plot(t, p_fault_kw, "r-", lw=0.7, label=f"R_f*i_f^2 (R_f={R_f:.4g})")
    axes[3, 0].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[3, 0].axhline(y=0, color="k", lw=0.5, ls="--")
    axes[3, 0].set(xlabel="Time, s", ylabel="Power, kW",
                    title="Fault branch power")
    axes[3, 0].legend(fontsize=8)

    # Row 3, col 1: Токи линии и сегмента замкнутой фазы
    axes[3, 1].plot(t, i_line_faulted, "b-", lw=0.7,
                    label=f"i_line (phase {itf.phase})")
    axes[3, 1].plot(t, i_seg2_faulted, "r--", lw=0.7,
                    label=f"i_seg2 (phase {itf.phase})")
    axes[3, 1].axvline(x=t_fault, color="k", lw=1.2, ls="--")
    axes[3, 1].axhline(y=0, color="k", lw=0.5, ls="--")
    axes[3, 1].set(xlabel="Time, s", ylabel="Current, A",
                    title=f"Line vs segment current (phase {itf.phase})")
    axes[3, 1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_main_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_main_path}")

    # Дополнительный график (3x2): фазы I + ток сегмента
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig2.suptitle(
        f"Stator phase currents & fault detail\n({res.scenario_name})",
        fontsize=13, fontweight="bold",
    )

    phase_data = [
        ("A", res.i1A, "b"),
        ("B", res.i1B, "r"),
        ("C", res.i1C, "g"),
    ]

    for row, (ph_name, i_phase, color) in enumerate(phase_data):
        # Колонка 0: фазный ток (эквивалентный)
        axes2[row, 0].plot(t, i_phase, color=color, lw=0.7,
                           label=f"i1{ph_name} (equiv)")
        axes2[row, 0].axvline(x=t_fault, color="k", lw=1.0, ls="--")
        axes2[row, 0].set_ylabel("A")
        axes2[row, 0].set_title(f"Phase {ph_name}: equivalent stator current")
        axes2[row, 0].legend(fontsize=8, loc="upper right")

        # Колонка 1: для замкнутой фазы — i_line и i_seg2;
        #             для здоровых — роторный ток
        if ph_name == itf.phase:
            axes2[row, 1].plot(t, i_line_faulted, "b-", lw=0.7,
                               label=f"i_line_{ph_name} (seg1)")
            axes2[row, 1].plot(t, i_seg2_faulted, "r--", lw=0.7,
                               label=f"i_seg2_{ph_name}")
            axes2[row, 1].plot(t, i_fault_loop, "m:", lw=0.7,
                               label="i_fault_loop")
            axes2[row, 1].set_title(
                f"Phase {ph_name}: line vs segment currents (faulted)"
            )
        else:
            rotor_map = {"A": res.i2a, "B": res.i2b, "C": res.i2c}
            i_rot = rotor_map[ph_name]
            axes2[row, 1].plot(t, i_rot, color=color, lw=0.7,
                               label=f"i2{ph_name.lower()}")
            axes2[row, 1].set_title(f"Phase {ph_name}: rotor current")
        axes2[row, 1].axvline(x=t_fault, color="k", lw=1.0, ls="--")
        axes2[row, 1].set_ylabel("A")
        axes2[row, 1].legend(fontsize=8, loc="upper right")

    axes2[2, 0].set_xlabel("Time, s")
    axes2[2, 1].set_xlabel("Time, s")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(save_segments_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {save_segments_path}")


# Запуск сценариев МВЗ
def run_interturn_cases(
    *,
    params: MachineParameters,
    cfg: SolverConfig,
    out_dir: Path,
    fast: bool,
) -> None:
    """Запускает сценарии межвитковых замыканий."""
    r_src = 0.02
    l_src = 2e-4

    if fast:
        t_end = 4.0
        t_step = 2.0
        t_fault = 2.0
    else:
        t_end = 8.0
        t_step = 4.0
        t_fault = 4.0

    cases = [
        # Case 1: ХХ, фаза A, mu=0.1, R_f=0.01
        (
            "no_load_itf_A_mu01_Rf001",
            MotorNoLoadThevenin(
                t_end=t_end,
                Mc_idle=0.0,
                Mc_friction=50.0,
                r_series=r_src,
                l_series=l_src,
            ),
            interturn_fault(phase="A", mu=0.1, R_f=0.01, t_fault=t_fault),
        ),
        # Case 2: ХХ, фаза A, mu=0.2, R_f=0.001 (более тяжелое КЗ)
        (
            "no_load_itf_A_mu02_Rf0001",
            MotorNoLoadThevenin(
                t_end=t_end,
                Mc_idle=0.0,
                Mc_friction=50.0,
                r_series=r_src,
                l_series=l_src,
            ),
            interturn_fault(phase="A", mu=0.2, R_f=0.001, t_fault=t_fault),
        ),
        # Case 3: Наброс нагрузки, фаза B, mu=0.1, R_f=0.01
        (
            "step_load_itf_B_mu01_Rf001",
            MotorStepLoadThevenin(
                t_end=t_end,
                t_step=t_step,
                Mc_load=None,
                Mc_idle=0.0,
                Mc_friction=50.0,
                r_series=r_src,
                l_series=l_src,
            ),
            interturn_fault(phase="B", mu=0.1, R_f=0.01, t_fault=t_fault),
        ),
        # Case 4: ХХ, фаза C, mu=0.3, R_f=0.005
        (
            "no_load_itf_C_mu03_Rf0005",
            MotorNoLoadThevenin(
                t_end=t_end,
                Mc_idle=0.0,
                Mc_friction=50.0,
                r_series=r_src,
                l_series=l_src,
            ),
            interturn_fault(phase="C", mu=0.3, R_f=0.005, t_fault=t_fault),
        ),
    ]

    itf_dir = out_dir / "interturn_fault"
    itf_dir.mkdir(parents=True, exist_ok=True)

    for label, scenario, itf in cases:
        print("\n")
        print(f"INTERTURN FAULT CASE: {label}")
        print(f"  phase={itf.phase}, mu={itf.mu:.3f}, R_f={itf.R_f:.4g}, "
              f"t_fault={itf.t_fault:.2f}")
        print("\n")

        res = (
            DAESimulationBuilder(params)
            .scenario(scenario)
            .interturn_fault(itf)
            .solver(ScipySolver("Radau", config=cfg))
            .run()
        )

        # Числовые характеристики
        i_fault_loop = res.extra["i_fault_loop"]
        mask_after = res.t > itf.t_fault + 0.05
        if np.any(mask_after):
            i_f_rms = np.sqrt(np.mean(i_fault_loop[mask_after] ** 2))
            i_f_max = np.max(np.abs(i_fault_loop[mask_after]))
            p_f_mean = itf.R_f * np.mean(i_fault_loop[mask_after] ** 2) / 1e3
            print(f"  I_fault_loop: RMS = {i_f_rms:.1f} A, "
                  f"max = {i_f_max:.1f} A")
            print(f"  P_fault (mean after fault) = {p_f_mean:.2f} kW")
        else:
            print("  WARNING: no data after t_fault")

        # Графики
        prefix = _slug(label)
        plot_interturn_fault_results(
            res,
            save_main_path=str(itf_dir / f"{prefix}_main.png"),
            save_segments_path=str(itf_dir / f"{prefix}_segments.png"),
        )


def main() -> None:
    """Запускает сценарии межвитковых замыканий."""
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

    run_interturn_cases(
        params=params,
        cfg=cfg,
        out_dir=out_dir,
        fast=args.fast,
    )

    print("\nSaved plots in:")
    print(f"  {os.path.abspath(str(out_dir))}")


if __name__ == "__main__":
    main()
