"""
Resistance sensitivity testing with DAE nonlinear model.

Scenarios:
1) no_load: MotorNoLoadScenario
2) step_load: MotorStepLoadScenario

Temperature sweeps in each scenario:
1) stator fixed at 20 C, rotor sweep: 20/50/75/100/125/150 C
2) rotor fixed at 20 C, stator sweep: 20/50/75/100/125/150 C

Outputs:
For each scenario, separate PNG files are generated in a dedicated subdirectory:
- <output-dir>/<scenario>/current_amp_i1_rotor_sweep.png
- <output-dir>/<scenario>/current_amp_i2_rotor_sweep.png
- <output-dir>/<scenario>/current_amp_i1_stator_sweep.png
- <output-dir>/<scenario>/current_amp_i2_stator_sweep.png
- <output-dir>/<scenario>/speed_comparison.png
- <output-dir>/<scenario>/flux_comparison.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from circuit import DAESimulationBuilder
from core.parameters import MachineParameters
from models.nonlinear import NonlinearInductionMachine
from models.parameter_laws import (
    ElectricalLawContext,
    ElectricalLawInputs,
    ElectricalParameterLaws,
)
from scenarios.base import Scenario
from scenarios.motor_no_load import MotorNoLoadScenario
from scenarios.motor_step_load import MotorStepLoadScenario
from solvers import ScipySolver, SolverConfig


TEMPERATURE_LEVELS = np.array([20.0, 50.0, 75.0, 100.0, 125.0, 150.0], dtype=float)
ALPHA_CU = 0.004
T_REF = 20.0
STEP_INSET_RANGE = (1.9, 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resistance-vs-temperature testing for nonlinear DAE model."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resistance_testing"),
        help="Directory for generated PNG plots (relative paths are resolved from project root).",
    )
    parser.add_argument("--method", type=str, default="RK45")
    parser.add_argument("--dt-out", type=float, default=2e-4)
    parser.add_argument("--max-step", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--t-end", type=float, default=4.0)
    parser.add_argument("--t-step", type=float, default=2.0)
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=("all", "no_load", "step_load"),
        help="Run no-load, step-load, or both.",
    )
    return parser.parse_args()


def _current_module(i_a: np.ndarray, i_b: np.ndarray, i_c: np.ndarray) -> np.ndarray:
    return np.sqrt((i_b - i_c) ** 2 / 3.0 + i_a ** 2)


def _flux_series(res) -> np.ndarray:
    machine = res.extra["machine"]
    return np.fromiter(
        (
            machine.flux_linkage_phaseA(
                res.i1A[k], res.i1B[k], res.i1C[k],
                res.i2a[k], res.i2b[k], res.i2c[k],
            )
            for k in range(res.N)
        ),
        dtype=float,
        count=res.N,
    )


def _add_time_inset(
    ax: plt.Axes,
    *,
    time_range: tuple[float, float] | None,
) -> None:
    if time_range is None:
        return

    t0, t1 = float(time_range[0]), float(time_range[1])
    if t1 <= t0:
        return

    inset = ax.inset_axes([0.57, 0.50, 0.40, 0.43])
    has_data = False

    for line in ax.get_lines():
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size == 0 or y.size == 0:
            continue
        mask = (x >= t0) & (x <= t1)
        if np.count_nonzero(mask) < 2:
            continue
        has_data = True
        inset.plot(
            x[mask],
            y[mask],
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=0.7,
            alpha=line.get_alpha() if line.get_alpha() is not None else 1.0,
        )

    inset.set_xlim(t0, t1)
    if has_data:
        inset.grid(True, alpha=0.25)
    else:
        inset.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=7, transform=inset.transAxes)
        inset.grid(False)

    inset.set_title(f"{t0:.1f}-{t1:.1f} s", fontsize=7)
    inset.tick_params(axis="both", labelsize=7)


def build_constant_temperature_laws(
    *,
    stator_temp_c: float,
    rotor_temp_c: float,
    alpha: float = ALPHA_CU,
) -> ElectricalParameterLaws:
    def temperature_placeholder(_: ElectricalLawContext) -> float:
        # Not used directly in Rs/Rr laws here, but kept for interface completeness.
        return stator_temp_c

    def l_profile(_: ElectricalLawContext) -> float:
        return 1.0

    def rs_law(inp: ElectricalLawInputs) -> float:
        return inp.rs0 * (1.0 + alpha * (stator_temp_c - T_REF))

    def rr_law(inp: ElectricalLawInputs) -> float:
        return inp.rr0 * (1.0 + alpha * (rotor_temp_c - T_REF))

    def lm_law(inp: ElectricalLawInputs) -> float:
        _ = inp.l_profile, inp.temperature
        return inp.lm0

    return ElectricalParameterLaws(
        temperature=temperature_placeholder,
        l_profile=l_profile,
        rs=rs_law,
        rr=rr_law,
        lm=lm_law,
    )


def run_case(
    *,
    params: MachineParameters,
    solver_cfg: SolverConfig,
    method: str,
    scenario: Scenario,
    stator_temp_c: float,
    rotor_temp_c: float,
):
    laws = build_constant_temperature_laws(
        stator_temp_c=stator_temp_c,
        rotor_temp_c=rotor_temp_c,
        alpha=ALPHA_CU,
    )
    return (
        DAESimulationBuilder(params)
        .model(NonlinearInductionMachine, laws=laws)
        .solver(ScipySolver(method, config=solver_cfg))
        .scenario(scenario)
        .run()
    )


def run_group(
    *,
    params: MachineParameters,
    solver_cfg: SolverConfig,
    method: str,
    scenario: Scenario,
    fixed_stator: float | None = None,
    fixed_rotor: float | None = None,
) -> dict[float, object]:
    if (fixed_stator is None) == (fixed_rotor is None):
        raise ValueError("Set exactly one of fixed_stator/fixed_rotor.")

    results: dict[float, object] = {}
    for temp in TEMPERATURE_LEVELS:
        if fixed_stator is not None:
            res = run_case(
                params=params,
                solver_cfg=solver_cfg,
                method=method,
                scenario=scenario,
                stator_temp_c=fixed_stator,
                rotor_temp_c=float(temp),
            )
        else:
            res = run_case(
                params=params,
                solver_cfg=solver_cfg,
                method=method,
                scenario=scenario,
                stator_temp_c=float(temp),
                rotor_temp_c=fixed_rotor,
            )
        results[float(temp)] = res
    return results


def plot_currents_amplitudes(
    *,
    sweep_results: dict[float, object],
    mode_label: str,
    legend_prefix: str,
    title: str,
    y_label: str,
    current_selector: str,
    inset_range: tuple[float, float] | None,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(f"{title} ({mode_label}, nonlinear DAE)", fontsize=13, fontweight="bold")

    for temp, res in sweep_results.items():
        if current_selector == "i1":
            i_mod = _current_module(res.i1A, res.i1B, res.i1C)
        elif current_selector == "i2":
            i_mod = _current_module(res.i2a, res.i2b, res.i2c)
        else:
            raise ValueError(f"Unsupported current_selector={current_selector!r}")
        ax.plot(res.t, i_mod, lw=0.8, label=f"{legend_prefix} T={temp:.0f}C")

    ax.set_xlabel("Time, s")
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)
    _add_time_inset(ax, time_range=inset_range)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_speed(
    *,
    rotor_sweep: dict[float, object],
    stator_sweep: dict[float, object],
    mode_label: str,
    inset_range: tuple[float, float] | None,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True, sharey=True)
    fig.suptitle(
        f"Rotor speed vs temperature ({mode_label}, nonlinear DAE)",
        fontsize=13,
        fontweight="bold",
    )

    for temp, res in rotor_sweep.items():
        n_rpm = res.omega_r * 60.0 / (2.0 * np.pi)
        axes[0].plot(res.t, n_rpm, lw=0.8, label=f"Rr T={temp:.0f}C")
    axes[0].set_title("Stator T=20C, rotor sweep")
    axes[0].set_xlabel("Time, s")
    axes[0].set_ylabel("Speed, rpm")
    axes[0].legend(fontsize=7)

    for temp, res in stator_sweep.items():
        n_rpm = res.omega_r * 60.0 / (2.0 * np.pi)
        axes[1].plot(res.t, n_rpm, lw=0.8, label=f"Rs T={temp:.0f}C")
    axes[1].set_title("Rotor T=20C, stator sweep")
    axes[1].set_xlabel("Time, s")
    axes[1].legend(fontsize=7)
    _add_time_inset(axes[0], time_range=inset_range)
    _add_time_inset(axes[1], time_range=inset_range)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_flux(
    *,
    rotor_sweep: dict[float, object],
    stator_sweep: dict[float, object],
    mode_label: str,
    inset_range: tuple[float, float] | None,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
    fig.suptitle(
        f"Magnetic flux (phase A linkage) vs temperature ({mode_label})",
        fontsize=13,
        fontweight="bold",
    )

    for temp, res in rotor_sweep.items():
        psi = _flux_series(res)
        axes[0].plot(res.t, psi, lw=0.8, label=f"Rr T={temp:.0f}C")
    axes[0].set_title("Stator T=20C, rotor sweep")
    axes[0].set_xlabel("Time, s")
    axes[0].set_ylabel("Psi_A, Wb")
    axes[0].legend(fontsize=7)

    for temp, res in stator_sweep.items():
        psi = _flux_series(res)
        axes[1].plot(res.t, psi, lw=0.8, label=f"Rs T={temp:.0f}C")
    axes[1].set_title("Rotor T=20C, stator sweep")
    axes[1].set_xlabel("Time, s")
    axes[1].legend(fontsize=7)
    _add_time_inset(axes[0], time_range=inset_range)
    _add_time_inset(axes[1], time_range=inset_range)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_and_plot_scenario(
    *,
    params: MachineParameters,
    solver_cfg: SolverConfig,
    method: str,
    scenario: Scenario,
    mode_slug: str,
    mode_label: str,
    output_dir: Path,
) -> list[Path]:
    scenario_dir = output_dir / mode_slug
    scenario_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running resistance-vs-temperature sweeps ({mode_label})...")
    print("  Group 1: stator fixed at 20C, rotor sweep")
    rotor_sweep = run_group(
        params=params,
        solver_cfg=solver_cfg,
        method=method,
        scenario=scenario,
        fixed_stator=20.0,
        fixed_rotor=None,
    )

    print("  Group 2: rotor fixed at 20C, stator sweep")
    stator_sweep = run_group(
        params=params,
        solver_cfg=solver_cfg,
        method=method,
        scenario=scenario,
        fixed_stator=None,
        fixed_rotor=20.0,
    )

    current_i1_rotor_path = scenario_dir / "current_amp_i1_rotor_sweep.png"
    current_i2_rotor_path = scenario_dir / "current_amp_i2_rotor_sweep.png"
    current_i1_stator_path = scenario_dir / "current_amp_i1_stator_sweep.png"
    current_i2_stator_path = scenario_dir / "current_amp_i2_stator_sweep.png"
    speed_path = scenario_dir / "speed_comparison.png"
    flux_path = scenario_dir / "flux_comparison.png"
    inset_range = STEP_INSET_RANGE if mode_slug == "step_load" else None

    plot_currents_amplitudes(
        sweep_results=rotor_sweep,
        mode_label=mode_label,
        legend_prefix="Rr",
        title="Stator current amplitude, stator T=20C, rotor sweep",
        y_label="|I1|, A",
        current_selector="i1",
        inset_range=inset_range,
        save_path=current_i1_rotor_path,
    )
    plot_currents_amplitudes(
        sweep_results=rotor_sweep,
        mode_label=mode_label,
        legend_prefix="Rr",
        title="Rotor current amplitude, stator T=20C, rotor sweep",
        y_label="|I2|, A",
        current_selector="i2",
        inset_range=inset_range,
        save_path=current_i2_rotor_path,
    )
    plot_currents_amplitudes(
        sweep_results=stator_sweep,
        mode_label=mode_label,
        legend_prefix="Rs",
        title="Stator current amplitude, rotor T=20C, stator sweep",
        y_label="|I1|, A",
        current_selector="i1",
        inset_range=inset_range,
        save_path=current_i1_stator_path,
    )
    plot_currents_amplitudes(
        sweep_results=stator_sweep,
        mode_label=mode_label,
        legend_prefix="Rs",
        title="Rotor current amplitude, rotor T=20C, stator sweep",
        y_label="|I2|, A",
        current_selector="i2",
        inset_range=inset_range,
        save_path=current_i2_stator_path,
    )
    plot_speed(
        rotor_sweep=rotor_sweep,
        stator_sweep=stator_sweep,
        mode_label=mode_label,
        inset_range=inset_range,
        save_path=speed_path,
    )
    plot_flux(
        rotor_sweep=rotor_sweep,
        stator_sweep=stator_sweep,
        mode_label=mode_label,
        inset_range=inset_range,
        save_path=flux_path,
    )
    return [
        current_i1_rotor_path,
        current_i2_rotor_path,
        current_i1_stator_path,
        current_i2_stator_path,
        speed_path,
        flux_path,
    ]


def main() -> None:
    args = parse_args()

    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    params = MachineParameters()
    solver_cfg = SolverConfig(
        dt_out=args.dt_out,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
    )
    saved_files: list[Path] = []

    if args.mode in ("all", "no_load"):
        no_load = MotorNoLoadScenario(t_end=args.t_end)
        saved_files.extend(
            run_and_plot_scenario(
                params=params,
                solver_cfg=solver_cfg,
                method=args.method,
                scenario=no_load,
                mode_slug="no_load",
                mode_label="no-load",
                output_dir=output_dir,
            )
        )

    if args.mode in ("all", "step_load"):
        step_load = MotorStepLoadScenario(t_end=args.t_end, t_step=args.t_step)
        saved_files.extend(
            run_and_plot_scenario(
                params=params,
                solver_cfg=solver_cfg,
                method=args.method,
                scenario=step_load,
                mode_slug="step_load",
                mode_label="step-load",
                output_dir=output_dir,
            )
        )

    print("Saved files:")
    for path in saved_files:
        print(f"  - {path.resolve()}")


if __name__ == "__main__":
    main()
