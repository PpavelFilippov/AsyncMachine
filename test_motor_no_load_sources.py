"""
Compare MotorNoLoadScenario with ideal vs non-ideal (Thevenin) source.

Outputs:
1) Two steady-state plots (ideal/non-ideal)
2) CSV with key steady-state metrics
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from core.parameters import MachineParameters
from plotting import plot_steady_state
from scenarios.motor_no_load import MotorNoLoadScenario
from simulation import SimulationBuilder
from solvers import ScipySolver, SolverConfig
from sources.three_phase_sine import ThreePhaseSineSource
from sources.three_phase_thevenin import ThreePhaseSineTheveninSource


class MotorNoLoadWithSourceScenario(MotorNoLoadScenario):
    """No-load scenario with injectable voltage source factory."""

    def __init__(self, *, source_factory, t_end: float, mc_idle: float):
        super().__init__(t_end=t_end, Mc_idle=mc_idle)
        self._source_factory = source_factory

    def voltage_source(self, params: MachineParameters):
        return self._source_factory(params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test MotorNoLoadScenario with ideal and Thevenin source."
    )
    parser.add_argument("--t-end", type=float, default=4.0, help="Simulation end time, s.")
    parser.add_argument("--mc-idle", type=float, default=0.0, help="No-load shaft torque, Nm.")
    parser.add_argument("--r-series", type=float, default=0.02, help="Thevenin source per-phase series resistance, Ohm.")
    parser.add_argument("--l-series", type=float, default=2e-4, help="Thevenin source per-phase series inductance, H.")
    parser.add_argument("--steady-frac", type=float, default=0.75, help="Steady-state window start as fraction of full interval.")
    parser.add_argument("--dt-out", type=float, default=2e-4, help="Output time step, s.")
    parser.add_argument("--max-step", type=float, default=2e-4, help="Max solver internal step, s.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "tests" / "motor_no_load_sources",
        help="Output directory.",
    )
    return parser.parse_args()


def _steady_metrics(res, steady_frac: float) -> dict[str, float]:
    ss = res.steady_state_slice(steady_frac)
    n_rpm = res.omega_r * 60.0 / (2.0 * np.pi)
    i1_mod = np.sqrt((res.i1B - res.i1C) ** 2 / 3.0 + res.i1A ** 2)
    i1a_rms = float(np.sqrt(np.mean(res.i1A[ss] ** 2)))
    i1_mod_p50 = float(np.median(i1_mod[ss]))

    mem_ss = float("nan")
    machine = res.extra.get("machine")
    if machine is not None:
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
        mem_ss = float(np.mean(mem[ss]))

    return {
        "n_rpm_ss": float(np.mean(n_rpm[ss])),
        "I1A_rms_A": i1a_rms,
        "I1_mod_p50_A": i1_mod_p50,
        "Mem_ss_Nm": mem_ss,
    }


def _run_case(
    *,
    case_name: str,
    params: MachineParameters,
    cfg: SolverConfig,
    scenario: MotorNoLoadScenario,
    output_dir: Path,
    steady_frac: float,
) -> dict[str, float]:
    res = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=cfg))
        .scenario(scenario)
        .run()
    )
    plot_path = output_dir / f"{case_name}_steady.png"
    plot_steady_state(res, save_path=str(plot_path))
    metrics = _steady_metrics(res, steady_frac)
    metrics["case"] = case_name
    metrics["plot_path"] = str(plot_path)
    return metrics


def _save_csv(rows: list[dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "n_rpm_ss",
        "I1A_rms_A",
        "I1_mod_p50_A",
        "Mem_ss_Nm",
        "plot_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    params = MachineParameters()
    cfg = SolverConfig(
        dt_out=args.dt_out,
        max_step=args.max_step,
        rtol=1e-6,
        atol=1e-8,
    )

    def ideal_source_factory(p: MachineParameters):
        return ThreePhaseSineSource(amplitude=p.Um, frequency=p.fn)

    def thevenin_source_factory(p: MachineParameters):
        return ThreePhaseSineTheveninSource(
            amplitude=p.Um,
            frequency=p.fn,
            r_series=args.r_series,
            l_series=args.l_series,
        )

    scenario_ideal = MotorNoLoadWithSourceScenario(
        source_factory=ideal_source_factory,
        t_end=args.t_end,
        mc_idle=args.mc_idle,
    )
    scenario_thevenin = MotorNoLoadWithSourceScenario(
        source_factory=thevenin_source_factory,
        t_end=args.t_end,
        mc_idle=args.mc_idle,
    )

    rows = [
        _run_case(
            case_name="ideal_source",
            params=params,
            cfg=cfg,
            scenario=scenario_ideal,
            output_dir=output_dir,
            steady_frac=args.steady_frac,
        ),
        _run_case(
            case_name="thevenin_source",
            params=params,
            cfg=cfg,
            scenario=scenario_thevenin,
            output_dir=output_dir,
            steady_frac=args.steady_frac,
        ),
    ]

    csv_path = output_dir / "motor_no_load_sources_metrics.csv"
    _save_csv(rows, csv_path)

    print("\nMotor no-load source comparison:")
    for r in rows:
        print(
            f"  {r['case']}: n_ss={r['n_rpm_ss']:.2f} rpm, "
            f"I1A_rms={r['I1A_rms_A']:.2f} A, "
            f"|I1|_p50={r['I1_mod_p50_A']:.2f} A, "
            f"Mem_ss={r['Mem_ss_Nm']:.2f} Nm"
        )
    print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()
