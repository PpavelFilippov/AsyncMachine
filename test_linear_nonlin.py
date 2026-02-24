"""
Скрипт test_linear_nonlin.py.

Сравнивает DAE-расчеты для моделей:
- models/linear.py
- models/nonlinear.py

Выводит максимальные расхождения состояний и замеры времени выполнения.
Дополнительно сохраняет отдельные графики тока намагничивания |Im| и imA.
"""
from __future__ import annotations

import argparse
import io
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from circuit import DAESimulationBuilder
from core.parameters import MachineParameters
from models.linear import LinearInductionMachine
from models.nonlinear import NonlinearInductionMachine
from models.parameter_laws import (
    ElectricalLawContext,
    ElectricalLawInputs,
    ElectricalParameterLaws,
    PiecewiseConstantTemperatureProfile,
)
from scenarios.motor_no_load import MotorNoLoadScenario
from scenarios.motor_start import MotorStartScenario
from scenarios.motor_step_load import MotorStepLoadScenario
from solvers import ScipySolver, SolverConfig


@dataclass(frozen=True)
class ScenarioCase:
    name: str
    factory: Callable[[], object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DAE runs for linear.py vs nonlinear.py"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="RK45",
        help="ODE method for solve_ivp (e.g. RK45, Radau, BDF).",
    )
    parser.add_argument("--dt-out", type=float, default=2e-4, help="Output step, s.")
    parser.add_argument("--max-step", type=float, default=2e-4, help="Solver max step, s.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Solver rtol.")
    parser.add_argument("--atol", type=float, default=1e-8, help="Solver atol.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of timed repeats per model/scenario.",
    )
    parser.add_argument(
        "--verbose-run",
        action="store_true",
        help="Do not suppress SimulationBuilder logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "nonlinear_compare",
        help="Directory for generated magnetizing current plots.",
    )
    return parser.parse_args()


def scenario_cases() -> list[ScenarioCase]:
    return [
        ScenarioCase(
            name="motor_start",
            factory=lambda: MotorStartScenario(
                t_end=1.2, t_load_start=0.35, t_load_ramp=0.2
            ),
        ),
        ScenarioCase(
            name="motor_no_load",
            factory=lambda: MotorNoLoadScenario(t_end=1.2, Mc_idle=0.0),
        ),
        ScenarioCase(
            name="motor_step_load",
            factory=lambda: MotorStepLoadScenario(t_end=1.6, t_step=0.8),
        ),
    ]


def build_stub_laws() -> ElectricalParameterLaws:
    """
    Демонстрация архитектуры:
    позже замените return в rs/rr/lm на реальные зависимости.
    """

    def temperature_profile(_: ElectricalLawContext) -> float:
        return 25.0

    def l_profile(ctx: ElectricalLawContext) -> float:
        _ = ctx.t, ctx.i1A, ctx.i2a, ctx.omega_r
        return 1.0

    def rs_law(inp: ElectricalLawInputs) -> float:
        _ = inp.temperature
        return inp.rs0

    def rr_law(inp: ElectricalLawInputs) -> float:
        _ = inp.temperature
        return inp.rr0

    def lm_law(inp: ElectricalLawInputs) -> float:
        _ = inp.l_profile, inp.temperature
        return inp.lm0

    return ElectricalParameterLaws(
        temperature=temperature_profile,
        l_profile=l_profile,
        rs=rs_law,
        rr=rr_law,
        lm=lm_law,
    )


def build_time_grid(
    *,
    t_span: tuple[float, float],
    dt_out: float,
) -> np.ndarray:
    t0, t1 = float(t_span[0]), float(t_span[1])
    if dt_out <= 0.0:
        raise ValueError(f"dt_out must be > 0, got {dt_out}")
    if t1 < t0:
        raise ValueError(f"t_span must satisfy t1 >= t0, got {t_span}")
    if abs(t1 - t0) <= 1e-15:
        return np.array([t0], dtype=float)

    n = int(np.floor((t1 - t0) / dt_out))
    t = t0 + np.arange(n + 1, dtype=float) * dt_out
    if t[-1] < t1:
        t = np.append(t, t1)
    else:
        t[-1] = t1
    return t


def build_piecewise_temperature_laws(
    *,
    t_grid: np.ndarray,
    temperature_values: np.ndarray,
) -> ElectricalParameterLaws:
    """
    Шаблон для кусочно-постоянной температуры:
    [t0, t_end] делится на N равных интервалов, где N = len(temperature_values).
    """
    laws = build_stub_laws()
    laws.temperature = PiecewiseConstantTemperatureProfile(
        t=t_grid,
        temperature_values=temperature_values,
    )
    return laws


def run_dae_case(
    *,
    model_cls,
    params: MachineParameters,
    scenario,
    solver_method: str,
    cfg: SolverConfig,
    quiet: bool,
    model_kwargs: dict | None = None,
):
    kwargs = model_kwargs or {}
    t0 = time.perf_counter()
    if quiet:
        with redirect_stdout(io.StringIO()):
            res = (
                DAESimulationBuilder(params)
                .model(model_cls, **kwargs)
                .solver(ScipySolver(solver_method, config=cfg))
                .scenario(scenario)
                .run()
            )
    else:
        res = (
            DAESimulationBuilder(params)
            .model(model_cls, **kwargs)
            .solver(ScipySolver(solver_method, config=cfg))
            .scenario(scenario)
            .run()
        )
    elapsed = time.perf_counter() - t0
    return res, elapsed


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _vector_module(iA: np.ndarray, iB: np.ndarray, iC: np.ndarray) -> np.ndarray:
    return np.sqrt((iB - iC) ** 2 / 3.0 + iA ** 2)


def _magnetizing_current_module(res) -> np.ndarray:
    imA = res.i1A + res.i2a
    imB = res.i1B + res.i2b
    imC = res.i1C + res.i2c
    return _vector_module(imA, imB, imC)


def _magnetizing_current_phase_a(res) -> np.ndarray:
    return res.i1A + res.i2a


def save_magnetizing_current_plot(
    *,
    case_name: str,
    res_lin,
    res_nonlin,
    output_dir: Path,
) -> Path:
    i_m_lin = _magnetizing_current_module(res_lin)
    i_m_nonlin = _magnetizing_current_module(res_nonlin)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(res_lin.t, i_m_lin, "b-", lw=0.8, label="|Im| linear")
    ax.plot(res_nonlin.t, i_m_nonlin, "r--", lw=0.8, label="|Im| nonlinear")
    ax.set_xlabel("Time, s")
    ax.set_ylabel("|Im|, A")
    ax.set_title(f"Magnetizing current module: {case_name}")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()

    path = output_dir / f"{case_name}_magnetizing_current.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_magnetizing_phase_a_plot(
    *,
    case_name: str,
    res_lin,
    res_nonlin,
    output_dir: Path,
) -> Path:
    imA_lin = _magnetizing_current_phase_a(res_lin)
    imA_nonlin = _magnetizing_current_phase_a(res_nonlin)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(res_lin.t, imA_lin, "b-", lw=0.8, label="imA linear")
    ax.plot(res_nonlin.t, imA_nonlin, "r--", lw=0.8, label="imA nonlinear")
    ax.set_xlabel("Time, s")
    ax.set_ylabel("imA, A")
    ax.set_title(f"Magnetizing current phase A: {case_name}")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()

    path = output_dir / f"{case_name}_magnetizing_current_imA.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def compare_results(res_lin, res_nonlin) -> tuple[dict[str, float], float]:
    if res_lin.t.shape != res_nonlin.t.shape:
        raise ValueError(
            f"Time grids have different lengths: {res_lin.t.shape} vs {res_nonlin.t.shape}"
        )
    if not np.allclose(res_lin.t, res_nonlin.t, rtol=0.0, atol=0.0):
        raise ValueError("Time grids differ, expected identical t_eval for both runs.")

    errs = {
        "i1A": _max_abs(res_lin.i1A, res_nonlin.i1A),
        "i1B": _max_abs(res_lin.i1B, res_nonlin.i1B),
        "i1C": _max_abs(res_lin.i1C, res_nonlin.i1C),
        "i2a": _max_abs(res_lin.i2a, res_nonlin.i2a),
        "i2b": _max_abs(res_lin.i2b, res_nonlin.i2b),
        "i2c": _max_abs(res_lin.i2c, res_nonlin.i2c),
        "omega_r": _max_abs(res_lin.omega_r, res_nonlin.omega_r),
    }
    overall = max(errs.values())
    return errs, overall


def print_case_report(
    *,
    case_name: str,
    t_lin_list: list[float],
    t_nonlin_list: list[float],
    errs: dict[str, float],
    overall_err: float,
) -> None:
    t_lin = float(np.mean(t_lin_list))
    t_nonlin = float(np.mean(t_nonlin_list))
    slowdown = (t_nonlin / t_lin) if t_lin > 0.0 else np.nan

    print(f"\nCASE: {case_name}")
    print(
        "  Time [s]: "
        f"linear={t_lin:.4f}, nonlinear={t_nonlin:.4f}, slowdown={slowdown:.3f}x"
    )
    print(
        "  Max abs diff: "
        f"i1A={errs['i1A']:.3e}, i1B={errs['i1B']:.3e}, i1C={errs['i1C']:.3e}, "
        f"i2a={errs['i2a']:.3e}, i2b={errs['i2b']:.3e}, i2c={errs['i2c']:.3e}, "
        f"omega={errs['omega_r']:.3e}"
    )
    print(f"  Overall max abs diff: {overall_err:.3e}")


def main() -> None:
    args = parse_args()

    if args.repeats < 1:
        raise ValueError(f"--repeats must be >= 1, got {args.repeats}")

    params = MachineParameters()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SolverConfig(
        dt_out=args.dt_out,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
    )

    print("DAE comparison: linear.py vs nonlinear.py")
    print(
        f"Solver={args.method}, dt_out={args.dt_out}, max_step={args.max_step}, "
        f"rtol={args.rtol}, atol={args.atol}, repeats={args.repeats}"
    )

    total_t_lin: list[float] = []
    total_t_nonlin: list[float] = []
    total_overall_err: list[float] = []

    for case in scenario_cases():
        t_lin_list: list[float] = []
        t_nonlin_list: list[float] = []

        res_lin_ref = None
        res_nonlin_ref = None

        for _ in range(args.repeats):
            scenario_lin = case.factory()
            res_lin, t_lin = run_dae_case(
                model_cls=LinearInductionMachine,
                params=params,
                scenario=scenario_lin,
                solver_method=args.method,
                cfg=cfg,
                quiet=not args.verbose_run,
            )
            t_lin_list.append(t_lin)

            scenario_nonlin = case.factory()
            t_grid = build_time_grid(
                t_span=scenario_nonlin.t_span(),
                dt_out=cfg.dt_out,
            )
            temperature_values = np.linspace(25.0, 85.0, num=8, dtype=float)
            res_nonlin, t_nonlin = run_dae_case(
                model_cls=NonlinearInductionMachine,
                params=params,
                scenario=scenario_nonlin,
                solver_method=args.method,
                cfg=cfg,
                quiet=not args.verbose_run,
                model_kwargs={
                    "laws": build_piecewise_temperature_laws(
                        t_grid=t_grid,
                        temperature_values=temperature_values,
                    )
                },
            )
            t_nonlin_list.append(t_nonlin)

            if res_lin_ref is None:
                res_lin_ref = res_lin
            if res_nonlin_ref is None:
                res_nonlin_ref = res_nonlin

        errs, overall_err = compare_results(res_lin_ref, res_nonlin_ref)
        print_case_report(
            case_name=case.name,
            t_lin_list=t_lin_list,
            t_nonlin_list=t_nonlin_list,
            errs=errs,
            overall_err=overall_err,
        )
        path_im = save_magnetizing_current_plot(
            case_name=case.name,
            res_lin=res_lin_ref,
            res_nonlin=res_nonlin_ref,
            output_dir=output_dir,
        )
        path_ima = save_magnetizing_phase_a_plot(
            case_name=case.name,
            res_lin=res_lin_ref,
            res_nonlin=res_nonlin_ref,
            output_dir=output_dir,
        )
        print(f"  Saved magnetizing current plot: {path_im}")
        print(f"  Saved imA plot: {path_ima}")

        total_t_lin.extend(t_lin_list)
        total_t_nonlin.extend(t_nonlin_list)
        total_overall_err.append(overall_err)

    mean_lin = float(np.mean(total_t_lin))
    mean_nonlin = float(np.mean(total_t_nonlin))
    slowdown_total = (mean_nonlin / mean_lin) if mean_lin > 0.0 else np.nan
    max_err_total = float(np.max(total_overall_err))

    print("\nSUMMARY")
    print(
        "  Mean time [s]: "
        f"linear={mean_lin:.4f}, nonlinear={mean_nonlin:.4f}, "
        f"slowdown={slowdown_total:.3f}x"
    )
    print(f"  Max overall abs diff across scenarios: {max_err_total:.3e}")
    print(f"  Magnetizing current plots dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
