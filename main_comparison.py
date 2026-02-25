"""
    Модуль main_comparison.py.
    Состав:
    Классы: нет.
    Функции: main.

    Финальное сравнение DAE-расчета линейной и нелинейной моделей:
    - нелинейная модель: температурные законы для R1/R2, Lm оставлена постоянной;
    - формируются два типа графиков:
      1) только для нелинейного расчета;
      2) сравнение linear vs nonlinear.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

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
from scenarios.motor_step_load import MotorStepLoadScenario
from solvers import ScipySolver, SolverConfig


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="DAE comparison for linear and nonlinear motor models."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "comparison")
    parser.add_argument("--method", type=str, default="RK45")
    parser.add_argument("--dt-out", type=float, default=2e-4)
    parser.add_argument("--max-step", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--t-end", type=float, default=4.0)
    parser.add_argument("--t-step", type=float, default=2.0)
    return parser.parse_args()


def build_time_grid(t_span: tuple[float, float], dt_out: float) -> np.ndarray:
    """Строит расчетную временную сетку как в ScipySolver."""
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


def build_temperature_laws(
    *,
    t_grid: np.ndarray,
    temperature_points_r1: np.ndarray,
    temperature_points_r2: np.ndarray,
    alpha: float,
) -> tuple[ElectricalParameterLaws, PiecewiseConstantTemperatureProfile, PiecewiseConstantTemperatureProfile]:
    """
    Создает законы параметров для nonlinear-модели:
    - два отдельных профиля температуры для R1 и R2 (объекты и массивы разные);
    - Lm остается постоянной (заглушка).
    """
    temp_profile_r1 = PiecewiseConstantTemperatureProfile(
        t=t_grid,
        temperature_values=temperature_points_r1,
    )
    temp_profile_r2 = PiecewiseConstantTemperatureProfile(
        t=t_grid,
        temperature_values=temperature_points_r2,
    )

    def l_profile(_: ElectricalLawContext) -> float:
        return 1.0

    def rs_law(inp: ElectricalLawInputs) -> float:
        t_deg = temp_profile_r1(inp.context)
        return inp.rs0 * (1.0 + alpha * (t_deg - 20.0))

    def rr_law(inp: ElectricalLawInputs) -> float:
        t_deg = temp_profile_r2(inp.context)
        return inp.rr0 * (1.0 + alpha * (t_deg - 20.0))

    def lm_law(inp: ElectricalLawInputs) -> float:
        _ = inp.l_profile, inp.temperature
        return inp.lm0

    laws = ElectricalParameterLaws(
        temperature=temp_profile_r1,
        l_profile=l_profile,
        rs=rs_law,
        rr=rr_law,
        lm=lm_law,
    )
    return laws, temp_profile_r1, temp_profile_r2


def _module_abc(iA: np.ndarray, iB: np.ndarray, iC: np.ndarray) -> np.ndarray:
    return np.sqrt((iB - iC) ** 2 / 3.0 + iA ** 2)


def _magnetizing_currents(res) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return res.i1A + res.i2a, res.i1B + res.i2b, res.i1C + res.i2c


def _torque_series(res) -> np.ndarray:
    machine = res.extra["machine"]
    return np.fromiter(
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


def _profile_to_series(profile, t_values: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (
            profile(
                ElectricalLawContext(
                    t=float(tk),
                    i1A=0.0, i1B=0.0, i1C=0.0,
                    i2a=0.0, i2b=0.0, i2c=0.0,
                    omega_r=0.0,
                )
            )
            for tk in t_values
        ),
        dtype=float,
        count=t_values.size,
    )


def plot_nonlinear_only(
    *,
    res_nonlin,
    params: MachineParameters,
    temp_profile_r1: PiecewiseConstantTemperatureProfile,
    temp_profile_r2: PiecewiseConstantTemperatureProfile,
    alpha: float,
    save_path: Path,
) -> None:
    """Строит графики только для нелинейного моделирования."""
    t = res_nonlin.t
    i1_mod = _module_abc(res_nonlin.i1A, res_nonlin.i1B, res_nonlin.i1C)
    i2_mod = _module_abc(res_nonlin.i2a, res_nonlin.i2b, res_nonlin.i2c)
    imA, imB, imC = _magnetizing_currents(res_nonlin)
    im_mod = _module_abc(imA, imB, imC)
    n_rpm = res_nonlin.omega_r * 60.0 / (2.0 * np.pi)
    mem = _torque_series(res_nonlin)

    temp_r1 = _profile_to_series(temp_profile_r1, t)
    temp_r2 = _profile_to_series(temp_profile_r2, t)
    r1_t = params.R1 * (1.0 + alpha * (temp_r1 - 20.0))
    r2_t = params.R2 * (1.0 + alpha * (temp_r2 - 20.0))

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Нелинейная модель (DAE): результаты", fontsize=13, fontweight="bold")

    axes[0, 0].plot(t, res_nonlin.i1A, "b-", lw=0.6, label="i1A")
    axes[0, 0].plot(t, res_nonlin.i1B, "r-", lw=0.6, label="i1B")
    axes[0, 0].plot(t, res_nonlin.i1C, "g-", lw=0.6, label="i1C")
    axes[0, 0].set(xlabel="Time, s", ylabel="Current, A", title="Stator currents")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t, i1_mod, "b-", lw=0.7, label="|I1|")
    axes[0, 1].plot(t, i2_mod, "r-", lw=0.7, label="|I2|")
    axes[0, 1].plot(t, im_mod, "g--", lw=0.7, label="|Im|")
    axes[0, 1].set(xlabel="Time, s", ylabel="Current, A", title="Current modules")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(t, n_rpm, "b-", lw=0.8)
    axes[1, 0].axhline(params.omega_sync * 60.0 / (2.0 * np.pi), color="r", ls="--", lw=0.8)
    axes[1, 0].set(xlabel="Time, s", ylabel="Speed, rpm", title="Rotor speed")

    axes[1, 1].plot(t, mem, "b-", lw=0.7, label="Mem")
    axes[1, 1].axhline(0.0, color="k", ls="--", lw=0.5)
    axes[1, 1].set(xlabel="Time, s", ylabel="Torque, Nm", title="Electromagnetic torque")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].plot(t, temp_r1, "b-", lw=0.7, label="T_R1")
    axes[2, 0].plot(t, temp_r2, "r--", lw=0.7, label="T_R2")
    axes[2, 0].set(xlabel="Time, s", ylabel="Temperature, C", title="Temperature profiles")
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].plot(t, r1_t, "b-", lw=0.7, label="R1(t)")
    axes[2, 1].plot(t, r2_t, "r--", lw=0.7, label="R2(t)")
    axes[2, 1].set(xlabel="Time, s", ylabel="Resistance, Ohm", title="R1/R2 from temperature")
    axes[2, 1].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_linear_vs_nonlinear(
    *,
    res_lin,
    res_nonlin,
    params: MachineParameters,
    save_path: Path,
) -> None:
    """Строит графики сравнения linear vs nonlinear."""
    t = res_lin.t
    if t.shape != res_nonlin.t.shape or not np.allclose(t, res_nonlin.t, rtol=0.0, atol=0.0):
        raise ValueError("Time grids of linear and nonlinear runs must match.")

    i1m_lin = _module_abc(res_lin.i1A, res_lin.i1B, res_lin.i1C)
    i1m_non = _module_abc(res_nonlin.i1A, res_nonlin.i1B, res_nonlin.i1C)
    imA_lin, imB_lin, imC_lin = _magnetizing_currents(res_lin)
    imA_non, imB_non, imC_non = _magnetizing_currents(res_nonlin)
    imm_lin = _module_abc(imA_lin, imB_lin, imC_lin)
    imm_non = _module_abc(imA_non, imB_non, imC_non)
    n_lin = res_lin.omega_r * 60.0 / (2.0 * np.pi)
    n_non = res_nonlin.omega_r * 60.0 / (2.0 * np.pi)
    mem_lin = _torque_series(res_lin)
    mem_non = _torque_series(res_nonlin)

    omega_sync = params.omega_sync
    slip_lin = (omega_sync - res_lin.omega_r) / omega_sync if abs(omega_sync) > 1e-12 else np.zeros_like(t)
    slip_non = (omega_sync - res_nonlin.omega_r) / omega_sync if abs(omega_sync) > 1e-12 else np.zeros_like(t)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Сравнение DAE: linear vs nonlinear", fontsize=13, fontweight="bold")

    axes[0, 0].plot(t, res_lin.i1A, "b-", lw=0.7, label="i1A linear")
    axes[0, 0].plot(t, res_nonlin.i1A, "r--", lw=0.7, label="i1A nonlinear")
    axes[0, 0].set(xlabel="Time, s", ylabel="Current, A", title="Stator phase A current")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t, i1m_lin, "b-", lw=0.7, label="|I1| linear")
    axes[0, 1].plot(t, i1m_non, "r--", lw=0.7, label="|I1| nonlinear")
    axes[0, 1].set(xlabel="Time, s", ylabel="Current, A", title="Stator current module")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(t, n_lin, "b-", lw=0.8, label="n linear")
    axes[1, 0].plot(t, n_non, "r--", lw=0.8, label="n nonlinear")
    axes[1, 0].set(xlabel="Time, s", ylabel="Speed, rpm", title="Rotor speed")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(t, mem_lin, "b-", lw=0.7, label="Mem linear")
    axes[1, 1].plot(t, mem_non, "r--", lw=0.7, label="Mem nonlinear")
    axes[1, 1].set(xlabel="Time, s", ylabel="Torque, Nm", title="Electromagnetic torque")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].plot(t, imm_lin, "b-", lw=0.7, label="|Im| linear")
    axes[2, 0].plot(t, imm_non, "r--", lw=0.7, label="|Im| nonlinear")
    axes[2, 0].set(xlabel="Time, s", ylabel="Current, A", title="Magnetizing current module")
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].plot(t, slip_lin, "b-", lw=0.7, label="slip linear")
    axes[2, 1].plot(t, slip_non, "r--", lw=0.7, label="slip nonlinear")
    axes[2, 1].set(xlabel="Time, s", ylabel="Slip", title="Slip comparison")
    axes[2, 1].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Запускает финальное DAE-сравнение linear и nonlinear моделей."""
    args = parse_args()

    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    params = MachineParameters()
    print(params.info())

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = MotorStepLoadScenario(t_end=args.t_end, t_step=args.t_step)
    cfg = SolverConfig(
        dt_out=args.dt_out,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
    )

    print("\nDAE run: linear model")
    res_linear = (
        DAESimulationBuilder(params)
        .model(LinearInductionMachine)
        .solver(ScipySolver(args.method, config=cfg))
        .scenario(scenario)
        .run()
    )

    print("\nDAE run: nonlinear model")
    temp_points_r1 = np.array([20.0, 502.0, 502.0, 1200.0, 1200.0, 20.0], dtype=float)
    temp_points_r2 = np.array([200.0,  200.0, 200.0, 200.0, 200.0, 200.0], dtype=float)
    alpha = 0.004
    t_grid = build_time_grid(scenario.t_span(), cfg.dt_out)
    laws, temp_profile_r1, temp_profile_r2 = build_temperature_laws(
        t_grid=t_grid,
        temperature_points_r1=temp_points_r1,
        temperature_points_r2=temp_points_r2,
        alpha=alpha,
    )

    print(f"  Temperature points R1: {temp_points_r1.tolist()}")
    print(f"  Temperature points R2: {temp_points_r2.tolist()}")

    res_nonlinear = (
        DAESimulationBuilder(params)
        .model(NonlinearInductionMachine, laws=laws)
        .solver(ScipySolver(args.method, config=cfg))
        .scenario(scenario)
        .run()
    )

    nonlinear_plot = out_dir / "nonlinear_only.png"
    comparison_plot = out_dir / "linear_vs_nonlinear.png"

    plot_nonlinear_only(
        res_nonlin=res_nonlinear,
        params=params,
        temp_profile_r1=temp_profile_r1,
        temp_profile_r2=temp_profile_r2,
        alpha=alpha,
        save_path=nonlinear_plot,
    )
    plot_linear_vs_nonlinear(
        res_lin=res_linear,
        res_nonlin=res_nonlinear,
        params=params,
        save_path=comparison_plot,
    )

    print("\nSaved comparison plots:")
    print(f"  - {os.path.abspath(str(nonlinear_plot))}")
    print(f"  - {os.path.abspath(str(comparison_plot))}")


if __name__ == "__main__":
    main()
