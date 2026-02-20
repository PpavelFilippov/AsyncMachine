"""
    Модуль motor_step_torque_sweep.py.
    Состав:
    Классы: нет.
    Функции: parse_args, phase_rms, current_module, steady_slice, evaluate_torque_point, run_for_multipliers,
    find_torque_for_in, save_csv, print_table, save_plot, main.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from core.parameters import MachineParameters
from models.linear import LinearInductionMachine
from scenarios import MotorStepLoadScenario
from simulation import SimulationBuilder
from solvers import ScipySolver, SolverConfig


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="Motor-step torque sweep with RMS/amplitude report."
    )
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        help="Torque multipliers relative to M_nom. Range must bracket In target.",
    )
    parser.add_argument("--t-end", type=float, default=4.0, help="Simulation end time, s.")
    parser.add_argument("--t-step", type=float, default=2.0, help="Load step time, s.")
    parser.add_argument("--dt-out", type=float, default=2e-4, help="Output step, s.")
    parser.add_argument("--max-step", type=float, default=2e-4, help="Solver max step, s.")
    parser.add_argument(
        "--steady-frac",
        type=float,
        default=0.75,
        help="Start of steady-state window as fraction of total time.",
    )
    parser.add_argument(
        "--bisection-iters",
        type=int,
        default=10,
        help="Bisection iterations for finding M_target where RMS == In.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "motor_step_torque_sweep",
        help="Output directory for csv/png.",
    )
    return parser.parse_args()


def phase_rms(values: np.ndarray) -> float:
    """Вычисляет действующее значение сигнала фазы."""
    return float(np.sqrt(np.mean(values ** 2)))


def current_module(iA: np.ndarray, iB: np.ndarray, iC: np.ndarray) -> np.ndarray:
    """Вычисляет модуль трехфазного вектора тока."""
    return np.sqrt((iB - iC) ** 2 / 3.0 + iA ** 2)


def steady_slice(n_points: int, steady_frac: float) -> slice:
    """Возвращает срез сигнала после окончания переходного процесса."""
    idx = int(n_points * steady_frac)
    return slice(idx, None)


def evaluate_torque_point(
    params: MachineParameters,
    t_end: float,
    t_step: float,
    cfg: SolverConfig,
    steady_frac: float,
    mc_load: float,
) -> dict[str, float]:
    """Проводит расчет для одного множителя момента нагрузки."""
    scenario = MotorStepLoadScenario(
        t_end=t_end,
        t_step=t_step,
        Mc_load=mc_load,
        Mc_idle=0.0,
    )
    res = (
        SimulationBuilder(params)
        .solver(ScipySolver("RK45", config=cfg))
        .scenario(scenario)
        .run()
    )

    ss = steady_slice(len(res.t), steady_frac)
    i1a_ss = res.i1A[ss]
    i1_mod_ss = current_module(res.i1A[ss], res.i1B[ss], res.i1C[ss])
    n_ss = res.omega_r[ss] * 60.0 / (2.0 * np.pi)

    machine = res.extra.get("machine")
    if machine is None:
        machine = LinearInductionMachine(params)
    mem_ss = np.fromiter(
        (
            machine.electromagnetic_torque(
                res.i1A[k], res.i1B[k], res.i1C[k],
                res.i2a[k], res.i2b[k], res.i2c[k],
            )
            for k in range(ss.start, len(res.t))
        ),
        dtype=float,
        count=len(res.t) - ss.start,
    )

    i1a_rms = phase_rms(i1a_ss)
    i1a_amp_abs_max = float(np.max(np.abs(i1a_ss)))
    i1a_amp_from_rms = float(math.sqrt(2.0) * i1a_rms)
    i1_mod_median = float(np.median(i1_mod_ss))

    return {
        "Mc_load_Nm": mc_load,
        "I1A_rms_A": i1a_rms,
        "I1A_amp_abs_max_A": i1a_amp_abs_max,
        "I1A_amp_from_rms_A": i1a_amp_from_rms,
        "I1_mod_median_A": i1_mod_median,
        "n_rpm_ss": float(np.mean(n_ss)),
        "Mem_ss_Nm": float(np.mean(mem_ss)),
    }


def run_for_multipliers(
    multipliers: Iterable[float],
    params: MachineParameters,
    t_end: float,
    t_step: float,
    cfg: SolverConfig,
    steady_frac: float,
) -> list[dict[str, float]]:
    """Выполняет расчеты для набора множителей момента."""
    rows: list[dict[str, float]] = []

    for k in multipliers:
        mc = float(k) * params.M_nom
        point = evaluate_torque_point(
            params=params,
            t_end=t_end,
            t_step=t_step,
            cfg=cfg,
            steady_frac=steady_frac,
            mc_load=mc,
        )
        point["k_Mnom"] = float(k)
        point["case"] = "sweep"
        rows.append(point)

    rows.sort(key=lambda r: r["k_Mnom"])
    return rows


def find_torque_for_in(
    params: MachineParameters,
    t_end: float,
    t_step: float,
    cfg: SolverConfig,
    steady_frac: float,
    k_low: float,
    k_high: float,
    iterations: int,
) -> dict[str, float]:
    """Подбирает множитель момента для заданного установившегося тока."""

    target = params.In
    low = k_low * params.M_nom
    high = k_high * params.M_nom

    p_low = evaluate_torque_point(params, t_end, t_step, cfg, steady_frac, low)
    p_high = evaluate_torque_point(params, t_end, t_step, cfg, steady_frac, high)

    if p_low["I1A_rms_A"] > target or p_high["I1A_rms_A"] < target:
        raise RuntimeError(
            "In target is outside multiplier bracket. "
            "Increase --multipliers range (min/max)."
        )

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        p_mid = evaluate_torque_point(params, t_end, t_step, cfg, steady_frac, mid)
        if p_mid["I1A_rms_A"] < target:
            low = mid
        else:
            high = mid

    best = evaluate_torque_point(
        params=params,
        t_end=t_end,
        t_step=t_step,
        cfg=cfg,
        steady_frac=steady_frac,
        mc_load=0.5 * (low + high),
    )
    best["k_Mnom"] = best["Mc_load_Nm"] / params.M_nom
    best["case"] = "target_In"
    return best


def save_csv(rows: list[dict[str, float]], csv_path: Path) -> None:
    """Сохраняет данные в файл."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "k_Mnom",
        "Mc_load_Nm",
        "I1A_rms_A",
        "I1A_amp_abs_max_A",
        "I1A_amp_from_rms_A",
        "I1_mod_median_A",
        "n_rpm_ss",
        "Mem_ss_Nm",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(
    rows: list[dict[str, float]],
    in_nom: float,
    target_row: dict[str, float],
) -> None:
    """Печатает таблицу результатов подбора."""

    headers = [
        "k*M_nom",
        "Mc, Nm",
        "I1A_rms, A",
        "I1A_amp|max|, A",
        "I1A_amp=sqrt(2)*RMS, A",
        "|I1|_median, A",
    ]
    print("\nSummary table (steady-state after step)")
    print(f"Nominal phase RMS In = {in_nom:.3f} A\n")
    print(
        f"{headers[0]:>8} | {headers[1]:>10} | {headers[2]:>11} | "
        f"{headers[3]:>15} | {headers[4]:>22} | {headers[5]:>13}"
    )
    print("-" * 98)
    for r in rows:
        print(
            f"{r['k_Mnom']:8.3f} | {r['Mc_load_Nm']:10.2f} | {r['I1A_rms_A']:11.3f} | "
            f"{r['I1A_amp_abs_max_A']:15.3f} | {r['I1A_amp_from_rms_A']:22.3f} | "
            f"{r['I1_mod_median_A']:13.3f}"
        )

    print("\nComputed torque for target I1A_rms = In")
    print(
        f"  k_target = {target_row['k_Mnom']:.6f}, "
        f"M_target = {target_row['Mc_load_Nm']:.3f} Nm, "
        f"I1A_rms = {target_row['I1A_rms_A']:.3f} A"
    )


def save_plot(
    rows: list[dict[str, float]],
    in_nom: float,
    target_row: dict[str, float],
    png_path: Path,
) -> None:
    """Сохраняет данные в файл."""

    x = np.array([r["k_Mnom"] for r in rows], dtype=float)
    y_rms = np.array([r["I1A_rms_A"] for r in rows], dtype=float)
    y_amp = np.array([r["I1A_amp_abs_max_A"] for r in rows], dtype=float)
    y_amp_from_rms = np.array([r["I1A_amp_from_rms_A"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, y_rms, marker="o", lw=2.0, label="I1A RMS (steady)")
    ax.plot(x, y_amp, marker="s", lw=1.8, label="I1A amplitude max|i| (steady)")
    ax.plot(x, y_amp_from_rms, marker="^", lw=1.4, ls="--", label="sqrt(2)*RMS")

    ax.axhline(in_nom, color="tab:red", lw=1.5, ls=":", label=f"In = {in_nom:.2f} A")
    ax.scatter(
        [target_row["k_Mnom"]],
        [target_row["I1A_rms_A"]],
        color="black",
        marker="x",
        s=80,
        label=(
            f"target: k={target_row['k_Mnom']:.4f}, "
            f"M={target_row['Mc_load_Nm']:.1f} Nm"
        ),
        zorder=5,
    )
    ax.set_xlabel("Torque multiplier k relative to M_nom")
    ax.set_ylabel("Current, A")
    ax.set_title("Motor-step sweep: RMS and amplitude vs shaft torque")
    ax.grid(True, alpha=0.35)
    ax.legend()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Запускает сценарий расчета из этого файла."""
    args = parse_args()


    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    params = MachineParameters()
    cfg = SolverConfig(
        dt_out=args.dt_out,
        max_step=args.max_step,
        rtol=1e-6,
        atol=1e-8,
    )

    rows = run_for_multipliers(
        multipliers=args.multipliers,
        params=params,
        t_end=args.t_end,
        t_step=args.t_step,
        cfg=cfg,
        steady_frac=args.steady_frac,
    )

    k_values = sorted(float(k) for k in args.multipliers)
    target_row = find_torque_for_in(
        params=params,
        t_end=args.t_end,
        t_step=args.t_step,
        cfg=cfg,
        steady_frac=args.steady_frac,
        k_low=k_values[0],
        k_high=k_values[-1],
        iterations=args.bisection_iters,
    )

    out_dir = args.output_dir
    csv_path = out_dir / "motor_step_torque_sweep_summary.csv"
    png_path = out_dir / "motor_step_torque_sweep_rms_amp.png"

    all_rows = [*rows, target_row]
    save_csv(all_rows, csv_path)
    save_plot(rows, params.In, target_row, png_path)
    print_table(rows, params.In, target_row)

    print("\nSaved:")
    print(f"  CSV: {csv_path}")
    print(f"  PNG: {png_path}")


if __name__ == "__main__":
    main()
