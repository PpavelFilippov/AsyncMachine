"""
    Модуль test.py.
    Тестирование модели с насыщением Lm.

    Часть 1: Валидация — nonlinear с Кн ≡ 1 должен совпасть с linear.
    Часть 2: Сравнительные графики linear vs nonlinear (реальное насыщение).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from core.parameters import MachineParameters
from core.results import SimulationResults
from models.linear import LinearInductionMachine
from models.nonlinear import NonlinearInductionMachine
from models.saturation import SaturationCharacteristic, I_OE_TABLE
from scenarios import MotorNoLoadScenario, MotorStepLoadScenario
from simulation import SimulationBuilder
from solvers import ScipySolver, SolverConfig

matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["figure.dpi"] = 150


# ---------------------------------------------------------------------------
#  Вспомогательные функции
# ---------------------------------------------------------------------------

def current_module(iA: np.ndarray, iB: np.ndarray, iC: np.ndarray) -> np.ndarray:
    """Модуль трёхфазного вектора тока."""
    return np.sqrt((iB - iC) ** 2 / 3.0 + iA ** 2)


def get_mem(res: SimulationResults) -> np.ndarray:
    """Электромагнитный момент по сохранённой модели."""
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


def get_n_rpm(res: SimulationResults) -> np.ndarray:
    """Скорость ротора в об/мин."""
    return res.omega_r * 60.0 / (2.0 * np.pi)


# ---------------------------------------------------------------------------
#  Запуск моделей
# ---------------------------------------------------------------------------

def run_linear(
    params: MachineParameters,
    scenario,
    solver_cfg: SolverConfig,
) -> SimulationResults:
    """Запускает linear-модель."""
    return (
        SimulationBuilder(params)
        .model(LinearInductionMachine)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(scenario)
        .run()
    )


def run_nonlinear(
    params: MachineParameters,
    scenario,
    solver_cfg: SolverConfig,
    sat: SaturationCharacteristic | None = None,
) -> SimulationResults:
    """Запускает nonlinear-модель с насыщением."""
    kwargs = {}
    if sat is not None:
        kwargs["saturation"] = sat
    return (
        SimulationBuilder(params)
        .model(NonlinearInductionMachine, **kwargs)
        .solver(ScipySolver("RK45", config=solver_cfg))
        .scenario(scenario)
        .run()
    )


# ---------------------------------------------------------------------------
#  Часть 1: валидация (Кн ≡ 1  ↔  linear)
# ---------------------------------------------------------------------------

def make_flat_saturation(params: MachineParameters) -> SaturationCharacteristic:
    """
    Создаёт характеристику с Кн ≡ 1 (без насыщения).
    K_sat_nominal = 1.0, таблица — все единицы → Lm = Lm_nominal всегда.
    """
    return SaturationCharacteristic(
        Lm_nominal=params.Lm,
        Im_nominal=params.Im_nominal,
        K_sat_nominal=1.0,
        i_oe_table=I_OE_TABLE.copy(),
        k_sat_table=np.ones_like(I_OE_TABLE),
    )


def compare_results(
    res_lin: SimulationResults,
    res_nonlin: SimulationResults,
    label: str,
) -> bool:
    """
    Сравнивает два набора результатов на общей сетке.
    Печатает макс. абсолютные и относительные ошибки.
    Возвращает True если ошибки < порога.
    """
    # Интерполяция на общую сетку (linear может иметь чуть другие точки)
    t_common = res_lin.t
    i1A_nl = np.interp(t_common, res_nonlin.t, res_nonlin.i1A)
    omega_nl = np.interp(t_common, res_nonlin.t, res_nonlin.omega_r)

    i1A_lin = res_lin.i1A
    omega_lin = res_lin.omega_r

    # Макс. абсолютные отклонения
    di = np.max(np.abs(i1A_lin - i1A_nl))
    dw = np.max(np.abs(omega_lin - omega_nl))

    # Относительные (нормировка по диапазону сигнала)
    i_range = np.max(np.abs(i1A_lin)) + 1e-12
    w_range = np.max(np.abs(omega_lin)) + 1e-12
    di_rel = di / i_range
    dw_rel = dw / w_range

    print(f"\n  [{label}]")
    print(f"    max|Δi1A|  = {di:.6e}  (отн. {di_rel:.6e})")
    print(f"    max|Δω_r|  = {dw:.6e}  (отн. {dw_rel:.6e})")

    tol = 1e-3
    passed = di_rel < tol and dw_rel < tol
    status = "PASSED" if passed else "FAILED"
    print(f"    Результат: {status} (порог {tol})")
    return passed


def run_validation(params, solver_cfg, output_dir):
    """Часть 1: Кн ≡ 1 vs linear."""
    print("\n" + "=" * 70)
    print("  ЧАСТЬ 1: ВАЛИДАЦИЯ (Кн ≡ 1  vs  Linear)")
    print("=" * 70)

    sat_flat = make_flat_saturation(params)

    scenarios = [
        ("No-Load", MotorNoLoadScenario(t_end=2.0)),
        ("Step-Load", MotorStepLoadScenario(t_end=3.0, t_step=1.5)),
    ]

    all_passed = True
    for name, scenario in scenarios:
        print(f"\n  --- Сценарий: {name} ---")
        res_lin = run_linear(params, scenario, solver_cfg)
        res_nl = run_nonlinear(params, scenario, solver_cfg, sat_flat)
        ok = compare_results(res_lin, res_nl, name)
        if not ok:
            all_passed = False

    print("\n" + "-" * 70)
    if all_passed:
        print("  ЧАСТЬ 1: ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    else:
        print("  ЧАСТЬ 1: ЕСТЬ НЕПРОЙДЕННЫЕ ТЕСТЫ")
    print("-" * 70)
    return all_passed


# ---------------------------------------------------------------------------
#  Часть 2: сравнительные графики linear vs nonlinear (реальное насыщение)
# ---------------------------------------------------------------------------

def plot_comparison(
    res_lin: SimulationResults,
    res_nl: SimulationResults,
    title: str,
    save_path: str,
) -> None:
    """Строит 2×2 графика: linear (синий) vs nonlinear (красный)."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # --- i1A ---
    axes[0, 0].plot(res_lin.t, res_lin.i1A, "b-", lw=0.5, label="Linear")
    axes[0, 0].plot(res_nl.t, res_nl.i1A, "r-", lw=0.5, alpha=0.7, label="Nonlinear (насыщ.)")
    axes[0, 0].set(xlabel="Время, с", ylabel="Ток, А", title="Ток фазы A статора i₁A")
    axes[0, 0].legend(fontsize=8)

    # --- |I1| ---
    i1_lin = current_module(res_lin.i1A, res_lin.i1B, res_lin.i1C)
    i1_nl = current_module(res_nl.i1A, res_nl.i1B, res_nl.i1C)
    axes[0, 1].plot(res_lin.t, i1_lin, "b-", lw=0.6, label="Linear")
    axes[0, 1].plot(res_nl.t, i1_nl, "r-", lw=0.6, alpha=0.7, label="Nonlinear (насыщ.)")
    axes[0, 1].set(xlabel="Время, с", ylabel="|I₁|, А", title="Модуль тока статора")
    axes[0, 1].legend(fontsize=8)

    # --- Mэм ---
    mem_lin = get_mem(res_lin)
    mem_nl = get_mem(res_nl)
    axes[1, 0].plot(res_lin.t, mem_lin, "b-", lw=0.5, label="Linear")
    axes[1, 0].plot(res_nl.t, mem_nl, "r-", lw=0.5, alpha=0.7, label="Nonlinear (насыщ.)")
    axes[1, 0].axhline(y=0, color="k", lw=0.5, ls="--")
    axes[1, 0].set(xlabel="Время, с", ylabel="Момент, Нм", title="Электромагнитный момент Mэм")
    axes[1, 0].legend(fontsize=8)

    # --- n(t) ---
    n_lin = get_n_rpm(res_lin)
    n_nl = get_n_rpm(res_nl)
    axes[1, 1].plot(res_lin.t, n_lin, "b-", lw=0.8, label="Linear")
    axes[1, 1].plot(res_nl.t, n_nl, "r-", lw=0.8, alpha=0.7, label="Nonlinear (насыщ.)")
    n_sync = res_lin.params.omega_sync * 60.0 / (2.0 * np.pi)
    axes[1, 1].axhline(y=n_sync, color="gray", lw=0.8, ls="--", label=f"n₀ = {n_sync:.0f}")
    axes[1, 1].set(xlabel="Время, с", ylabel="Скорость, об/мин", title="Частота вращения ротора")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Сохранено: {save_path}")
    plt.close(fig)


def run_comparison(params, solver_cfg, output_dir):
    """Часть 2: linear vs nonlinear (реальное насыщение)."""
    print("\n" + "=" * 70)
    print("  ЧАСТЬ 2: СРАВНЕНИЕ LINEAR vs NONLINEAR (реальное насыщение)")
    print("=" * 70)

    sat = SaturationCharacteristic.from_params(params)
    print(f"\n  Lm_nominal   = {sat.Lm_nominal * 1e3:.3f} мГн")
    print(f"  Lm_unsaturated = {sat.Lm_unsaturated * 1e3:.3f} мГн")
    print(f"  K_sat_nominal  = {params.K_sat_nominal}")
    print(f"  Im_nominal     = {params.Im_nominal} А")

    scenarios = [
        (
            "No-Load",
            MotorNoLoadScenario(t_end=2.0),
            os.path.join(output_dir, "test_noload_comparison.png"),
        ),
        (
            "Step-Load",
            MotorStepLoadScenario(t_end=3.0, t_step=1.5),
            os.path.join(output_dir, "test_step_comparison.png"),
        ),
    ]

    for name, scenario, save_path in scenarios:
        print(f"\n  --- Сценарий: {name} ---")
        res_lin = run_linear(params, scenario, solver_cfg)
        res_nl = run_nonlinear(params, scenario, solver_cfg, sat)

        # Печать сводки установившегося режима
        ss = res_lin.steady_state_slice()
        n_lin = get_n_rpm(res_lin)
        n_nl = get_n_rpm(res_nl)
        i1_lin = current_module(res_lin.i1A, res_lin.i1B, res_lin.i1C)
        i1_nl = current_module(res_nl.i1A, res_nl.i1B, res_nl.i1C)
        mem_lin = get_mem(res_lin)
        mem_nl = get_mem(res_nl)

        print(f"\n    {'Параметр':<30} {'Linear':>12} {'Nonlinear':>12}")
        print(f"    {'-' * 54}")
        print(f"    {'n (уст.), об/мин':<30} {np.mean(n_lin[ss]):>12.1f} {np.mean(n_nl[ss]):>12.1f}")
        print(f"    {'|I1| (уст., P50), А':<30} {np.median(i1_lin[ss]):>12.1f} {np.median(i1_nl[ss]):>12.1f}")
        print(f"    {'Mэм (уст.), Нм':<30} {np.mean(mem_lin[ss]):>12.1f} {np.mean(mem_nl[ss]):>12.1f}")

        plot_comparison(
            res_lin, res_nl,
            title=f"Linear vs Nonlinear (насыщение Lm)\n{scenario.name()}",
            save_path=save_path,
        )


# ---------------------------------------------------------------------------
#  Часть 3: осциллограмма i1A нелинейной модели
# ---------------------------------------------------------------------------

def plot_nonlinear_waveform(
    res_nl: SimulationResults,
    title: str,
    save_path: str,
    t_window: float = 0.1,
) -> None:
    """Строит осциллограмму i1A нелинейной модели (последние t_window секунд)."""

    t = res_nl.t
    t_start = max(0.0, t[-1] - t_window)
    mask = t >= t_start
    tt = (t[mask] - t[mask][0]) * 1000.0  # мс

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax.plot(tt, res_nl.i1A[mask], "b-", lw=0.8)
    ax.set(
        xlabel="Время, мс",
        ylabel="Ток, А",
        title=f"Ток фазы A статора i₁A (последние {t_window * 1000:.0f} мс)",
    )
    ax.axhline(y=0, color="k", lw=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Сохранено: {save_path}")
    plt.close(fig)


def run_waveforms(params, solver_cfg, output_dir):
    """Часть 3: осциллограммы i1A нелинейной модели."""
    print("\n" + "=" * 70)
    print("  ЧАСТЬ 3: ОСЦИЛЛОГРАММА i1A (Nonlinear)")
    print("=" * 70)

    scenarios = [
        (
            "No-Load",
            MotorNoLoadScenario(t_end=2.0),
            os.path.join(output_dir, "test_noload_waveform.png"),
        ),
        (
            "Step-Load",
            MotorStepLoadScenario(t_end=3.0, t_step=1.5),
            os.path.join(output_dir, "test_step_waveform.png"),
        ),
    ]

    for name, scenario, save_path in scenarios:
        print(f"\n  --- Сценарий: {name} ---")
        res_nl = run_nonlinear(params, scenario, solver_cfg)
        plot_nonlinear_waveform(
            res_nl,
            title=f"Nonlinear (насыщение Lm) — {scenario.name()}",
            save_path=save_path,
        )


# ---------------------------------------------------------------------------
#  Часть 4: гармонический анализ тока статора (установившийся режим)
# ---------------------------------------------------------------------------

def harmonic_analysis(
    signal: np.ndarray,
    dt: float,
    f_fund: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Гармонический анализ сигнала.

    Берёт целое число периодов основной частоты для чистого БПФ.

    Возвращает:
        freqs   — массив частот (Гц)
        amps    — амплитуды (А, пиковые)
        thd     — THD (%), отношение суммы высших гармоник к основной
    """
    T_period = 1.0 / f_fund
    samples_per_period = int(round(T_period / dt))
    n_periods = len(signal) // samples_per_period
    if n_periods < 1:
        raise ValueError("Недостаточно данных для одного полного периода.")

    # Обрезаем до целого числа периодов (с конца сигнала)
    N = n_periods * samples_per_period
    seg = signal[-N:]

    # БПФ
    spectrum = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(N, d=dt)
    amps = 2.0 * np.abs(spectrum) / N
    amps[0] *= 0.5  # DC-компонента без удвоения

    # THD: корень из суммы квадратов высших гармоник / амплитуда основной
    idx_fund = int(round(f_fund * N * dt))
    a_fund = amps[idx_fund]
    # Берём гармоники: 2*f_fund, 3*f_fund, ... до Найквиста
    harmonics_sq = 0.0
    for h in range(2, 50):
        idx_h = int(round(h * f_fund * N * dt))
        if idx_h >= len(amps):
            break
        harmonics_sq += amps[idx_h] ** 2
    thd = np.sqrt(harmonics_sq) / a_fund * 100.0 if a_fund > 1e-12 else 0.0

    return freqs, amps, thd


def plot_harmonics(
    res_lin: SimulationResults,
    res_nl: SimulationResults,
    title: str,
    save_path: str,
    n_harmonics: int = 20,
) -> None:
    """
    Строит:
      верх — осциллограммы i1A (последние 100 мс) linear vs nonlinear
      низ  — гармонический спектр (столбцы) до n-й гармоники
    """
    dt = res_lin.t[1] - res_lin.t[0]
    f_fund = res_lin.params.fn

    # Установившийся участок
    ss_lin = res_lin.steady_state_slice()
    ss_nl = res_nl.steady_state_slice()

    freqs_l, amps_l, thd_l = harmonic_analysis(res_lin.i1A[ss_lin], dt, f_fund)
    freqs_n, amps_n, thd_n = harmonic_analysis(res_nl.i1A[ss_nl], dt, f_fund)

    # Извлекаем амплитуды гармоник 1..n_harmonics
    harm_nums = np.arange(1, n_harmonics + 1)
    a_lin = np.zeros(n_harmonics)
    a_nl = np.zeros(n_harmonics)

    N_lin = len(res_lin.i1A[ss_lin])
    samples_per_period_lin = int(round(1.0 / (f_fund * dt)))
    N_fft_lin = (N_lin // samples_per_period_lin) * samples_per_period_lin

    N_nl = len(res_nl.i1A[ss_nl])
    N_fft_nl = (N_nl // samples_per_period_lin) * samples_per_period_lin

    for i, h in enumerate(harm_nums):
        idx_l = int(round(h * f_fund * N_fft_lin * dt))
        idx_n = int(round(h * f_fund * N_fft_nl * dt))
        if idx_l < len(amps_l):
            a_lin[i] = amps_l[idx_l]
        if idx_n < len(amps_n):
            a_nl[i] = amps_n[idx_n]

    # --- Графики ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Верх: осциллограмма (последние 100 мс)
    t_window = 0.1
    t_start_l = max(0, res_lin.t[-1] - t_window)
    mask_l = res_lin.t >= t_start_l
    t_start_n = max(0, res_nl.t[-1] - t_window)
    mask_n = res_nl.t >= t_start_n

    tt_l = (res_lin.t[mask_l] - res_lin.t[mask_l][0]) * 1000
    tt_n = (res_nl.t[mask_n] - res_nl.t[mask_n][0]) * 1000

    axes[0].plot(tt_l, res_lin.i1A[mask_l], "b-", lw=0.8, label="Linear")
    axes[0].plot(tt_n, res_nl.i1A[mask_n], "r-", lw=0.8, alpha=0.7, label="Nonlinear (насыщ.)")
    axes[0].set(
        xlabel="Время, мс",
        ylabel="Ток, А",
        title="Ток фазы A статора i₁A (последние 100 мс, уст. режим)",
    )
    axes[0].axhline(y=0, color="k", lw=0.3)
    axes[0].legend(fontsize=9)

    # Низ: столбчатый спектр гармоник
    bar_w = 0.35
    x = harm_nums
    axes[1].bar(x - bar_w / 2, a_lin, bar_w, color="steelblue", label=f"Linear (THD={thd_l:.2f}%)")
    axes[1].bar(x + bar_w / 2, a_nl, bar_w, color="indianred", label=f"Nonlinear (THD={thd_n:.2f}%)")
    axes[1].set(
        xlabel="Номер гармоники",
        ylabel="Амплитуда, А",
        title="Гармонический спектр тока i₁A (установившийся режим)",
    )
    axes[1].set_xticks(harm_nums)
    axes[1].set_xticklabels([str(h) for h in harm_nums], fontsize=8)
    axes[1].legend(fontsize=9)

    # Аннотация с амплитудами основных гармоник
    print(f"\n    {'Гармоника':<12} {'Linear, А':>12} {'Nonlinear, А':>12}")
    print(f"    {'-' * 36}")
    for i, h in enumerate(harm_nums):
        if a_lin[i] > 0.1 or a_nl[i] > 0.1:
            print(f"    {h:<12d} {a_lin[i]:>12.2f} {a_nl[i]:>12.2f}")
    print(f"    {'THD, %':<12} {thd_l:>12.2f} {thd_n:>12.2f}")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Сохранено: {save_path}")
    plt.close(fig)


def run_harmonics(params, solver_cfg, output_dir):
    """Часть 4: гармонический анализ тока."""
    print("\n" + "=" * 70)
    print("  ЧАСТЬ 4: ГАРМОНИЧЕСКИЙ АНАЛИЗ ТОКА i1A (уст. режим)")
    print("=" * 70)

    sat = SaturationCharacteristic.from_params(params)

    scenarios = [
        (
            "No-Load",
            MotorNoLoadScenario(t_end=2.0),
            os.path.join(output_dir, "test_noload_harmonics.png"),
        ),
        (
            "Step-Load",
            MotorStepLoadScenario(t_end=3.0, t_step=1.5),
            os.path.join(output_dir, "test_step_harmonics.png"),
        ),
    ]

    for name, scenario, save_path in scenarios:
        print(f"\n  --- Сценарий: {name} ---")
        res_lin = run_linear(params, scenario, solver_cfg)
        res_nl = run_nonlinear(params, scenario, solver_cfg, sat)
        plot_harmonics(
            res_lin, res_nl,
            title=f"Гармонический анализ i₁A — {scenario.name()}\nLinear vs Nonlinear (насыщение Lm)",
            save_path=save_path,
        )


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    params = MachineParameters()
    print(params.info())

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    solver_cfg = SolverConfig(dt_out=2e-4)

    # Часть 1
    run_validation(params, solver_cfg, output_dir)

    # Часть 2
    run_comparison(params, solver_cfg, output_dir)

    # Часть 3
    run_waveforms(params, solver_cfg, output_dir)

    # Часть 4
    run_harmonics(params, solver_cfg, output_dir)

    print(f"\n  Графики сохранены в: {output_dir}/")
    for f_name in sorted(os.listdir(output_dir)):
        if f_name.startswith("test_"):
            print(f"    - {f_name}")


if __name__ == "__main__":
    main()
