"""
    Модуль plotting/standard_plots.py.
    Состав:
    Классы: нет.
    Функции: plot_motor_start, plot_steady_state, plot_waveforms.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from core.results import SimulationResults

matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['figure.dpi'] = 150


def _current_module(iA: np.ndarray, iB: np.ndarray, iC: np.ndarray) -> np.ndarray:
    """Вычисляет модуль трехфазного вектора."""
    return np.sqrt((iB - iC) ** 2 / 3.0 + iA ** 2)


def _get_mem(res: SimulationResults) -> Optional[np.ndarray]:
    """Возвращает временной ряд электромагнитного момента."""

    machine = res.extra.get("machine")
    if machine is not None:
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
    return None


def _get_mc(res: SimulationResults) -> Optional[np.ndarray]:
    """Возвращает временной ряд момента нагрузки."""

    load = res.extra.get("load")
    if load is not None:
        return np.fromiter(
            (load(res.t[k], res.omega_r[k]) for k in range(res.N)),
            dtype=float,
            count=res.N,
        )
    return None


def _get_n_rpm(res: SimulationResults) -> np.ndarray:
    """Возвращает механическую скорость в оборотах в минуту."""
    return res.omega_r * 60.0 / (2.0 * np.pi)


def _get_slip(res: SimulationResults) -> Optional[np.ndarray]:
    """Вычисляет скольжение по текущей механической скорости."""

    omega_sync = _sync_omega_mech(res)
    if abs(omega_sync) <= 1e-10:
        return None
    return (omega_sync - res.omega_r) / omega_sync


def _sync_omega_mech(res: SimulationResults) -> float:
    """Возвращает синхронную механическую угловую скорость."""

    omega_sync = res.params.omega_sync
    source = res.extra.get("source")
    if source is not None:
        f_src = source.electrical_frequency_hz()
        if f_src is not None:
            omega_sync = 2.0 * np.pi * float(f_src) / float(res.params.p)
    return float(omega_sync)


def _sync_speed_rpm(res: SimulationResults) -> float:
    """Возвращает синхронную скорость в оборотах в минуту."""
    return _sync_omega_mech(res) * 60.0 / (2.0 * np.pi)


def _get_i1_mod(res: SimulationResults) -> np.ndarray:
    """Возвращает модуль тока статора."""
    return _current_module(res.i1A, res.i1B, res.i1C)


def _get_i2_mod(res: SimulationResults) -> np.ndarray:
    """Возвращает модуль тока ротора."""
    return _current_module(res.i2a, res.i2b, res.i2c)


def _get_im_phases(res: SimulationResults) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает токи намагничивания по фазам."""
    return res.i1A + res.i2a, res.i1B + res.i2b, res.i1C + res.i2c


def _get_im_mod(res: SimulationResults) -> np.ndarray:
    """Возвращает модуль тока намагничивания."""
    imA, imB, imC = _get_im_phases(res)
    return _current_module(imA, imB, imC)


def _get_psi1A(res: SimulationResults) -> Optional[np.ndarray]:
    """Вычисляет потокосцепление фазы A по результатам."""

    machine = res.extra.get("machine")
    if machine is not None:
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
    return None


def _get_u_phases(
    res: SimulationResults,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Возвращает фазные напряжения источника на сетке времени."""

    source = res.extra.get("source")
    rhs = res.extra.get("rhs")
    if source is not None:
        R_src = res.extra.get("source_r_matrix")
        L_src = res.extra.get("source_l_matrix")
        if R_src is None:
            R_src = np.asarray(source.series_resistance_matrix(), dtype=float)
        if L_src is None:
            L_src = np.asarray(source.series_inductance_matrix(), dtype=float)
        R_src = np.asarray(R_src, dtype=float)
        L_src = np.asarray(L_src, dtype=float)
        if R_src.shape != (3, 3):
            raise ValueError(
                f"source_r_matrix has shape {R_src.shape}, expected (3, 3)."
            )
        if L_src.shape != (3, 3):
            raise ValueError(
                f"source_l_matrix has shape {L_src.shape}, expected (3, 3)."
            )
        has_source_drop = bool(np.any(np.abs(R_src) > 0.0) or np.any(np.abs(L_src) > 0.0))
        if has_source_drop and rhs is None:
            return None

        U1A = np.zeros(res.N)
        U1B = np.zeros(res.N)
        U1C = np.zeros(res.N)
        for k in range(res.N):
            u = np.asarray(source(res.t[k]), dtype=float)
            if u.shape != (3,):
                raise ValueError(
                    f"{source.__class__.__name__} returned shape {u.shape}, expected (3,)."
                )
            if has_source_drop:
                yk = np.array([
                    res.i1A[k], res.i1B[k], res.i1C[k],
                    res.i2a[k], res.i2b[k], res.i2c[k],
                    res.omega_r[k],
                ])
                di_stator = rhs(res.t[k], yk)[0:3]
                i_stator = np.array([res.i1A[k], res.i1B[k], res.i1C[k]])
                u = u - R_src @ i_stator - L_src @ di_stator
            U1A[k], U1B[k], U1C[k] = u[0], u[1], u[2]
        return U1A, U1B, U1C

    return None


def plot_motor_start(res: SimulationResults, save_path: Optional[str] = None):
    """Строит графики по данным моделирования."""

    t = res.t
    params = res.params
    i1_mod = _get_i1_mod(res)
    mem = _get_mem(res)
    n_rpm = _get_n_rpm(res)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{res.scenario_name}\n(Прямой)',
                 fontsize=13, fontweight='bold')

    axes[0, 0].plot(t, res.i1A, 'b-', lw=0.5)
    axes[0, 0].set(xlabel='Время, с', ylabel='Ток, А',
                    title='Ток фазы A статора i1A')

    axes[0, 1].plot(t, i1_mod, 'b-', lw=0.6)
    axes[0, 1].set(xlabel='Время, с', ylabel='|I1|, А',
                    title='Модуль результирующего тока статора')

    if mem is not None:
        axes[1, 0].plot(t, mem, 'b-', lw=0.5)
        axes[1, 0].axhline(y=0, color='k', lw=0.5, ls='--')
        axes[1, 0].set(xlabel='Время, с', ylabel='Момент, Нм',
                        title='Электромагнитный момент Mэм')

    axes[1, 1].plot(t, n_rpm, 'b-', lw=0.8)
    n_sync = _sync_speed_rpm(res)
    axes[1, 1].axhline(y=n_sync, color='r', lw=0.8, ls='--',
                        label=f'n₀ = {n_sync:.0f}')
    axes[1, 1].axhline(y=params.n_nom, color='g', lw=0.8, ls='--',
                        label=f'nн = {params.n_nom:.0f}')
    axes[1, 1].set(xlabel='Время, с', ylabel='Скорость, об/мин',
                    title='Частота вращения ротора')
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save_and_close(fig, save_path)
    return fig


def plot_steady_state(res: SimulationResults, save_path: Optional[str] = None):
    """Строит графики по данным моделирования."""

    t = res.t
    i1_mod = _get_i1_mod(res)
    i2_mod = _get_i2_mod(res)
    im_mod = _get_im_mod(res)
    mem = _get_mem(res)
    n_rpm = _get_n_rpm(res)
    slip = _get_slip(res)
    u_phases = _get_u_phases(res)
    psi1A = _get_psi1A(res)
    mc = _get_mc(res)

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(f'Результаты моделирования\n({res.scenario_name})',
                 fontsize=13, fontweight='bold')


    axes[0, 0].plot(t, res.i1A, 'b-', lw=0.5, label='i1A')
    axes[0, 0].plot(t, res.i1B, 'r-', lw=0.5, label='i1B')
    axes[0, 0].plot(t, res.i1C, 'g-', lw=0.5, label='i1C')
    axes[0, 0].set(xlabel='Время, с', ylabel='Ток, А',
                    title='Фазные токи статора')
    axes[0, 0].legend(fontsize=8)

                  
    axes[0, 1].plot(t, i1_mod, 'b-', lw=0.6, label='|I1|')
    axes[0, 1].plot(t, i2_mod, 'r-', lw=0.6, label='|I2|')
    axes[0, 1].plot(t, im_mod, 'g--', lw=0.6, label='|Im|')
    axes[0, 1].set(xlabel='Время, с', ylabel='Ток, А',
                    title='Модули результирующих токов')
    axes[0, 1].legend(fontsize=8)

            
    if mem is not None:
        axes[1, 0].plot(t, mem, 'b-', lw=0.6)
        axes[1, 0].axhline(y=0, color='k', lw=0.5, ls='--')
    axes[1, 0].set(xlabel='Время, с', ylabel='Момент, Нм',
                    title='Электромагнитный момент Mэм')

              
    axes[1, 1].plot(t, n_rpm, 'b-', lw=0.8)
    n_sync = _sync_speed_rpm(res)
    axes[1, 1].axhline(y=n_sync, color='r', lw=0.8, ls='--',
                        label=f'n синхр = {n_sync:.0f}')
    axes[1, 1].set(xlabel='Время, с', ylabel='Скорость, об/мин',
                    title='Частота вращения ротора')
    axes[1, 1].legend(fontsize=8)

                
    if slip is not None:
        axes[2, 0].plot(t, slip, 'b-', lw=0.6)
        axes[2, 0].axhline(y=0, color='r', lw=0.8, ls='--', label='s = 0')
    axes[2, 0].set(xlabel='Время, с', ylabel='Скольжение s',
                    title='Скольжение s(t)')
    if slip is not None:
        axes[2, 0].legend(fontsize=8)

                                        
    if u_phases is not None:
        u1A, _, _ = u_phases
        t_start = max(0, t[-1] - 0.06)
        mask = t >= t_start
        axes[2, 1].plot(t[mask] * 1000, u1A[mask], 'b-', lw=0.6, label='U1A')
        axes[2, 1].plot(t[mask] * 1000, res.i1A[mask], 'r-', lw=0.6, label='i1A')
        axes[2, 1].set(xlabel='Время, мс', ylabel='В / А',
                        title='U и I фазы A (последние 60 мс)')
        axes[2, 1].legend(fontsize=8)

                     
    if psi1A is not None:
        axes[3, 0].plot(t, psi1A, 'b-', lw=0.6)
    axes[3, 0].set(xlabel='Время, с', ylabel='psi, Вб',
                    title='Потокосцепление фазы A статора')

                             
    if mc is not None:
        axes[3, 1].plot(t, mc, 'r-', lw=0.8, label='Mc (нагрузка)')
    if mem is not None:
        axes[3, 1].plot(t, mem, 'b-', lw=0.4, alpha=0.5, label='Mэм')
    axes[3, 1].axhline(y=0, color='k', lw=0.5, ls='--')
    axes[3, 1].set(xlabel='Время, с', ylabel='Момент, Нм',
                    title='Момент нагрузки на валу Mc(t)')
    if mc is not None or mem is not None:
        axes[3, 1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_and_close(fig, save_path)
    return fig


def plot_waveforms(res: SimulationResults, save_path: Optional[str] = None):
    """Строит графики по данным моделирования."""

    t = res.t
    t_start = max(0, t[-1] - 0.1)
    mask = t >= t_start
    tt = (t[mask] - t[mask][0]) * 1000
    u_phases = _get_u_phases(res)
    imA, imB, imC = _get_im_phases(res)
    psi1A = _get_psi1A(res)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'ОСЦИЛЛОГРАММЫ {res.scenario_name}\n'
                 '(последние 100 мс)',
                 fontsize=13, fontweight='bold')

                  
    axes[0, 0].plot(tt, res.i1A[mask], 'b-', lw=0.8, label='i1A')
    axes[0, 0].plot(tt, res.i1B[mask], 'r-', lw=0.8, label='i1B')
    axes[0, 0].plot(tt, res.i1C[mask], 'g-', lw=0.8, label='i1C')
    axes[0, 0].set(xlabel='Время, мс', ylabel='Ток, А',
                    title='Токи статора')
    axes[0, 0].legend(fontsize=8)

                 
    axes[0, 1].plot(tt, res.i2a[mask], 'b-', lw=0.8, label='i2a')
    axes[0, 1].plot(tt, res.i2b[mask], 'r-', lw=0.8, label='i2b')
    axes[0, 1].plot(tt, res.i2c[mask], 'g-', lw=0.8, label='i2c')
    axes[0, 1].set(xlabel='Время, мс', ylabel='Ток, А',
                    title='Токи ротора')
    axes[0, 1].legend(fontsize=8)

                  
    if u_phases is not None:
        U1A, U1B, U1C = u_phases
        ax2 = axes[1, 0].twinx()
        ln1, = axes[1, 0].plot(tt, U1A[mask], 'b-', lw=0.8, label='U1A')
        ln2, = ax2.plot(tt, res.i1A[mask], 'r-', lw=0.8, label='i1A')
        axes[1, 0].set(xlabel='Время, мс', ylabel='Напряжение, В')
        ax2.set_ylabel('Ток, А', color='r')
        axes[1, 0].set_title('U1A и i1A')
        axes[1, 0].legend(handles=[ln1, ln2], fontsize=8)

                         
    axes[1, 1].plot(tt, imA[mask], 'b-', lw=0.8, label='imA')
    axes[1, 1].plot(tt, imB[mask], 'r-', lw=0.8, label='imB')
    axes[1, 1].plot(tt, imC[mask], 'g-', lw=0.8, label='imC')
    axes[1, 1].set(xlabel='Время, мс', ylabel='Ток, А',
                    title='Токи намагничивания')
    axes[1, 1].legend(fontsize=8)

                     
    if psi1A is not None:
        axes[2, 0].plot(tt, psi1A[mask], 'b-', lw=0.8)
    axes[2, 0].set(xlabel='Время, мс', ylabel='psi1_A, Вб',
                    title='Потокосцепление фазы A')

                         
    if u_phases is not None:
        U1A, U1B, U1C = u_phases
        P_inst = (U1A[mask] * res.i1A[mask]
                  + U1B[mask] * res.i1B[mask]
                  + U1C[mask] * res.i1C[mask])
        axes[2, 1].plot(tt, P_inst / 1e3, 'b-', lw=0.8)
        axes[2, 1].axhline(
            y=np.mean(P_inst) / 1e3, color='r', ls='--', lw=1,
            label=f'Pср = {np.mean(P_inst) / 1e3:.1f} кВт')
    axes[2, 1].set(xlabel='Время, мс', ylabel='Мощность, кВт',
                    title='Мгновенная электрическая мощность')
    if u_phases is not None:
        axes[2, 1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_and_close(fig, save_path)
    return fig


def _half_period_envelope(signal: np.ndarray, dt: float, fn: float) -> np.ndarray:
    """Вычисляет огибающую амплитуды за каждый полупериод."""

    half_n = max(1, int(round(0.5 / fn / dt)))
    env = np.empty_like(signal)
    for i in range(len(signal)):
        lo = max(0, i - half_n)
        hi = min(len(signal), i + half_n + 1)
        env[i] = np.max(np.abs(signal[lo:hi]))
    return env


def _amplitude_spectrum(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Вычисляет односторонний амплитудный спектр с окном Ханна."""

    N = len(signal)
    win = np.hanning(N)
    scale = 2.0 / win.sum()
    spectrum = np.abs(np.fft.rfft(signal * win)) * scale
    freqs = np.fft.rfftfreq(N, d=dt)
    return freqs, spectrum


def plot_phase_ab_zoom(
    res: SimulationResults,
    t_step: Optional[float] = None,
    save_path: Optional[str] = None,
):
    """Строит токи фаз A и B + спектр в трёх временных окнах по 10 периодов."""

    t = res.t
    fn = res.params.fn
    window = 10.0 / fn
    dt = t[1] - t[0] if len(t) > 1 else 1e-4

    t1_s, t1_e = 0.0, window

    if t_step is not None:
        t2_s = max(0.0, t_step - window / 2)
        t2_e = t2_s + window
        label2 = f'Скачок ({t2_s * 1000:.0f}\u2026{t2_e * 1000:.0f} мс)'
    else:
        t_mid = (t[0] + t[-1]) / 2.0
        t2_s = t_mid - window / 2
        t2_e = t_mid + window / 2
        label2 = f'Середина ({t2_s * 1000:.0f}\u2026{t2_e * 1000:.0f} мс)'

    t3_e = t[-1]
    t3_s = t3_e - window

    masks = [
        (t >= t1_s) & (t <= t1_e),
        (t >= t2_s) & (t <= t2_e),
        (t >= t3_s) & (t <= t3_e),
    ]
    titles_t = [
        f'Пуск ({t1_s * 1000:.0f}\u2026{t1_e * 1000:.0f} мс)',
        label2,
        f'Уст. режим ({t3_s * 1000:.0f}\u2026{t3_e * 1000:.0f} мс)',
    ]

    env_A = _half_period_envelope(res.i1A, dt, fn)
    env_B = _half_period_envelope(res.i1B, dt, fn)

    i1_mod = _current_module(res.i1A, res.i1B, res.i1C)

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle(f'Токи фаз A и B: осциллограммы и спектр\n({res.scenario_name})',
                 fontsize=13, fontweight='bold')

    f_max = 500.0

    for row, (mask, title) in enumerate(zip(masks, titles_t)):
        tt = t[mask] * 1000
        i1A_w = res.i1A[mask]
        i1B_w = res.i1B[mask]
        i1mod_w = i1_mod[mask]

        ax_t = axes[row, 0]
        ax_t.plot(tt, i1A_w, 'b-', lw=0.5, alpha=0.5, label='i1A')
        ax_t.plot(tt, i1B_w, 'r-', lw=0.5, alpha=0.5, label='i1B')
        ax_t.plot(tt, env_A[mask], 'b-', lw=1.5, label='|i1A| огиб.')
        ax_t.plot(tt, -env_A[mask], 'b-', lw=1.5)
        ax_t.plot(tt, env_B[mask], 'r--', lw=1.5, label='|i1B| огиб.')
        ax_t.plot(tt, -env_B[mask], 'r--', lw=1.5)
        ax_t.set(xlabel='Время, мс', ylabel='Ток, А', title=title)
        ax_t.legend(fontsize=7, loc='upper right')

        ax_f = axes[row, 1]
        freqs_A, spec_A = _amplitude_spectrum(i1A_w, dt)
        freqs_B, spec_B = _amplitude_spectrum(i1B_w, dt)
        freqs_m, spec_m = _amplitude_spectrum(i1mod_w, dt)

        f_mask = freqs_A <= f_max
        ax_f.semilogy(freqs_A[f_mask], spec_A[f_mask], 'b-', lw=1.0, label='i1A')
        ax_f.semilogy(freqs_B[f_mask], spec_B[f_mask], 'r-', lw=1.0, label='i1B')
        ax_f.semilogy(freqs_m[f_mask], spec_m[f_mask], 'k--', lw=1.0, alpha=0.7, label='|I1|')

        for harm in range(1, int(f_max / fn) + 1):
            ax_f.axvline(x=harm * fn, color='gray', lw=0.5, ls=':', alpha=0.5)

        ax_f.set(xlabel='Частота, Гц', ylabel='Амплитуда, А',
                 title=f'Спектр: {title}')
        ax_f.set_xlim(0, f_max)
        ax_f.legend(fontsize=7, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_and_close(fig, save_path)
    return fig


def _save_and_close(fig, save_path: Optional[str]):
    """Сохраняет данные в файл."""

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Сохранено: {save_path}")
    plt.close(fig)
