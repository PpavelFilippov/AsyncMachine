"""
Стандартные графики результатов моделирования
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


def plot_motor_start(res: SimulationResults, save_path: Optional[str] = None):
    """Графики пуска АД"""
    t = res.t
    params = res.params

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{res.scenario_name}\n(Прямой)',
                 fontsize=13, fontweight='bold')

    axes[0, 0].plot(t, res.i1A, 'b-', lw=0.5)
    axes[0, 0].set(xlabel='Время, с', ylabel='Ток, А',
                    title='Ток фазы A статора i1A')

    if res.I1_mod is not None:
        axes[0, 1].plot(t, res.I1_mod, 'b-', lw=0.6)
        axes[0, 1].set(xlabel='Время, с', ylabel='|I1|, А',
                        title='Модуль результирующего тока статора')

    if res.Mem is not None:
        axes[1, 0].plot(t, res.Mem, 'b-', lw=0.5)
        axes[1, 0].axhline(y=0, color='k', lw=0.5, ls='--')
        axes[1, 0].set(xlabel='Время, с', ylabel='Момент, Нм',
                        title='Электромагнитный момент Mэм')

    if res.n_rpm is not None:
        axes[1, 1].plot(t, res.n_rpm, 'b-', lw=0.8)
        n_sync = 60 * params.fn / params.p
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
    """Графики для установившегося режима (моторный/генераторный)"""
    t = res.t

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(f'Результаты моделирования\n({res.scenario_name})',
                 fontsize=13, fontweight='bold')

    # Фазные токи статора
    axes[0, 0].plot(t, res.i1A, 'b-', lw=0.5, label='i1A')
    axes[0, 0].plot(t, res.i1B, 'r-', lw=0.5, label='i1B')
    axes[0, 0].plot(t, res.i1C, 'g-', lw=0.5, label='i1C')
    axes[0, 0].set(xlabel='Время, с', ylabel='Ток, А',
                    title='Фазные токи статора')
    axes[0, 0].legend(fontsize=8)

    # Модули токов
    if res.I1_mod is not None:
        axes[0, 1].plot(t, res.I1_mod, 'b-', lw=0.6, label='|I1|')
    if res.I2_mod is not None:
        axes[0, 1].plot(t, res.I2_mod, 'r-', lw=0.6, label='|I2|')
    if res.Im_mod is not None:
        axes[0, 1].plot(t, res.Im_mod, 'g--', lw=0.6, label='|Im|')
    axes[0, 1].set(xlabel='Время, с', ylabel='Ток, А',
                    title='Модули результирующих токов')
    axes[0, 1].legend(fontsize=8)

    # Момент
    if res.Mem is not None:
        axes[1, 0].plot(t, res.Mem, 'b-', lw=0.6)
        axes[1, 0].axhline(y=0, color='k', lw=0.5, ls='--')
    axes[1, 0].set(xlabel='Время, с', ylabel='Момент, Нм',
                    title='Электромагнитный момент Mэм')

    # Скорость
    if res.n_rpm is not None:
        axes[1, 1].plot(t, res.n_rpm, 'b-', lw=0.8)
        n_sync = 60 * res.params.fn / res.params.p
        axes[1, 1].axhline(y=n_sync, color='r', lw=0.8, ls='--',
                            label=f'n синхр = {n_sync:.0f}')
    axes[1, 1].set(xlabel='Время, с', ylabel='Скорость, об/мин',
                    title='Частота вращения ротора')
    axes[1, 1].legend(fontsize=8)

    # Скольжение
    if res.slip is not None:
        axes[2, 0].plot(t, res.slip, 'b-', lw=0.6)
        axes[2, 0].axhline(y=0, color='r', lw=0.8, ls='--', label='s = 0')
    axes[2, 0].set(xlabel='Время, с', ylabel='Скольжение s',
                    title='Скольжение s(t)')
    axes[2, 0].legend(fontsize=8)

    # Напряжение + ток (последние 60 мс)
    if res.U1A is not None:
        t_start = max(0, t[-1] - 0.06)
        mask = t >= t_start
        axes[2, 1].plot(t[mask] * 1000, res.U1A[mask], 'b-', lw=0.6, label='U1A')
        axes[2, 1].plot(t[mask] * 1000, res.i1A[mask], 'r-', lw=0.6, label='i1A')
        axes[2, 1].set(xlabel='Время, мс', ylabel='В / А',
                        title='U и I фазы A (последние 60 мс)')
        axes[2, 1].legend(fontsize=8)

    # Потокосцепление
    if res.Psi1A is not None:
        axes[3, 0].plot(t, res.Psi1A, 'b-', lw=0.6)
    axes[3, 0].set(xlabel='Время, с', ylabel='psi, Вб',
                    title='Потокосцепление фазы A статора')

    # Мощности
    if res.P_elec is not None:
        axes[3, 1].plot(t, res.P_elec / 1e3, 'b-', lw=0.5, label='P электр.')
    if res.P_mech is not None:
        axes[3, 1].plot(t, res.P_mech / 1e3, 'r-', lw=0.5, label='P мех.')
    axes[3, 1].axhline(y=0, color='k', lw=0.5, ls='--')
    axes[3, 1].set(xlabel='Время, с', ylabel='Мощность, кВт',
                    title='Электрическая и механическая мощность')
    axes[3, 1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_and_close(fig, save_path)
    return fig


def plot_waveforms(res: SimulationResults, save_path: Optional[str] = None):
    """Детальные осциллограммы (последние 100 мс)"""
    t = res.t
    t_start = max(0, t[-1] - 0.1)
    mask = t >= t_start
    tt = (t[mask] - t[mask][0]) * 1000

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'ОСЦИЛЛОГРАММЫ {res.scenario_name}\n'
                 '(последние 100 мс)',
                 fontsize=13, fontweight='bold')

    # Токи статора
    axes[0, 0].plot(tt, res.i1A[mask], 'b-', lw=0.8, label='i1A')
    axes[0, 0].plot(tt, res.i1B[mask], 'r-', lw=0.8, label='i1B')
    axes[0, 0].plot(tt, res.i1C[mask], 'g-', lw=0.8, label='i1C')
    axes[0, 0].set(xlabel='Время, мс', ylabel='Ток, А',
                    title='Токи статора')
    axes[0, 0].legend(fontsize=8)

    # Токи ротора
    axes[0, 1].plot(tt, res.i2a[mask], 'b-', lw=0.8, label='i2a')
    axes[0, 1].plot(tt, res.i2b[mask], 'r-', lw=0.8, label='i2b')
    axes[0, 1].plot(tt, res.i2c[mask], 'g-', lw=0.8, label='i2c')
    axes[0, 1].set(xlabel='Время, мс', ylabel='Ток, А',
                    title='Токи ротора')
    axes[0, 1].legend(fontsize=8)

    # U и I фазы A
    if res.U1A is not None:
        ax2 = axes[1, 0].twinx()
        ln1, = axes[1, 0].plot(tt, res.U1A[mask], 'b-', lw=0.8, label='U1A')
        ln2, = ax2.plot(tt, res.i1A[mask], 'r-', lw=0.8, label='i1A')
        axes[1, 0].set(xlabel='Время, мс', ylabel='Напряжение, В')
        ax2.set_ylabel('Ток, А', color='r')
        axes[1, 0].set_title('U1A и i1A')
        axes[1, 0].legend(handles=[ln1, ln2], fontsize=8)

    # Токи намагничивания
    if res.imA is not None:
        axes[1, 1].plot(tt, res.imA[mask], 'b-', lw=0.8, label='imA')
        axes[1, 1].plot(tt, res.imB[mask], 'r-', lw=0.8, label='imB')
        axes[1, 1].plot(tt, res.imC[mask], 'g-', lw=0.8, label='imC')
    axes[1, 1].set(xlabel='Время, мс', ylabel='Ток, А',
                    title='Токи намагничивания')
    axes[1, 1].legend(fontsize=8)

    # Потокосцепление
    if res.Psi1A is not None:
        axes[2, 0].plot(tt, res.Psi1A[mask], 'b-', lw=0.8)
    axes[2, 0].set(xlabel='Время, мс', ylabel='psi1_A, Вб',
                    title='Потокосцепление фазы A')

    # Мгновенная мощность
    if res.U1A is not None:
        P_inst = (res.U1A[mask] * res.i1A[mask]
                  + res.U1B[mask] * res.i1B[mask]
                  + res.U1C[mask] * res.i1C[mask])
        axes[2, 1].plot(tt, P_inst / 1e3, 'b-', lw=0.8)
        axes[2, 1].axhline(
            y=np.mean(P_inst) / 1e3, color='r', ls='--', lw=1,
            label=f'Pср = {np.mean(P_inst) / 1e3:.1f} кВт')
    axes[2, 1].set(xlabel='Время, мс', ylabel='Мощность, кВт',
                    title='Мгновенная электрическая мощность')
    axes[2, 1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_and_close(fig, save_path)
    return fig


def _save_and_close(fig, save_path: Optional[str]):
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Сохранено: {save_path}")
    plt.close(fig)