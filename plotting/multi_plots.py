"""
Графики для мульти-машинного моделирования.

Сравнительные графики двух (и более) машин на одном рисунке,
включая зуммированные области вокруг неисправностей
"""
from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from core.multi_results import MultiMachineResults
from faults.base import Fault

matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['figure.dpi'] = 150

_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']


def _get_labels(mres: MultiMachineResults) -> List[str]:
    """Извлечь короткие метки машин из scenario_name"""
    labels = []
    for i, r in enumerate(mres.machines):
        if "[" in r.scenario_name:
            labels.append(r.scenario_name.split("[")[-1].rstrip("]"))
        else:
            labels.append(f"М{i + 1}")
    return labels


def _find_fault_window(
    faults_by_machine: List[List[Fault]],
    margin: float = 0.15,
) -> Optional[Tuple[float, float]]:
    """
    Найти временное окно, покрывающее все неисправности + отступ.
    Возвращает (t_left, t_right) или None если faults нет
    """
    t_min = float('inf')
    t_max = float('-inf')
    has_fault = False

    for faults in faults_by_machine:
        for f in faults:
            has_fault = True
            t_min = min(t_min, f.t_start)
            t_end = f.t_end if f.t_end is not None else f.t_start + 1.0
            t_max = max(t_max, t_end)

    if not has_fault:
        return None

    span = t_max - t_min
    return (t_min - margin * span - margin,
            t_max + margin * span + margin)


def _shade_fault_zones(
    ax,
    faults_by_machine: List[List[Fault]],
    labels: List[str],
):
    """Подсветить зоны неисправностей на графике"""
    for i, faults in enumerate(faults_by_machine):
        for f in faults:
            t_end = f.t_end if f.t_end is not None else ax.get_xlim()[1]
            ax.axvspan(f.t_start, t_end,
                       alpha=0.12, color=_COLORS[i % len(_COLORS)],
                       label=f'{labels[i]}: {f.describe()}')


def _plot_on_ax_mem(ax, mres, labels, t_slice=None):
    """Нарисовать Mэм на заданных осях"""
    t = mres.t
    mask = np.ones(len(t), dtype=bool) if t_slice is None else \
        (t >= t_slice[0]) & (t <= t_slice[1])
    for i, res in enumerate(mres.machines):
        if res.Mem is not None:
            ax.plot(t[mask], res.Mem[mask],
                    color=_COLORS[i % len(_COLORS)], lw=0.6, label=labels[i])
    ax.axhline(y=0, color='k', lw=0.5, ls='--')
    ax.set(ylabel='Момент, Нм')


def _plot_on_ax_i1(ax, mres, labels, t_slice=None):
    """Нарисовать |I1| на заданных осях"""
    t = mres.t
    mask = np.ones(len(t), dtype=bool) if t_slice is None else \
        (t >= t_slice[0]) & (t <= t_slice[1])
    for i, res in enumerate(mres.machines):
        if res.I1_mod is not None:
            ax.plot(t[mask], res.I1_mod[mask],
                    color=_COLORS[i % len(_COLORS)], lw=0.6, label=labels[i])
    ax.set(ylabel='|I1|, А')


def plot_multi_comparison(
    mres: MultiMachineResults,
    faults_by_machine: Optional[List[List[Fault]]] = None,
    save_path: Optional[str] = None,
):
    """
    Сравнительные графики N машин:
      Строка 1: Mэм (полный) | Mэм (зум на fault)
      Строка 2: |I1| (полный) | |I1| (зум на fault)
      Строка 3: скорость | скольжение
      Строка 4: суммарный ток от источника | (пусто / напряжение шины)
    """
    t = mres.t
    labels = _get_labels(mres)

    # Определить окно зума
    fault_window = None
    if faults_by_machine is not None:
        fault_window = _find_fault_window(faults_by_machine)
    if fault_window is None:
        fault_window = _find_fault_window_from_results(mres)

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(mres.scenario_name, fontsize=14, fontweight='bold')

    #Строка 0: Mэм
    ax = axes[0, 0]
    _plot_on_ax_mem(ax, mres, labels)
    ax.set(xlabel='Время, с', title='Электромагнитный момент Mэм (полный)')
    ax.legend(fontsize=8)
    if fault_window and faults_by_machine:
        _shade_fault_zones(ax, faults_by_machine, labels)

    ax = axes[0, 1]
    if fault_window:
        _plot_on_ax_mem(ax, mres, labels, t_slice=fault_window)
        ax.set(xlabel='Время, с',
               title=f'Mэм (зум: {fault_window[0]:.2f}–{fault_window[1]:.2f} с)')
        ax.legend(fontsize=8)
        if faults_by_machine:
            _shade_fault_zones(ax, faults_by_machine, labels)
    else:
        ax.set_visible(False)

    #Строка 1: |I1|
    ax = axes[1, 0]
    _plot_on_ax_i1(ax, mres, labels)
    ax.set(xlabel='Время, с', title='Модуль тока статора |I1| (полный)')
    ax.legend(fontsize=8)
    if fault_window and faults_by_machine:
        _shade_fault_zones(ax, faults_by_machine, labels)

    ax = axes[1, 1]
    if fault_window:
        _plot_on_ax_i1(ax, mres, labels, t_slice=fault_window)
        ax.set(xlabel='Время, с',
               title=f'|I1| (зум: {fault_window[0]:.2f}–{fault_window[1]:.2f} с)')
        ax.legend(fontsize=8)
        if faults_by_machine:
            _shade_fault_zones(ax, faults_by_machine, labels)
    else:
        ax.set_visible(False)

    #Строка 2: скорость | скольжение
    ax = axes[2, 0]
    for i, res in enumerate(mres.machines):
        if res.n_rpm is not None:
            ax.plot(t, res.n_rpm, color=_COLORS[i % len(_COLORS)],
                    lw=0.8, label=labels[i])
    n_sync = 60 * mres.machines[0].params.fn / mres.machines[0].params.p
    ax.axhline(y=n_sync, color='k', lw=0.8, ls='--',
               label=f'n синхр = {n_sync:.0f}')
    ax.set(xlabel='Время, с', ylabel='Скорость, об/мин',
           title='Частота вращения роторов')
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    for i, res in enumerate(mres.machines):
        if res.slip is not None:
            ax.plot(t, res.slip, color=_COLORS[i % len(_COLORS)],
                    lw=0.6, label=labels[i])
    ax.axhline(y=0, color='k', lw=0.5, ls='--')
    ax.set(xlabel='Время, с', ylabel='Скольжение',
           title='Скольжение s(t)')
    ax.legend(fontsize=8)

    #Строка 3: суммарный ток | напряжения шины
    ax = axes[3, 0]
    if mres.I_total_A is not None:
        ax.plot(t, mres.I_total_A, 'b-', lw=0.5, label='ΣI_A')
        ax.plot(t, mres.I_total_B, 'r-', lw=0.5, label='ΣI_B')
        ax.plot(t, mres.I_total_C, 'g-', lw=0.5, label='ΣI_C')
    ax.set(xlabel='Время, с', ylabel='Ток, А',
           title='Суммарные фазные токи от источника')
    ax.legend(fontsize=8)

    ax = axes[3, 1]
    if mres.U_bus_A is not None:
        ax.plot(t, mres.U_bus_A, 'b-', lw=0.5, label='U_bus_A')
        ax.plot(t, mres.U_bus_B, 'r-', lw=0.5, label='U_bus_B')
        ax.plot(t, mres.U_bus_C, 'g-', lw=0.5, label='U_bus_C')
    ax.set(xlabel='Время, с', ylabel='Напряжение, В',
           title='Напряжения общей шины')
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_and_close(fig, save_path)
    return fig


def _find_fault_window_from_results(
    mres: MultiMachineResults,
    margin: float = 0.15,
) -> Optional[Tuple[float, float]]:
    """
    Fallback: попытаться определить зону неисправности
    по резкому скачку |I1| или Mem у любой машины
    """
    # Простая эвристика: ищем точку максимального |dMem/dt|
    best_t = None
    best_val = 0
    for res in mres.machines:
        if res.Mem is not None and len(res.Mem) > 10:
            dM = np.abs(np.diff(res.Mem))
            idx = np.argmax(dM)
            if dM[idx] > best_val:
                best_val = dM[idx]
                best_t = res.t[idx]
    if best_t is not None and best_val > 100:
        half = 0.3
        return (best_t - half, best_t + half + margin)
    return None


def plot_multi_phase_currents(
    mres: MultiMachineResults,
    save_path: Optional[str] = None,
):
    """
    Фазные токи каждой машины + суммарные - детально.
    По строке на машину, последняя строка - суммарные
    """
    t = mres.t
    n = mres.n_machines
    labels = _get_labels(mres)

    n_rows = n + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))
    fig.suptitle(f'{mres.scenario_name}\nФазные токи',
                 fontsize=13, fontweight='bold')

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, res in enumerate(mres.machines):
        ax = axes[i, 0]
        ax.plot(t, res.i1A, 'b-', lw=0.5, label='i1A')
        ax.plot(t, res.i1B, 'r-', lw=0.5, label='i1B')
        ax.plot(t, res.i1C, 'g-', lw=0.5, label='i1C')
        ax.set(xlabel='Время, с', ylabel='Ток, А',
               title=f'{labels[i]}: токи статора')
        ax.legend(fontsize=7)

        ax = axes[i, 1]
        if res.U1A is not None:
            ax.plot(t, res.U1A, 'b-', lw=0.5, label='U1A')
            ax.plot(t, res.U1B, 'r-', lw=0.5, label='U1B')
            ax.plot(t, res.U1C, 'g-', lw=0.5, label='U1C')
        ax.set(xlabel='Время, с', ylabel='Напряжение, В',
               title=f'{labels[i]}: напряжения статора')
        ax.legend(fontsize=7)

    ax = axes[n, 0]
    if mres.I_total_A is not None:
        ax.plot(t, mres.I_total_A, 'b-', lw=0.5, label='sum I_A')
        ax.plot(t, mres.I_total_B, 'r-', lw=0.5, label='sum I_B')
        ax.plot(t, mres.I_total_C, 'g-', lw=0.5, label='sum I_C')
    ax.set(xlabel='Время, с', ylabel='Ток, А',
           title='Суммарные токи от источника')
    ax.legend(fontsize=7)

    ax = axes[n, 1]
    if mres.U_bus_A is not None:
        ax.plot(t, mres.U_bus_A, 'b-', lw=0.5, label='U_bus_A')
        ax.plot(t, mres.U_bus_B, 'r-', lw=0.5, label='U_bus_B')
        ax.plot(t, mres.U_bus_C, 'g-', lw=0.5, label='U_bus_C')
    ax.set(xlabel='Время, с', ylabel='Напряжение, В',
           title='Напряжения общей шины')
    ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_and_close(fig, save_path)
    return fig


def _save_and_close(fig, save_path: Optional[str]):
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Сохранено: {save_path}")
    plt.close(fig)