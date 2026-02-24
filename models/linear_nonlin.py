"""
    Модуль models/linear_nonlin.py.
    Состав:
    Классы: LinearInductionMachineNonlinear.
    Функции: нет.

    Файл оставлен для обратной совместимости.
    Основная реализация находится в models/nonlinear.py.
"""
from __future__ import annotations

from .nonlinear import NonlinearInductionMachine


class LinearInductionMachineNonlinear(NonlinearInductionMachine):
    """Совместимое имя класса для существующих скриптов."""

