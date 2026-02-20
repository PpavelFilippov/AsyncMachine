"""
Модульный схемный ассемблер для моделирования асинхронной машины.

Пакет реализует DAE-подход: ОДУ двигателя не меняются,
а внешняя схема (источник, контуры КЗ) собирается через
CircuitTopology + CircuitAssembler.
"""
from .components import MotorComponent, SourceComponent, FaultBranchComponent
from .topology import CircuitTopology
from .assembler import CircuitAssembler
from .simulation_dae import DAESimulationBuilder
