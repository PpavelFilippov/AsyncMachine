# Математическая модель межвиткового замыкания и вычислительный алгоритм

Документ описывает новые файлы, связанные с межвитковым замыканием:

1. Математическую модель `models/linear_segmented.py`.
2. Сборку уравнений через `circuit/components_seg.py` и `circuit/assembler_seg.py`.
3. Алгоритм расчета в `DAESimulationBuilder._run_with_interturn_fault` (`circuit/simulation_dae.py`).
4. Параметры дефекта из `faults/interturn.py`.

Формулы даны в текстовом виде без LaTeX.

## 1. Модель segmented для межвиткового замыкания

### 1.1 Вектор состояния

Внутренний полный вектор модели:

```text
y_full = [i_A1, i_A2, i_B1, i_B2, i_C1, i_C2, i_2a, i_2b, i_2c, omega_r]^T
```

где:

- `i_A1, i_A2, ...` - токи сегментов статорных фаз.
- `i_2a, i_2b, i_2c` - токи ротора.
- `omega_r` - механическая угловая скорость.

### 1.2 Параметры сегментации

Для каждой фазы задается доля выделенного сегмента:

```text
mu_A, mu_B, mu_C,    0 < mu_X < 1
```

Коэффициенты долей витков:

```text
a1 = 1 - mu_A,  a2 = mu_A
b1 = 1 - mu_B,  b2 = mu_B
c1 = 1 - mu_C,  c2 = mu_C
```

Эквивалентные фазные токи статора:

```text
i_eq = T_s * i_seg
i_seg = [i_A1, i_A2, i_B1, i_B2, i_C1, i_C2]^T

T_s =
[[a1, a2, 0,  0,  0,  0 ],
 [0,  0,  b1, b2, 0,  0 ],
 [0,  0,  0,  0,  c1, c2]]
```

то есть:

```text
i_A_eq = a1*i_A1 + a2*i_A2
i_B_eq = b1*i_B1 + b2*i_B2
i_C_eq = c1*i_C1 + c2*i_C2
```

### 1.3 Матрица индуктивностей segmented

Базовые 3x3 подматрицы `Lss`, `Lrr`, `Lsr` берутся из `linear.py`:

```text
Mm    = 2*Lm/3
m_off = -Mm/2
```

Далее строится 9x9 матрица segmented:

```text
L_ss_seg = T_s^T * Lss * T_s
L_sr_seg = T_s^T * Lsr
L_rr_seg = Lrr

L_seg =
[[L_ss_seg,     L_sr_seg],
 [L_sr_seg^T,   L_rr_seg]]
```

### 1.4 Сопротивления

Сопротивления сегментов статора масштабируются долей витков:

```text
R_A1 = a1*R1, R_A2 = a2*R1
R_B1 = b1*R1, R_B2 = b2*R1
R_C1 = c1*R1, R_C2 = c2*R1
R_2a = R_2b = R_2c = R2
```

### 1.5 ЭДС ротора и электрическая подсистема

Магнитные токи считаются через эквивалентные токи статора:

```text
imA = i_A_eq + i_2a
imB = i_B_eq + i_2b
imC = i_C_eq + i_2c
omega_e = p * omega_r
```

ЭДС ротора:

```text
Ea = omega_e/sqrt(3) * (L2sigma*(i_2b - i_2c) + Lm*(imB - imC))
Eb = omega_e/sqrt(3) * (L2sigma*(i_2c - i_2a) + Lm*(imC - imA))
Ec = omega_e/sqrt(3) * (L2sigma*(i_2a - i_2b) + Lm*(imA - imB))
```

Электрическое уравнение внутренней модели:

```text
L_seg * di_full_dt = b0_full
```

где `di_full_dt` относится к 9 электрическим переменным, а:

```text
b0_full =
[-R_A1*i_A1,
 -R_A2*i_A2,
 -R_B1*i_B1,
 -R_B2*i_B2,
 -R_C1*i_C1,
 -R_C2*i_C2,
 -R2*i_2a - Ea,
 -R2*i_2b - Eb,
 -R2*i_2c - Ec]^T
```

### 1.6 Механическая подсистема

Электромагнитный момент берется по эквивалентным статорным токам:

```text
Mem = (p*Lm/sqrt(3)) *
      ((i_A_eq*i_2c + i_B_eq*i_2a + i_C_eq*i_2b) -
       (i_A_eq*i_2b + i_B_eq*i_2c + i_C_eq*i_2a))

domega_r_dt = (Mem - Mc(t, omega_r))/J
```

## 2. Приведенная DAE система для расчета

## 2.1 Constraint-представление

В DAE расчете используется вектор независимых токов `i_ind`, а полный 9-вектор восстанавливается:

```text
i_full = C * i_ind
```

Здоровый режим:

```text
C_h: 9x6
i_ind_h = [i_A, i_B, i_C, i_2a, i_2b, i_2c]^T
```

Межвитковое замыкание в фазе X:

```text
C_f: 9x7
i_ind_f = [ ... , i_X1, i_X2, ... , i_2a, i_2b, i_2c]^T
```

У замкнутой фазы `i_X1` и `i_X2` становятся независимыми, у здоровых фаз сегменты остаются последовательно связаны.

## 2.2 Приведение матриц

Из полной модели строятся:

```text
L_red  = C^T * L_seg * C
b0_red = C^T * b0_full
```

Это делает `SegmentedMotorComponent`.

## 2.3 Подключение источника

Источник задается матрицами `R_src`, `L_src` и ЭДС `e_src(t)` (3 фазы).  
Они добавляются только в уравнения линейных токов.

```text
L_ext = L_red + embed(L_src, line_indices)
```

```text
i_line = select(i_ind, line_indices)
src = e_src - R_src * i_line
rhs_ext = b0_red
rhs_ext[line_indices] += src
```

`line_indices` зависят от режима:

- healthy: токи `[i_A, i_B, i_C]`;
- fault: токи линий с учетом отдельного `i_X1`.

## 2.4 Модель ветви межвиткового замыкания

В режиме МВЗ добавляется сопротивление `R_f` между током линии замкнутой фазы и током выделенного сегмента:

```text
i_fault_loop = i_line_X - i_X2
u_fault      = R_f * i_fault_loop
```

В правую часть добавляются члены:

```text
rhs[idx_seg2] += R_f * (i_line_X - i_X2)
rhs[idx_line] -= R_f * (i_line_X - i_X2)
```

Таким образом учитывается перераспределение напряжения между основным и выделенным сегментом замкнутой фазы.

Механика остается той же:

```text
domega_r_dt = (Mem - Mc)/J
```

## 3. Алгоритм в `DAESimulationBuilder` для interturn

`DAESimulationBuilder.interturn_fault(...)` включает отдельный путь `_run_with_interturn_fault()`.

### 3.1 Сегмент 1 (до `t_fault`)

1. Создается `SegmentedLinearInductionMachine`.
2. Выбирается `C_h`, затем `SegmentedMotorComponent` и `SegmentedCircuitAssembler(fault=None)`.
3. Решается система размерности 7:

```text
[i_A, i_B, i_C, i_2a, i_2b, i_2c, omega_r]
```

на интервале `[t_start, t_fault]`.

### 3.2 Переход в точке замыкания

Из состояния в момент `t_fault` формируется начальное условие для fault-режима:

```text
i_X1(t_fault+) = i_X(t_fault-)
i_X2(t_fault+) = i_X(t_fault-)
```

То есть токи двух сегментов замкнутой фазы непрерывны и равны до момента появления ветви `R_f`.

### 3.3 Сегмент 2 (после `t_fault`)

1. Выбирается `C_f` для замкнутой фазы.
2. Создается `SegmentedCircuitAssembler(..., fault=InterTurnFaultDescriptor)`.
3. Решается система размерности 8:

```text
[i_ind_f(7), omega_r]
```

на интервале `[t_fault, t_end]`.

### 3.4 Склейка и формирование выхода

После интегрирования:

1. Сегменты по времени объединяются.
2. Из `i_ind_f` восстанавливается полный 9-вектор и эквивалентные фазные токи:

```text
i_full = C_f * i_ind_f
i_eq   = T_s * i_full_stator_segments
```

3. В итоговый `SimulationResults` записывается базовый 7-вектор:

```text
[i_A_eq, i_B_eq, i_C_eq, i_2a, i_2b, i_2c, omega_r]
```

4. Дополнительно сохраняются диагностические ряды:

```text
i_line_faulted
i_seg2_faulted
i_fault_loop = i_line_faulted - i_seg2_faulted
```

## 4. Параметры дефекта `faults/interturn.py`

Дескриптор МВЗ:

```text
InterTurnFaultDescriptor(
    phase in {A, B, C},
    mu in (0, 1),
    R_f >= 0,
    t_fault >= 0
)
```

Смысл параметров:

- `phase` - фаза, где разделяются сегменты и появляется ветвь МВЗ.
- `mu` - доля витков выделенного сегмента этой фазы.
- `R_f` - сопротивление межвитковой ветви.
- `t_fault` - время включения дефекта.

## 5. Концептуальный итог

- Модель МВЗ реализована как segmented-расширение базовой линейной модели.
- Физическая связь с базовой моделью сохраняется через эквивалентные фазные токи и тот же закон момента.
- Отличие МВЗ вводится через:
  - разрыв последовательной связи сегментов замкнутой фазы;
  - дополнительный токовый контур `i_fault_loop` через `R_f`.
- Вычислительно это реализовано двухсегментной интеграцией: healthy -> fault с непрерывным переходом токов в `t_fault`.
