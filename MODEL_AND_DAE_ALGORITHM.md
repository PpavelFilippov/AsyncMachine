# Математическая модель и вычислительный алгоритм

Документ описывает:

1.  Математическую модель из `models/linear.py`.
2.  Концептуальный алгоритм текущей реализации (legacy) без КЗ и с КЗ.
3.  DAE модель и сборку системы в `circuit/*`.

Формулы даны в текстовом виде без LaTeX-разметки. Такой формат корректно отображается в любом Markdown-рендере.

## 1. Математическая модель `linear.py`

### 1.1 Вектор состояния

```text
y = [i1A, i1B, i1C, i2a, i2b, i2c, omega_r]^T
```

где:

-   `i1A, i1B, i1C` - токи статора.
-   `i2a, i2b, i2c` - токи ротора.
-   `omega_r` - механическая угловая скорость ротора.

### 1.2 Матрица индуктивностей

```text
Mm    = 2*Lm/3
m_off = -0.5*Mm

L1_diag = L1sigma + Mm
L2_diag = L2sigma + Mm

Lss = [[L1_diag, m_off,   m_off  ],
       [m_off,   L1_diag, m_off  ],
       [m_off,   m_off,   L1_diag]]

Lrr = [[L2_diag, m_off,   m_off  ],
       [m_off,   L2_diag, m_off  ],
       [m_off,   m_off,   L2_diag]]

Lsr = [[Mm,    m_off, m_off],
       [m_off, Mm,    m_off],
       [m_off, m_off, Mm   ]]

L_machine_6 = [[Lss, Lsr],
               [Lsr, Lrr]]
```

### 1.3 ЭДС вращения ротора

```text
omega_e = p * omega_r
imA = i1A + i2a
imB = i1B + i2b
imC = i1C + i2c

Ea = omega_e/sqrt(3) * (L2sigma*(i2b - i2c) + Lm*(imB - imC))
Eb = omega_e/sqrt(3) * (L2sigma*(i2c - i2a) + Lm*(imC - imA))
Ec = omega_e/sqrt(3) * (L2sigma*(i2a - i2b) + Lm*(imA - imB))
```

### 1.4 Электрическая часть

Базовая правая часть:

```text
b0 = [-R1*i1A,
      -R1*i1B,
      -R1*i1C,
      -R2*i2a - Ea,
      -R2*i2b - Eb,
      -R2*i2c - Ec]
```

С учетом источника:

```text
L_src_6 = [[L_src, 0],
           [0,     0]]

i_s = [i1A, i1B, i1C]^T

(L_machine_6 + L_src_6) * di_dt =
    b0 + [E_src(t) - R_src*i_s,
          0]
```

### 1.5 Механическая часть

```text
Mem = (p*Lm/sqrt(3)) *
      ((i1A*i2c + i1B*i2a + i1C*i2b) -
       (i1A*i2b + i1B*i2c + i1C*i2a))

domega_r_dt = (Mem - Mc(t, omega_r)) / J
```

### 1.6 Потокосцепление фазы A

```text
psi_A = L1sigma*i1A
      + Mm*(i1A - (i1B + i1C)/2)
      + Mm*(i2a - (i2b + i2c)/2)
```

## 2. Концептуальный алгоритм legacy реализации

### 2.1 Без КЗ (`SimulationBuilder`)

1.  Получить из сценария `y0`, `source`, `load`, `t_span`.
2.  Создать модель `LinearInductionMachine`.
3.  На каждом шаге интегратора собрать `rhs(t, y)`:
    -   вычислить `L_machine_6` и `b0`;
    -   добавить вклад источника `R_src`, `L_src`, `E_src(t)`;
    -   решить линейную систему для `di_dt`;
    -   вычислить `domega_r_dt`.
4.  Интегрировать систему решателем.
5.  Сформировать `SimulationResults`.

### 2.2 С КЗ (`FaultSimulationBuilder`)

1.  Сегмент 1 до `t_fault`: решается обычная 7-мерная система.
2.  В момент `t_fault` состояние расширяется:

```text
y_ext = [i1A, i1B, i1C, i2a, i2b, i2c, omega_r, i_fault]^T
```

3.  Сегмент 2 после `t_fault`: решается расширенная система через `LinearInductionMachineSC` (`models/linear_sc.py`).
4.  Два сегмента склеиваются в единый результат.

## 3. DAE модель и сборка `circuit/*`

### 3.1 Компоненты

-   `MotorComponent`: базовые уравнения машины.
-   `SourceComponent`: ЭДС источника и матрицы `R_src`, `L_src`.
-   `FaultBranchComponent`: описание ветви КЗ (`D`, `R_fault`, число доп. токов).
-   `CircuitTopology`: назначает структуру и смещения состояний.
-   `CircuitAssembler`: собирает расширенные матрицы и правую часть.

### 3.2 Расширенное состояние

```text
i_fault in R^m
i_src_eff = i_s + D^T * i_fault
```

где `m` - число дополнительных токов ветвей КЗ.

### 3.3 Расширенная электрическая система

```text
L_ext * dI_ext_dt = rhs_ext

I_ext = [i_s, i_r, i_fault]^T
```

Структура матрицы:

```text
L_ext = [[L_machine_6 + L_src_6,   C      ],
         [C^T,                     D*L_src*D^T]]

C = [[L_src*D^T],
     [0      ]]
```

Правая часть:

```text
r_s = b0_s + E_src - R_src*i_src_eff
r_r = b0_r
r_f = D*E_src - (D*R_src)*i_src_eff - R_fault*i_fault

rhs_ext = [r_s, r_r, r_f]^T
```

### 3.4 Механика в DAE

Механическая часть та же:

```text
domega_r_dt = (Mem - Mc)/J
```

Момент `Mem` вычисляется базовой моделью `LinearInductionMachine`.

### 3.5 Алгоритм DAE без КЗ

1.  Собрать топологию: мотор + источник.
2.  `CircuitAssembler` строит `rhs`.
3.  Решить 7-мерную систему.
4.  Получить результат.

### 3.6 Алгоритм DAE с КЗ

1.  Сегмент 1 до `t_fault`: топология без активной ветви КЗ.
2.  В `t_fault`: расширить состояние, доп. токи инициализировать нулем.
3.  Сегмент 2 после `t_fault`: топология с `FaultBranchComponent`.
4.  Решать расширенную систему через `L_ext`.
5.  Склеить сегменты, сохранить `i_fault` в `results.extra`.

### 3.7 Проверка корректности контура КЗ

Перед расчетом с КЗ выполняется проверка:

```text
min_eigenvalue(D * L_src * D^T) > 0
```

Если условие не выполнено, контур КЗ вырожден (обычно из-за нулевой индуктивности источника), и расчет останавливается.

## 4. Итог

-   Без КЗ: legacy и DAE используют один и тот же физический закон.
-   С КЗ:
    -   legacy путь использует специализированный `linear_sc.py`;
    -   DAE путь использует универсальную матричную сборку схемы.
-   DAE подход проще масштабировать на новые ветви и топологии.