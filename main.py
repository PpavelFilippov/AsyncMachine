import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['figure.dpi'] = 150


class MachineParameters:
    """Параметры асинхронной машины (тяговый АД метрополитена)."""

    def __init__(self):
        self.P2n = 170e3  # Номинальная мощность на валу, Вт
        self.Unom_line = 570.0  # Номинальное линейное напряжение, В
        self.Unom_phase = 570.0 / np.sqrt(3)  # Фазное напряжение, В
        self.In = 204.42  # Номинальный ток статора, А
        self.fn = 50.0  # Номинальная частота, Гц
        self.omega_n = 2 * np.pi * self.fn  # Угловая частота сети, рад/с
        self.cos_phi = 0.8993  # Коэффициент мощности
        self.eta = 0.93664  # КПД
        self.n_nom = 1481.0  # Номинальная частота вращения, об/мин
        self.n_sync = 1500.0  # Синхронная частота вращения, об/мин
        self.p = 2  # Число пар полюсов
        self.m = 3  # Число фаз
        self.s_nom = 0.0127  # Номинальное скольжение

        self.R1 = 0.0279  # Активное сопротивление статора, Ом
        self.L1sigma = 0.000509  # Индуктивность рассеяния статора, Гн
        self.R2 = 0.0207  # Приведённое акт. сопротивление ротора, Ом
        self.L2sigma = 0.000458  # Приведённая индуктивность рассеяния ротора, Гн
        self.Rm = 1e-12  # Активное сопротивление ветви намагничивания, Ом
        self.Lm = 0.01668  # Индуктивность ветви намагничивания, Гн

        self.J = 2.875  # Момент инерции ротора, кг·м²

        self.Kmm = 3.07  # Кратность максимального момента
        self.Kmp = 1.34  # Кратность пускового момента
        self.Kip = 6.22  # Кратность пускового тока

        self.Rs_alt = 0.0291  # Ом
        self.Rr_alt = 0.02017  # Ом
        self.Xs = 0.171  # Ом (индуктивное сопротивление статора)
        self.Xr = 0.171  # Ом (индуктивное сопротивление ротора)
        self.Ls_alt = 0.544e-3  # Гн
        self.Lr_alt = 0.476e-3  # Гн
        self.Lm_alt = 0.228e-3  # Гн

    def info(self):
        """Вывод основных параметров."""
        print("=" * 60)
        print("  ПАРАМЕТРЫ МАШИНЫ")
        print("=" * 60)
        print(f"  P2н = {self.P2n / 1e3:.0f} кВт")
        print(f"  Uн(лин) = {self.Unom_line:.0f} В, Uн(фаз) = {self.Unom_phase:.1f} В")
        print(f"  Iн = {self.In:.2f} А")
        print(f"  fн = {self.fn:.0f} Гц, p = {self.p}")
        print(f"  cos_phi = {self.cos_phi:.4f}, η = {self.eta * 100:.2f}%")
        print(f"  nном = {self.n_nom:.0f} об/мин, sн = {self.s_nom:.4f}")
        print("-" * 60)
        print(f"  R1 = {self.R1:.4f} Ом")
        print(f"  L1s = {self.L1sigma * 1e3:.4f} мГн")
        print(f"  R2' = {self.R2:.4f} Ом")
        print(f"  L2s' = {self.L2sigma * 1e3:.4f} мГн")
        print(f"  Lm = {self.Lm * 1e3:.3f} мГн")
        print(f"  J = {self.J:.3f} кг·м^2")
        print("=" * 60)


class AsyncGeneratorModel:
    """
    Модель трёхфазного асинхронного генератора
    в фазных координатах ABC.

    Вектор состояния y = [i1A, i1B, i1C, i2a, i2b, i2c, omega_r]
      i1A, i1B, i1C — фазные токи статора (A)
      i2a, i2b, i2c — приведённые фазные токи ротора (A)
      omega_r       — угловая скорость вращения ротора (рад/с)
    """

    def __init__(self, params: MachineParameters):
        self.p_obj = params
        self.R1 = params.R1
        self.R2 = params.R2
        self.L1s = params.L1sigma
        self.L2s = params.L2sigma
        self.Lm = params.Lm
        self.J = params.J
        self.p = params.p
        self.fn = params.fn
        self.omega1 = 2 * np.pi * params.fn  # Синхронная угловая частота поля

        self.L1 = self.L1s + self.Lm  # Полная индуктивность статора
        self.L2 = self.L2s + self.Lm  # Полная индуктивность ротора

        self.Um = params.Unom_phase * np.sqrt(2)  # Амплитуда фазного напряжения

    def _build_inductance_matrix(self):
        """
        Построение матрицы индуктивностей 6×6 для системы:
        [L1s+Lm    0      0    Lm   0    0  ] [di1A/dt]
        [  0    L1s+Lm    0     0   Lm   0  ] [di1B/dt]
        [  0       0   L1s+Lm   0    0   Lm ] [di1C/dt]
        [ Lm       0      0  L2s+Lm  0    0 ] [di2a/dt]
        [  0       Lm     0     0  L2s+Lm  0] [di2b/dt]
        [  0       0      Lm    0    0  L2s+Lm] [di2c/dt]

        Уравнение для i1A:
          U1A = R1*i1A + L1s*di1A/dt + Lm*(di1A/dt + di2a/dt)
              = R1*i1A + (L1s+Lm)*di1A/dt + Lm*di2a/dt

        Аналогично для ротора, плюс ЭДС вращения.
        """
        L1 = self.L1s + self.Lm
        L2 = self.L2s + self.Lm
        Lm = self.Lm

        L = np.array([
            [L1, 0, 0, Lm, 0, 0],
            [0, L1, 0, 0, Lm, 0],
            [0, 0, L1, 0, 0, Lm],
            [Lm, 0, 0, L2, 0, 0],
            [0, Lm, 0, 0, L2, 0],
            [0, 0, Lm, 0, 0, L2]
        ])
        return L

    def stator_voltages(self, t, freq=None, amplitude=None):
        """
        Трёхфазное синусоидальное напряжение статора.

        U1A = Um * sin(w1*t)
        U1B = Um * sin(w1*t - 2pi/3)
        U1C = Um * sin(w1*t + 2pi/3)
        """
        if freq is None:
            freq = self.fn
        if amplitude is None:
            amplitude = self.Um

        omega = 2 * np.pi * freq
        U1A = amplitude * np.sin(omega * t)
        U1B = amplitude * np.sin(omega * t - 2 * np.pi / 3)
        U1C = amplitude * np.sin(omega * t + 2 * np.pi / 3)
        return np.array([U1A, U1B, U1C])

    def rotor_emf(self, i2a, i2b, i2c, imA, imB, imC, omega_mech):
        """
        Противо-ЭДС в обмотках ротора от вращения.

        w2 = w_мех * p  (электрическая угловая скорость ротора)

        E_a = w2/sqrt(3) * [L2s'*(i2b - i2c) + Lm*(imB - imC)]
        E_b = w2/sqrt(3) * [L2s'*(i2c - i2a) + Lm*(imC - imA)]
        E_c = w2/sqrt(3) * [L2s'*(i2a - i2b) + Lm*(imA - imB)]
        """
        omega_e = omega_mech * self.p  # Электрическая угловая скорость
        sqrt3 = np.sqrt(3)

        Ea = omega_e / sqrt3 * (self.L2s * (i2b - i2c) + self.Lm * (imB - imC))
        Eb = omega_e / sqrt3 * (self.L2s * (i2c - i2a) + self.Lm * (imC - imA))
        Ec = omega_e / sqrt3 * (self.L2s * (i2a - i2b) + self.Lm * (imA - imB))

        return np.array([Ea, Eb, Ec])

    def electromagnetic_torque(self, i1A, i1B, i1C, i2a, i2b, i2c):
        """
        Электромагнитный момент (из формулы п.3.4):

        Mэм = p·Lm/sqrt(3) · [(i1A*i2c + i1B*i2a + i1C*i2b)
                          -(i1A*i2b + i1B*i2c + i1C*i2a)]

        Альтернативная форма:
        Mэм = sqrt(3)/2 * p*Lm * [i2a*(i1B - i1C) - i1A*(i2b - i2c)]
        """
        sqrt3 = np.sqrt(3)
        Mem = (self.p * self.Lm / sqrt3) * (
                (i1A * i2c + i1B * i2a + i1C * i2b)
                - (i1A * i2b + i1B * i2c + i1C * i2a)
        )
        return Mem

    def result_current_module(self, iA, iB, iC):
        """
        Модуль результирующего вектора тока (из МПСУ_ЭС_с_АМ.pdf):

        I = sqrt((in+1 - in+2)^2/3 + in^2)

        Для оси A: I = sqrt((iB - iC)^2/3 + iA^2)
        """
        return np.sqrt((iB - iC) ** 2 / 3.0 + iA ** 2)

    def result_voltage_module(self, uA, uB, uC):
        """Модуль результирующего вектора напряжения."""
        return np.sqrt((uB - uC) ** 2 / 3.0 + uA ** 2)

    def flux_linkage_phaseA(self, i1A, i1B, i1C, i2a, i2b, i2c):
        """
        Потокосцепление фазы A статора:
        psi1A = L'1A*i1A + Mм*[i1A - (i1B+i1C)/2] + Mм*[i2a - (i2b+i2c)/2]
        где Mм = 2*Lm/3
        """
        Mm = 2.0 * self.Lm / 3.0
        psi = (self.L1s * i1A
               + Mm * (i1A - (i1B + i1C) / 2.0)
               + Mm * (i2a - (i2b + i2c) / 2.0))
        return psi

    def ode_system(self, t, y, Mc_func, U_func=None):
        """
        Правая часть системы ОДУ.

        y = [i1A, i1B, i1C, i2a, i2b, i2c, omega_r]

        Система:
          [L]*[di/dt] = [U] - [R*i] - [EMF_rotation]
          J*dw/dt = Mэм - Mc

        Где для генератора (короткозамкнутый ротор): U2a = U2b = U2c = 0
        """
        i1A, i1B, i1C = y[0], y[1], y[2]
        i2a, i2b, i2c = y[3], y[4], y[5]
        omega_r = y[6]

        imA = i1A + i2a
        imB = i1B + i2b
        imC = i1C + i2c

        if U_func is not None:
            Us = U_func(t)
        else:
            Us = self.stator_voltages(t)

        Ur = np.array([0.0, 0.0, 0.0])
        E_rot = self.rotor_emf(i2a, i2b, i2c, imA, imB, imC, omega_r)

        b = np.zeros(6)
        b[0] = Us[0] - self.R1 * i1A
        b[1] = Us[1] - self.R1 * i1B
        b[2] = Us[2] - self.R1 * i1C
        b[3] = Ur[0] - self.R2 * i2a - E_rot[0]
        b[4] = Ur[1] - self.R2 * i2b - E_rot[1]
        b[5] = Ur[2] - self.R2 * i2c - E_rot[2]

        L_matrix = self._build_inductance_matrix()
        di_dt = np.linalg.solve(L_matrix, b)

        Mem = self.electromagnetic_torque(i1A, i1B, i1C, i2a, i2b, i2c)
        Mc = Mc_func(t, omega_r)

        domega_dt = (Mem - Mc) / self.J

        dydt = np.zeros(7)
        dydt[0:6] = di_dt
        dydt[6] = domega_dt

        return dydt


def simulate_generator_mode(params=None, t_end=2.0, dt_out=1e-4):
    """
    Моделирование генераторного режима асинхронной машины.
    Возвращает словарь временных рядов для анализа и построения графиков.
    """
    if params is None:
        params = MachineParameters()

    model = AsyncGeneratorModel(params)

    omega_sync = 2 * np.pi * params.fn / params.p  # рад/с

    print(f"\nСинхронная скорость: w1/p = {omega_sync:.2f} рад/с "
          f"({omega_sync * 60 / (2 * np.pi):.0f} об/мин)")

    Mc_max = -1200.0  # Отрицательный момент задает привод в генераторном режиме.
    t_ramp = 0.5  # Время нарастания момента привода, с

    def Mc_func(t, omega_r):
        """
        Момент нагрузки.
        Отрицательный момент — ускоряет ротор выше синхронной скорости.
        """
        if t < t_ramp:
            return Mc_max * (t / t_ramp)
        else:
            return Mc_max

    omega0 = omega_sync  # Начальная скорость = синхронная
    y0 = np.zeros(7)
    y0[6] = omega0

    print(f"\nЗапуск моделирования генераторного режима...")
    print(f"  Время моделирования: {t_end:.1f} с")
    print(f"  Момент привода: {Mc_max:.0f} Нм (нарастание за {t_ramp:.1f} с)")
    print(f"  Начальная скорость: {omega0:.2f} рад/с")

    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt_out)

    sol = solve_ivp(
        fun=lambda t, y: model.ode_system(t, y, Mc_func),
        t_span=t_span,
        y0=y0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
        max_step=1e-4
    )

    if not sol.success:
        print(f"  ОШИБКА: {sol.message}")
        return None

    print(f"  Решение получено. Точек: {len(sol.t)}")

    t = sol.t
    i1A = sol.y[0]
    i1B = sol.y[1]
    i1C = sol.y[2]
    i2a = sol.y[3]
    i2b = sol.y[4]
    i2c = sol.y[5]
    omega_r = sol.y[6]

    N = len(t)
    Mem = np.zeros(N)
    I1_mod = np.zeros(N)
    I2_mod = np.zeros(N)
    U1A_arr = np.zeros(N)
    U1B_arr = np.zeros(N)
    U1C_arr = np.zeros(N)
    U_mod = np.zeros(N)
    slip = np.zeros(N)
    Psi1A = np.zeros(N)
    imA_arr = np.zeros(N)
    imB_arr = np.zeros(N)
    imC_arr = np.zeros(N)
    Im_mod = np.zeros(N)

    for k in range(N):
        Mem[k] = model.electromagnetic_torque(
            i1A[k], i1B[k], i1C[k], i2a[k], i2b[k], i2c[k])
        I1_mod[k] = model.result_current_module(i1A[k], i1B[k], i1C[k])
        I2_mod[k] = model.result_current_module(i2a[k], i2b[k], i2c[k])

        Us = model.stator_voltages(t[k])
        U1A_arr[k] = Us[0]
        U1B_arr[k] = Us[1]
        U1C_arr[k] = Us[2]
        U_mod[k] = model.result_voltage_module(Us[0], Us[1], Us[2])

        omega_sync_t = 2 * np.pi * params.fn / params.p
        if abs(omega_sync_t) > 1e-10:
            slip[k] = (omega_sync_t - omega_r[k]) / omega_sync_t
        else:
            slip[k] = 0

        Psi1A[k] = model.flux_linkage_phaseA(
            i1A[k], i1B[k], i1C[k], i2a[k], i2b[k], i2c[k])

        imA_arr[k] = i1A[k] + i2a[k]
        imB_arr[k] = i1B[k] + i2b[k]
        imC_arr[k] = i1C[k] + i2c[k]
        Im_mod[k] = model.result_current_module(imA_arr[k], imB_arr[k], imC_arr[k])

    n_rpm = omega_r * 60 / (2 * np.pi)

    P_elec = (U1A_arr * i1A + U1B_arr * i1B + U1C_arr * i1C)
    P_mech = Mem * omega_r


    print("\n")
    print("  РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ (установившийся режим)")

    idx_ss = int(0.75 * N)

    print(f"  Скорость: {np.mean(n_rpm[idx_ss:]):.1f} об/мин")
    print(f"  Скольжение: {np.mean(slip[idx_ss:]):.5f}")
    print(f"  Мэм (среднее): {np.mean(Mem[idx_ss:]):.1f} Нм")
    print(f"  |I1| (среднее): {np.mean(I1_mod[idx_ss:]):.1f} А")
    print(f"  |I1| (max): {np.max(I1_mod[idx_ss:]):.1f} А")
    print(f"  |I2| (среднее): {np.mean(I2_mod[idx_ss:]):.1f} А")
    print(f"  |Im| (среднее): {np.mean(Im_mod[idx_ss:]):.1f} А")
    print(f"  |U| (среднее): {np.mean(U_mod[idx_ss:]):.1f} В")
    print(f"  P_элек (среднее): {np.mean(P_elec[idx_ss:]) / 1e3:.1f} кВт")
    print(f"  P_мех (среднее): {np.mean(P_mech[idx_ss:]) / 1e3:.1f} кВт")
    print("=" * 60)

    results = {
        't': t, 'i1A': i1A, 'i1B': i1B, 'i1C': i1C,
        'i2a': i2a, 'i2b': i2b, 'i2c': i2c,
        'omega_r': omega_r, 'n_rpm': n_rpm,
        'Mem': Mem, 'slip': slip,
        'I1_mod': I1_mod, 'I2_mod': I2_mod, 'Im_mod': Im_mod,
        'U1A': U1A_arr, 'U1B': U1B_arr, 'U1C': U1C_arr, 'U_mod': U_mod,
        'Psi1A': Psi1A, 'P_elec': P_elec, 'P_mech': P_mech,
        'imA': imA_arr, 'imB': imB_arr, 'imC': imC_arr,
        'model': model, 'params': params
    }
    return results


def simulate_motor_start(params=None, t_end=1.5, dt_out=1e-4):
    """
    Моделирование пуска асинхронного двигателя.

    Сценарий: прямой пуск от сети с постоянным моментом сопротивления.
    """
    if params is None:
        params = MachineParameters()

    model = AsyncGeneratorModel(params)
    Mc_load = 1105.0  # Номинальный момент нагрузки, Нм

    def Mc_func(t, omega_r):
        """Момент нагрузки: нарастает после 0.5 с (после разгона)."""
        if t < 0.8:
            return 50.0  # Минимальный момент трения при пуске
        else:
            return Mc_load * min(1.0, (t - 0.8) / 0.3) + 50.0

    y0 = np.zeros(7)
    y0[6] = 0.0  # Старт с нуля

    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt_out)

    print(f"\nМоделирование пуска АД...")

    sol = solve_ivp(
        fun=lambda t, y: model.ode_system(t, y, Mc_func),
        t_span=t_span,
        y0=y0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
        max_step=1e-4
    )

    if not sol.success:
        print(f"  ОШИБКА: {sol.message}")
        return None

    t = sol.t
    N = len(t)

    Mem = np.zeros(N)
    I1_mod = np.zeros(N)
    for k in range(N):
        Mem[k] = model.electromagnetic_torque(
            sol.y[0][k], sol.y[1][k], sol.y[2][k],
            sol.y[3][k], sol.y[4][k], sol.y[5][k])
        I1_mod[k] = model.result_current_module(
            sol.y[0][k], sol.y[1][k], sol.y[2][k])

    n_rpm = sol.y[6] * 60 / (2 * np.pi)
    omega_sync = 2 * np.pi * params.fn / params.p
    slip = (omega_sync - sol.y[6]) / omega_sync

    results = {
        't': t, 'i1A': sol.y[0], 'i1B': sol.y[1], 'i1C': sol.y[2],
        'i2a': sol.y[3], 'i2b': sol.y[4], 'i2c': sol.y[5],
        'omega_r': sol.y[6], 'n_rpm': n_rpm, 'slip': slip,
        'Mem': Mem, 'I1_mod': I1_mod,
        'model': model, 'params': params
    }
    return results

def plot_generator_results(res, save_path=None):
    """Построение графиков генераторного режима."""

    t = res['t']
    t_ms = t * 1000  # в мс для детальных графиков

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle('Результаты моделирования\n'
                 '(Трёхфазные координаты)',
                 fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(t, res['i1A'], 'b-', lw=0.5, label='i1A')
    ax.plot(t, res['i1B'], 'r-', lw=0.5, label='i1B')
    ax.plot(t, res['i1C'], 'g-', lw=0.5, label='i1C')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Ток, А')
    ax.set_title('Фазные токи статора i1A, i1B, i1C')
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, res['I1_mod'], 'b-', lw=0.6, label='|I1|')
    ax.plot(t, res['I2_mod'], 'r-', lw=0.6, label='|I2|')
    ax.plot(t, res['Im_mod'], 'g--', lw=0.6, label='|Im|')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Ток, А')
    ax.set_title('Модули результирующих токов')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(t, res['Mem'], 'b-', lw=0.6)
    ax.axhline(y=0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Момент, Нм')
    ax.set_title('Электромагнитный момент Mэм')

    ax = axes[1, 1]
    ax.plot(t, res['n_rpm'], 'b-', lw=0.8)
    omega_sync = 2 * np.pi * res['params'].fn / res['params'].p
    n_sync = omega_sync * 60 / (2 * np.pi)
    ax.axhline(y=n_sync, color='r', lw=0.8, ls='--', label=f'n синхр = {n_sync:.0f} об/мин')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Скорость, об/мин')
    ax.set_title('Частота вращения ротора')
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(t, res['slip'], 'b-', lw=0.6)
    ax.axhline(y=0, color='r', lw=0.8, ls='--', label='s = 0 (синхр.)')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Скольжение s')
    ax.set_title('Скольжение s(t)')
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    t_start = max(0, t[-1] - 0.06)  # Последние 60 мс
    mask = t >= t_start
    ax.plot(t[mask] * 1000, res['U1A'][mask], 'b-', lw=0.6, label='U1A')
    ax.plot(t[mask] * 1000, res['i1A'][mask] * 1.0, 'r-', lw=0.6, label='i1A')
    ax.set_xlabel('Время, мс')
    ax.set_ylabel('В / А')
    ax.set_title(f'Напряжение и ток фазы A (последние 60 мс)')
    ax.legend(fontsize=8)

    ax = axes[3, 0]
    ax.plot(t, res['Psi1A'], 'b-', lw=0.6)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('psi, Вб')
    ax.set_title('Потокосцепление фазы A статора psi1A')

    ax = axes[3, 1]
    ax.plot(t, res['P_elec'] / 1e3, 'b-', lw=0.5, label='P электр.')
    ax.plot(t, res['P_mech'] / 1e3, 'r-', lw=0.5, label='P мех.')
    ax.axhline(y=0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Мощность, кВт')
    ax.set_title('Электрическая и механическая мощность')
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nГрафик сохранён: {save_path}")

    plt.close(fig)
    return fig


def plot_motor_start_results(res, save_path=None):
    """Построение графиков пуска АД."""

    t = res['t']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ПУСК\n'
                 '(Прямой)',
                 fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(t, res['i1A'], 'b-', lw=0.5)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Ток, А')
    ax.set_title('Ток фазы A статора i1A')

    ax = axes[0, 1]
    ax.plot(t, res['I1_mod'], 'b-', lw=0.6)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('|I1|, А')
    ax.set_title('Модуль результирующего тока статора |I1|')

    ax = axes[1, 0]
    ax.plot(t, res['Mem'], 'b-', lw=0.5)
    ax.axhline(y=0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Момент, Нм')
    ax.set_title('Электромагнитный момент Mэм')

    ax = axes[1, 1]
    ax.plot(t, res['n_rpm'], 'b-', lw=0.8)
    params = res['params']
    n_sync = 60 * params.fn / params.p
    ax.axhline(y=n_sync, color='r', lw=0.8, ls='--', label=f'n₀ = {n_sync:.0f}')
    ax.axhline(y=params.n_nom, color='g', lw=0.8, ls='--', label=f'nн = {params.n_nom:.0f}')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Скорость, об/мин')
    ax.set_title('Частота вращения ротора')
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nГрафик сохранён: {save_path}")

    plt.close(fig)
    return fig


def plot_detailed_waveforms(res, save_path=None):
    """Детальные осциллограммы установившегося режима генератора."""

    t = res['t']
    t_start = max(0, t[-1] - 0.1)  # Последние 100 мс
    mask = t >= t_start
    tt = (t[mask] - t[mask][0]) * 1000  # мс от начала окна

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('ОСЦИЛЛОГРАММЫ УСТАНОВИВШЕГОСЯ ГЕНЕРАТОРНОГО РЕЖИМА\n'
                 '(последние 100 мс моделирования)',
                 fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(tt, res['i1A'][mask], 'b-', lw=0.8, label='i1A')
    ax.plot(tt, res['i1B'][mask], 'r-', lw=0.8, label='i1B')
    ax.plot(tt, res['i1C'][mask], 'g-', lw=0.8, label='i1C')
    ax.set_xlabel('Время, мс')
    ax.set_ylabel('Ток, А')
    ax.set_title('Токи статора (3 фазы)')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(tt, res['i2a'][mask], 'b-', lw=0.8, label='i2a')
    ax.plot(tt, res['i2b'][mask], 'r-', lw=0.8, label='i2b')
    ax.plot(tt, res['i2c'][mask], 'g-', lw=0.8, label='i2c')
    ax.set_xlabel('Время, мс')
    ax.set_ylabel('Ток, А')
    ax.set_title('Токи ротора (3 фазы)')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax2 = ax.twinx()
    ln1, = ax.plot(tt, res['U1A'][mask], 'b-', lw=0.8, label='U1A')
    ln2, = ax2.plot(tt, res['i1A'][mask], 'r-', lw=0.8, label='i1A')
    ax.set_xlabel('Время, мс')
    ax.set_ylabel('Напряжение, В', color='b')
    ax2.set_ylabel('Ток, А', color='r')
    ax.set_title('Напряжение U1A и ток i1A фазы A')
    ax.legend(handles=[ln1, ln2], fontsize=8)

    ax = axes[1, 1]
    ax.plot(tt, res['imA'][mask], 'b-', lw=0.8, label='imA')
    ax.plot(tt, res['imB'][mask], 'r-', lw=0.8, label='imB')
    ax.plot(tt, res['imC'][mask], 'g-', lw=0.8, label='imC')
    ax.set_xlabel('Время, мс')
    ax.set_ylabel('Ток, А')
    ax.set_title('Токи намагничивания im = i1 + i2')
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(tt, res['Psi1A'][mask], 'b-', lw=0.8)
    ax.set_xlabel('Время, мс')
    ax.set_ylabel('psi1_A, Вб')
    ax.set_title('Потокосцепление фазы A')

    ax = axes[2, 1]
    P_inst = (res['U1A'][mask] * res['i1A'][mask]
              + res['U1B'][mask] * res['i1B'][mask]
              + res['U1C'][mask] * res['i1C'][mask])
    ax.plot(tt, P_inst / 1e3, 'b-', lw=0.8)
    ax.axhline(y=np.mean(P_inst) / 1e3, color='r', ls='--', lw=1,
               label=f'Pср = {np.mean(P_inst) / 1e3:.1f} кВт')
    ax.set_xlabel('Время, мс')
    ax.set_ylabel('Мощность, кВт')
    ax.set_title('Мгновенная электрическая мощность')
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nГрафик сохранён: {save_path}")

    plt.close(fig)
    return fig



if __name__ == '__main__':

    print("═" * 62)

    params = MachineParameters()
    params.info()
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    print("пуск АД")

    res_motor = simulate_motor_start(params, t_end=2.5, dt_out=2e-4)
    if res_motor is not None:
        plot_motor_start_results(res_motor,
                                 save_path=os.path.join(output_dir, 'motor_start.png'))

        I1_max = np.max(res_motor['I1_mod'])
        M_max = np.max(res_motor['Mem'])
        print(f"\n  Проверка (пуск АД):")
        print(f"    |I1|макс = {I1_max:.0f} А "
              f"(кратность: {I1_max / params.In:.2f}, "
              f"расч: {params.Kip:.2f})")
        print(f"    Mмакс = {M_max:.0f} Нм "
              f"(кратность: {M_max / (params.P2n / (2 * np.pi * params.n_nom / 60)):.2f})")


    print("  ЭТАП 2: Генераторный режим")

    res_gen = simulate_generator_mode(params, t_end=2.0, dt_out=2e-4)
    if res_gen is not None:
        plot_generator_results(res_gen,
                               save_path=os.path.join(output_dir, 'generator_mode.png'))
        plot_detailed_waveforms(res_gen,
                                save_path=os.path.join(output_dir, 'generator_waveforms.png'))

    print(f"\nГрафики сохранены в папку: {output_dir}")
    print("   Готовые файлы:")
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"   - {f}")
