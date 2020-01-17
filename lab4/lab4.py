import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def EulerIntegrator(h, y0, f):
    return y0 + h * f(y0)


def oneStepErrorPlot(f, y, integrator):
    eps = np.finfo(float).eps
    steps = np.logspace(-10, 0, 50)  # шаги интегрирования
    y0 = y(0)  # начальное значение
    yPrecise = [y(t) for t in steps]  # точные значения решения
    yApproximate = [integrator(t, y0, f) for t in steps]  # приближенные решения
    h = [np.maximum(np.max(np.abs(yp - ya)), eps) for yp, ya in zip(yPrecise, yApproximate)]
    plt.loglog(steps, h, '-')
    plt.xlabel(u"Шаг интегрирования")
    plt.ylabel(u"Погрешность одного шага")


def firstOrderPlot():
    ax = plt.gca()
    steps = np.asarray(ax.get_xlim())
    plt.loglog(steps, steps, '--r')


def integrate(N, delta, f, y0, integrator):
    for n in range(N):
        y0 = integrator(delta, y0, f)
    return y0


def intervalErrorPlot(f, y, integrator, T=1, maxNumberOfSteps=1000, numberOfPointsOnPlot=16):
    eps = np.finfo(float).eps
    numberOfSteps = np.logspace(0, np.log10(maxNumberOfSteps), numberOfPointsOnPlot).astype(np.int)
    steps = T / numberOfSteps  # шаги интегрирования
    y0 = y(0)  # начальное значение
    yPrecise = y(T)  # точнре значения решения на правом конце
    yApproximate = [integrate(N, T / N, f, y0, integrator) for N in numberOfSteps]  # приближенные решения
    h = [np.maximum(np.max(np.abs(yPrecise - ya)), eps) for ya in yApproximate]
    plt.loglog(steps, h, '.-')
    plt.xlabel("Шаг интегрирования")
    plt.ylabel("Погрешность интегрования на интервале")


def NewtonIntegrator(h, y0, f):
    return y0 + h * f[0](y0) + f[0](y0) * f[1](y0) * h * h / 2


def ModifiedEulerIntegrator(h, y0, f):
    yIntermediate = y0 + f(y0) * h / 2
    return y0 + h * f(yIntermediate)


def RungeKuttaIntegrator(h, y0, f):
    k1 = f(y0)
    k2 = f(y0 + k1 * h / 2)
    k3 = f(y0 + k2 * h / 2)
    k4 = f(y0 + k3 * h)
    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6


def NewtonMethod(F, x0):
    for i in range(100):  # ограничиваем максимальное число итераций
        x = x0 - F[0](x0) / F[1](x0)
        if x == x0: break  # достигнута максимальная точность
        x0 = x
    return x0


def BackwardEulerIntegrator(h, y0, f):
    F = (lambda y: y0 + h * f[0](y) - y, lambda y: h * f[1](y) - 1)
    return NewtonMethod(F, y0)


alpha = -10
f = (lambda y: alpha * y, lambda y: alpha)
yExact = lambda t: np.exp(alpha * t)

# Решение методом Эйлера.
f = (lambda y: np.cos(y), lambda y: 2 * np.arctan(np.tanh((0 + c) / 2)))
# Аналитическое решение
c = 2 * np.arctanh(np.tan(1 / 2))
yExact = lambda t: 2 * np.arctan(np.tanh((t + c) / 2))

intervalErrorPlot(f[0], yExact, EulerIntegrator)
intervalErrorPlot(f, yExact, NewtonIntegrator)
intervalErrorPlot(f[0], yExact, ModifiedEulerIntegrator)
intervalErrorPlot(f[0], yExact, RungeKuttaIntegrator)
intervalErrorPlot(f[0], yExact, RungeKuttaIntegrator)
firstOrderPlot()
plt.legend([u"Метод Эйлера", u"Метод Ньютона", u"Мод. Эйлер", u"Метод Рунге-Кутты",
            u"Первый порядок"], loc=2)
plt.show()

# Строим график ошибок
oneStepErrorPlot(f[0], yExact, EulerIntegrator)
oneStepErrorPlot(f[0], yExact, ModifiedEulerIntegrator)
oneStepErrorPlot(f[0], yExact, RungeKuttaIntegrator)
firstOrderPlot()
plt.legend([u"Метод Эйлера", u"Мод. Эйлер", u"Метод Рунге-Кутты", u"Первый порядок"], loc=2)
plt.show()


# начальные условия


def f(t, u):
    return -u


def exact(u0, du0, t):
    # analytical solution
    return u0 * math.cos(t) + du0 * math.sin(t)


def iterate(func, u, v, tmax, n):
    dt = tmax / (n - 1)
    t = 0.0

    for i in range(n):
        u, v = func(u, v, t, dt)
        t += dt

    return u


def euler_iter(u, v, t, dt):
    v_new = v + dt * f(t, u)
    u_new = u + dt * v
    return u_new, v_new


def rk_iter(u, v, t, dt):
    k1 = f(t, u)
    k2 = f(t + dt * 0.5, u + k1 * 0.5 * dt)
    k3 = f(t + dt * 0.5, u + k2 * 0.5 * dt)
    k4 = f(t + dt, u + k3 * dt)

    v += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # v doesn't explicitly depend on other variables
    k1 = k2 = k3 = k4 = v

    u += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return u, v


euler = lambda u, v, tmax, n: iterate(euler_iter, u, v, tmax, n)
runge_kutta = lambda u, v, tmax, n: iterate(rk_iter, u, v, tmax, n)


def plot_result(u, v, tmax, n):
    dt = tmax / (n - 1)
    t = 0.0
    allt = []
    error_euler = []
    error_rk = []
    r_exact = []
    r_euler = []
    r_rk = []

    u0 = u_euler = u_rk = u
    v0 = v_euler = v_rk = v

    for i in range(n):
        u = exact(u0, v0, t)
        u_euler, v_euler = euler_iter(u_euler, v_euler, t, dt)
        u_rk, v_rk = rk_iter(u_rk, v_rk, t, dt)
        allt.append(t)
        error_euler.append(abs(u_euler - u))
        error_rk.append(abs(u_rk - u))
        r_exact.append(u)
        r_euler.append(u_euler)
        r_rk.append(u_rk)
        t += dt

    _plot("error.png", "Ошибка", "Время t", "Ошибка e", allt, error_euler, error_rk, u_euler)
    # _plot("result.png", "Result", "time t", "u(t)", allt, r_euler, r_rk, r_exact)


def _plot(out, title, xlabel, ylabel, allt, euler, rk, r, exact=None):
    plt.title(title)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.plot(allt, euler, 'b-', label="Эйлер")
    plt.plot(allt, rk, 'r--', label="Рунге-Кутта")

    if exact:
        plt.plot(allt, exact, 'g.', label='Точный')

    plt.legend(loc=4)
    plt.grid(True)

    plt.savefig(out, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False)
    plt.show()


u0 = 1
du0 = v0 = 0
tmax = 10.0
n = 2000

print("Приближения:")
print("время t=", tmax)
print("методом Эйлера =", euler(u0, v0, tmax, n))
print("методом Рунге-Кутты =", runge_kutta(u0, v0, tmax, n))
print("точное значение =", exact(u0, v0, tmax))

plot_result(u0, v0, tmax * 2, n * 2)
