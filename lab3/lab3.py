import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

eps = 1e-3
n = 100 / eps
x = np.linspace(1, 1 + eps, 100)
print(x)
y = np.log(x)
plt.semilogx(x, y)
plt.semilogx(x ** (-n), -n * y)
plt.semilogx(x ** n, n * y)
plt.xlabel('$x$')
plt.ylabel('$y=\ln x$')
plt.show()


def relative_error(x0, x): return np.abs(x0 - x) / np.abs(x0)


def log_teylor_series(x, n=5):
    print(n)
    a = x - 1
    a_k = a  # x в степени k. Сначала k=1
    y = a  # Значене логарифма, пока для k=1.
    for k in range(2, n):  # сумма по степеням
        a_k = -a_k * a  # последовательно увеличиваем степень и учитываем множитель со знаком
        y = y + a_k / k
    return y


def get_n(eps):
    return (1 - eps) / eps


x0 = np.logspace(-5, 5, 1000, dtype=np.double)
epsilon = np.finfo(np.float).eps
best_precision = (epsilon / 2) * np.abs(1. / np.log(x0))

x = np.logspace(-5, 1, 1001)
y0 = np.log(x)
y = log_teylor_series(x, n=int(get_n(10e-3)))
plt.loglog(x, relative_error(y0, y), '-k')
plt.loglog(x0, best_precision, '--r')
plt.xlabel('$x$')
plt.ylabel('$(y-y_0)/y_0$')
plt.legend(["$Достиг.\;погрешность$", "$Мин.\;погрешность$"], loc=5)
plt.show()

from math import pi

import matplotlib.pyplot as plt
import numpy as np


def log_teylor_series(x, N=5):
    print(N)
    a = x - 1
    a_k = a  # x в степени k. Сначала k=1
    y = a  # Значене логарифма, пока для k=1.
    for k in range(2, N):  # сумма по степеням
        a_k = -a_k * a  # последовательно увеличиваем степень и учитываем множитель со знаком
        y = y + a_k / k
    return y


def get_N(eps):
    return (1 - eps) / eps


# Узлы итерполяции
N = 5
xn = 1 + 1. / (1 + np.arange(N))
zn = np.arange(N)
print(zn)
un = 3 * (xn - 1) / (2 * (1 + xn))


def opt_u(a, b, zn):
    return (a + b) / 2 + (b - a) / 2 * np.cos(((2 * zn + 1) * pi) / (2 * (zn + 1)))


opt = opt_u(0, 1, zn)
print(opt)
print(xn)
print(un)
yn = np.log(xn)
yyn = np.log(un)
optyn = np.log(opt)

x = np.linspace(xn[4], xn[0], 1000)
xu = np.linspace(un[4], un[0], 1000)
xuu = np.linspace(opt[4], opt[0], 1000)
y = np.log(x)
yy = np.log(xu)
yyy = np.log(xuu)

L = scipy.interpolate.lagrange(xn, yn)
L1 = scipy.interpolate.lagrange(un, yyn)
Lopt = scipy.interpolate.lagrange(opt, optyn)
yl = L(x)
y2 = L1(xu)
y3 = Lopt(xuu)
plt.plot(x, y, '-k')
plt.plot(xn, yn, '.b')
plt.plot(x, yl, '-r')
# plt.plot(xu, y2, '-g')
# plt.plot(x, y3)
plt.xlabel("$x$")
plt.ylabel("$y=\ln x$")
plt.show()

plt.plot(xu, yy, '-k')
plt.plot(un, yyn, '.b')
plt.plot(xu, y2, '-g')
plt.xlabel("$x$")
plt.ylabel("$y=\ln x$")
plt.show()

plt.plot(xuu, yyy, '-k')
plt.plot(opt, optyn, '.b')
plt.plot(xuu, y3, '-g')
plt.xlabel("$x$")
plt.ylabel("$y=\ln x$")
plt.show()

plt.semilogy(x, relative_error(y, yl), '-r')
plt.semilogy(xu, relative_error(yy, y2), '-g')
plt.semilogy(x, relative_error(yyy, y3), '-b')
plt.xlabel("$Аргумент$")
plt.ylabel("$Относ.\;погрешность$")
plt.show()


def log_newton(x, eps=6.66133814776e-16):
    y = 1  # начальное приближение
    while True:
        diff = -1 + x / np.exp(y)
        '''print(diff)
        print('______________________________')'''
        a, b = min(diff), max(diff)
        m = max(abs(a), abs(b))
        # print(m, np.sqrt(2*np.exp(-10)*eps/np.exp(10)))
        if m < eps:
            break
        y = y - 1 + x / np.exp(y)
    return y


x = np.logspace(-3, 3, 1000)
y0 = np.log(x)
y = log_newton(x)
plt.loglog(x, relative_error(y0, y), '-k')
plt.xlabel("$Аргумент$")
plt.ylabel("$Относ.\;погрешность$")
plt.show()
