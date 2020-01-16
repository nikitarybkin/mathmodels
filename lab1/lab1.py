import numpy as np
import matplotlib.pyplot as plt


def relative_error(x0, x): return np.abs(x0 - x) / np.abs(x0)


eps = np.finfo(np.double).eps
print("Машинная точность:", eps)

x0 = np.logspace(-4, 4, 100, dtype=np.double)


print(1 < np.inf)


def plot_error(x0, err):
    mask = np.logical_and(err > 0, err < np.inf)
    plt.loglog(x0[mask], err[mask], ".k")
    plt.loglog(x0, [eps] * len(err), "--r")  # машинная точность для сравнения
    plt.xlabel("$Значение\;аргумента$")
    plt.ylabel("$Относительная\;погрешность$")
    plt.show()


def f_sqrt_sqr(x, n=52):
    for k in range(n): x = np.sqrt(x)
    for k in range(n): x = x * x
    return x


x = f_sqrt_sqr(x0)
err = relative_error(x0, x)
plot_error(x0, err)


class LogNumber(object):
    def __init__(self, zeta):
        """Конструктор принимает zeta, но обьект соответствует числу двоичный логарифм от zeta."""
        self.zeta = zeta

    def __str__(self):
        """На экран выводится значение x, которое может быть менее точно,
        чем храниемое значение."""
        return "{}".format(self.to_float())

    def from_float(x):
        """Создает число со значением, равным x."""
        return LogNumber(np.log2(x))

    def to_float(self):
        """Преобразует число в формат с плавающей запятой"""
        return pow(2, self.zeta)

    def __mul__(self, other):
        """Перезагрузка операции умножения."""
        return LogNumber(self.zeta + other.zeta)

    def __pow__(self, power, modulo=None):
        "Перезагрузка степени."
        return LogNumber(power * self.zeta)


def f_sqrt_sqr_log(x, n=52):
    x1=LogNumber.from_float(x)
    for k in range(n): x1 = x1.__pow__(0.5)
    for k in range(n): x1 = x1 * x1
    return x1.to_float()


x = f_sqrt_sqr_log(x0)
err = relative_error(x0, x)
plot_error(x0, err)

print("Число с плавающей запятой:", np.pi)
pi = LogNumber.from_float(np.pi)
print("Наше представление числа: ", pi)
print("Квадрат в арифметике с плавающей запятой:", np.pi * np.pi)
print("Квадрат в нашем представлении:           ", pi * pi)
print(pi.__pow__(1/2)*pi.__pow__(1/2))

