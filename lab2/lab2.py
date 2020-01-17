import numpy as np
import matplotlib.pyplot as plt

base = 10  # параметр, может принимать любые целые значения > 1


def exact_sum(k):
    """Точное значение суммы всех элементов."""
    l = np.sin((k + 1) / 2) * np.sin(k / 2) / np.sin(0.5)
    return l


def sins(k):
    """"Элементы выборки sin n от 1 до k"."""
    x = np.empty(k)
    for i in range(k):
        x[i] = np.sin(i + 1)
    return x


def direct_sum(x):
    """Последовательная сумма всех элементов вектора x"""
    s = 0.
    for e in x:
        s += e
    return s


def number_of_samples(K):
    """Число элементов в выборке"""
    return np.sum([base ** k for k in range(0, K)])


def exact_mean(K):
    """Значение среднего арифметического по выборке с близкой к машинной точностью."""
    return 1. / number_of_samples(K)


def exact_variance(K):
    """Значение оценки дисперсии с близкой к машинной точностью."""
    # разные значения элементов выборки
    values = np.asarray([float(base) ** (-k) / K for k in range(0, K)], dtype=np.double)
    # сколько раз значение встречается в выборке
    count = np.asarray([base ** k for k in range(0, K)])
    return np.sum(count * (values - exact_mean(K)) ** 2) / number_of_samples(K)


k = 1111111  # число слагаемых
x = sins(k)  # сохраняем выборку в массив
print("Число элементов:", len(x))
print("Самое маленькое и большое значения:", np.min(x), np.max(x))
print(x)

exact_sum_for_x = exact_sum(k)  # значение суммы с близкой к машинной погрешностью
print(exact_sum_for_x)
direct_sum_for_x = direct_sum(x)  # сумма всех элементов по порядку


def relative_error(x0, x):
    """Погрешность x при точном значении x0"""
    return np.abs(x0 - x) / np.abs(x)


print("Погрешность прямого суммирования:", relative_error(exact_sum_for_x, direct_sum_for_x))

sorted_x = x[np.argsort(x)]
sorted_sum_for_x = direct_sum(sorted_x)
print("Погрешность суммирования по возрастанию:", relative_error(exact_sum_for_x, sorted_sum_for_x))

sorted_x = x[np.argsort(x)[::-1]]
sorted_sum_for_x = direct_sum(sorted_x)
print("Погрешность суммирования по убыванию:", relative_error(exact_sum_for_x, sorted_sum_for_x))


def Kahan_sum(x):
    s = 0.0  # частичная сумма
    c = 0.0  # сумма погрешностей
    for i in x:
        y = i - c  # первоначально y равно следующему элементу последовательности
        t = s + y  # сумма s может быть велика, поэтому младшие биты y будут потеряны
        c = (t - s) - y  # (t-s) отбрасывает старшие биты, вычитание y восстанавливает младшие биты
        s = t  # новое значение старших битов суммы
    return s


Kahan_sum_for_x = Kahan_sum(x)  # сумма всех элементов по порядку
print("Погрешность суммирования по Кэхэну:", relative_error(exact_sum_for_x, Kahan_sum_for_x))

# параметры выборки
mean = 1e6 # среднее
delta = 1e-5  # величина отклонения от среднего


def samples(N_over_two):
    """Генерирует выборку из 2*N_over_two значений с данным средним и среднеквадратическим
    отклонением."""
    x = np.full((2 * N_over_two,), mean, dtype=np.double)
    x[:N_over_two] += delta
    x[N_over_two:] -= delta
    return np.random.permutation(x)


def exact_mean():
    """Значение среднего арифметического по выборке с близкой к машинной точностью."""
    return mean


def exact_variance():
    """Значение оценки дисперсии с близкой к машинной точностью."""
    return delta ** 2


print("Погрешность суммирования по Кэхэну для массива по возрастанию:",
      relative_error(exact_sum_for_x, Kahan_sum(sorted(x))))
print("Погрешность суммирования по Кэхэну для массива по убыванию:",
      relative_error(exact_sum_for_x, Kahan_sum(sorted(x, reverse=True))))

x = samples(1000000)

print("Размер выборки:", len(x))
print("Среднее значение:", exact_mean())
print("Оценка дисперсии:", exact_variance())
print("Ошибка среднего для встроенной функции:", relative_error(exact_mean(), np.mean(x)))
print("Ошибка дисперсии для встроенной функции:", relative_error(exact_variance(), np.var(x)))


def direct_mean(x):
    """Среднее через последовательное суммирование."""
    return direct_sum(x) / len(x)


print("Ошибка среднего для последовательного суммирования:", relative_error(exact_mean(), direct_mean(x)))


def direct_second_var(x):
    """Вторая оценка дисперсии через последовательное суммирование."""
    return direct_mean(x ** 2) - direct_mean(x) ** 2


def online_second_var(x):
    """Вторая оценка дисперсии через один проход по выборке"""
    m = x[0]  # накопленное среднее
    m2 = x[0] ** 2  # накопленное среднее квадратов
    for n in range(1, len(x)):
        m = (m * (n - 1) + x[n]) / n
        m2 = (m2 * (n - 1) + x[n] ** 2) / n
    return m2 - m ** 2


print("Ошибка второй оценки дисперсии для последовательного суммирования:",
      relative_error(exact_variance(), direct_second_var(x)))
print("Ошибка второй оценки дисперсии для однопроходного суммирования:",
      relative_error(exact_variance(), online_second_var(x)))


def direct_first_var(x):
    """Первая оценка дисперсии через последовательное суммирование."""
    return direct_mean((x - direct_mean(x)) ** 2)


print("Ошибка первой оценки дисперсии для последовательного суммирования:",
      relative_error(exact_variance(), direct_first_var(x)))
