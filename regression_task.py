import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple
import numpy as np


def test_data(k: float = 1.0, b: float = 0.1, half_disp: float = 0.05, n: int = 100, x_step: float = 0.01) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Генерируюет линию вида y = k*x + b + dy, где dy - аддитивный шум с амплитудой half_disp
    :param k: наклон линии
    :param b: смещение по y
    :param half_disp: амплитуда разброса данных
    :param n: количество точек
    :param x_step: шаг между соседними точками
    :return: кортеж значенией по x и y
    """
    x = np.arange(-(n * x_step) / 2, (n * x_step) / 2, x_step)
    return x, np.asarray([k * xi + b + np.random.normal(scale=half_disp) for xi in x])

def test_data_2d(kx: float = -2.0, ky: float = 2.0, b: float = 12.0, half_disp: float = 1.01, n: int = 100,
                 x_step: float = 0.01, y_step: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерирует плоскость вида z = kx*x + ky*y + b + dz, где dz - аддитивный шум с амплитудой half_disp
    :param kx: наклон плоскости по x
    :param ky: наклон плоскости по y
    :param b: смещение по z
    :param half_disp: амплитуда разброса данных
    :param n: количество точек
    :param x_step: шаг между соседними точками по х
    :param y_step: шаг между соседними точками по y
    :returns: кортеж значенией по x, y и z
    """
    # ??? 🤔
    # x = np.arange(-(n * x_step) / 2, (n * x_step) / 2, x_step)
    # y = np.arange(-(n * y_step) / 2, (n * y_step) / 2, y_step)
    x = np.random.rand(n)
    y = np.random.rand(n)
    return x, y, np.asarray([kx * xi + ky * yi + b + np.random.normal(scale=half_disp) for xi, yi in zip(x, y)])

def test_data_nd(k: np.ndarray = [1, 2, 3], b: float = 12, dim = 3, half_disp: float = 1.01, n: int = 100):
    points = np.asarray([np.random.randn(n) for _ in range(dim)])

    f = []
    for row in range(n):
        res = 0
        for i in range(dim):
            res += k[i] * points[i, row]
        f.append(res + b + np.random.normal(scale=half_disp))

    data_rows = []
    for row in range(n):
        curr = []
        for i in range(dim):
            curr.append(points[i, row])
        curr.append(f[row])
        data_rows.append(curr)
   
    return np.asarray(data_rows)

def distance_sum(x: np.ndarray, y: np.ndarray, k: float, b: float) -> float:
    """
    Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b при фиксированных k и b
    по формуле: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
    :param x: массив значений по x
    :param y: массив значений по y
    :param k: значение параметра k (наклон)
    :param b: значение параметра b (смещение)
    :returns: F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5
    """
    return np.asarray(sum((yi - (k * xi + b)) ** 2 for xi, yi in zip(x.flat, y.flat)) ** 0.5)

def distance_field(x: np.ndarray, y: np.ndarray, k: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Вычисляет сумму квадратов расстояний от набора точек до линии вида y = k*x + b, где k и b являются диапазонами
    значений. Формула расстояния для j-ого значения из набора k и k-ого значения из набора b:
    F(k_j, b_k) = (Σ(yi -(k_j * xi + b_k))^2)^0.5 (суммирование по i)
    :param x: массив значений по x
    :param y: массив значений по y
    :param k: массив значений параметра k (наклоны)
    :param b: массив значений параметра b (смещения)
    :returns: поле расстояний вида F(k, b) = (Σ(yi -(k * xi + b))^2)^0.5 (суммирование по i)
    """
    return np.asarray([[distance_sum(x, y, ki, bi) for bi in b.flat] for ki in k.flat])

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Линейная регрессия.\n
    Основные формулы:\n
    yi - xi*k - b = ei\n
    yi - (xi*k + b) = ei\n
    (yi - (xi*k + b))^2 = yi^2 - 2*yi*(xi*k + b) + (xi*k + b)^2 = ei^2\n
    yi^2 - 2*(yi*xi*k + yi*b) + (xi^2 * k^2 + 2 * xi * k * b + b^2) = ei^2\n
    yi^2 - 2*yi*xi*k - 2*yi*b + xi^2 * k^2 + 2 * xi * k * b + b^2 = ei^2\n
    d ei^2 /dk = - 2*yi*xi + 2 * xi^2 * k + 2 * xi * b = 0\n
    d ei^2 /db = - 2*yi + 2 * xi * k + 2 * b = 0\n\n
    
	d ei^2 /dk = (yi - xi * k - b) * xi = 0\n
    d ei^2 /db =  yi - xi * k - b = 0\n\n
    
	Σ(yi - xi * k - b) * xi = 0\n
    Σ yi - xi * k - b = 0\n\n
    
	Σ(yi - xi * k - b) * xi = 0\n
    Σ(yi - xi * k) = n * b\n\n
    
	Σyi - k * Σxi = n * b\n
    Σxi*yi - xi^2 * k - xi*b = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*b = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*(Σyi - k * Σxi) / n = 0\n
    Σxi*yi - Σxi^2 * k - Σxi*Σyi / n + k * (Σxi)^2 / n = 0\n
    Σxi*yi - Σxi*Σyi / n + k * ((Σxi)^2 / n - Σxi^2)  = 0\n
    Σxi*yi - Σxi*Σyi / n = -k * ((Σxi)^2 / n - Σxi^2)\n
    (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n) = k\n
    окончательно:\n
    k = (Σxi*yi - Σxi*Σyi / n) / (Σxi^2 - (Σxi)^2 / n)\n
    b = (Σyi - k * Σxi) /n\n
    :param x: массив значений по x
    :param y: массив значений по y
    :returns: возвращает пару (k, b), которая является решением задачи (Σ(yi -(k * xi + b))^2)->min
    """
    x_summ = x.sum()
    y_summ = y.sum()
    xx_summ = (x * x).sum()
    xy_summ = (x * y).sum()
    one_over_n = 1.0 / x.size
    k = (xy_summ - x_summ * y_summ * one_over_n) / (xx_summ - x_summ * x_summ * one_over_n) 
    b = (y_summ - k * x_summ) * one_over_n

    #k = (sum([xi * yi for xi, yi in zip(x.flat, y.flat)]) - sum(x.flat) * sum(y.#flat) / len(x.flat)) / (sum([xi * xi for xi in x.flat]) - sum(x.flat) ** 2 / #len(x.flat))
    # b = (sum(y.flat) - k * sum(x.flat)) / len(x.flat)
    return k, b

def bi_linear_regression(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> (float, float, float):
    """
    Билинейная регрессия.\n
    Основные формулы:\n
    zi - (yi * ky + xi * kx + b) = ei\n
    zi^2 - 2*zi*(yi * ky + xi * kx + b) + (yi * ky + xi * kx + b)^2 = ei^2\n
    ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + ((yi*ky)^2 + 2 * (xi*kx*yi*ky + b*yi*ky) + (xi*kx + b)^2)\n
    ei^2 = zi^2 - 2*yi*zi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx + b)^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
  
	d Σei^2 /dkx = Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b = 0\n
    d Σei^2 /dky = Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi = 0\n
    d Σei^2 /db  = Σ-zi + yi*ky + xi*kx + b = 0\n\n

    d Σei^2 /dkx / dkx = Σ xi^2\n
    d Σei^2 /dkx / dky = Σ xi*yi\n
    d Σei^2 /dkx / db  = Σ xi\n\n

    d Σei^2 /dky / dkx = Σ xi*yi\n
    d Σei^2 /dky / dky = Σ yi^2\n
    d Σei^2 /dky / db  = Σ yi\n\n

    d Σei^2 /db / dkx = Σ xi\n
    d Σei^2 /db / dky = Σ yi\n
    d Σei^2 /db / db  = n\n\n

    Hesse matrix:\n
    || d Σei^2 /dkx / dkx;  d Σei^2 /dkx / dky;  d Σei^2 /dkx / db ||\n
    || d Σei^2 /dky / dkx;  d Σei^2 /dky / dky;  d Σei^2 /dky / db ||\n
    || d Σei^2 /db  / dkx;  d Σei^2 /db  / dky;  d Σei^2 /db  / db ||\n\n

    Hesse matrix:\n
                   | Σ xi^2;  Σ xi*yi; Σ xi |\n
    H(kx, ky, b) = | Σ xi*yi; Σ yi^2;  Σ yi |\n
                   | Σ xi;    Σ yi;    n    |\n\n
    
                      | Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b |\n
    grad(kx, ky, b) = | Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi |\n
                      | Σ-zi + yi*ky + xi*kx + b            |\n\n
 
  
    Окончательно решение:\n
    |kx|   |1|\n
    |ky| = |1| -  H(1, 1, 0)^-1 * grad(1, 1, 0)\n
    | b|   |0|\n

    :param x: массив значений по x
    :param y: массив значений по y
    :param z: массив значений по z
    :returns: возвращает тройку (kx, ky, b), которая является решением задачи (Σ(zi - (yi * ky + xi * kx + b))^2)->min
    """
    kx = 1
    ky = 1
    b = 0
    x_s = x.sum()
    y_s = y.sum()
    xy_s = (x * y).sum()
    xx_s = (x * x).sum()
    yy_s = (y * y).sum()
    zx_s = (z * x).sum()
    zy_s = (z * y).sum()
    z_s  = z.sum()
    
    hessian = np.array([[xx_s, xy_s, x_s],
                        [xy_s, yy_s, y_s],
                        [x_s, y_s, x.size]])
    """
                      | Σ-zi * xi + xi * yi + xi^2     |\n
    grad(kx, ky, b) = | Σ-zi * yi + yi^2    + xi * yi  |\n
                      | Σ-zi      + yi      + xi           |\n\n
 
    """
    grad = np.array([-zx_s + xy_s + xx_s,
                     -zy_s + yy_s + xy_s,
                     -z_s + y_s + x_s])
    print(np.array((1, 1, 0)) - np.linalg.inv(hessian) @ grad)
    
    return np.array((1, 1, 0)) - np.linalg.inv(hessian) @ grad
   

    H = np.linalg.inv(np.matrix([
        [sum([xi ** 2 for xi in x.flat]), sum([xi * yi for xi, yi in zip(x.flat, y.flat)]), sum(x.flat)],
        [sum([xi * yi for xi, yi in zip(x.flat, y.flat)]), sum([yi ** 2 for yi in y.flat]), sum(y.flat)],
        [sum(x.flat), sum(y.flat), len(x.flat)]
        ]))

    grad = np.matrix([
            [sum([-1 * zi * xi + ky * xi * yi + kx * xi ** 2 + xi * b for xi, yi, zi in zip(x.flat, y.flat, z.flat)])],
            [sum([-1 * zi * yi + ky * yi ** 2 + kx * xi * yi + yi * b for xi, yi, zi in zip(x.flat, y.flat, z.flat)])],
            [sum([-1 * zi + yi * ky + xi * kx + b for xi, yi, zi in zip(x.flat, y.flat, z.flat)])]
            ])
        
    res = np.matrix([[kx], [ky], [b]]) - H @ grad

    return (res[0, 0], res[1, 0], res[2, 0])
		
def n_linear_regression(data_rows: np.ndarray) -> np.ndarray:
    """
    H_ij = Σx_i * x_j, i in [0, rows - 1] , j in [0, rows - 1]
    H_ij = Σx_i, j = rows i in [rows, :]
    H_ij = Σx_j, j in [:, rows], i = rows

           | Σkx * xi^2    + Σky * xi * yi + b * Σxi - Σzi * xi|\n
    grad = | Σkx * xi * yi + Σky * yi^2    + b * Σyi - Σzi * yi|\n
           | Σyi * ky      + Σxi * kx                - Σzi     |\n

    x_0 = [1,...1, 0] =>

           | Σ xi^2    + Σ xi * yi - Σzi * xi|\n
    grad = | Σ xi * yi + Σ yi^2    - Σzi * yi|\n
           | Σxi       + Σ yi      - Σzi     |\n

    :param data_rows:  состоит из строк вида: [x_0,x_1,...,x_n, f(x_0,x_1,...,x_n)]
    :return:
    """
    rows, cols = data_rows.shape

    x_0 = np.ones(shape=(cols, 1))
    x_0[-1] = 0

    H = np.zeros(shape=(cols, cols))
    
    for row in range(cols):
        for col in range(cols):
            if (row == cols - 1):
                H[row, col] = sum(data_rows[:, col])
            elif (col == cols - 1):
                H[row, col] = sum(data_rows[:, row])
            else:   
                H[row, col] = sum(data_rows[:, row] * data_rows[:, col])
    H[cols - 1, cols - 1] = rows
    
    grad = np.zeros(shape=(cols, 1))

    for row in range(cols):
        for col in range(cols):
            if col == cols - 1:
                grad[row] -= sum(data_rows[:, -1] * data_rows[:, row]) if row != cols - 1 else sum(data_rows[:, -1])
            else:
                grad[row] += sum(data_rows[:, row] * data_rows[:, col]) if row != cols - 1 else sum(data_rows[:, col])

    return np.asarray(x_0 - np.linalg.inv(H) @ grad).flat
    
def poly_regression(x: np.ndarray, y: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Полином: y = Σ_j x^j * bj
    Отклонение: ei =  yi - Σ_j xi^j * bj
    Минимизируем: Σ_i(yi - Σ_j xi^j * bj)^2 -> min
    Σ_i(yi - Σ_j xi^j * bj)^2 = Σ_iyi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2
    условие минимума: d/dbj Σ_i ei = d/dbj (Σ_i yi^2 - 2 * yi * Σ_j xi^j * bj +(Σ_j xi^j * bj)^2) = 0
    :param x: массив значений по x
    :param y: массив значений по y
    :param order: порядок полинома
    :return: набор коэффициентов bi полинома y = Σx^i*bi
    """
    # https://habr.com/ru/articles/414245/ 😎
    # A = np.zeros((order, order))
    # C = np.zeros((order, 1))
    # n = x.size

    # for row in range(order):
    #     C[row] = sum(y * x ** row) / n

    # A[0, 0] = 1
    # prev = 1
    # for col in range(1, order):
    #     A[0, col] = sum(prev * x) / n
    #     prev *= x

    # for row in range(1, order):
    #     for col in range(0, order):
    #         if col == order - 1:
    #             A[row, col] = sum(prev * x) / n
    #             prev *= x
    #         else:
    #             A[row, col] = A[row - 1, col + 1]

    # return np.linalg.inv(A) @ C

    # https://math.stackexchange.com/questions/2572460/2d-polynomial-regression-with-condition 😎
    # X = np.ones(shape=(x.size, order))
    # for row in range(x.size):
    #     for col in range(order):
    #         X[row, col] = x[row] ** col
    # return np.linalg.inv(X.T @ X) @ X.T @ y

    # 2 строчки 😎
    X = np.asarray([[xi ** col for col in range(order)] for xi in x])
    return np.linalg.inv(X.T @ X) @ X.T @ y

def quadratic_regression_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray, order: int = 5) -> np.ndarray:
    """
    https://math.stackexchange.com/questions/2572460/2d-polynomial-regression-with-condition God bless 🙏 
    """
    # A = [1, x, y, x ** 2, x * y, y ** 2, ...]
    # A = []
    # for xi, yi in zip(x, y):
    #     row = []
    #     for power in range(order):
    #         for i in range(power + 1):
    #             row.append(xi ** (power - i) * yi ** i)
    #     A.append(row)
    # A = np.asarray(A)
    
    # 2 строчки 😎
    A = np.asarray([[xi ** (power - i) * yi ** i for power in range(order) for i in range(power + 1)] for xi, yi in zip(x, y)])
    return np.linalg.inv(A.T @ A) @ A.T @ z   

def polynom(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    :param x: массив значений по x
    :param b: массив коэффициентов полинома
    :returns: возвращает полином yi = Σxi^j*bj
    """
    return np.asarray([sum(xi ** j * bj for j, bj in enumerate(b)) for xi in x])

def distance_field_test() -> None:
    """
    Функция проверки поля расстояний:
    1) Посчитать тестовыe x и y используя функцию test_data
    2) Задать интересующие нас диапазоны k и b (np.linspace...)
    3) Рассчитать поле расстояний (distance_field) и вывести в виде изображения.
    4) Проанализировать результат (смысл этой картинки в чём...)
    :return:
    """
    k = 1
    b = -0.5
    x, y = test_data(k=k, b=b)
    k_range = np.linspace(-10, 10, 100)
    b_range = np.linspace(-10, 10, 100)
    Z = distance_field(x, y, k_range, b_range)

    X, Y = np.meshgrid(k_range, b_range)

    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax.set_xlabel('b')
    ax.set_ylabel('k')
    ax.set_zlabel('dist')
    ax.set_title(f'Distance field for k = {k} and b = {b}')

    plt.show() 

def linear_reg_test() -> None:
    """
    Функция проверки работы метода линейной регрессии:
    1) Посчитать тестовыe x и y используя функцию test_data
    2) Получить с помошью linear_regression значения k и b
    3) Вывести на графике x и y в виде массива точек и построить
       регрессионную прямую вида y = k*x + b
    :return:
    """
    k_true = 2.3
    b_true = -3.5
    
    x, y = test_data(k=k_true, b=b_true)
    k_pred, b_pred = linear_regression(x=x, y=y)
    
    plt.figure(figsize=(16, 9))
    plt.scatter(x, y, color="red")
    plt.xlabel('x')
    plt.ylabel('y')

    y_pred = k_pred * x + b_pred
    plt.plot(x, y_pred, color="blue")
    plt.title(f"k_true = {k_true} and b_true = {b_true}, prediction k_pred = {k_pred} and b_pred = {b_pred}")

    plt.show()

def bi_linear_reg_test() -> None:
    """
    Функция проверки работы метода билинейной регрессии:
    1) Посчитать тестовыe x, y и z используя функцию test_data_2d
    2) Получить с помошью bi_linear_regression значения kx, ky и b
    3) Вывести на трёхмерном графике x, y и z в виде массива точек и построить
       регрессионную плоскость вида z = kx*x + ky*y + b
    :return:
    """
    kx = 4
    ky = 2
    b = 11

    X, Y, Z_true = test_data_2d(kx=kx, ky=ky, b=b)

    kx_pred, ky_pred, b_pred = bi_linear_regression(X, Y, Z_true)

    _, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.scatter(X, Y, Z_true, c="blue")

    X, Y = np.meshgrid(X, Y)
    Z_pred = kx_pred * X + ky_pred * Y + b_pred

    ax.plot_surface(X, Y, Z_pred, cmap="Reds")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(f"z_true = {kx:.2f} * x + {ky:.2f} * y + {b:.2f} and\n z_pred = {kx_pred:.2f} * x + {ky_pred:.2f} * y + {b_pred:.2f}")

    plt.show()

def n_linear_reg_test() -> None:
    """
    Функция проверки работы метода регрессии произвольного размера:
    """
    k = [2, -3, 5, 6, 8]
    b = 5
    dim = len(k)
    data_rows = test_data_nd(k=k, dim=dim, b=b)
    pred = n_linear_regression(data_rows)
    print(f"f\t = {''.join([f'{ki:.2f} * x_{i} + ' for i, ki in enumerate(k)])}{b}")
    print(f"f_pred   = {''.join([f'{ki:.2f} * x_{i} + ' for i, ki in enumerate(pred[:-1])])}{pred[-1]}")

def poly_reg_test() -> None:
    """
    Функция проверки работы метода полиномиальной регрессии:
    1) Посчитать тестовыe x, y используя функцию test_data
    2) Посчитать набор коэффициентов bi полинома y = Σx^i*bi используя функцию poly_regression
    3) Вывести на графике x и y в виде массива точек и построить
       регрессионную кривую. Для построения кривой использовать метод polynom
    :return:
    """
    x, y = test_data()
    b = poly_regression(x, y)

    y_pred = polynom(x, b)
    plt.figure(figsize=(16, 9))
    plt.scatter(x, y, color="red")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot(x, y_pred, color="blue")
    plt.title(f"F(x) = {''.join([f'{bi:.4f} * x ^ {i} + ' for i, bi in enumerate(b.flat)])[:-2]}")

    plt.show()

def quadratic_regression_2d_test() -> None:
    """Very cool function 😎™"""
    x, y, z = test_data_2d(half_disp=0)
    order = 5
    b = quadratic_regression_2d(x, y, z, order=order)
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.scatter(x, y, z, c="blue")

    X, Y = np.meshgrid(x, y)

    Z_pred = b[0]
    idx = 1
    for power in range(1, order):
        for i in range(power + 1):
            Z_pred += b[idx] * (X ** (power - i) * Y ** i)
            idx += 1
       

    ax.plot_surface(X, Y, Z_pred, cmap="Reds")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

if __name__ == "__main__":
    # distance_field_test()
    # linear_reg_test()
    # bi_linear_reg_test()
    # n_linear_reg_test()
    # poly_reg_test()
    # quadratic_regression_2d_test()
    pass
