import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def test_data(k: float = 1.0, b: float = 0.1, half_disp: float = 0.05, n: int = 100, x_step: float = 0.01) -> \
        (np.ndarray, np.ndarray):
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
    return (x, np.asarray([k * xi + b + half_disp for xi in x]))


def test_data_2d(kx: float = -2.0, ky: float = 2.0, b: float = 12.0, half_disp: float = 1.01, n: int = 100,
                 x_step: float = 0.01, y_step: float = 0.01) -> (np.ndarray, np.ndarray, np.ndarray):
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
    x = np.arange(-(n * x_step) / 2, (n * x_step) / 2, x_step)
    y = np.arange(-(n * y_step) / 2, (n * y_step) / 2, y_step)
    return (x, y, np.asarray([kx * xi + ky * yi + b + half_disp for xi, yi in zip(x, y)]))


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
    return sum((yi - (k * xi + b)) ** 2 for xi, yi in zip(x.flat, y.flat)) ** 0.5


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


def linear_regression(x: np.ndarray, y: np.ndarray) -> (float, float):
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
    k = 1
    b = 2
    return (k, b)


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
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b+ b^2\n
    ei^2 =\n
    zi^2 - 2*zi*yi*ky - 2*zi*xi*kx - 2*zi*b + (yi*ky)^2 + 2*xi*kx*yi*ky + 2*b*yi*ky + (xi*kx)^2 + 2*xi*kx*b + b^2\n\n
  
	d Σei^2 /dkx = Σ-zi*xi + ky*xi*yi + kx*xi^2 + xi*b = 0\n
    d Σei^2 /dky = Σ-zi*yi + ky*yi^2 + kx*xi*yi + b*yi = 0\n
    d Σei^2 /db  = Σ-zi + yi*ky + xi*kx = 0\n\n

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
                      | Σ-zi + yi*ky + xi*kx                |\n\n
 
	Окончательно решение:\n
    |kx|   |1|\n
    |ky| = |1| -  H(1, 1, 0)^-1 * grad(1, 1, 0)\n
    | b|   |0|\n

    :param x: массив значений по x
    :param y: массив значений по y
    :param z: массив значений по z
    :returns: возвращает тройку (kx, ky, b), которая является решением задачи (Σ(zi - (yi * ky + xi * kx + b))^2)->min
    """
    pass
	
	
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
    pass
	
def quadratic_regression_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
	"""
	x^T * A * x = 0 ...
	"""
pass


def polynom(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    :param x: массив значений по x
    :param b: массив коэффициентов полинома
    :returns: возвращает полином yi = Σxi^j*bj
    """
    return sum(xi ** j * bj for j, xi, bj in enumerate(zip(x, b)))


def distance_field_test(k: int, b: int):
    """
    Функция проверки поля расстояний:
    1) Посчитать тестовыe x и y используя функцию test_data
    2) Задать интересующие нас диапазоны k и b (np.linspace...)
    3) Рассчитать поле расстояний (distance_field) и вывести в виде изображения.
    4) Проанализировать результат (смысл этой картинки в чём...)
    :return:
    """
    x, y = test_data(k=k, b=b)
    k_range = np.linspace(-10, 10, 100)
    b_range = np.linspace(-10, 10, 100)
    Z = distance_field(x, y, k_range, b_range)

    X, Y = np.meshgrid(k_range, b_range)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax.set_xlabel('b')
    ax.set_ylabel('k')
    ax.set_zlabel('dist')
    ax.set_title(f'Distance field for k = {k} and b = {b}')

    plt.show()
    


def linear_reg_test():
    """
    Функция проверки работы метода линейной регрессии:
    1) Посчитать тестовыe x и y используя функцию test_data
    2) Получить с помошью linear_regression значения k и b
    3) Вывести на графике x и y в виде массива точек и построить
       регрессионную прямую вида y = k*x + b
    :return:
    """
    pass


def bi_linear_reg_test():
    """
    Функция проверки работы метода билинейной регрессии:
    1) Посчитать тестовыe x, y и z используя функцию test_data_2d
    2) Получить с помошью bi_linear_regression значения kx, ky и b
    3) Вывести на трёхмерном графике x, y и z в виде массива точек и построить
       регрессионную плоскость вида z = kx*x + ky*y + b
    :return:
    """
    pass


def poly_reg_test():
    """
    Функция проверки работы метода полиномиальной регрессии:
    1) Посчитать тестовыe x, y используя функцию test_data
    2) Посчитать набор коэффициентов bi полинома y = Σx^i*bi используя функцию poly_regression
    3) Вывести на графике x и y в виде массива точек и построить
       регрессионную кривую. Для построения кривой использовать метод polynom
    :return:
    """
    pass


if __name__ == "__main__":
    # distance_field_test(k=5, b=5)
    # linear_reg_test()
    # bi_linear_reg_test()
    # poly_reg_test()
    # x, y, z = test_data_2d(kx=1, ky=1)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    #                    linewidth=0, antialiased=False)
    # # fig = plt.figure(figsize=(16, 9))
    # ax = plt.axes(projection='3d')
    # ax.plot3D(x, y, z, 'gray')
    # plt.show()
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # # Make data.
    # X = np.arange(-5, 5, 0.25)
    # Y = np.arange(-5, 5, 0.25)
    # X, Y = np.meshgrid(X, Y)
    
    # R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)
    # print(len(X))
    # print(len(Y))
    # print(len(Z))

    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                     linewidth=0, antialiased=False)

    # plt.show()
