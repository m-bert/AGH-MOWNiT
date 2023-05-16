import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class StopCondition(Enum):
    DIFF = 0
    ABS = 1


START = -1.6
END = 0.9

EPS_ZERO = 10**(-20)

ITERATIONS = 1000


#  FUNCTION

def f(x):
    return np.power(x, 2) - 35 * np.power(np.sin(x), 11)


def f_derivative(x):
    return 2*x + - 385 * np.power(np.sin(x), 10) * np.cos(x)


def secant_method(x0, x1, stop_condition, RO):
    for i in range(ITERATIONS):
        diff = f(x1) - f(x0)

        if np.abs(diff) < EPS_ZERO:
            return [None, None]

        x2 = x1 - f(x1) * (x1 - x0) / diff
        x0, x1 = x1, x2

        if (stop_condition == StopCondition.DIFF and np.abs(x0 - x1) < RO) \
                or (stop_condition == StopCondition.ABS and np.abs(f(x1)) < RO):

            return [x2, i+1]

    return [None, None]


def Newton_Raphson(x0, stop_condition, RO):
    for i in range(ITERATIONS):
        y_prime = f_derivative(x0)

        if np.abs(y_prime) < EPS_ZERO:
            return [None, None]

        x = x0 - f(x0) / y_prime

        if (stop_condition == StopCondition.DIFF and np.abs(x0 - x) < RO) \
                or (stop_condition == StopCondition.ABS and np.abs(f(x)) < RO):

            return [x, i+1]

        x0 = x

    return [None, None]


def perform_test(method, current_value, stop_condition, ro, left_hand_side):

    if method == Newton_Raphson:
        result = Newton_Raphson(current_value, stop_condition, ro)
    else:
        if left_hand_side:
            result = secant_method(START, current_value, stop_condition, ro)
        else:
            result = secant_method(current_value, END, stop_condition, ro)

    if result[0] is None:
        result = [-(10.0)**10, -1]
    else:
        result[0] = '{:.20f}'.format(result[0])

    method_name = "Newtona" if method == Newton_Raphson else "Siecznych"

    str_result = f"{method_name} {current_value} {result[0]} {result[1]} {ro} {stop_condition}\n"

    print(str_result)

    with open("results.txt", "a") as f:
        f.write(str_result)

    return


criterias = [StopCondition.ABS, StopCondition.DIFF]
ro_values = [10**(-9), 10**(-12), 10**(-15)]
step = 0.1

for criteria in criterias:
    for ro in ro_values:
        current_value = START

        while (current_value <= END):
            perform_test(Newton_Raphson, current_value, criteria, ro, None)
            perform_test(secant_method, current_value, criteria, ro, True)
            perform_test(secant_method, current_value, criteria, ro, False)

            current_value += step
            current_value = round(current_value, 2)


def draw_function():
    EPS = 10**(-5)

    T = np.linspace(START, END, 10**6)

    plt.title(
        f"f'(x) = 2x - 385sin^10(x)cos(x)")
    plt.grid()
    plt.plot(T, f_derivative(T), color="blue")

    P = []

    for t in T:
        y = f_derivative(t)

        if abs(y) < EPS:
            P.append((t, y))

    for p in P:
        plt.plot(p[0], p[1], marker="o", color="red")

    plt.show()

    return


# draw_function()
