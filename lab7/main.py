import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

START = 0
END = 3 * np.pi
INTERVAL_LENGTH = END - START

EPS = 10**(-5)

#  FUNCTION


def f(x):
    return np.sin(2 * x) * np.sin(x**2 / np.pi)


def draw_function():
    T = np.linspace(START, END, 10**6)

    plt.title(
        f"f(x) = sin(2x) * sin(x^2/pi)")
    plt.grid()
    plt.plot(T, f(T), color="green")

    P = []

    for t in T:
        y = f(t)

        if abs(y) < EPS:
            P.append((t, y))

    for p in P:
        plt.plot(p[0], p[1], marker="o", color="red")

    plt.show()

    return


draw_function()
