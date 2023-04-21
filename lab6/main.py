import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

START = 0
END = 3 * np.pi
INTERVAL_LENGTH = END - START

#  FUNCTION


def f(x):
    return np.sin(2 * x) * np.sin(x**2 / np.pi)


# NODES

def evenSpace(n, a, b):
    return np.linspace(a, b, n)


def getPoints(nodesPositionFn, n, a, b):
    X = nodesPositionFn(n, a, b)

    P = [(x, f(x)) for x in X]

    return P

# APPROXIMATION


def scale_point(x):

    x /= INTERVAL_LENGTH
    x *= 2 * np.pi
    x -= np.pi + (2*np.pi * START / INTERVAL_LENGTH)

    return x


def calculate_coefficients(points):
    def calculate_A(points):
        return [2 * sum([p[1] * np.cos(j * p[0]) for p in points]) / len(points)
                for j in range(len(points))]

    def calculate_B(points):
        return [2 * sum([p[1] * np.sin(j * p[0]) for p in points]) / len(points)
                for j in range(len(points))]

    scaled_points = [(scale_point(p[0]), p[1]) for p in points]

    A = calculate_A(scaled_points)
    B = calculate_B(scaled_points)

    return A, B


def approx(x, m, A, B):
    scaled_x = scale_point(x)

    return (A[0]/2) + sum(A[j] * np.cos(j * scaled_x) + B[j] * np.sin(j * scaled_x) for j in range(1, m+1))


def LSA_Trig(X, m, A, B):
    return [approx(x, m, A, B) for x in X]


def draw_custom_plot(n, START, END, nodesPosition, m):
    T = np.linspace(START, END, 1000)

    P = getPoints(nodesPosition, n, START, END)
    A, B = calculate_coefficients(P)

    plt.title(
        f"Aproksymacja średniokwadratowa wielomianami algebraicznymi, n={n}, m={m}")
    plt.grid()
    plt.plot(T, LSA_Trig(T, m, A, B), color="blue")
    plt.plot(T, f(T), color="green")

    for point in P:
        plt.plot(point[0], point[1], marker="o", color="red")

    plt.show()

    return


def generatePlots(nodesPosition, N, M):
    T = np.linspace(START, END, 1000)

    for n in N:
        for m in M:
            if m >= n:
                continue

            P = getPoints(nodesPosition, n, START, END)
            A, B = calculate_coefficients(P)

            plt.title(
                f"Aproksymacja średniokwadratowa, n={n}, m={m}")
            plt.grid()
            plt.plot(T, LSA_Trig(T, m, A, B), color="blue")
            plt.plot(T, f(T), color="green")

            for point in P:
                plt.plot(point[0], point[1], marker="o", color="red")

            plt.savefig(
                f"./plots/LSA_{n}n_{m}m.jpg"
            )

            plt.clf()

    return


def getErrors(nodesPosition, N, M):
    ERROR_POINTS = 1000
    T = np.linspace(START, END, ERROR_POINTS)

    for n in N:
        for m in M:
            if m >= n:
                continue

            P = getPoints(nodesPosition, n, START, END)
            A, B = calculate_coefficients(P)

            max_error_fn = np.vectorize(
                lambda x: np.abs(f(x) - approx(x, m, A, B))
            )

            avg_error_fn = np.vectorize(
                lambda x: (f(x) - approx(x, m, A, B)) ** 2
            )

            max_error = np.max(max_error_fn(T))
            avg_error = (np.sqrt(np.sum(avg_error_fn(T)))) / (ERROR_POINTS-1)

            variant = f"LSA_TRIG_{n}n_{m}m"

            print(max_error, avg_error, variant)

    return


# draw_custom_plot(100, START, END, evenSpace, 10)

N = [4, 10, 15, 20, 30, 50, 75, 100]
M = [2, 4, 6, 10]

getErrors(evenSpace, N, M)
generatePlots(evenSpace, N, M)
