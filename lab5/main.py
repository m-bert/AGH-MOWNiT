import numpy as np
import matplotlib.pyplot as plt

START = 0
END = 3 * np.pi

#  FUNCTION


def f(x):
    return np.sin(2 * x) * np.sin(x**2 / np.pi)


def w(x):
    return 1


# NODES


def chebyshevKthZero(k, n, a, b):
    return (1 / 2) * (a + b) + (1 / 2) * (a - b) * np.cos(np.pi * (2 * k - 1) / (2 * n))


def chebyshevZeros(n, a, b):
    return np.array([chebyshevKthZero(x, n, a, b) for x in range(1, n + 1)])


def evenSpace(n, a, b):
    return np.linspace(a, b, n)


def getPoints(nodesPositionFn, n, a, b):
    X = nodesPositionFn(n, a, b)

    P = [(x, f(x)) for x in X]

    return P

# APPROXIMATION


def calculateCoefficientsMatrix(m, points):
    n = len(points)

    G = [[0 for _ in range(m+1)] for _ in range(m+1)]

    for row in range(m+1):
        for column in range(m+1):
            for i in range(n):
                G[row][column] += w(points[i][0]) * \
                    np.power(points[i][0], row + column)

    B = [0 for _ in range(m+1)]

    for row in range(m+1):
        for i in range(n):
            B[row] += w(points[i][0]) * points[i][1] * \
                np.power(points[i][0], row)

    A = np.linalg.solve(G, B)

    return A


def approx(x, m, A):
    return np.sum([A[i] * x**i for i in range(m+1)])


def LSA(X, m, A):
    return [approx(x, m, A) for x in X]


# VISUALIZATION


def draw_custom_plot(n, START, END, nodesPosition, m):
    t = np.linspace(START, END, 1000)

    P = getPoints(nodesPosition, n, START, END)
    A = calculateCoefficientsMatrix(m, P)

    plt.title(f"Chuj")
    plt.grid()
    plt.plot(t, LSA(t, m, A), color="blue")
    plt.plot(t, f(t), color="green")

    for point in P:
        plt.plot(point[0], point[1], marker="o", color="red")

    plt.show()

    return


draw_custom_plot(50, START, END, evenSpace, 6)
