import matplotlib.pyplot as plt
import numpy as np

# Getting Nodes


def chebyshevKthZero(k, n, a, b):
    return (1 / 2) * (a + b) + (1 / 2) * (a - b) * np.cos(np.pi * (2 * k - 1) / (2 * n))


def chebyshevZeros(n, a, b):
    return np.array([chebyshevKthZero(x, n, a, b) for x in range(1, n + 1)])


def rangeNodes(n, a, b):
    # step = (b - a) / (n - 1)
    # return np.arange(a, b + step, step)
    return np.linspace(a, b, n)


#  Lagrange Polynomials


def l(j, x, zeros):
    k = len(zeros)
    value = 1

    for m in range(k):
        if m == j:
            continue

        value *= (x - zeros[m]) / (zeros[j] - zeros[m])

    return value


def L(x, points):
    k = len(points)
    zeros = [p[0] for p in points]

    value = 0

    for j in range(k):
        value += points[j][1] * l(j, x, zeros)

    return value


# Newton Polynomial


def dividedDifference(points, size):
    differences = [[0 for j in range(size)] for i in range(size)]

    for i in range(size):
        differences[i][0] = points[i][1]

    for j in range(1, size):
        for i in range(size - j):
            differences[i][j] = (differences[i + 1][j - 1] - differences[i][j - 1]) / (
                points[i + j][0] - points[i][0]
            )

    return differences[0][size - 1]


def n(j, x, zeros):
    if j == 0:
        return 1

    value = 1

    for i in range(j):
        value *= x - zeros[i]

    return value


def N(x, points):
    k = len(points)
    zeros = [p[0] for p in points]

    value = 0

    for j in range(k):
        # print(dividedDifference(points, j + 1))
        # print(n(j, x, zeros))
        value += dividedDifference(points, j + 1) * n(j, x, zeros)

    return value


# TESTING


def f(x):
    return np.sin(2 * x) * np.sin(x**2 / np.pi)


a = 0
b = 3 * np.pi
t = np.arange(a, b, 0.01)

interpolations = [L, N]
nodesPositions = [chebyshevZeros, rangeNodes]
nodes = [3, 4, 5, 7, 10, 15, 20]

for interpolation in interpolations:
    for nodePosition in nodesPositions:
        for nodeAmount in nodes:
            methodName = "Lagrange" if interpolation == L else "Newton"
            nodesPosition = (
                "równomierny" if nodePosition == rangeNodes else "zer Czebyszewa"
            )

            X = nodePosition(nodeAmount, a, b)
            points = [(x, f(x)) for x in X]

            if nodeAmount == 10 and nodePosition == rangeNodes:
                print(points)

            plt.title(
                f"Interpolacja {methodName}'a rozkład {nodesPosition} n = {nodeAmount}"
            )
            plt.grid()
            plt.plot(t, interpolation(t, points), color="blue")
            plt.plot(t, f(t), color="green")

            for point in points:
                plt.plot(point[0], point[1], marker="o", color="red")

            plt.savefig(f"./screens/{methodName}_{nodesPosition}_{nodeAmount}.jpg")

            plt.clf()
