import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sin(2 * x) * np.sin(x**2 / np.pi)


def derivative(x):
    return (2 * x * np.sin(2 * x) * np.cos(x**2 / np.pi)) / np.pi + 2 * np.sin(
        x**2 / np.pi
    ) * np.cos(2 * x)


A = 0
B = 3 * np.pi
T = np.linspace(A, B, 1000)


def chebyshevKthZero(k, n, a, b):
    return (1 / 2) * (a + b) + (1 / 2) * (a - b) * np.cos(np.pi * (2 * k - 1) / (2 * n))


def chebyshevZeros(n, a, b):
    return np.array([chebyshevKthZero(x, n, a, b) for x in range(1, n + 1)])


def rangeNodes(n, a, b):
    return np.linspace(a, b, n)


def getPoints(nodesPositionFn, n, a, b):
    X = nodesPositionFn(n, a, b)

    P = [(x, [f(x), derivative(x)]) for x in X]

    return P


def dividedDifference(points, size):
    differences = [[0 for _ in range(size)] for _ in range(size)]
    derivatives_amount = 2

    n = len(points)
    index = 0

    for i in range(n):
        for _ in range(derivatives_amount):
            differences[index][0] = points[i][1][0]

            index += 1

            if index >= size:
                break

        if index >= size:
            break

    for j in range(1, size):
        for i in range(size - j):
            starting_index = (i + j) // derivatives_amount
            variable_index = i // derivatives_amount

            if points[starting_index][0] - points[variable_index][0] == 0:
                differences[i][j] = points[variable_index][1][j] / np.math.factorial(j)
            else:
                differences[i][j] = (
                    differences[i + 1][j - 1] - differences[i][j - 1]
                ) / (points[starting_index][0] - points[variable_index][0])

    return differences[0][size - 1]


def h(j, x, Z):
    if j == 0:
        return 1

    value = 1

    for i in range(j):
        value *= x - Z[i]

    return value


def H(x, points):
    derivatives_amount = 2
    n = len(points)
    s = 0

    Z = []

    for i in range(n):
        for j in range(derivatives_amount):
            Z.append(points[i][0])

    value = 0

    for i in range(n):
        for j in range(derivatives_amount):
            value += dividedDifference(points, s + j + 1) * h(s + j, x, Z)

        s += derivatives_amount

    return value


points = getPoints(chebyshevZeros, 15, A, B)

print(points)


# plt.title(f"Interpolacja Hermite'a")
# plt.grid()
# plt.plot(T, H(T, points), color="blue")
# plt.plot(T, f(T), color="green")
# for point in points:
#     plt.plot(point[0], point[1][0], marker="o", color="red")
# plt.show()

nodesPositions = [chebyshevZeros, rangeNodes]
nodes = [3, 4, 5, 7, 10, 15, 20]


def generate_plots():
    for nodePosition in nodesPositions:
        for nodeAmount in nodes:
            methodName = "Hermit"
            nodesPosition = (
                "równomierny" if nodePosition == rangeNodes else "zer Czebyszewa"
            )

            X = nodePosition(nodeAmount, A, B)
            points = [(x, f(x)) for x in X]
            interpolationNodes = getPoints(nodePosition, nodeAmount, A, B)

            plt.title(
                f"Interpolacja {methodName}'a rozkład {nodesPosition} n = {nodeAmount}"
            )
            plt.grid()
            plt.plot(T, H(T, interpolationNodes), color="blue")
            plt.plot(T, f(T), color="green")

            for point in points:
                plt.plot(point[0], point[1], marker="o", color="red")

            plt.savefig(
                f"./hermit_screens/{methodName}_{nodesPosition}_{nodeAmount}.jpg"
            )

            plt.clf()

    return


generate_plots()
