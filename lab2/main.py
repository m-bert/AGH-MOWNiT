import matplotlib.pyplot as plt
import numpy as np

# Getting Nodes


def chebyshevKthZero(k, n, a, b):
    return (1 / 2) * (a + b) + (1 / 2) * (a - b) * np.cos(np.pi * (2 * k - 1) / (2 * n))


def chebyshevZeros(n, a, b):
    return np.array([chebyshevKthZero(x, n, a, b) for x in range(1, n + 1)])


def rangeNodes(n, a, b):
    step = (b - a) / n
    return np.arange(a, b + step, step)


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


points = [(-9, 5), (-4, 2), (-1, -2), (7, 9)]

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


t = np.arange(-10, 8, 0.1)

plt.plot(t, L(t, points))
# plt.plot(t, N(t, points))

for point in points:
    plt.plot(point[0], point[1], marker="o", color="red")

plt.show()
