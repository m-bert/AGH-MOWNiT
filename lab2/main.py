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


def L(x, zeros, values):
    k = len(zeros)
    value = 0

    for j in range(k):
        value += values[j] * l(j, x, zeros)

    return value


# zeros = [-9, -4, -1, 7]
# values = [5, 2, -2, 9]

# t = np.arange(-9, 7, 0.1)
# plt.plot(t, L(t, zeros, values))
# plt.show()
