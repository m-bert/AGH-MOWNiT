import numpy as np
import matplotlib.pyplot as plt

#  FUNCTION


def f(x):
    return np.sin(2 * x) * np.sin(x**2 / np.pi)

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

# CUBIC INTERPOLATION


def delta(degree, i, points):
    if degree == 1:
        return (points[i+1][1] - points[i][1])/(points[i+1][0] - points[i][0])
    elif degree == 2:
        return (delta(1, i+1, points) - delta(1, i, points)) / (points[i+1][0] - points[i-1][0])
    elif degree == 3:
        return (delta(2, i+1, points) - delta(2, i, points)) / (points[i+2][0] - points[i-1][0])


def h(points, i):
    return points[i+1][0] - points[i][0]


def prepareMatrices(points, type):
    n = len(points)

    hMatrix = np.zeros((n, n))
    deltaMatrix = np.zeros((n, 1))

    for i in range(1, n-1):
        hMatrix[i][i-1] = h(points, i-1)
        hMatrix[i][i] = 2 * (h(points, i-1) + h(points, i))
        hMatrix[i][i+1] = h(points, i)

        deltaMatrix[i] = delta(1, i, points) - delta(1, i-1, points)

    if type == "cubic":
        hMatrix[0][0] = -h(points, 0)
        hMatrix[0][1] = h(points, 0)

        hMatrix[n-1][n-2] = h(points, n-2)
        hMatrix[n-1][n-1] = -h(points, n-2)

        deltaMatrix[0] = np.power(h(points, 0), 2) * delta(3, 0, points)
        deltaMatrix[n-1] = -np.power(h(points, n-2), 2) * delta(3, n-4, points)

        sigmaMatrix = np.linalg.solve(hMatrix, deltaMatrix)

    elif type == "natural":
        hMatrix[0][0] = 1
        hMatrix[n-1][n-1] = 1

        sigmaMatrix = np.linalg.solve(hMatrix, deltaMatrix)

    return sigmaMatrix


def binarySearch(p, q, x, points):
    if p > q:
        return p-1

    mid = (p + q)//2

    if mid == len(points):
        return mid - 1

    if x >= points[mid][0]:
        return binarySearch(mid + 1, q, x, points)
    else:
        return binarySearch(p, mid - 1, x, points)


def s(x, points, sigmaMatrix):
    n = len(points)

    i = min(binarySearch(0, n, x, points), n-2)
    b = (points[i+1][1] - points[i][1])/h(points, i) - \
        h(points, i) * (sigmaMatrix[i+1] + 2*sigmaMatrix[i])
    c = 3 * sigmaMatrix[i]
    d = (sigmaMatrix[i+1] - sigmaMatrix[i]) / h(points, i)

    return points[i][1] + b * (x - points[i][0]) + c * np.power((x-points[i][0]), 2) + d * np.power((x - points[i][0]), 3)


def S(X, points, sigmaMatrix):
    return [s(x, points, sigmaMatrix) for x in X]


# QUADRATIC INTERPOLATION

def gamma(i, points):
    return delta(1, i-1, points)


def a(type, i, points):
    return (b(type, i+1, points) - b(type, i, points)) / (2*(points[i+1][0] - points[i][0]))


def b(type, i, points):
    if i == 0:
        return 0 if type == "natural" else gamma(1, points)

    return 2 * gamma(i, points) - b(type, i-1, points)


def q(x, points, type):
    n = len(points)

    i = min(binarySearch(0, n, x, points), n-2)

    _a = a(type, i, points)
    _b = b(type, i, points)

    return _a * np.power((x - points[i][0]), 2) + _b * (x - points[i][0]) + points[i][1]


def Q(X, points, type):
    return [q(x, points, type) for x in X]

# UTILS


def draw_custom_plot(n, A, B, nodesPosition):
    t = np.linspace(A, B, 1000)

    nodesPositionName = (
        "równomierny" if nodesPosition == evenSpace else "zer Czebyszewa"
    )

    points = getPoints(nodesPosition, n, A, B)
    sigmaMatrix = prepareMatrices(points, "cubic")

    plt.title(
        f"Interpolacja sześcienna rozkład {nodesPositionName} n = {n}"
    )
    plt.grid()
    plt.plot(t, Q(t, points, "natural"), color="blue")
    plt.plot(t, f(t), color="green")

    for point in points:
        plt.plot(point[0], point[1], marker="o", color="red")

    plt.show()

    return


draw_custom_plot(20, 0, 3*np.pi, evenSpace)
