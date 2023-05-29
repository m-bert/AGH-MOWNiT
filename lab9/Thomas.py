import numpy as np


def TDMA(a, b, c, d):
    n = len(d)
    w = np.zeros(n - 1, float)
    g = np.zeros(n, float)
    p = np.zeros(n, float)

    w[0] = c[0] / b[0]
    g[0] = d[0] / b[0]

    for i in range(1, n - 1):
        w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
    p[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        p[i - 1] = g[i - 1] - w[i - 1] * p[i]
    return p


def createADiagonals(n, m, size):
    diagonal = [(-m * i * n) for i in range(size)]
    lowerDiagonal = [m / i for i in range(1, size)]
    upperDiagonal = [i for i in range(size - 1)]

    print(diagonal)
    print(lowerDiagonal)
    print(upperDiagonal)


def createAMatrix(size, precision):
    A = np.zeros(size)
    return


def getXVector(n, precision):
    x = np.random.randint(2, size=n)
    x[x == 0] = -1

    return np.array(x).astype(precision)


def calculateBVector(A, X):
    return A @ X


size = 10

X = getXVector(size, np.float64)
A = createAMatrix(11, 35, size)
B = calculateBVector(A, X)

print()
