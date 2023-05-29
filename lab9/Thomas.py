import numpy as np
from timeit import default_timer as timer


def calculateError(expected, got):
    return np.sqrt(np.sum(np.power(expected-got, 2)))


def TDMA(n, m, size, solutions, precision):
    lowerDiagonal, mainDiagonal, upperDiagonal = createDiagonals(
        n, m, size, precision)

    startTime = timer()

    c = np.zeros(shape=size-1).astype(precision)
    c[0] = upperDiagonal[0] / mainDiagonal[0]

    for i in range(1, size-1):
        c[i] = upperDiagonal[i] / (mainDiagonal[i] - c[i-1])

    d = np.zeros(shape=size).astype(precision)
    d[0] = solutions[0] / mainDiagonal[0]

    for i in range(1, size):
        d[i] = (solutions[i] - lowerDiagonal[i-1] * d[i-1]) / \
            (mainDiagonal[i] - lowerDiagonal[i-1] * c[i-1])

    x = np.zeros(shape=size).astype(precision)
    x[size-1] = d[size-1]

    for i in range(size-2, -1, -1):
        x[i] = d[i] - c[i] * x[i+1]

    endTime = timer()

    return x, endTime - startTime


def createDiagonals(n, m, size, precision):
    diagonal = np.array([(-m * (i+1) * n)
                        for i in range(size)]).astype(precision)
    lowerDiagonal = np.array([m / (i+1)
                             for i in range(1, size)]).astype(precision)
    upperDiagonal = np.array([(i+1)
                             for i in range(size - 1)]).astype(precision)

    return lowerDiagonal, diagonal, upperDiagonal


def createAMatrix(n, m, size, precision):
    A = np.zeros(shape=(size, size)).astype(precision)

    for i in range(size):
        A[i][i] = (-m * (i+1) * n)

        if i+1 < size and i+1 >= 0:
            A[i][i+1] = (i+1)
        if i-1 < size and i-1 >= 0:
            A[i][i-1] = m/(i+1)

    return A


def getXVector(n, precision):
    x = np.random.randint(2, size=n)
    x[x == 0] = -1

    return np.array(x).astype(precision)


def calculateBVector(A, X):
    return A @ X


N = 11
M = 35
SIZE = 10
PREC = np.float64

X = getXVector(SIZE, PREC)
A = createAMatrix(N, M, SIZE, PREC)
B = calculateBVector(A, X)

result, elapsedTime = TDMA(N, M, SIZE, B, PREC)

print(calculateError(X, result), elapsedTime)
