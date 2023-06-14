import numpy as np
from timeit import default_timer as timer
from enum import Enum


class StopConditions(Enum):
    DIFF = 0
    LINEAR = 1


K = 10
M = 1.5

CONDITIONS = [StopConditions.DIFF, StopConditions.LINEAR]
RO_VALUES = [10**(-9), 10**(-12), 10**(-15)]
SIZES = [5, 10, 15, 20, 25, 30, 50, 100, 350, 500, 1000, 2500]


def createAMatrix(n):
    return np.array([[
        M / (n - (row + 1) - (col + 1) + 0.5) if col != row else K
        for col in range(n)
    ] for row in range(n)])


def getXVector(n):
    x = np.random.randint(2, size=n)
    x[x == 0] = -1

    return np.array(x)


def calculateBVector(A, X):
    return A @ X


def norm(x1, x2):
    return np.sqrt(np.sum(np.power(x1-x2, 2)))


def Jacobi(A, B, RO, stopCondition, iterations=1000):
    x = np.zeros(len(A))

    D = np.diagflat(np.diag(A))
    invD = np.linalg.inv(D)

    M = invD @ (A - D)
    spectralRadius = np.max(np.abs(np.linalg.eigvals(M)))

    startTime = timer()

    for iteration in range(iterations):
        new_x = invD @ B - M @ x

        if stopCondition == StopConditions.DIFF and norm(new_x, x) < RO \
                or stopCondition == StopConditions.LINEAR and norm(A @ new_x, B) < RO:
            endTime = timer()
            return new_x, spectralRadius, iteration + 1, endTime - startTime

        x = new_x

    endTime = timer()

    return x, spectralRadius, iteration + 1, endTime - startTime


with open("results.txt", "w") as f:
    for condition in CONDITIONS:
        for size in SIZES:
            A = createAMatrix(size)

            for ro in RO_VALUES:
                X = getXVector(size)
                B = calculateBVector(A, X)

                result, spectralRadius, iterations, time = Jacobi(
                    A, B, ro, condition)

                error = norm(X, result)

                resultStr = f"{condition}\t{size}\t{ro}\t{spectralRadius}\t{error}\t{'MAX' if iterations == 1000 else iterations}\t{time}\n"

                print(resultStr)
                f.write(resultStr)
