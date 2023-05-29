import numpy as np
from timeit import default_timer as timer


def printMatrix(M):
    for row in M:
        print(row)

    return


def createMatrixA1(n, precision):
    return np.array(
        [[1 if i == 0 else 1 / (i + j + 1) for j in range(n)]
         for i in range(n)]
    ).astype(precision)


def createMatrixA2(n, precision):
    return np.array(
        [
            [
                2 * (i + 1) / (j + 1) if j >= i else 2 * (j + 1) / (i + 1)
                for j in range(n)
            ]
            for i in range(n)
        ]
    ).astype(precision)


def getXVector(n, precision):
    x = np.random.randint(2, size=n)
    x[x == 0] = -1

    return np.array(x).astype(precision)


def calculateBVector(A, X):
    return A @ X


def calculateError(expected, got):
    return np.sqrt(np.sum(np.power(expected-got, 2)))


def GaussianElimination(A, B):
    n = len(B)

    A = np.hstack((A, B.reshape(-1, 1)))

    startTime = timer()

    currentRow = 0
    currentCol = 0

    while currentRow < n and currentCol < n + 1:
        iMax = np.argmax(np.abs(A[currentRow:, currentCol])) + currentRow

        if A[iMax][currentCol] == 0:
            currentCol += 1
            continue

        A[[iMax, currentRow]] = A[[currentRow, iMax]]

        for row in range(currentRow + 1, n):
            ratio = A[row][currentCol] / A[currentRow][currentCol]
            A[row][currentCol] = 0

            for col in range(currentCol + 1, n + 1):
                A[row][col] -= A[currentRow][col] * ratio

        currentRow += 1
        currentCol += 1

    X = np.zeros(n)

    for i in range(n - 1, -1, -1):
        X[i] = A[i][n] / A[i][i]

        for j in range(i - 1, -1, -1):
            A[j][n] -= A[j][i] * X[i]

    endTime = timer()

    return X, startTime - endTime


MATRIX_TYPES = [createMatrixA1, createMatrixA2]
PRECISIONS = [np.float32, np.float64, np.float128]
SIZES = [i for i in range(3, 30 + 1)]


with open("results.txt", "w") as f:
    for matrixType in MATRIX_TYPES:
        for precision in PRECISIONS:
            for size in SIZES:
                A = matrixType(size, precision)
                X = getXVector(size, precision)
                B = calculateBVector(A, X)

                calculatedX, elapsedTime = GaussianElimination(A, B)

                task = 1 if matrixType == createMatrixA1 else 2

                print(task, precision.__name__, size)
                print(X)
                print(calculatedX)

                str1 = f"{task}\t{precision.__name__}\t{size}\n"
                str2 = f"{X}\n"
                str3 = f"{calculatedX}\n"
                str4 = (
                    f"Wskaźnik uwarunkowania: {np.linalg.cond(A, p='fro')}\n"
                    if precision != np.float128
                    else "\n"
                )
                # str5 = f"Błąd: {np.linalg.norm(X - calculatedX)}\n"
                str5 = f"Błąd: {calculateError(X, calculatedX)}\n"

                f.write(
                    "==============================================================================\n"
                )
                f.write(str1)
                f.write(str2)
                f.write(str3)
                f.write(str4)
                f.write(str5)
                f.write(
                    "==============================================================================\n"
                )
                f.write("\n")
