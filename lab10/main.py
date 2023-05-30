import numpy as np


def Jacobi(A, B, iterations=50):
    x = np.zeros(len(A))

    D = np.diagflat(np.diag(A))
    invD = np.linalg.inv(D)

    LU = A - D

    for _ in range(iterations):
        x = invD @ (B - LU @ x)

    return x


A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]])

b = np.array([6., 25., -11., 15.])


print(Jacobi(A, b, 100))
