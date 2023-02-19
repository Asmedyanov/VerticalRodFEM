# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt

N = 1000
h = 1.0 / N
i_ = np.arange(N)
x_ = i_ * h
E = 150.0e3 * np.exp(-x_*x_*10)  # Pa
G = 9.8 * 1.0e3 * 1.0 * np.ones(N)  # Pa
G[len(G)//2] = 2.0e3 * G[0]


def phi_(i: int, x):
    """Form functions matrix"""
    phi_i_1 = (x_[i] - x) / h
    phi_i = (x - x_[i]) / h
    return phi_i_1, phi_i


def S_(i: int):
    """Kinematic connection matrix"""
    S = np.zeros((2, N))
    S[0][i - 1] = 1
    S[1][i] = 1
    return S


def K_(i: int):
    """Hardness matrix"""
    K = E[i] * np.array([[1.0, -1.0], [-1.0, 1.0]])
    return K / h


def F_(i: int):
    """Load vector"""
    F = np.ones(2).transpose()
    return 0.5 * G[i] * h * F


K_f = 0
for i in i_:
    matrix = \
        np.dot(K_(i), S_(i))
    K_f += \
        np.dot(S_(i).transpose(), matrix)
F_f = 0
for i in i_:
    F_f += \
        S_(i).transpose().dot(F_(i))
K_f = K_f[1:].transpose()[1:].transpose()
F_f = F_f[:-1]
K_f_inv = np.linalg.inv(K_f)
U_f = np.dot(K_f_inv, F_f)
plt.plot(U_f)
plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
