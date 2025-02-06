import numpy as np
import matplotlib.pyplot as plt

ts = np.linspace(0, 2*np.pi, 201)


def x(t):
    return t


def y(t):
    return 0.3 * np.sin(t)


def implicit(points):
    return points[:, 1] - y(points[:, 0])


xs = x(ts)
ys = y(ts)

t_A, t_B, t_C = 1, 3, 5
A = np.array([x(t_A), y(t_A)])
B = np.array([x(t_B), y(t_B)])
C = np.array([x(t_C), y(t_C)])

ts_1 = np.linspace(t_A, t_B, 201)
xs_1 = x(ts_1)
ys_1 = y(ts_1)
color_1 = "green"
proj_1 = (A + B)/2 + np.array([0, -1])

ts_2 = np.linspace(t_B, t_C, 201)
xs_2 = x(ts_2)
ys_2 = y(ts_2)
color_2 = "purple"
proj_2 = (B + C)/2 + np.array([0, -1])

plt.plot(xs, ys, "k-")
plt.plot(*A, "k.")
plt.plot(*B, "k.")
plt.plot(*C, "k.")

plt.plot(*proj_1, "*", color=color_1)
plt.plot(xs_1, ys_1, "-", color=color_1)
plt.plot(*np.block([[A], [B]]).T, "--", color=color_1)

plt.plot(*proj_2, "*", color=color_2)
plt.plot(xs_2, ys_2, "-", color=color_2)
plt.plot(*np.block([[B], [C]]).T, "--", color=color_2)
