import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
from .. import TimeDomain_Start_Stop_MaxSpacing, Euler, RK4, Trapezoidal
from ..backward_differentiation import BDF3
from ..adams_bashforth import AB5
from ..adams_moulton import AM4
from tqdm import tqdm

PROBLEM = """
Solve the second order equation
x'' = -x
x(0) = 0
x'(0) = 1

which has the solution
x(t) = sin(t)
"""

x0 = 0
y0 = 1
t0, tf = 0, 5

A = np.array(
    [
        [0, 1],
        [-1, 0],
    ]
)

u0 = np.array([x0, y0])


def rhs(t, u):
    return A @ u


def exact(t):  # exact solution for testing
    return np.sin(t)


def sine_example():
    print(PROBLEM)
    plt.ion()
    # define the system and parameters
    # simple example
    max_time_step = 1e-2
    time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, max_time_step)
    solver = RK4()
    print(f"Sovle using {solver.name}")
    xs = np.array([u[0] for u in solver.solve(u0, rhs, time)])
    fig, (ax_sol, ax_err) = plt.subplots(2, 1, sharex=True)
    ax_sol.plot(time.array, xs, label=solver.name)
    ax_sol.plot(time.array, exact(time.array), "--", linewidth=3, label="True")
    ax_err.semilogy(time.array, np.abs(xs - exact(time.array)))
    ax_sol.legend(loc="lower left")
    ax_err.set_xlabel("$t$")
    ax_err.set_ylabel("error")
    plt.show(block=True)

    print("Do many simultanious solves.")
    # list of solvers to use
    solvers = [
        Euler(),
        Trapezoidal(),
        BDF3(seed=RK4(), seed_steps_per_step=1),
        RK4(),
        AB5(seed=RK4(), seed_steps_per_step=2),
        AM4(seed=RK4(), seed_steps_per_step=2),
    ]

    # Multiple simultainous solves
    max_time_step = 3e-2
    time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, max_time_step)
    colors = TABLEAU_COLORS.values()

    plt.ion()
    fig, (ax_sol, ax_err) = plt.subplots(2, 1, sharex=True)
    ax_sol.plot(time.array, exact(time.array), "k-", label="exact")

    for solver, color in zip(solvers, colors):
        ax_sol.plot([], [], marker=".", color=color, label=f"{solver.name}")
        ax_err.semilogy([], [], marker=".", color=color, label=f"{solver.name}")

    for ax in [ax_sol, ax_err]:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax_err.set_xlabel("$t$")
    ax_err.set_ylabel("error")
    plt.tight_layout()

    for t, *us in zip(
        time.array, *(solver.solution_generator(u0, rhs, time) for solver in solvers)
    ):
        for color, (x, _) in zip(colors, us):
            ax_sol.plot(t, x, color=color, marker=".")
            x_true = exact(t)
            err = abs(x - x_true)
            ax_err.semilogy(t, err, color=color, marker=".")

        plt.pause(1e-3)
    plt.show(block=True)

    # convergence plot
    print("Test convergence.")
    x_true = exact(tf)
    time_steps = np.logspace(-3, -1, 10)
    plt.figure("Convergence")
    for solver, color in zip(solvers, colors):
        errs = []
        my_steps = []
        for k in tqdm(time_steps):
            time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, k)
            try:
                u = solver.t_final(u0, rhs, time)
                my_steps.append(k)
            except ValueError:
                continue
            x = u[0]
            err = abs((x - x_true) / x_true)
            errs.append(err)
        plt.loglog(my_steps, errs, "o", color=color, label=solver.name)
        order_err = errs[-1] * np.exp(
            solver.order * np.log(time_steps[0] / time_steps[-1])
        )
        plt.plot(
            [time_steps[0], time_steps[-1]],
            [order_err, errs[-1]],
            color=color,
            label=f"$\\mathcal{{O}}({solver.order})$",
        )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.xlabel("Time Step")
    plt.ylabel("Relative Error")
    plt.title("Convergence")
    plt.grid()
    plt.show(block=True)
