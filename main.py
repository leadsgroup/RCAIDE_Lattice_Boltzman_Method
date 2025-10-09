import sys
import time

import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

from RLBM.case import Case2D
from RLBM.solver import Solver2D
from RLBM.solver.schemes import D2Q5, D2Q9


def main():
    if len(sys.argv) != 3:
        print("FORMAT: rlbm <case> <geometry>")

    case_file = sys.argv[1]
    geom_file = sys.argv[2]

    case = Case2D(D2Q9(), 80, (3000, 2000), 15000)
    solver = Solver2D.initialize(case)
    plot_interval = case.grid_shape[1] // 10

    @jax.jit
    def simulate_n(s):
        return s.run_iterations(plot_interval)

    start = time.time()
    for i in range(50):
        plt.imshow(solver.rho, vmin=1, vmax=2)
        plt.colorbar()
        plt.title(r"$\rho_{max} = $" + str(jnp.max(solver.rho)) + f" n={i * plot_interval}")
        plt.show()

        solver = simulate_n(solver)

    plt.imshow(solver.rho, vmin=1, vmax=2)
    plt.colorbar()
    plt.title(r"$\rho_{max} = $" + str(jnp.max(solver.rho)) + f" n={50 * plot_interval}")
    plt.show()
    print(time.time() - start)

if __name__ == "__main__":
    main()
