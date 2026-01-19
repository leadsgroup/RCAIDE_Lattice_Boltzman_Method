import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from RLBM.case import Case3D
from RLBM.solver import Solver3D
from RLBM.solver.schemes import *


def main():
    if len(sys.argv) != 3:
        print("FORMAT: rlbm <case> <geometry>")

    case_file = sys.argv[1]
    geom_file = sys.argv[2]

    case = Case3D(D3Q19(), 80, (150, 100, 100), 15000)
    solver = Solver3D.initialize(case)
    plot_interval = 10

    @jax.jit
    def simulate_n(s):
        return s.run_iterations(plot_interval)

    for i in range(50):
        plt.imshow(solver.rho[:, :, 50], vmin=1, vmax=2)
        plt.colorbar()
        plt.title(r"$\rho_{max} = $" + str(jnp.max(solver.rho)) + f" n={i * plot_interval}")
        plt.show()
        solver = simulate_n(solver)


if __name__ == "__main__":
    main()
