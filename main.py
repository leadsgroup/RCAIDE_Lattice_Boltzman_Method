import sys
import matplotlib.pyplot as plt
import jax.numpy as jnp

from RLBM.case import Case2D
from RLBM.solver import Solver2D
from RLBM.solver.schemes import D2Q9


def main():
    if len(sys.argv) != 3:
        print("FORMAT: rlbm <case> <geometry>")

    case_file = sys.argv[1]
    geom_file = sys.argv[2]

    case = Case2D(D2Q9(), 80, (300, 200), 15000)
    solver = Solver2D.initialize(case)

    print(solver.rho.shape)
    plt.imshow(solver.rho)
    plt.colorbar()
    plt.show()

    for i in range(1000):
        solver.iterate()
        if i % 20 == 0:
            plt.imshow(solver.rho, vmin=1, vmax=2)
            plt.colorbar()
            plt.xlabel(r"$\rho_{max} = $" + str(jnp.max(solver.rho)))
            plt.show()


if __name__ == "__main__":
    main()
