import sys
import jax.numpy as jnp

from RLBM.solver import Solver2D


def main():
    if len(sys.argv) != 3:
        print("FORMAT: rlbm <case> <geometry>")

    case_file = sys.argv[1]
    geom_file = sys.argv[2]

    solver = Solver2D()

if __name__ == "__main__":
    main()
