from RLBM.solver import *
import jax.numpy as jnp

# Array convention
# N - idx of point in x
# M - idx of point in y
# q - number of discrete velocities
# d - number of dimensions (2)
# f[N][M][q]
# u[N][M][d]
# rho[N][M]
# c[N][M][q]


class Solver2D(Solver):
    def __init__(self, case):
        self.rho = []
        self.discrete_velocities = []
        self.u = []

    def solve(self):
        pass

    def iterate(self):
        pass

    # ρ = ∑ᵢ fᵢ
    def get_density(self):
        self.rho = jnp.einsum("NMq->NM", self.discrete_velocities)

    # u = 1/ρ ∑ᵢ fᵢ cᵢ
    def get_macroscopic_velocities(self):
        self.u = jnp.einsum()
