import jax.numpy as jnp

from RLBM.case import Case2D
from RLBM.solver import *


# Array convention
# n - idx of point in x
# m - idx of point in y
# q - number of discrete velocities
# d - number of dimensions (2)
# f[n][m][q] - (discrete velocities)
# f_eq[n][m][q] - (equilibrium discrete velocities)
# u[n][m][d] - (macroscopic velocities)
# rho[n][m] - (density)
# c[d][q] - (lattice velocities)
# w[q] - (lattice weights)

class Solver2D(Solver):
    def __init__(self, case):
        assert isinstance(case, Case2D)
        self.scheme = case.scheme

        self.f = jnp.zeros(case.grid_shape + (self.scheme.Q,))
        self.f_eq = jnp.zeros(case.grid_shape + (self.scheme.Q,))
        self.u = jnp.zeros(case.grid_shape + (2,))
        self.rho = jnp.ones(case.grid_shape)
        self.c = self.scheme.LATTICE_VELOCITIES
        self.w = self.scheme.LATTICE_WEIGHTS
        self.tau = 5

        nx, ny = case.grid_shape
        ctr_x, ctr_y = nx // 2, ny // 2
        r_cone = ny // 4

        # Create coordinate grids
        x = jnp.arange(nx)[:, None]  # shape (nx,1)
        y = jnp.arange(ny)[None, :]  # shape (1,ny)

        # Compute squared distance from center
        r_sq = (x - ctr_x) ** 2 + (y - ctr_y) ** 2
        mask = r_sq < r_cone ** 2

        # Compute radial distance only where needed
        r = jnp.sqrt(r_sq)
        rho_update = 2 - r / r_cone

        # Apply masked update (no Python loops)
        self.rho = jnp.where(mask, rho_update, self.rho)

        print("Done initializing rho")
        self.calc_f_eq()
        self.f = self.f_eq

    def run_iterations(self, n=10):
        for i in range(n):
            self.iterate()

    def iterate(self):
        self.calc_rho()
        self.calc_u()
        self.calc_f_eq()

        # Collision step
        f_star = self.f - (self.f - self.f_eq) / self.tau

        # Stream step
        f_stream = f_star
        for i in range(self.scheme.Q):
            f_stream = f_stream.at[:, :, i].set(
                jnp.roll(
                    jnp.roll(
                        f_star[:, :, i],
                        self.c[0, i],
                        axis=0
                    ),
                    self.c[1, i],
                    axis=1
                )
            )

        self.f = f_stream

    def calc_rho(self):
        self.rho = jnp.einsum("nmq->nm", self.f)

    def calc_u(self):
        self.u = jnp.einsum("nmq,dq->nmd", self.f, self.c) / self.rho[..., jnp.newaxis]

    def calc_f_eq(self):
        self.f_eq = self.rho[..., jnp.newaxis] * self.w[jnp.newaxis, jnp.newaxis, :] * (
                1
                + 3 * jnp.einsum("nmd,dq->nmq", self.u, self.c)
                + 9 / 2 * jnp.einsum("nmd,dq->nmq", self.u, self.c) ** 2
                - 3 / 2 * jnp.sum(self.u ** 2, axis=-1, keepdims=True)
        )
