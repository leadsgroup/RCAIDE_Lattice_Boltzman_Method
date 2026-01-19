import jax.numpy as jnp
import jax
from flax import struct

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


@struct.dataclass
class Solver2D:
    f: jnp.ndarray
    f_eq: jnp.ndarray
    u: jnp.ndarray
    rho: jnp.ndarray
    c: jnp.ndarray
    w: jnp.ndarray
    tau: float = 1.0

    def replace(self, **updates):
        """IDE stub — replaced at runtime by flax.struct.dataclass"""
        return self

    @staticmethod
    def initialize(case: Case2D):
        """Pure initializer returning a new Solver2D instance."""
        assert isinstance(case, Case2D)
        scheme = case.scheme
        nx, ny = case.grid_shape

        u = jnp.zeros((nx, ny, 2))
        rho = jnp.ones((nx, ny))
        c = scheme.LATTICE_VELOCITIES
        w = scheme.LATTICE_WEIGHTS
        tau = 1

        # --- Initialize density cone ---
        ctr_x, ctr_y = nx // 2, ny // 2
        r_cone = ny // 4
        x = jnp.arange(nx)[:, None]
        y = jnp.arange(ny)[None, :]
        r_sq = (x - ctr_x) ** 2 + (y - ctr_y) ** 2
        mask = r_sq < r_cone ** 2
        r = jnp.sqrt(r_sq)
        rho_update = 2 - r / r_cone
        rho = jnp.where(mask, rho_update, rho)

        # --- Initialize equilibrium and f ---
        f_eq = rho[..., jnp.newaxis] * w[jnp.newaxis, jnp.newaxis, :] * (
                1
                + 3 * jnp.einsum("nmd,dq->nmq", u, c)
                + 4.5 * jnp.einsum("nmd,dq->nmq", u, c) ** 2
                - 1.5 * jnp.sum(u ** 2, axis=-1, keepdims=True)
        )
        f = f_eq

        return Solver2D(f=f, f_eq=f_eq, u=u, rho=rho, c=c, w=w, tau=tau)

    # ρ = ∑ᵢ fᵢ
    def calc_rho(self):
        rho = jnp.sum(self.f, axis=-1)
        return self.replace(rho=rho)

    # u = 1/ρ ∑ᵢ fᵢ cᵢ
    def calc_u(self):
        u = jnp.einsum("nmq,dq->nmd", self.f, self.c) / self.rho[..., jnp.newaxis]
        return self.replace(u=u)

    # Equilibrium populations
    def calc_f_eq(self):
        u_proj = jnp.einsum("nmd,dq->nmq", self.u, self.c)
        u_mag_sq = jnp.sum(self.u ** 2, axis=-1, keepdims=True)
        f_eq = (
                self.rho[..., jnp.newaxis] * self.w[jnp.newaxis, jnp.newaxis, :] *
                (1 + 3 * u_proj + 4.5 * u_proj ** 2 - 1.5 * u_mag_sq)
        )
        return self.replace(f_eq=f_eq)

    def iterate(self):
        s = self.calc_rho().calc_u().calc_f_eq()

        # Collision
        f_star = s.f - (s.f - s.f_eq) / s.tau

        # Streaming
        def stream_one(i, f):
            return f.at[:, :, i].set(
                jnp.roll(f_star[..., i], s.c[:, i], axis=(0,1))
            )

        f_stream = jax.lax.fori_loop(0, s.c.shape[1], stream_one, f_star)
        return s.replace(f=f_stream)

    def run_iterations(self, n):
        def body(_, state):
            return state.iterate()

        return jax.lax.fori_loop(0, n, body, self)
