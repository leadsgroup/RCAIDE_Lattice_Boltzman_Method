from RLBM.solver.schemes import *
import jax.numpy as jnp


class D1Q3(Scheme1D):
    LATTICE_VELOCITIES = jnp.array([[0, +1, -1]])

    LATTICE_WEIGHTS = jnp.array([
        2/3,
        1/6,  1/6
    ])
    
    LATTICE_INDICES, OPPOSITE_LATTICE_INDICES = compute_lattice_weights(LATTICE_VELOCITIES)
