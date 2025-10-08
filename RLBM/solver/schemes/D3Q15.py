from RLBM.solver.schemes import *
import jax.numpy as jnp


class D3Q15(Scheme3D):
    LATTICE_VELOCITIES = jnp.array([
        [0, -1, +1,  0,  0,  0,  0, -1, +1, -1, +1, -1, +1, +1, -1],
        [0,  0,  0, -1, +1,  0,  0, -1, +1, -1, +1, +1, -1, -1, +1],
        [0,  0,  0,  0,  0, -1, +1, -1, +1, +1, -1, -1, +1, -1, +1]
    ])

    LATTICE_WEIGHTS = jnp.array([
        2/9,
        1/9,  1/9,  1/9,  1/9,  1/9,  1/9,
        1/72, 1/72, 1/72, 1/72, 1/72, 1/72, 1/72, 1/72
    ])
    
    LATTICE_INDICES, OPPOSITE_LATTICE_INDICES = compute_lattice_weights(LATTICE_VELOCITIES)
