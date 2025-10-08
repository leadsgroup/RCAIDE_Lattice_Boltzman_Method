from RLBM.solver.schemes import *
import jax.numpy as jnp


class D3Q27(Scheme3D):
    LATTICE_VELOCITIES = jnp.array([
        [ 0,  +1,  -1,   0,   0,   0,   0,  +1,  -1,  +1,  -1,   0,   0,  +1,  -1,  +1,  -1,   0,   0,  +1,  -1,  +1,  -1,  +1,  -1,  -1,  +1],
        [ 0,   0,   0,  +1,  -1,   0,   0,  +1,  -1,   0,   0,  +1,  -1,  -1,  +1,   0,   0,  +1,  -1,  +1,  -1,  +1,  -1,  -1,  +1,  +1,  -1],
        [ 0,   0,   0,   0,   0,  +1,  -1,   0,   0,  +1,  -1,  +1,  -1,   0,   0,  -1,  +1,  -1,  +1,  +1,  -1,  -1,  +1,  +1,  -1,  +1,  -1]
    ])

    LATTICE_WEIGHTS = jnp.array([
        8/27,
        2/27, 2/27, 2/27, 2/27, 2/27, 2/27,
        1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54,
        1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216
    ])
    
    LATTICE_INDICES, OPPOSITE_LATTICE_INDICES = compute_lattice_weights(LATTICE_VELOCITIES)
