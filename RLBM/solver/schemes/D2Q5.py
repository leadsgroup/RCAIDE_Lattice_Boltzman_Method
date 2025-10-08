from RLBM.solver.schemes import *
import jax.numpy as jnp

class D2Q5(Scheme2D):
    LATTICE_VELOCITIES = jnp.array([
        [ 0,  1,  0, -1,  0],
        [ 0,  0,  1,  0, -1]
    ])

    LATTICE_WEIGHTS = jnp.array([
        1/3,                        # Center Velocity [0,]
        1/6,  1/6,  1/6,  1/6,      # Axis-Aligned Velocities [1, 2, 3, 4]
    ])
    
    LATTICE_INDICES, OPPOSITE_LATTICE_INDICES = compute_lattice_weights(LATTICE_VELOCITIES)
    Q = 5
