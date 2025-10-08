from RLBM.solver.schemes import *
import jax.numpy as jnp

class D2Q9(Scheme2D):
    LATTICE_VELOCITIES = jnp.array([
        [ 0,  1,  0, -1,  0,  1, -1, -1,  1,],
        [ 0,  0,  1,  0, -1,  1,  1, -1, -1,]
    ])

    LATTICE_WEIGHTS = jnp.array([
        4/9,                        # Center Velocity [0,]
        1/9,  1/9,  1/9,  1/9,      # Axis-Aligned Velocities [1, 2, 3, 4]
        1/36, 1/36, 1/36, 1/36,     # 45 ° Velocities [5, 6, 7, 8]
    ])
    
    LATTICE_INDICES, OPPOSITE_LATTICE_INDICES = compute_lattice_weights(LATTICE_VELOCITIES) 
