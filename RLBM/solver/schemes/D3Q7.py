from solver.schemes.scheme import *
import jax.numpy as jnp


class D3Q7(Scheme.D3):
    LATTICE_VELOCITIES = jnp.array([
        [0,  1,  0, -1,  0,  0,  0],
        [0,  0,  1,  0, -1,  0,  0]
        [0,  0,  0,  0,  0,  1, -1]
    ])

    LATTICE_WEIGHTS = jnp.array([
        1/4,                                # Center Velocity [0,]
        1/8,  1/8,  1/8,  1/8,  1/8,  1/8   # Axis-Aligned Velocities [1, 2, 3, 4, 5, 6]
    ])
    
    LATTICE_INDICES, OPPOSITE_LATTICE_INDICES = Scheme.compute_lattice_weights(LATTICE_VELOCITIES)
