import jax.numpy as jnp


def compute_lattice_weights(velocities: jnp.ndarray):  # type: ignore
    indices = jnp.arange(len(velocities[0]))
    opp_indices = jnp.zeros_like(indices)

    # Yes, for loops are slow, but it doesn't matter here
    for i in indices:
        idx = jnp.argmax(jnp.sum(velocities.T == -velocities.T[i], axis=1))
        opp_indices = opp_indices.at[i].set(idx)

    return indices, opp_indices


class Scheme:
    LATTICE_VELOCITIES = jnp.array([])
    LATTICE_INDICES = jnp.array([])
    OPPOSITE_LATTICE_INDICES = jnp.array([])
    LATTICE_WEIGHTS = jnp.array([])


class Scheme1D(Scheme):
    pass


class Scheme2D(Scheme):
    pass


class Scheme3D(Scheme):
    pass
