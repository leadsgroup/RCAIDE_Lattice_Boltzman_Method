import jax.numpy as jnp


class Scheme:
    LATTICE_VELOCITIES = jnp.array([])
    LATTICE_INDICES = jnp.array([])
    OPPOSITE_LATTICE_INDICES = jnp.array([])
    LATTICE_WEIGHTS = jnp.array([])

    def compute_lattice_weights(vels):
        indices = jnp.arange(len(vels[0]))

        opp_indices = jnp.zeros_like(indices)
        
        # Yes, for loops are slow but it doesn't matter here
        for i in indices:
            idx = jnp.argmax(jnp.sum(vels.T == -vels.T[i], axis=1))
            opp_indices = opp_indices.at[i].set(idx)

        return indices, opp_indices


class D2(Scheme):
    pass


class D3(Scheme):
    pass


Scheme.D2 = D2
Scheme.D3 = D3
