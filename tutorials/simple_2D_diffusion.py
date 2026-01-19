import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

from RLBM.case import Case2D
from RLBM.solver import Solver2D
from RLBM.solver.schemes import D2Q9

case = Case2D(D2Q9(), 80, (300, 200), 15000)
solver = Solver2D.initialize(case)
plot_interval = case.grid_shape[1] // 10


@jax.jit
def simulate_n(s):
    return s.run_iterations(plot_interval)


for i in range(50):
    plt.imshow(solver.rho, vmin=1, vmax=2)
    plt.colorbar()
    plt.title(r"$\rho_{max} = $" + str(jnp.max(solver.rho)) + f" n={i * plot_interval}")
    plt.show()
    solver = simulate_n(solver)

plt.imshow(solver.rho, vmin=1, vmax=2)
plt.colorbar()
plt.title(r"$\rho_{max} = $" + str(jnp.max(solver.rho)) + f" n={50 * plot_interval}")
plt.show()
