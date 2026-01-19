from RLBM.mesher import Mesh2D


class Case:
    def __init__(self, scheme, reynolds_num, grid_shape, n_iterations, _mesh=None, _boundary_conditions=None,
                 _velocity_profile_initial=None):
        self.scheme = scheme
        self.reynolds_num = reynolds_num
        self.grid_shape = grid_shape
        self.n_iterations = n_iterations
        self.mesh = _mesh
        self.boundary_conditions = _boundary_conditions
        self.u0 = _velocity_profile_initial


class Case2D(Case):
    def __init__(self, scheme, reynolds_num, grid_shape, n_iterations, _mesh=None, _boundary_conditions=None,
                 _velocity_profile_initial=None):
        super().__init__(scheme, reynolds_num, grid_shape, n_iterations, _mesh, _boundary_conditions,
                         _velocity_profile_initial)

        # Input format assertions
        assert len(grid_shape) == 2
        if _mesh:
            assert isinstance(_mesh, Mesh2D)


class Case3D(Case):
    pass
