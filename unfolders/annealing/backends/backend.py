
class Backend:
    """Backend to solve a QUBO"""
    def __init__(self, solver_parameters={}):
        """Create a backend with some solver_parameters

        Parameters
        ----------
        solver_parameters : dict
            Parameters accessible when solving a QUBO
        """
        self.solver_parameters = solver_parameters

    def solve(self, qubo_matrix):
        """Solve a qubo_matrix using this backend

        Parameters
        ----------
        qubo_matrix : np.array
            Matrix representing the objective function
        """
        raise NotImplementedError("Solver not implemented")
