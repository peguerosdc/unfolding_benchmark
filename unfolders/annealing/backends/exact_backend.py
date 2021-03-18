import dimod
from .backend import Backend
from .utils import dimod_extract_best_fit


class ExactBackend(Backend):
    """Finds the exact solutions of a QUBO using dimod.ExactSolver"""

    def __str__(self):
        return f"<ExactBackend solver='dimod.ExactSolver'>"

    @dimod_extract_best_fit
    def solve(self, qubo_matrix):
        print("Finding exact solutions...")
        # translate the problem to a BQM
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
        # find exact solutions
        return dimod.ExactSolver().sample(bqm, **self.solver_parameters)
