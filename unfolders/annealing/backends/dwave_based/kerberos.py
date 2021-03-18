import dimod
from hybrid.reference.kerberos import KerberosSampler
from ..utils import dimod_extract_best_fit


class KerberosBackend:
    """
    Solver params accepted by this solver are:
        - max_iter
        - convergence
    """

    @dimod_extract_best_fit
    def solve(self, qubo_matrix):
        print("Sampling states with the lowest energy using Kerberos...")
        # find embedding of subproblem-sized complete graph to the QPU
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
        # Use the reference kerberos sampler
        return KerberosSampler().sample(
            bqm, qpu_sampler=self.hardware_sampler, **self.solver_parameters
        )
