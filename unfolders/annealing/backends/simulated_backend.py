import dimod
import neal
from .backend import Backend


class SimulatedAnnealingBackend(Backend):
    """Finds the exact solutions of a QUBO sampling from a
    Boltzmann Distribution"""
    def __init__(self, num_reads):
        super().__init__()
        self.num_reads = num_reads

    def __str__(self):
        return f"<SimulatedAnnealingBackend num_reads={self.num_reads}>"

    def solve(self, qubo_matrix):
        # translate the problem to a BQM and create a sampler
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
        sampler = neal.SimulatedAnnealingSampler()
        # sample solutions
        return sampler.sample(bqm, num_reads=self.num_reads).aggregate()
