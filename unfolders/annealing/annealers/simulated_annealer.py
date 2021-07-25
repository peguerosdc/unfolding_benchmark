import dimod
import neal
from .annealer import Annealer
from .utils import dimod_as_numpy_array


class SimulatedAnnealer(Annealer):
    """Finds the exact solutions of a QUBO sampling from a
    Boltzmann Distribution"""

    def __init__(self, num_reads):
        super().__init__()
        self.num_reads = num_reads

    def __str__(self):
        return f"<SimulatedAnnealingBackend num_reads={self.num_reads}>"

    @dimod_as_numpy_array
    def solve(self, qubo_matrix):
        # translate the problem to a BQM and create a sampler
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
        sampler = neal.SimulatedAnnealingSampler()
        # sample solutions
        return sampler.sample(bqm, num_reads=self.num_reads)
