from .backend import AnnealingBackend
from .annealers import SimulatedQuantumAnnealer


class SimulatedQuantumAnnealingBackend(AnnealingBackend):
    def __init__(self, n_bits, iterations, *args, **kwargs):
        super().__init__(n_bits, *args, **kwargs)
        self.iterations = iterations

    def get_annealer(self):
        return SimulatedQuantumAnnealer(self.iterations)