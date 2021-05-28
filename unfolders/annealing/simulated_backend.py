from .backend import AnnealingBackend
from .annealers import SimulatedAnnealer


class SimulatedAnnealingBackend(AnnealingBackend):
    def __init__(self, n_bits, num_reads, *args, **kwargs):
        super().__init__(n_bits, *args, **kwargs)
        self.num_reads = num_reads

    def get_annealer(self):
        print(f"num reads: {self.num_reads}")
        return SimulatedAnnealer(self.num_reads)