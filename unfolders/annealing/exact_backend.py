from .backend import AnnealingBackend
from .annealers import ExactAnnealer


class ExactBackend(AnnealingBackend):
    def __init__(self, n_bits, *args, **kwargs):
        super().__init__(n_bits, *args, **kwargs)

    def get_annealer(self):
        return ExactAnnealer()