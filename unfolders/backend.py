from .result import UnfoldingResult
from abc import ABC, abstractmethod


class Backend(ABC):
    @abstractmethod
    def solve(self, data, statcov, xini, bini, R):
        """
        Solves the given inverse problem. Must return an instance of UnfoldingResult
        """
        return
