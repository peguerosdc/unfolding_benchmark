import numpy as np


def dimod_extract_best_fit(solver):
    """
    Extracts the best solution of a dimod sample set
    """

    def aux(*args, **kwargs):
        # Call the solver
        solutions = solver(*args, **kwargs)
        # Retrieve best fit
        best_fit = solutions.first
        return np.array(list(best_fit.sample.values()))

    return aux
