import numpy as np


def dimod_extract_best_fit(solver):
    """
    Extracts best solution from the sample set
    """

    def aux(*args, **kwargs):
        # Call the solver
        solutions = solver(*args, **kwargs)
        # Retrieve best fit
        best_fit = solutions.first
        return np.array(list(best_fit.sample.values()))

    return aux


def dimod_as_numpy_array(solver):
    """
    Map dimods solutions to a numpy array sorted by accuracy
    """

    def aux(*args, **kwargs):
        # Call the solver
        solutions = solver(*args, **kwargs)
        # Convert to numpy array
        return (
            solutions.to_pandas_dataframe()
            .sort_values("energy")
            .drop(columns=["energy", "num_occurrences"])
            .to_numpy()
        )

    return aux