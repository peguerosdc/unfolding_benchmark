import numpy as np


def compute_numpy_covariance_matrix(histogram):
    """
    Compute the covariance matrix of the given histogram
    assuming a Poisson distribution

    Parameters
    ----------
    histogram : np.array
        Histogram to compute its covariance matrix
    """
    nbins = histogram.shape[0]
    statcov = np.zeros((nbins, nbins))
    np.fill_diagonal(statcov, histogram)
    return statcov