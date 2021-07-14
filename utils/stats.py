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
    # nbins = histogram.shape[0]
    # statcov = np.zeros((nbins, nbins))
    # np.fill_diagonal(statcov, histogram)
    return np.diagflat(histogram)


def transform_covariance_matrix(transformation, histogram):
    """
    Compute the covariance matrix of a new histogram given by:

    new_histogram = transformation * histogram
    """
    A = transformation
    V = compute_numpy_covariance_matrix(histogram)
    return A * V * A.T
