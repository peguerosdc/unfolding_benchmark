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


def mape(observed, expected):
    """
    Compute the Mean Absolute Percentage Error
    """
    vals = []
    for o, e in zip(observed, expected):
        if e != 0:
            vals += [np.abs((o - e) / e)]
    return 100 * np.mean(vals)


def chi_square(observed, expected):
    """
    Compute the chi square test
    """
    from scipy import stats

    # glen cowan pp61
    mychi = 0
    for i, (n, nu) in enumerate(zip(observed, expected)):
        if nu != 0:
            mychi += ((n - nu) ** 2) / nu
    # compute p value
    p = stats.chi2.sf(mychi, len(observed) - 1)
    return mychi, p