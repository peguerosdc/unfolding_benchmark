import root_numpy
import ROOT


def histogram_to_python(hist):
    """
    Converts a ROOT THx to a numpy array.

    Returns: histogram, edges
    """
    return root_numpy.hist2array(hist, return_edges=True)


def compute_covariance_matrix(histogram):
    """
    Computes the covariance matrix of a given histogram based on the
    Poisson hypothesis.

    Returns a TH2D histogram

    Parameters
    ----------
    histogram : TH1D
        The ROOT histogram to compute its covariance matrix
    """
    # Retrieve histograms metadata
    name = histogram.GetName()
    nbins = histogram.GetNbinsX()
    rmin = histogram.GetBinLowEdge(1)
    rmax = histogram.GetBinLowEdge(nbins + 1)
    # Create covariance matrix
    statcov = ROOT.TH2D(
        f"statcov_{name}",
        f"covariance matrix of {name}",
        nbins,
        rmin,
        rmax,
        nbins,
        rmin,
        rmax,
    )
    # Fill the covariance matrix
    for i in range(1, nbins):
        statcov.SetBinContent(i, i, histogram.GetBinError(i) * histogram.GetBinError(i))
    return statcov