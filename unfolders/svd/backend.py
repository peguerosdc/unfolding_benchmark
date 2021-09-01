from ..backend import Backend
from ..result import UnfoldingResult
import ROOT
import numpy as np
import root_numpy
import math


class SVDBackend(Backend):
    def __init__(self, regularization_param, bins_min, bins_max):
        """
        Parameters
        ----------
        regularization_param : number
            Regularization parameter to use in the unfolding method
        bins_min : number
            Lower limit of the histograms' bins
        bins_max : number
            Upper limit of the histograms' bins
        """
        self.kreg = regularization_param
        self.bins_min = bins_min
        self.bins_max = bins_max

    def solve(self, data, statcov, xini, bini, R):
        bins_min = self.bins_min
        bins_max = self.bins_max
        # Transform the reponse matrix from probabilities to events
        # R = np.multiply(xini, R, where=xini != 0)
        # Transform python arrays to ROOT TH1D
        datar = ROOT.TH1D("data", "data", data.shape[0], bins_min, bins_max)
        root_numpy.array2hist(data, datar)
        xinir = ROOT.TH1D("xini", "xini", xini.shape[0], bins_min, bins_max)
        root_numpy.array2hist(xini, xinir)
        binir = ROOT.TH1D("bini", "bini", bini.shape[0], bins_min, bins_max)
        root_numpy.array2hist(bini, binir)
        Adet = ROOT.TH2D(
            "R",
            "R",
            xini.shape[0],
            bins_min,
            bins_max,
            bini.shape[0],
            bins_min,
            bins_max,
        )
        root_numpy.array2hist(R, Adet)
        # Compute covariance matrix assuming the amount of events in each bin
        # come from a Poisson distribution
        statcovr = ROOT.TH2D(
            "statcov",
            "statcov",
            data.shape[0],
            bins_min,
            bins_max,
            data.shape[0],
            bins_min,
            bins_max,
        )
        root_numpy.array2hist(statcov, statcovr)
        # Create TSVDUnfold object and initialise
        tsvdunf = ROOT.TSVDUnfold(datar, statcovr, binir, xinir, Adet)
        # It is possible to normalise unfolded spectrum to unit area
        tsvdunf.SetNormalize(ROOT.kFALSE)
        # Perform the unfolding with regularisation parameter kreg = self.kreg
        # - the larger kreg, the finer grained the unfolding, but the more fluctuations occur
        # - the smaller kreg, the stronger is the regularisation and the bias
        unfres = tsvdunf.Unfold(self.kreg)
        # Get the distribution of the d to cross check the regularization
        # - choose kreg to be the point where |d_i| stop being statistically significantly >>1
        ddist = tsvdunf.GetD()
        self.kreg_distribution = root_numpy.hist2array(ddist)
        # Get the distribution of the singular values
        svdist = tsvdunf.GetSV()
        # Compute the error matrix for the unfolded spectrum using toy MC
        # using the measured covariance matrix as input to generate the toys
        # 100 toys should usually be enough
        # The same method can be used for different covariance matrices separately.
        ustatcov = tsvdunf.GetUnfoldCovMatrix(statcovr, 100)
        # Now compute the error matrix on the unfolded distribution originating
        # from the finite detector matrix statistics
        uadetcov = tsvdunf.GetAdetCovMatrix(100)
        # Sum up the two (they are uncorrelated)
        ustatcov.Add(uadetcov)
        # Get the computed regularized covariance matrix (always corresponding to total uncertainty passed in constructor) and add uncertainties from finite MC statistics.
        utaucov = tsvdunf.GetXtau()
        utaucov.Add(uadetcov)
        # Get the computed inverse of the covariance matrix
        uinvcov = tsvdunf.GetXinv()
        # Convert result to a python array
        result = root_numpy.hist2array(unfres)
        error = root_numpy.hist2array(utaucov)
        # the std to plot is just:
        # np.sqrt(np.diagonal(error))
        return UnfoldingResult(result, error)
