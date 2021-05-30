import numpy as np
import ROOT

# Random number generator
R = ROOT.TRandom3()

bins_min = -10
bins_max = 10


def reconstruct(xt):
    """
    Simulates the detector effect with an efficienty cut and a Gaussian
    smearing

    Returns: None if the event didn't survive the efficiency cut and its
    smeared value otherwise
    """
    xeff = 0.3 + (1.0 - 0.3) / 20.0 * (xt + 10.0)
    # Return the smeared event if the event survived the efficiency cut
    return None if R.Rndm() > xeff else xt + R.Gaus(-2.5, 0.2)


def generate_initial_samples(nbins, amount=100000):
    """
    Generates a truth distribution (xini) from a Breit-Wigner and its
    measured distribution (bini) after passing through a detector with
    response matrix Adet (see reconstruct() for more details)

    Returns: xini, bini, Adet
    """
    # create ROOT objects
    xini = ROOT.TH1D("xini", "MC truth", nbins, bins_min, bins_max)
    bini = ROOT.TH1D("bini", "MC reco", nbins, bins_min, bins_max)
    Adet = ROOT.TH2D(
        "Adet",
        "detector response",
        nbins,
        bins_min,
        bins_max,
        nbins,
        bins_min,
        bins_max,
    )
    # Fill the MC using a Breit-Wigner with mean=0.3 and width=2.5.
    for i in range(amount):
        xt = R.BreitWigner(0.3, 2.5)
        xini.Fill(xt)
        # Pass it through the detector
        x = reconstruct(xt)
        if x:
            # If the detector was able to see the event, store it
            Adet.Fill(x, xt)
            bini.Fill(x)
    return xini, bini, Adet


def generate_test_samples(nbins, amount=10000):
    """
    Generates a test distribution (datatrue) from a Gaussian and its
    measured distribution (data) after passing through a detector
    (see reconstruct() for more details).

    Returns: datatrue, data, statcov
    Where statcov is the covariance matrix of the measured histogram
    (data) assuming they come from a Poisson distribution
    """
    # create ROOT objects
    datatrue = ROOT.TH1D("datatrue", "data truth", nbins, bins_min, bins_max)
    data = ROOT.TH1D("data", "data", nbins, bins_min, bins_max)
    statcov = ROOT.TH2D(
        "statcov",
        "covariance matrix",
        nbins,
        bins_min,
        bins_max,
        nbins,
        bins_min,
        bins_max,
    )
    # Fill the "data" using a Gaussian with mean=0 and width=2
    for i in range(amount):
        xt = R.Gaus(0.0, 2.0)
        datatrue.Fill(xt)
        # Pass it through the detector
        x = reconstruct(xt)
        if x:
            data.Fill(x)
    # Fill the data covariance matrix assuming the amount of events in each bin
    # come from a Poisson distribution
    for i in range(1, data.GetNbinsX()):
        statcov.SetBinContent(i, i, data.GetBinError(i) * data.GetBinError(i))
    return datatrue, data, statcov
