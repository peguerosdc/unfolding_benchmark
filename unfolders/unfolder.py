import numpy as np


class Unfolder(object):
    def __init__(self, data, statcov, xini, bini, R):
        """
        Parameters
        ----------
        data :
            This is the data/histogram to be unfolded
        statcov :
            This is the covariance matrix of the data
        xini :
            This is how an unfolded histogram is supposed to look
            like (i.e. a simulation of "data" without the effects
            of the detector)
        bini :
            This is how xini looks like with the effects of the detector
        R :
            This is the response matrix of probabilities used to go from
            xini to bini
        """
        # Check that R is a matrix of probabilities
        are_probabilities = np.logical_and(R >= 0, R <= 1).all()
        if not are_probabilities:
            raise ValueError("The R matrix should be a matrix of probabilities")
        # save the variables to use them later
        self.data = data
        self.statcov = statcov
        self.xini = xini
        self.bini = bini
        self.R = R

    def unfold(self, backend):
        """
        Solves the unfolding problem for the data defined in this instance.
        Returns an instance of UnfoldingResult

        Parameters
        ----------
        backend : Backend
            A Backend instance capable of performing the unfolding
        """
        return backend.solve(self.data, self.statcov, self.xini, self.bini, self.R)
