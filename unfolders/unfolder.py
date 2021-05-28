class Unfolder(object):
    def __init__(self, data, xini, bini, R):
        """
        Parameters
        ----------
        data :
            This is the data/histogram to be unfolded
        xini :
            This is how an unfolded histogram looks like (without the effects
            of the detector)
        bini :
            This is how xini looks like with the effects of the detector
        R :
            This is the response matrix used to go from xini to bini
        """
        # save the variables to use them later
        self.data = data
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
        return backend.solve(self.data, self.xini, self.bini, self.R)
