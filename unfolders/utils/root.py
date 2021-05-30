import root_numpy


def histogram_to_python(hist):
    """
    Converts a ROOT THx to a numpy array.

    Returns: histogram, edges
    """
    return root_numpy.hist2array(hist, return_edges=True)