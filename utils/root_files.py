import ROOT
from array import array
import root_numpy


def _get_list_of_branches(chain):
    """
    Returns a list with the names of all the branches of the given TChain
    """
    return [b.GetName() for b in chain.GetListOfBranches()]


def _create_branches(tree, variables):
    """For the given tree, create a branch for each variable in the "variables" list and return the arrays where the values
    of each branch are meant to be stored"""
    # arrays to store the values
    arrays = []
    # create a branch per variable
    for var in variables:
        temp = array("f", [0.0])
        tree.Branch(var, temp, f"{var}/F")
        # store the array which is going to hold the value
        arrays.append(temp)
    return arrays


def split_into_test_and_training(
    source_file, target_file, source_tree_name, branches_to_store=[]
):
    """
    Splits the given ROOT file into a target file with two trees: one with
    a test sample set and one with a training sample set. The events are
    splitted 50% and 50%.
    Returns the amount of entries splitted

    Parameters
    ----------
    source_file : string
        The path of the ROOT file which is going to be splitted
    target_file : string
        The path of the ROOT file which will contain the splitted result.
        If the file already exists, it will be replaced
    source_tree_name : string
        The name of the tree containing the branches to split in the source_file
    branches_to_store : list(string), optional
        A list containing the variables/leafs to split in the source_tree_name.
        If None or empty, all variables will be stored in the target_file.
    """
    # Source file
    chain = ROOT.TChain(source_tree_name)
    chain.AddFile(source_file)

    # Store all the branches if no branches_to_store is given
    if not branches_to_store:
        branches_to_store = _get_list_of_branches(chain)

    # Target file
    f = ROOT.TFile(target_file, "RECREATE")

    # Create the tree to hold the training set
    training_tree = ROOT.TTree(
        f"{source_tree_name}_training", f"Training {source_file}"
    )
    # Create the training branches
    training_branches = _create_branches(training_tree, branches_to_store)

    # Create the tree to hold the sample set
    test_tree = ROOT.TTree(f"{source_tree_name}_test", f"Test {source_file}")
    # Create the sample branches
    test_branches = _create_branches(test_tree, branches_to_store)

    # Iterate all events
    entries = chain.GetEntries()
    for i in range(entries):
        # Retrieve event to be accessible by the chain
        chain.GetEntry(i)
        # Decide if the event belongs to the training branches or to the test branches
        branch = training_branches if i % 2 == 0 else test_branches
        tree = training_tree if i % 2 == 0 else test_tree
        # Store every variable in the corresponding array and in its corresponding tree
        for this_array, variable in zip(branch, branches_to_store):
            this_array[0] = getattr(chain, variable)
        tree.Fill()

    # Save the trees
    training_tree.Write("", ROOT.TObject.kOverwrite)
    test_tree.Write("", ROOT.TObject.kOverwrite)
    f.Close()
    return entries


def compute_response_and_histograms(
    source_file,
    source_tree_name,
    get_truth,
    get_detected,
    histogram_metadata=(0, 0, 1),
    variable_name="no_name",
):
    """
    From a given ROOT file, computes the histogram of two variabes (the "truth"
    and the "detected" variables) and stores them with its response matrix in
    another ROOT file.

    Returns (as a 3-tuple of ROOT objects):
        (TH1D truth histogram, TH1D detected histogram, TH2D response matrix)

    Parameters
    ----------
    source_file : string
        Path of the input ROOT file
    source_tree_name : string
        Name of the tree in the source_file that contains the data
    get_truth : (entry) => number
        Function to receive an entry/event of the source_file and to return
        the truth variable
    get_detected : (entry) => number
        Function to receive an entry/event of the source_file and to return
        the detected version of the truth variable
    histogram_metadata : tuple(number, number, number)
        Tuple containing the following metadata of the histograms to build:
        (amount_of_bins, bin_min, bin_max)
    variable_name : string, optional
        Name of the variables to be used as titles of the histograms
    """
    # Source file
    chain = ROOT.TChain(source_tree_name)
    chain.Add(source_file)

    # Define the histograms metadata
    bins, rmin, rmax = histogram_metadata

    # Define the histograms
    hist_mc = ROOT.TH1D("truth", f"Truth {variable_name}", bins, rmin, rmax)
    hist_detected = ROOT.TH1D("detected", f"Detected {variable_name}", bins, rmin, rmax)
    response = ROOT.TH2D(
        "response",
        f"Response matrix of {variable_name}",
        bins,
        rmin,
        rmax,
        bins,
        rmin,
        rmax,
    )

    # Iterate all events
    for i in range(chain.GetEntries()):
        # Retrieve event to be accessible by the chain
        chain.GetEntry(i)
        # Add the value as generated by the simulation
        truth = get_truth(chain)
        hist_mc.Fill(truth)
        # Add the value as detected/reconstructed by the experiment
        detected = get_detected(chain)
        hist_detected.Fill(detected)
        # Add the value to the response matrix
        response.Fill(detected, truth)

    return hist_mc, hist_detected, response


def load_response_and_histograms(source_file):
    """
    Loads the histograms and the response matrix from a ROOT file as generated by
    compute_response_and_histograms.

    Returns the following array:

        [truth histogram, detected histogram, response matrix]

    where each element is a tuple of numpy elements with the following shape:

        (histogram, edges)

    Parameters
    ----------
    source_file : string
        Path of the ROOT file to load
    """
    # Read the histograms of the given file
    storage = ROOT.TFile(source_file, "READ")

    # Turn ROOT objects into python objects
    truth = root_numpy.hist2array(storage.truth, return_edges=True)
    data = root_numpy.hist2array(storage.detected, return_edges=True)
    response = root_numpy.hist2array(storage.response, return_edges=True)

    return [truth, data, response]


def store_root_objects_in_file(filename, *objects):
    """
    Stores the given objects in a ROOT file

    Parameters:
    ----------
    filename : string
        Path of the ROOT file to store the objects
    objects : ROOT objects
        Instances of ROOT objects to store in "filename".
        Must have the .Write() method.
    """
    storage = ROOT.TFile(filename, "RECREATE")
    for obj in objects:
        obj.Write()
    storage.Close()
    del storage