# Utilities
from examples import example
from unfolders.utils import root as root_utils
import numpy as np

# Unfolding libraries
from unfolders.unfolder import Unfolder
from unfolders.annealing import (
    SimulatedAnnealingBackend,
    SimulatedQuantumAnnealingBackend,
)

# Plotting libraries
import pylab as plt

# toy generation
nbins = 40
# Generate initial distribution and response matrix
xini, bini, Adet = example.maker.generate_initial_samples(nbins)
# Generate test distribution (what is measured in the experiment)
datatrue, data, statcov = example.maker.generate_test_samples(nbins)

# Convert everything into python objects
xini, xini_edges = root_utils.histogram_to_python(xini)
bini, _ = root_utils.histogram_to_python(bini)
data, _ = root_utils.histogram_to_python(data)
statcov, _ = root_utils.histogram_to_python(statcov)
datatrue, _ = root_utils.histogram_to_python(datatrue)
R, _ = root_utils.histogram_to_python(Adet)
R_probabilities = np.true_divide(R, xini, where=xini != 0)

# Define backend
backend = SimulatedQuantumAnnealingBackend(4, 50, weight_regularization=1)

# Turn Adetpy from number of events to probabilities (to use it with quantum annealing)

# Perform unfolding
unfolder = Unfolder(data, statcov, datatrue + 1, bini, R_probabilities)
result = unfolder.unfold(backend)

# Plot the result
axis = xini_edges[0][:-1]
plt.figure(figsize=(15, 7))
# Plot the original distribution
plt.step(axis, datatrue, fillstyle="bottom", label="True data")
plt.fill_between(axis, datatrue, step="pre", alpha=0.4)
# Plot the unfolded
plt.errorbar(
    axis,
    result.solution,
    yerr=result.error,
    fmt="o",
    color="black",
    label="Unfolded Data",
)
plt.legend(prop={"size": 13})
plt.title("Annealing Example")
plt.show()