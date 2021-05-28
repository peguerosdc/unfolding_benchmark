# Utilities
from examples import example
from utils import root_utils

# Unfolding libraries
from unfolders.unfolder import Unfolder
from unfolders.svd.backend import SVDBackend

# Plotting libraries
import pylab as plt

# toy generation
nbins = 40
# Generate initial distribution and response matrix
xini, binir, Adet = example.maker.generate_initial_samples(nbins)
# Generate test distribution (what is measured in the experiment)
datatrue, data, statcov = example.maker.generate_test_samples(nbins)

# Convert everything into python objects
xini, xini_edges = root_utils.histogram_to_python(xini)
bini, _ = root_utils.histogram_to_python(binir)
data, _ = root_utils.histogram_to_python(data)
datatrue, _ = root_utils.histogram_to_python(datatrue)
R, _ = root_utils.histogram_to_python(Adet)

# Define backend
backend = SVDBackend(13, -10, 10)

# Perform unfolding
unfolder = Unfolder(data, xini, bini, R)
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
plt.title("SVD Example")
plt.show()