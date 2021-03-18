"""
This example benchmarks different methods to deconvolute a sigmoid and a gaussian
"""
# annealing unfolder
from unfolders.annealing import QUBOUnfolder, backends

# utils
import matplotlib.pyplot as plt
import traceback
from utils import convolution_matrix

# numpy
import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=500)


# Utilities


def build_axis(n_points):
    return np.linspace(-10, 10, n_points)


def build_diagonal_covariance(histogram):
    n_bins_b = len(histogram)
    B = np.zeros(shape=(n_bins_b, n_bins_b))
    std_dev = np.sqrt(histogram)
    for i in range(n_bins_b):
        B[i, i] = std_dev[i] * std_dev[i]
    return B


# Functions


def sigmoid(x, amplitude=1, smoothness=1, plot_reference=False):
    # plot a reference of the sigmoid if required
    if plot_reference:
        x_ref = build_axis(100)
        z_ref = sigmoid(x_ref, amplitude=amplitude, smoothness=smoothness)
        plt.plot(x_ref, z_ref, dashes=[6, 2], label="ref_sigmoid")
    # return real sigmoid
    return amplitude / (1 + np.exp(x / smoothness))


def gaussian(x, mu, sig, amplitude=1, plot_reference=False):
    # plot a reference of the gaussian if required
    if plot_reference:
        x_ref = build_axis(100)
        g_ref = gaussian(x_ref, mu, sig, amplitude)
        plt.plot(x_ref, g_ref, dashes=[6, 2], label="ref_gauss")
    # return real gaussian
    return amplitude * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


# Program


def main(n_bins, bits, sigmoid_smoothness=2, gaussian_sigma=2, lambda_factor=0):
    """
    This example tries to recover a sigmoid from a gaussian-convoluted signal

    Parameters
    ----------
    n_bins : integer
        Amount of "bins" to use
    bits : integer
        Amount of bits to code the samples
    sigmoid_smoothness : number
        Steepness factor of the sigmoid. The higher, the smoother.
    gaussian_sigma : number
        Standard deviation of the gaussian filter
    lambda_factor : number
        Weight to apply to the regularization penalties as descrubed by the quantum annealing unfolder
    """
    x = build_axis(n_bins)
    # build the functions we want to fold/unfold
    z_ini = sigmoid(x, amplitude=1, smoothness=sigmoid_smoothness, plot_reference=True)
    z = z_ini + 0.01 * np.random.random(size=(n_bins))
    g = gaussian(x, 0, gaussian_sigma, amplitude=1, plot_reference=True)
    # Build reponse matrix
    r = convolution_matrix(g, mode="same")
    # Fold and plot for reference
    z_folded = np.dot(r, z)
    plt.step(x, z_folded, label="data")

    # Unfold with simulated annealing
    unfolder_annealing = QUBOUnfolder(
        z_ini, r, z_folded, n_bits=bits, weight_regularization=lambda_factor
    )
    annealer = backends.SimulatedAnnealingBackend(1000)
    plt.scatter(
        x,
        unfolder_annealing.solve(annealer),
        label=f"SA(λ={lambda_factor},bits={bits})",
    )

    # TODO: Unfold with SVD

    # show plot
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.title(
        f"Sigmoid(a={sigmoid_smoothness}) + Gaussian(sigma={gaussian_sigma})."
        f"{n_bins} bins"
    )
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(n_bins=10, bits=4, sigmoid_smoothness=2)
