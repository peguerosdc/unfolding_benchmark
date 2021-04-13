import numpy as np


def covariance_matrix_of_result(R, regularization_factor, data_covariance):
    """
    Computes the covariance matrix of the fit based on Glenn Cowan's secc 11.6, but modified
    to the quantum annealing minimization expression (log likelihood is not used as the main
    objective function as the covariance is removed).

    The function to minimize x is considered to be:

        phi = |Rx - n|² + regularization_factor * S(x)

    where:

        * R = response matrix of the detector
        * the norm |.| is the L-2 norm
        * S(x) is the second derivative squared

    And the covariance matrix is given by:

        C_{i,j} = partiald(mu_i, n_k) * partiald(mu_j, n_l) * Cov(n_k, n_l)

    where (labeled as "partial1" in the code):

        partiald(mu_i, n_k) = - (A^-1 * B)_{i,k}

    where:

        * A = partiald(phi, (mu_i, mu_j))
        * B = partiald(phi, (n_j, mu_i))

    where:

        * partiald(phi, (mu_i, mu_j)) = 2 * (R.T * R)_{j,i} - 2 * regularization_factor * G_{i,j}
        * partiald(phi, (n_i, mu_j)) = - 2 * R_{i, j}

    Where G is the (n_bins * n_bins) matrix given by Glenn Cowan's eq 11.48
    """
    # Build G matrix
    n_bins = R.shape[1]
    G = G_matrix(n_bins)
    # Build A matrix
    A = 2 * np.transpose(R.T.dot(R)) - 2 * regularization_factor * G
    # Build B matrix
    B = -2 * R
    # Build partiald(mu_i, n_k)
    partial1 = -np.linalg.inv(A).dot(B)
    # Build C_{i,j}
    C = np.einsum("ik,jl,kl", partial1, partial1, data_covariance)
    return C


def G_matrix(M):
    """
    Returns a G matrix given by Glenn Cowan's eq 11.48 of size (M * M)
    """
    G = np.zeros(shape=(M, M))
    # First, fill the borders
    G[0, 0], G[-1, -1] = 1, 1
    G[1, 1], G[-2, -2] = 5, 5
    G[0, 1], G[1, 0], G[M - 1, M - 2], G[M - 2, M - 1] = -2, -2, -2, -2
    # Now, fill the center
    rows, cols = np.diag_indices_from(G)
    i = rows[2:-2]
    G[i, i] = 6
    G[i, i + 1], G[i, i - 1], G[i + 1, i], G[i - 1, i] = -4, -4, -4, -4
    G[i, i + 2], G[i, i - 2], G[i + 2, i], G[i - 2, i] = 1, 1, 1, 1
    return G
