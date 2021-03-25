import numpy as np
from .decimal2binary import BinaryEncoder, laplacian
import logging

logger = logging.getLogger(__name__)


class QUBOSystematics:
    """
    Systematics wrapper. Contains a pair of shifts and n_bits to encode
    """

    def __init__(self, h_syst: np.array, n_bits: int):
        """
        :param h_syst:      systematic shifts wrt nominal
        :param n_bits:      encoding
        """
        self.h_syst = h_syst
        self.n_bits = n_bits


class QUBOUnfolder(object):
    """
    QUBO Data Unfolder based (after some refactoring) on https://github.com/rdisipio/quantum_unfolding
    """

    def __init__(
        self,
        truth,
        R,
        data,
        n_bits,
        weight_regularization=0.0,
        systematics=[],
        weight_systematics=0.0,
        syst_range=2.0,
    ):
        """
        Creates an unfolder based on the minimization of a QUBO function

        Parameters
        ----------
        truth : numpy.array
            True distribution. This represents what the bins look like without
            the effects of the detector. This param is only used to get the amount
            of bins
        R : numpy.array
            Response matrix. This represents what the detector does to the data
        data : numpy.array
            This is the data/distribution to be unfolded.
        n_bits : int or np.array
            Amount of bits to encode each bin
        weight_regularization : float
            Weight to apply to the regularization penalties
        systematics : list[QUBOSystematics]
            List of systematics to apply
        weight_systematics : float
            Weight to apply to the systematic penalties
        syst_range : float
            Systematics range in units of standard deviation
        """

        # Store the data
        self.R = R
        self.data = data
        # validate the amount of bins in the truth distribution
        self.n_bins_truth = truth.shape[0]
        if not self.R.shape[1] == self.n_bins_truth:
            raise ValueError(
                "Number of bins at truth level do not match between 1D spectrum (%i) and response matrix (%i)"
                % (self.n_bins_truth, self.R.shape[1])
            )

        # Tikhonov regularization and its penalty weight
        self.D = []
        self.lmbd = weight_regularization

        # Systematics
        self.syst_range = syst_range
        self.gamma = weight_systematics
        # Systematics binary encoding
        self.syst = []
        self.rho_systs = []
        for syst in systematics:
            self.syst.append(np.copy(syst.h_syst))
            self.rho_systs.append(int(syst.n_bits))

        # if encoding is still a int number, change it to
        # an array of Nbits per bin
        self.rho = (
            np.array([n_bits] * self.n_bins_truth)
            if isinstance(n_bits, int)
            else n_bits
        )
        # binary encoding
        self._encoder = BinaryEncoder(self.rho, auto_scaling=0.5)
        self._encoder.auto_encode(truth)

        # convert the problem into a binary problem
        self._add_systematics_to_problem()
        # make QUBO matrix
        self.qubo_matrix = self._make_qubo_matrix()

    def _add_systematics_to_problem(self):
        """
        Add the systematics to the encoded problem
        Auto encoding derives best-guess values for alpha_i and beta_ia
        based on a scaling parameter (e.g. +- 50% ) and the truth signal distribution
        """
        n_syst = len(self.rho_systs)
        # add systematics (if any)
        if n_syst > 0:
            # Convert list to a numpy array
            self.rho_systs = np.array(self.rho_systs, dtype="uint")

            n_bits_syst = np.sum(self.rho_systs)
            alpha_syst = np.zeros(n_syst)
            beta_syst = np.zeros([n_syst, n_bits_syst])

            for isyst in range(n_syst):
                # Fill alpha systematics
                alpha_syst[isyst] = -self.syst_range
                # Fill beta systematics
                n_bits = self.rho_systs[isyst]
                w = 2.0 * abs(self.syst_range) / np.power(2, n_bits)
                for j in range(n_bits):
                    a = int(np.sum(self.rho_systs[:isyst]) + j)
                    beta_syst[isyst][a] = w * np.power(2, n_bits - j - 1)
                # Update rho
                self._encoder.rho = np.append(self._encoder.rho, [n_bits])

            # Update alpha systematics
            self._encoder.alpha = np.append(self._encoder.alpha, alpha_syst)
            # Update beta systematics
            n_bins = self._encoder.beta.shape[0]
            n_bits_0 = self._encoder.beta.shape[1]
            self._encoder.beta = np.block(
                [
                    [self._encoder.beta, np.zeros([n_bins, n_bits_syst])],
                    [np.zeros([n_syst, n_bits_0]), beta_syst],
                ]
            )
            # Debugging
            logger.debug("Encoding systematics...")
            logger.debug(f"Rho systs: {self.rho_systs}")
            logger.debug(f"Alpha systs: {alpha_syst}")
            logger.debug(f"Beta systs: {beta_syst}")

        logger.debug(f"Alpha matrix = {self._encoder.alpha}")
        logger.debug(f"Beta matrix = \n{self._encoder.beta}")

    def _make_qubo_matrix(self):

        Nbins = self.n_bins_truth

        # regularization (Laplacian matrix)
        self.D = laplacian(self.n_bins_truth)

        # include systematics (if any)
        self.S = np.zeros([Nbins, Nbins])
        Nsyst = len(self.rho_systs)
        if Nsyst > 0:
            # matrix of systematic shifts
            T = np.vstack(self.syst).T
            logger.debug("Matrix of systematic shifts:")
            logger.debug(T)

            # update response uber-matrix
            self.R = np.block([self.R, T])
            logger.debug("Response uber-matrix:")
            logger.debug(self.R)

            # in case Nsyst>0, extend vectors and laplacian
            self.D = np.block(
                [
                    [self.D, np.zeros([Nbins, Nsyst])],
                    [np.zeros([Nsyst, Nbins]), np.zeros([Nsyst, Nsyst])],
                ]
            )

            self.S = np.block(
                [
                    [np.zeros([Nbins, Nbins]), np.zeros([Nbins, Nsyst])],
                    [np.zeros([Nsyst, Nbins]), np.eye(Nsyst)],
                ]
            )

            logger.debug(f"Systematics penalty matrix with strength {self.gamma}:")
            logger.debug(self.S)

        # Show Laplacian operator
        logger.debug(
            f"Laplacian operator with regularization strength {self.lmbd}: \n {self.D}"
        )

        # From now on, we will be using Einstein notation
        d = self.data
        alpha = self._encoder.alpha
        beta = self._encoder.beta
        R = self.R
        D = self.D
        S = self.S

        # Build W_jk = (R_ij R_ik + lambda * D_ij D_ik + gamma * S_ij S_ik)
        # to be used in equations A.10 and A.11 / A.12
        W = (
            np.einsum("ij,ik", R, R)
            + self.lmbd * np.einsum("ij,ik", D, D)
            + self.gamma * np.einsum("ij,ik", S, S)
        )
        logger.debug(f"W_ij = \n {W}")

        # Build coefficients of quadratic constraints (A.11)
        c_ab = 2 * np.einsum("jk,ja,kb->ab", W, beta, beta)
        c_ab = np.triu(c_ab)
        np.fill_diagonal(c_ab, 0.0)
        logger.debug(f"Quadratic coeff c_ab = \n {c_ab}")

        # Build coefficients of linear constraints (A.12)
        c_aa = (
            2 * np.einsum("jk,k,ja->a", W, alpha, beta)
            + np.einsum("jk,ja,ka->a", W, beta, beta)
            - 2 * np.einsum("ij,i,ja->a", R, d, beta)
        )
        c_aa = np.diag(c_aa)
        logger.debug(f"Linear coeff c_aa =\n {c_aa}")

        # Total coefficient matrix (right before A.11)
        Q = c_ab + c_aa

        logger.debug(f"Matrix {Q.shape} of QUBO coefficents Q_ab = \n {Q}")

        return Q

    def solve(self, backend):
        result = backend.solve(self.qubo_matrix)
        # return the decoded solution
        return self._encoder.decode(result)
