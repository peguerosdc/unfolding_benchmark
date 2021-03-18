import numpy as np
from .decimal2binary import BinaryEncoder, laplacian


class QUBOUnfolder(object):

    def __init__(self, truth, R, data, n_bits,
                 weight_regularization=0.0,
                 weight_systematics=0.0):
        """
        Creates an unfolder based on the minimization of a QUBO function

        Parameters
        ----------
        truth : numpy.array
            True distribution. This represents what the bins look like without
            the effects of the detector
        R : numpy.array
            Response matrix. This represents what the detector does to the data
        data : numpy.array
            This is the data/distribution to be unfolded.
        n_bits : int or np.array
            Amount of bits to encode each bin
        weight_regularization : float
            Weight to apply to the regularization penalties
        weight_systematics : float
            Weight to apply to the systematic penalties
        """

        # Store the data
        self.x = truth
        self.R = R
        self.data = data
        # Store penalty weight for Tikhonov regularization
        self.lmbd = weight_regularization
        # validate the amount of bins in the truth distribution
        self.n_bins_truth = self.x.shape[0]
        if not self.R.shape[1] == self.n_bins_truth:
            raise Exception(
                "Number of bins at truth level do not match between 1D spectrum (%i) and response matrix (%i)"
                % (self.n_bins_truth, self.R.shape[1]))

        # Tikhonov regularization
        self.D = []

        # Systematics
        self.n_syst = 0
        self.syst_range = 2.  # units of standard deviation
        self.syst = []
        self.gamma = weight_systematics
        # Systematics binary encoding
        self.rho_systs = []

        # if encoding is still a int number, change it to
        # an array of Nbits per bin
        if isinstance(n_bits, int):
            N = self.x.shape[0]
            n_bits = np.array([n_bits] * N)
        self.rho = n_bits
        # binary encoding
        self._encoder = BinaryEncoder(self.rho, auto_scaling=0.5)
        self._encoder.auto_encode(truth)

        # convert the problem into a binary problem
        self.convert_to_binary()
        # make QUBO matrix
        self.qubo_matrix = self.make_qubo_matrix()

    def add_syst_1sigma(self, h_syst: np.array, n_bits=4):
        '''
        :param h_syst:      systematic shifts wrt nominal
        :param n_bits:      encoding
        :param syst_range:  range of systematic variation in units of standard deviation
        '''
        self.syst.append(np.copy(h_syst))
        self.rho_systs.append(int(n_bits))
        self.n_syst += 1

    def convert_to_binary(self):
        '''
        auto_encode derives best-guess values for alpha_i and beta_ia
        based on a scaling parameter (e.g. +- 50% ) and the truth signal distribution
        '''

        ############################################################
        # add systematics (if any)
        self.rho_systs = np.array(self.rho_systs, dtype='uint')

        n_bits_syst = np.sum(self.rho_systs)
        beta_syst = np.zeros([self.n_syst, n_bits_syst])
        alpha_syst = np.zeros(self.n_syst)

        if self.n_syst > 0:
            print("DEBUG: systematics encodings:")
            print(self.rho_systs)

        for isyst in range(self.n_syst):
            alpha_syst[isyst] = -self.syst_range
        self._encoder.alpha = np.append(self._encoder.alpha, alpha_syst)

        for isyst in range(self.n_syst):
            n_bits = self.rho_systs[isyst]

            w = 2.*abs(self.syst_range) / np.power(2, n_bits)
            #w = abs(self.syst_range) / float(n_bits)
            for j in range(n_bits):
                a = int(np.sum(self.rho_systs[:isyst]) + j)
                beta_syst[isyst][a] = w * np.power(2, n_bits - j - 1)

            self._encoder.rho = np.append(self._encoder.rho, [n_bits])

        if self.n_syst > 0:
            print("alpha (syst only):")
            print(alpha_syst)

            print("beta (syst only):")
            print(beta_syst)

            n_bins = self._encoder.beta.shape[0]
            n_bits_0 = self._encoder.beta.shape[1]

            self._encoder.beta = np.block(
                [[self._encoder.beta,
                  np.zeros([n_bins, n_bits_syst])],
                 [np.zeros([self.n_syst, n_bits_0]), beta_syst]])
        ############################################################
        print("INFO: alpha =", self._encoder.alpha)
        print("INFO: beta =")
        print(self._encoder.beta)

    def make_qubo_matrix(self):

        Nbins = self.n_bins_truth
        Nsyst = self.n_syst

        # regularization (Laplacian matrix)
        self.D = laplacian(self.n_bins_truth)

        # systematics
        self.S = np.zeros([Nbins, Nbins])

        if self.n_syst > 0:

            # matrix of systematic shifts
            T = np.vstack(self.syst).T
            print("INFO: matrix of systematic shifts:")
            print(T)

            # update response uber-matrix
            self.R = np.block([self.R, T])
            print("INFO: response uber-matrix:")
            print(self.R)

            # in case Nsyst>0, extend vectors and laplacian
            self.D = np.block(
                [[self.D, np.zeros([Nbins, Nsyst])],
                 [np.zeros([Nsyst, Nbins]),
                  np.zeros([Nsyst, Nsyst])]])

            self.S = np.block(
                [[np.zeros([Nbins, Nbins]),
                  np.zeros([Nbins, Nsyst])],
                 [np.zeros([Nsyst, Nbins]),
                  np.eye(Nsyst)]])

            print("INFO: systematics penalty matrix:")
            print(self.S)
            print("INFO: systematics penalty strength:", self.gamma)

        print("INFO: Laplacian operator:")
        print(self.D)
        print("INFO: regularization strength:", self.lmbd)

        d = self.data
        alpha = self._encoder.alpha
        beta = self._encoder.beta
        R = self.R
        D = self.D
        S = self.S

        W = np.einsum('ij,ik', R, R) + \
            self.lmbd*np.einsum('ij,ik', D, D) + \
            self.gamma*np.einsum('ij,ik', S, S)
        print("DEBUG: W_ij =")
        print(W)

        # Using Einstein notation

        # quadratic constraints
        Qq = 2 * np.einsum('jk,ja,kb->ab', W, beta, beta)
        Qq = np.triu(Qq)
        np.fill_diagonal(Qq, 0.)
        print("DEBUG: quadratic coeff Qq =")
        print(Qq)

        # linear constraints
        Ql = 2*np.einsum('jk,k,ja->a', W, alpha, beta) + \
            np.einsum('jk,ja,ka->a', W, beta, beta) - \
            2*np.einsum('ij,i,ja->a', R, d, beta)
        Ql = np.diag(Ql)
        print("DEBUG: linear coeff Ql =")
        print(Ql)

        # total coeff matrix:
        Q = Qq + Ql

        print("DEBUG: matrix of QUBO coefficents Q_ab =:")
        print(Q)
        print("INFO: size of the QUBO coeff matrix is", Q.shape)

        return Q

    def solve(self, backend, raw_results=False):
        Q = self.qubo_matrix
        # print("INFO: solving the QUBO model (size=%i)..." % len(self._bqm))
        results = backend.solve(Q)
        # decode the results if required
        if not raw_results:
            # get the best solution (i.e. the one with the lowest energy)
            best_fit = results.first
            # decode
            q = np.array(list(best_fit.sample.values()))
            results = self._encoder.decode(q)

        # return the results
        return results
