import numpy as np
import logging

logger = logging.getLogger(__name__)


def laplacian(n):
    return (
        np.diag(2 * np.ones(n))
        + np.diag(-1 * np.ones(n - 1), 1)
        + np.diag(-1 * np.ones(n - 1), -1)
    )


class BinaryEncoder(object):
    def __init__(self, rho, auto_scaling, alpha=None, beta=None):
        """
        Creates a decimal to binary encoder

        Parameters
        ----------
        auto_scaling : float
            Scaling factor to perform the encoding
        alpha : np.array
        beta : np.array
        rho : int or np.array
            Amount of bits to encode each bin/data element
        """
        # Get the amount of bins
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.auto_scaling = auto_scaling

    def auto_encode(self, x):
        """
        if range is [ x*(1-h), x*(1+h)], e.g. x +- 50%
        alpha = x*(1-h) = lowest possible value
        width = x*(1+h) - x*(1-h) = x + hx -x + hx = 2hx
        then divide the range in n steps
        """
        auto_range = self.auto_scaling
        # rho is the number of bits
        N = len(self.rho)
        n_bits_total = sum(self.rho)
        self.alpha = np.zeros(N)
        self.beta = np.zeros([N, n_bits_total])

        for i in range(N - 1, -1, -1):
            # n is the number of bits of this entry
            n = self.rho[i]
            # the offset alpha for this entry is now set to
            # alpha is the lowest possible value as the bits:
            #    zeros = [0, 0, 0, 0, ...]
            # result in:
            #    alpha + beta.*zeros = alpha
            self.alpha[i] = (1.0 - auto_range) * x[i]
            # w = 2 * auto_range*x[i] / float(n)
            # w is the width?
            w = 2 * auto_range * x[i] / np.power(2, n)

            for j in range(n):
                a = np.sum(self.rho[:i]) + j
                self.beta[i][a] = w * np.power(2, n - j - 1)

        x_b = self.encode(x)

        return x_b

    def encode(self, x):
        """
        :param alpha: offeset
        :param beta: scaling
        :param rho: n-bits encoding
        :param x: vector in base-10
        :return: Returns binary-encoded vector
        """
        from decimal import Decimal

        # number of bins/entries
        N = len(x)
        # np.savez("test_encoding", x=x, a=self.alpha, b=self.beta)
        # l = np.load("test_encoding.npz")
        # x,alpha,beta = l['x'], l['a'],l['b']

        n_bits_total = int(sum(self.rho))
        # x_b(inary) will store the result
        x_b = np.zeros(n_bits_total, dtype="uint")
        logger.debug(f"Encoding: {x}")
        # in the loop, x[i] is the entry to encode
        for i in range(N - 1, -1, -1):
            # rho contains the amount of bits to encode each entry of x
            # with this, each bin can be encoded with a different amount
            # of bits
            n = int(self.rho[i])
            # remove the offset of this bin to this entry
            x_d = x[i] - self.alpha[i]

            logger.debug(f"- Encoding x[{i}]={x[i]}:")
            logger.debug(
                f"  - x_d = x[{i}] - alpha[{i}] = {x[i]} - {self.alpha[i]} = {x_d}"
            )
            logger.debug(f"  - Beta = {self.beta[i]}")
            # get the value of each bit
            for j in range(0, n, 1):
                # "a" is the index of the bit currently encoding in the
                # final array
                a = int(np.sum(self.rho[:i]) + j)
                beta = self.beta[i][a]
                more_than = Decimal(x_d) // Decimal(beta)
                equal_to = np.isclose(x_d, beta)

                # set the bit in the encoded
                x_b[a] = min([1, more_than or equal_to])
                x_d = x_d - x_b[a] * self.beta[i][a]
                logger.debug(
                    f"    - Bit {a}: x_d={x_d} encoded to {x_b[a]}. Rest = {x_d}"
                )

                # when the value can be encoded exactly, stop looking.
                # this is a workaround to prevent comparing small numbers in
                # np.isclose, but an appropiate atol shouldb e set to
                # handle this case. see:
                # https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
                if equal_to:
                    break

        logger.debug(f"Encoded: {x_b}")
        return x_b

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def decode(self, x_b):
        """
        :param alpha: offeset (vector)
        :param beta: scaling (array)
        :param rho: n-bits encoding (vector)
        :param x: binary vector
        :return: Returns decoded vector
        """

        N = len(self.alpha)
        x = np.zeros(N)
        for i in range(N - 1, -1, -1):
            x[i] = self.alpha[i]
            n = int(self.rho[i])
            for j in range(0, n, 1):
                a = int(np.sum(self.rho[:i]) + j)
                x[i] += self.beta[i][a] * x_b[a]

        return x
