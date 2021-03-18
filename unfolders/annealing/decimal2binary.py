import numpy as np
import random as rnd


def laplacian(n):

    lap = np.diag(2*np.ones(n)) + \
        np.diag(-1*np.ones(n-1), 1) + \
        np.diag(-1*np.ones(n-1), -1)

    return lap


def discretize_vector(x, encoding=[]):
    N = len(x)
    n = 0

    if len(encoding) == 0:
        n = 4
        encoding = np.array([n] * N)

    q = np.zeros(N * n)
    for i in range(N - 1, -1, -1):
        x_d = int(x[i])
        j = n - 1
        while x_d > 0:
            k = i * n + j
            q[k] = x_d % 2
            x_d = x_d // 2
            j -= 1
    return np.uint8(q)


def binary_matmul(A, x):

    n = x.shape[0]
    y = np.zeros(n, dtype='uint8')

    for i in range(n - 1, -1, -1):
        c = 0
        for j in range(n - 1, -1, -1):
            p = A[i][j] & x[j]
            c = y[i] & p  # carry bit if y=01+01=10=2
            y[i] ^= p
            y[i - 1] ^= c
    return y


def compact_vector(q, encoding=[]):
    n = 0
    N = 0
    if len(encoding) == 0:
        n = 4
        N = q.shape[0] // n
        encoding = np.array([n] * N)

    x = np.zeros(N, dtype='uint8')
    for i in range(N):
        for j in range(n):
            p = np.power(2, n - j - 1)
            x[i] += p * q[(n * i + j)]
    return x


def discretize_matrix(A, encoding=[]):
    # x has N elements (decimal)
    # q has Nx elements (binary)
    # A has N columns
    # D has Nn columns
    # Ax = Dq

    N = A.shape[0]
    M = A.shape[1]

    n = 0
    if len(encoding) == 0:
        n = 4

    D = np.zeros([N, M * n])

    for i in range(M):
        for j in range(n):  #-->bits
            k = (i) * n + j
            D[:, k] = np.power(2, n - j - 1) * A[:, i]
    return D


def d2b(a, n_bits=8):
    '''Convert a list or a list of lists to binary representation
    '''
    A = np.array(a, dtype='uint8')

    n_cols = A.shape[0]

    n_vectors = n_cols * n_bits

    R_b = np.zeros([n_vectors, n_vectors], dtype='uint8')
    # up to here, it seems that every entry of the matrix will be encoded with 8 bits as the
    # dimension of R_b is (n_vectors X n_vectors)

    # the complete space is spanned by (n_cols X n_bits) standard basis vectors v, i.e.:
    # ( 0, 0, ..., 1 )
    # ( 0, 1, ..., 0 )
    # ( 1, 0, ..., 0 )

    # Multiplying Rv "extracts" the column corresponding the non-zero element
    # By iteration, we can convert R from decimal to binary

    for i in range(n_vectors):
        v_bin = np.zeros(n_vectors, dtype='uint8')
        v_bin[i] = 1
        # print(v_bin)

        #v_dec = np.packbits(v_bin)
        # we are passing to compact_vector:
        # - v_bin:  the basis vector. i.e. ( 1, 0, ..., 0 ) for i=0
        # - n_bits: the amount of bits per entry
        v_dec = compact_vector(v_bin, n_bits)
        # print(x_dec)

        u_dec = np.dot(A, v_dec)
        #u_bin = np.unpackbits(u_dec)
        u_bin = discretize_vector(u_dec, n_bits)

        R_b[:, i] = u_bin

    return R_b


#####################################


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

    def set_alpha(self, alpha: np.array):
        self.alpha = np.copy(alpha)

    def set_rho(self, rho: np.array):
        self.rho = np.copy(rho)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def encode(self, x):
        '''
        :param alpha: offeset
        :param beta: scaling
        :param rho: n-bits encoding
        :param x: vector in base-10
        :return: Returns binary-encoded vector
        '''
        from decimal import Decimal
        # number of bins/entries
        N = len(x)
        np.savez("test_encoding", x=x, a=self.alpha, b=self.beta)
        # l = np.load("test_encoding.npz")
        # x,alpha,beta = l['x'], l['a'],l['b']

        n_bits_total = int(sum(self.rho))
        # x_b(inary) will store the result
        x_b = np.zeros(n_bits_total, dtype='uint')
        print(f"Encoding: {x}")
        # in the loop, x[i] is the entry to encode
        for i in range(N - 1, -1, -1):
            # rho contains the amount of bits to encode each entry of x
            # with this, each bin can be encoded with a different amount
            # of bits
            n = int(self.rho[i])
            # remove the offset of this bin to this entry
            x_d = x[i] - self.alpha[i]
            print(f"Beta = {self.beta[i]}")

            print(f"Encoding x[{i}]={x[i]} now as {x_d}")
            # get the value of each bit
            for j in range(0, n, 1):
                # "a" is the index of the bit currently encoding in the
                # final array
                a = int(np.sum(self.rho[:i]) + j)
                print(f"Bit {a} to encode x_d={x_d}")
                beta = self.beta[i][a]
                print(f"beta {beta}")
                more_than = Decimal(x_d) // Decimal(beta)
                equal_to = np.isclose(x_d, beta)

                print(f"more_than = {more_than}")
                print(f"equal_to = {equal_to}")
                # set the bit in the encoded 
                x_b[a] = min([1, more_than or equal_to])

                x_d = x_d - x_b[a] * self.beta[i][a]

                # when the value can be encoded exactly, stop looking.
                # this is a workaround to prevent comparing small numbers in
                # np.isclose, but an appropiate atol shouldb e set to
                # handle this case. see:
                # https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
                if equal_to:
                    break

        return x_b

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def auto_encode(self, x):
        ''' 
        if range is [ x*(1-h), x*(1+h)], e.g. x +- 50%
        alpha = x*(1-h) = lowest possible value
        width = x*(1+h) - x*(1-h) = x + hx -x + hx = 2hx
        then divide the range in n steps
        '''
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
            self.alpha[i] = (1. - auto_range) * x[i]
            #w = 2 * auto_range*x[i] / float(n)
            # w is the width?
            w = 2 * auto_range * x[i] / np.power(2, n)

            for j in range(n):
                a = np.sum(self.rho[:i]) + j
                self.beta[i][a] = w * np.power(2, n - j - 1)

        x_b = self.encode(x)

        return x_b

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def decode(self, x_b):
        '''
        :param alpha: offeset (vector)
        :param beta: scaling (array)
        :param rho: n-bits encoding (vector)
        :param x: binary vector
        :return: Returns decoded vector
        '''

        N = len(self.alpha)
        x = np.zeros(N)
        for i in range(N - 1, -1, -1):
            x[i] = self.alpha[i]
            n = int(self.rho[i])
            for j in range(0, n, 1):
                a = int(np.sum(self.rho[:i]) + j)
                x[i] += self.beta[i][a] * x_b[a]

        return x
