import numpy as np
from scipy import stats
import random

n_gen_norm_1 = 1_000_000
n_gen_norm_2 = 1_000_000
n_test_norm_1 = 100_000
n_test_norm_2 = 50_000
norm_gen_params_1 = [-8, 5.5]
norm_gen_params_2 = [5, 8.5]
norm_test_params_1 = [-15, 6.5]
norm_test_params_2 = [7, 5.5]


class DocProblem:
    def __init__(
        self,
        norm_test_params_1=[-15, 6.5],
        norm_test_params_2=[7, 5.5],
        smearing=(-2.0, 0.5),
    ):
        self.norm_test_params_1 = norm_test_params_1
        self.norm_test_params_2 = norm_test_params_2
        self.smearing = smearing

    def set_bins_x_ini(self, n_bins_x, bin_low_x, bin_high_x):
        self.n_bins_x = n_bins_x
        self.__bin_low_x, self.__bin_high_x = bin_low_x, bin_high_x
        self.bins_x, self.bin_centers_x = self.__bins_and_centers(
            n_bins_x, bin_low_x, bin_high_x
        )

    def set_bins_b_ini(self, n_bins_b, bin_low_b, bin_high_b):
        self.n_bins_b = n_bins_b
        self.__bin_low_b, self.__bin_high_b = bin_low_b, bin_high_b
        self.bins_b, self.bin_centers_b = self.__bins_and_centers(
            n_bins_b, bin_low_b, bin_high_b
        )

    def setup_example(self):
        self.__generate_initial_MC()
        self.__b_ini_gen = self.__generate_data(self.__x_ini_gen)
        self.__generate_test_distribution()
        self.__b_test_gen = self.__generate_data(self.__x_test_gen)
        self.__generate_response_matrix()
        self.__B = self.__build_diagonal_covariance(self.__b_test_gen)

    def get_initial_MC(self):
        return self.__x_ini_gen

    def get_initial_MC_data(self):
        return self.__b_ini_gen

    def get_response_matrix(self):
        return self.__A

    def get_test_distribution(self):
        return self.__x_test_gen

    def get_test_data(self):
        return self.__b_test_gen

    def get_test_data_covariance(self):
        return self.__B

    def true_distribution(self, x):
        peak1_integral = self.__histogram_integral(self.__x_ini_1)
        peak2_integral = self.__histogram_integral(self.__x_ini_2)
        return peak1_integral * stats.norm.pdf(
            x, norm_gen_params_2[0], norm_gen_params_2[1]
        ) + peak2_integral * stats.norm.pdf(
            x, norm_gen_params_1[0], norm_gen_params_1[1]
        )

    def test_distribution(self, x):
        peak1_integral = self.__histogram_integral(self.__x_test_1)
        peak2_integral = self.__histogram_integral(self.__x_test_2)
        return peak1_integral * stats.norm.pdf(
            x, self.norm_test_params_1[0], self.norm_test_params_1[1]
        ) + peak2_integral * stats.norm.pdf(
            x, self.norm_test_params_2[0], self.norm_test_params_2[1]
        )

    def __generate_initial_MC_peak1(self):
        self.__x_ini_1 = stats.norm.rvs(
            norm_gen_params_2[0], norm_gen_params_2[1], n_gen_norm_2
        )

    def __generate_initial_MC_peak2(self):
        self.__x_ini_2 = stats.norm.rvs(
            norm_gen_params_1[0], norm_gen_params_1[1], n_gen_norm_1
        )

    def __histogram_integral(self, gen_distr):
        hist = np.histogram(gen_distr, self.bins_x)
        integral = hist[0].sum() * (hist[1][1] - hist[1][0])
        return integral

    def __generate_initial_MC(self):
        self.__generate_initial_MC_peak1()
        self.__generate_initial_MC_peak2()
        self.__x_ini_gen = np.append(self.__x_ini_1, self.__x_ini_2)

    def __generate_test_distribution_peak1(self):
        self.__x_test_1 = stats.norm.rvs(
            self.norm_test_params_1[0], self.norm_test_params_1[1], n_test_norm_1
        )

    def __generate_test_distribution_peak2(self):
        self.__x_test_2 = stats.norm.rvs(
            self.norm_test_params_2[0], self.norm_test_params_2[1], n_test_norm_2
        )

    def __generate_test_distribution(self):
        self.__generate_test_distribution_peak1()
        self.__generate_test_distribution_peak2()
        self.__x_test_gen = np.append(self.__x_test_1, self.__x_test_2)

    def __generate_data(self, x):
        b_gen = []
        for i in x:
            i = self.__efficiency_cut(i)
            b_gen.append(self.__add_smearing(i))
        return b_gen

    def __generate_response_matrix(self):
        self.__A = np.histogram2d(
            self.__x_ini_gen, self.__b_ini_gen, bins=[self.bins_x, self.bins_b]
        )

    def __build_diagonal_covariance(self, data_gen):
        data = np.histogram(
            data_gen, bins=self.n_bins_b, range=(self.__bin_low_b, self.__bin_high_b)
        )[0]
        B = np.zeros(shape=(self.n_bins_b, self.n_bins_b))
        std_dev = np.sqrt(data)
        for i in range(self.n_bins_b):
            B[i, i] = std_dev[i] * std_dev[i]
        return B

    def __bins_and_centers(self, nbins, low, high):
        bins = np.linspace(low, high, nbins + 1)
        bin_centers = (bins + (bins[1] - bins[0]) / 2)[:-1]
        return bins, bin_centers

    def __add_smearing(self, x):
        a, b = self.smearing
        smear = np.random.normal(a, b)
        return x + smear

    def __efficiency_cut(self, x):
        eff = 0.2 + (1.0 - 0.2) / (self.bins_b[-1] - self.bins_b[0]) * (
            x + self.bins_b[-1]
        )
        eff_x = random.random()
        if eff_x > eff:
            x = -999999
        return x
