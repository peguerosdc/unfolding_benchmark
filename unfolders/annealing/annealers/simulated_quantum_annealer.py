import sqaod as sq
from sqaod import algorithm as algo
from .annealer import Annealer


class SimulatedQuantumAnnealer(Annealer):
    """Finds the exact solutions of a QUBO sampling from a
    Quantum Distribution obtained via a Path-Integral approach"""

    def __init__(self, trotters, Ginit=5.0, Gfin=0.01, beta=1.0 / 0.02, tau=0.99):
        """
        Parameters
        ----------
        trotters : integer
            Amount of replicas (P in the path integral formulation)
        Ginit : number
            Initial strength of the transverse magnetic field
            (Gamma_0 in the path integral formulation)
        Gfin : number
            Final strength of the transverse magnetic field. A schedule of
            magnetic fields will be created in the range [Ginit, Gfin]
        beta : number
            Inverse temperature without the Boltzmann constant
            (1/T in the path integral formulation)
        tau : number < 1
            Percentage of the field strength to reduce on every step. The
            smaller the value, the faster the annealing will be performed
        """
        super().__init__()
        self.trotters = trotters
        self.Ginit = Ginit
        self.Gfin = Gfin
        self.beta = beta
        self.tau = tau

    def __str__(self):
        return f"<SimulatedQuantumAnnealer trotters={self.trotters}>"

    def solve(self, qubo_matrix):
        # choose the solver
        solver = sq.cpu
        # instanciate solver and set the algorithm to use
        ann = solver.dense_graph_annealer(
            qubo_matrix, sq.minimize, n_trotters=self.trotters
        )
        ann.set_preferences(algorithm=algo.naive)
        # prepare to run the algorithm
        ann.prepare()
        # randomize or set x(0 or 1) to set the initial state
        ann.randomize_spin()
        # perform annealing according to the magnetic fields schedule
        G = self.Ginit
        while self.Gfin <= G:
            # call anneal_one_step to try flipping spins for (n_bits x n_trotters) times.
            ann.anneal_one_step(G, self.beta)
            G *= self.tau
        # retrieve the solutions
        # summary = sq.make_summary(ann)
        # return summary.xlist
        return ann.get_x()
