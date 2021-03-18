import dimod
from dwave.system import FixedEmbeddingComposite
from dwave_qbsolv import QBSolv


class QBBackend:
    """
    Solver params accepted by this solver are:
        - num_repeats
        - solver_limit
    """

    def solve(self, qubo_matrix):
        # find embedding of subproblem-sized complete graph to the QPU
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
        edges_list = bqm.to_qubo()[0]
        embedding = self.get_best_embedding(self.edges_list)
        # create a sampler with the best embedding found
        sampler = FixedEmbeddingComposite(self.hardware_sampler, embedding)
        return QBSolv().sample_qubo(
                edges_list,
                solver=sampler, **self.solver_parameters)
