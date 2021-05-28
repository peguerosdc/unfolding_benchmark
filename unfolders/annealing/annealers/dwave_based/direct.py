import dimod
from dwave.system import FixedEmbeddingComposite
from ..utils import dimod_extract_best_fit


class DirectAnnealer:
    @dimod_extract_best_fit
    def solve(self, qubo_matrix):
        # find the best minor embedding for this problem
        print(f"Finding optimal minor embedding for {self.topology} topology ")
        # Solve
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
        edges_list = bqm.to_qubo()[0]
        embedding = self.get_best_embedding(edges_list)
        # create a sampler with the best embedding found
        sampler = FixedEmbeddingComposite(self.hardware_sampler, embedding)
        return sampler.sample(bqm, **self.solver_parameters).aggregate()
