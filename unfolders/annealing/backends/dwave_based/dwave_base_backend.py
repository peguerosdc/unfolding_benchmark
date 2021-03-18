from ..backend import Backend
from dwave.system import DWaveSampler
import minorminer


class DWaveBaseBackend(Backend):

    def __init__(self, topology, solver_parameters={}):
        super().__init__(solver_parameters)
        # Check if the topology is valid
        if topology not in ["chimera", "pegasus"]:
            raise ValueError("Topology must be either 'chimera' or 'pegasus'")
        self.topology = topology
        self.hardware_sampler = self.get_hardware_sampler()

    def get_hardware_sampler(self):
        return DWaveSampler(solver={'topology__type': self.topology})

    def get_best_embedding(self, source_edges):
        return minorminer.find_embedding(
            source_edges, self.hardware_sampler.edgelist)
