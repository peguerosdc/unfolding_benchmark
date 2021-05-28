from braket.ocean_plugin import BraketDWaveSampler
from .dwave_base_annealer import DWaveBaseAnnealer


class AWSAnnealer(DWaveBaseAnnealer):
    def __init__(self, s3_bucket, s3_bucket_folder, topology, solver_parameters={}):
        # Validate the S3 bucket and its folder are passed
        if s3_bucket is None or s3_bucket_folder is None:
            raise ValueError(
                "To use an AWS Annealer, s3_bucket and s3_bucket_folder " "are required"
            )
        # Store them for further use
        self.s3_folder = (s3_bucket, s3_bucket_folder)
        # Forward init process to the base class
        super().__init__(topology, solver_parameters)

    def get_hardware_sampler(self):
        # if the topology exists, create a hardware sampler for it
        if self.topology == "pegasus":
            device = "arn:aws:braket:::device/qpu/d-wave/Advantage_system1"
        else:
            device = "arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6"
        return BraketDWaveSampler(self.s3_folder, device)
