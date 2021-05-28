from .dwave_base_annealer import DWaveBaseAnnealer
from .direct import DirectAnnealer
from .qbsolver import QBAnnealer
from .kerberos import KerberosAnnealer


class DWaveDirectAnnealer(DirectAnnealer, DWaveBaseAnnealer):
    pass


class DWaveQBAnnealer(QBAnnealer, DWaveBaseAnnealer):
    pass


class DWaveKerberosAnnealer(KerberosAnnealer, DWaveBaseAnnealer):
    pass
