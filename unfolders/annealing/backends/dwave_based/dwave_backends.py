from .dwave_base_backend import DWaveBaseBackend
from .direct import DirectBackend
from .qbsolver import QBBackend
from .kerberos import KerberosBackend


class DWaveDirectBackend(DirectBackend, DWaveBaseBackend):
    pass


class DWaveQBBackend(QBBackend, DWaveBaseBackend):
    pass


class DWaveKerberosBackend(KerberosBackend, DWaveBaseBackend):
    pass
