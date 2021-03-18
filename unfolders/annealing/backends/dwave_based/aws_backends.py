from .aws_base_backend import AWSBackend
from .direct import DirectBackend
from .qbsolver import QBBackend
from .kerberos import KerberosBackend


class AwsDirectBackend(DirectBackend, AWSBackend):
    pass


class AwsQBBackend(QBBackend, AWSBackend):
    pass


class AwsKerberosBackend(KerberosBackend, AWSBackend):
    pass
