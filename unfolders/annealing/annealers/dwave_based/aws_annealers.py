from .aws_base_annealer import AWSAnnealer
from .direct import DirectAnnealer
from .qbsolver import QBAnnealer
from .kerberos import KerberosAnnealer


class AwsDirectAnnealer(DirectAnnealer, AWSAnnealer):
    pass


class AwsQBAnnealer(QBAnnealer, AWSAnnealer):
    pass


class AwsKerberosAnnealer(KerberosAnnealer, AWSAnnealer):
    pass
