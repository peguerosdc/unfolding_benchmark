# Unfolding benchmark

Comparison of the following data unfolding algorithms applied to a High-Energy physics case:

* **Simulated Quantum Annealing:** this algorithm is from the paper [arXiv:1908.08519](https://arxiv.org/abs/1908.08519) and the implementation is adapted (after some refactoring) from its computational appendix [/rdisipio/quantum_unfolding](https://github.com/rdisipio/quantum_unfolding) but using [/shinmorino/sqaod](https://github.com/shinmorino/sqaod) as the simulator.

* **Simulated Annealing:** this algorithm is based on the same idea as quantum annealing, but samples are taken from a classical Boltzmann distribution (hence it's not "quantum"). D-Wave's implementation is used as found in [/dwavesystems/dimod](https://github.com/dwavesystems/dimod).

* **SVD:** this algorithm based on "Singular Value Decomposition" is proposed at [arXiv:1112.2226](https://arxiv.org/abs/1112.2226) and the implementation is taken from ROOT's [TSVDUnfold](https://root.cern.ch/doc/master/classTSVDUnfold.html).

Additionaly, the Quantum Annealing algorithm can be run on real D-Wave's quantum computers (as adapted from the original code) and on AWS's backends.

## Installation

All python dependencies are listed in `requirements.txt` so running `pip install requirements.txt` should be enough (if you find some are missing, please create an issue so I can include them). For details on what is required for each algorithm, please visit the corresponding repositories.

Additionally, to run the Simulated Quantum Annealing algorithm, follow [sqaod's installation guide](https://github.com/shinmorino/sqaod/wiki/Installation) (or [build it from source](https://github.com/shinmorino/sqaod/wiki/Build-from-source)) as some C dependencies are required. Also, to run the SVD algorithm, follow the installation instructions at [ROOT's website](https://root.cern/install/).

If you want to run the algorithms on a D-Wave's QPU, please follow the configuration instructions of the [Ocean SDK](https://docs.ocean.dwavesys.com/en/stable/overview/install.html). Instructions to use AWS backends can be found on the Amazon's documentation, but a brief summary is provided below.

### Setting up AWS backends

1. Create an AWS account

2. Create an IAM user with `AmazonBraketFullAccess` according to ["Managing access to Amazon Braket"](https://docs.aws.amazon.com/braket/latest/developerguide/braket-manage-access.html).

3. Set up [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html):

    3.1. Perform the installation `pip install boto3`
    
    3.2. Create the configuration files with the access and secret keys of the user created in step 2.

4. Install the [amazon-braket-sdk](https://github.com/aws/amazon-braket-sdk-python): `pip install amazon-braket-sdk`

5. Create an [S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) with a folder to store the results of the experiments.

6. Test the `amazon-braket-sdk` with the test circuit in the ["Usage"](https://github.com/aws/amazon-braket-sdk-python#usage) section using the S3 bucket created in step 5.

## License

Licensed under the [MIT License](https://github.com/peguerosdc/unfolding_benchmark/blob/master/LICENSE).