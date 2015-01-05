import os
import multiprocessing

NUM_CPUS = multiprocessing.cpu_count() - 1

CPU_FREE_TOL = 0.1
RAM_FREE_TOL = 0.1
if 'GPU_FREE_TOL' in os.environ:
    GPU_FREE_TOL = float(os.environ['GPU_FREE_TOL'])
else:
    GPU_FREE_TOL = 0.02

SSH_TIMEOUT = 10
SSH_CMD = 'ssh -q -x -o ConnectTimeout=10 -o ServerAliveInterval=3'

SLEEP_SEC = 30

PYTHON_CMD = '/afs/cs.stanford.edu/u/zxie/virtualenvs/scl/bin/python'

# TODO Setup command

CLUSTER_NODES = {
    'gorgon': range(39, 70),
    'deep': list(range(1, 13)) + list(range(14, 17))# + range(23, 25)
}

GPU_CLUSTER_TO_USE = 'deep'

INACTIVE_GPU_NUM_CHECKS = 5
INACTIVE_GPU_WAIT_SEC = 150
