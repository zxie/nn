import os
import cudamat as cm

def claim_gpu():
    if 'CUDA_DEVICE' in os.environ:
        cm.cuda_set_device(int(os.environ['CUDA_DEVICE']))
    else:
        cm.cuda_set_device(0)
