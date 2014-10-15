import os
import gnumpy as gnp
from log_utils import get_logger

logger = get_logger()

# TODO Can replace board_id with function that checks GPU
# usage and uses one that's free

def gnumpy_setup():
    # Claim GPU
    if 'CUDA_DEVICE' in os.environ:
        gnp.board_id_to_use = int(os.environ['CUDA_DEVICE'])
        logger.info('Claiming gpu %d' % gnp.board_id_to_use)
    else:
        pass
    # Turn off expensive checks
    #gnp.expensive_check_probability = 0
