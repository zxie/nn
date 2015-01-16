import os
from os.path import join as pjoin
import numpy as np
import argparse
from ops import array, as_np
from model_utils import get_model_class_and_params
from optimizer import OptimizerHyperparams
from run_utils import CfgStruct
from run_utils import load_config
from train import MODEL_TYPE
import matplotlib.pyplot as plt
from bounce import bounce_vec, show_V
np.random.seed(5)

'''
Sample video from RNN generator
'''

# FIXME PARAM
FEAT_DIM = 256


def generate_next_frame(v, model):
    data = v.reshape(-1, 1)
    model.T = 1
    model.bsize = 1

    _, out = model.cost_and_grad(data, None, prev_h0=model.last_h)
    out = np.squeeze(as_np(out))

    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', help='config file with run data for model to use')
    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    model_class, model_hps = get_model_class_and_params(MODEL_TYPE)
    opt_hps = OptimizerHyperparams()
    model_hps.set_from_dict(cfg)
    opt_hps.set_from_dict(cfg)
    cfg = CfgStruct(**cfg)

    # Construct network
    model = model_class(None, model_hps, opt_hps, train=False)

    with open(pjoin(os.path.dirname(args.cfg_file), 'params_save_every.pk'), 'rb') as fin:
        model.from_file(fin)

    model.last_h = None

    SAMPLE_LENGTH = 100
    frames = np.zeros((FEAT_DIM, SAMPLE_LENGTH))
    start_frame = bounce_vec(int(np.sqrt(FEAT_DIM)), T=1)
    frames[:, 0] = start_frame

    for k in range(1, SAMPLE_LENGTH):
        next_frame = generate_next_frame(frames[:, -1], model)
        frames[:, k] = next_frame

    #frames = bounce_vec(int(np.sqrt(FEAT_DIM)), T=100)

    show_V(frames.T)
