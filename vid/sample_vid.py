import os
from os.path import join as pjoin
import numpy as np
import argparse
import cv2
from ops import as_np
from model_utils import get_model_class_and_params
from optimizer import OptimizerHyperparams
from run_utils import CfgStruct
from run_utils import load_config
from train import MODEL_TYPE
from bounce import bounce_vec
np.random.seed(5)

'''
Sample video from RNN generator
'''

# FIXME PARAM
FEAT_DIM = 256
FPS = 5
OUT_SIZE = (256, 256)
SAMPLE_LENGTH = 100


def generate_next_frame(v, model):
    data = v.reshape((-1, 1))
    model.T = 1
    model.bsize = 1

    _, out = model.cost_and_grad(data, None, prev_h0=model.last_h)
    out = np.squeeze(as_np(out))

    return out


# NOTE Have output layer do this
def normalize_frame(v):
    v = (v - np.min(v)) / np.max(v)
    return v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', help='config file with run data for model to use')
    parser.add_argument('--out_file', default='out.avi')
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

    frames = np.zeros((FEAT_DIM, SAMPLE_LENGTH))
    edge_size = int(np.sqrt(FEAT_DIM))
    start_frames = bounce_vec(edge_size, n=1, T=2)
    frames[:, 0:2] = start_frames.T
    next_frame = generate_next_frame(frames[:, 0], model)

    for k in range(2, SAMPLE_LENGTH):
        next_frame = generate_next_frame(frames[:, k-1], model)
        frames[:, k] = next_frame

    # Write the video

    fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    writer = cv2.VideoWriter()
    writer.open(args.out_file, fourcc, FPS, OUT_SIZE)
    for k in xrange(frames.shape[1]):
        frame = frames[:, k].reshape((edge_size, edge_size))
        #frame = normalize_frame(frame)
        frame = np.array(frame * 256, dtype=np.uint8)
        frame = np.dstack((frame, frame, frame))
        frame = cv2.resize(frame, OUT_SIZE, interpolation=cv2.cv.CV_INTER_NN)
        writer.write(frame)
    writer.release()
