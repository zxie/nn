import os
from os.path import join as pjoin
import h5py
import numpy as np
import argparse
from char_stream import CharStream, CONTEXT
from utt_char_stream import UttCharStream
from optimizer import OptimizerHyperparams
from log_utils import get_logger
from run_utils import CfgStruct
from run_utils import load_config
from gpu_utils import gnumpy_setup
from ops import as_np
from train import MODEL_TYPE
from model_utils import get_model_class_and_params

'''
Takes a trained model and writes the likelihoods
'''

logger = get_logger()
gnumpy_setup()

def write_likelihoods(likelihoods, labels, out_file):
    f = h5py.File(out_file, 'w')
    dset = f.create_dataset('likelihoods', likelihoods.shape, dtype='float32')
    dset[...] = likelihoods
    dset = f.create_dataset('labels', labels.shape, dtype='int32')
    dset[...] = labels
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', help='config file with run data for model to use')
    parser.add_argument('--params_file', help='optionally specify params file instead of using default params.pk in cfg_file directory')
    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    model_class, model_hps = get_model_class_and_params(MODEL_TYPE)
    opt_hps = OptimizerHyperparams()

    model_hps.set_from_dict(cfg)
    opt_hps.set_from_dict(cfg)
    cfg = CfgStruct(**cfg)

    # Load dataset
    if 'rnn' not in MODEL_TYPE:
        dataset = CharStream(CONTEXT, model_hps.batch_size, subset='test')
    else:
        dataset = UttCharStream(model_hps.batch_size, subset='test')

    # Construct network
    model = model_class(dataset, model_hps, opt_hps, train=False)
    # Load parameters
    if args.params_file:
        params_file = args.params_file
    else:
        params_file = pjoin(os.path.dirname(args.cfg_file), 'params.pk')
    logger.info('Loading params from %s' % params_file)
    with open(params_file, 'rb') as fin:
        model.from_file(fin)

    likelihoods = None
    labels = None
    it = 0
    while dataset.data_left():
        cost, probs = model.run(back=False)

        if 'rnn' in MODEL_TYPE:
            if MODEL_TYPE == 'brnn':
                llt = as_np(probs).reshape((probs.shape[0], model.T, -1))
            else:
                llt = np.zeros((probs[0].shape[0], len(probs), probs[0].shape[1]))
                for t in xrange(len(probs)):
                    llt[:, t, :] = as_np(probs[t])

            # Deal with sequences in batch being of different lengths
            ll = llt[:, 0:len(model.dset.batch_labels[0]), 0].reshape((llt.shape[0], -1))
            j = 1
            for sl in model.dset.batch_labels[1:]:
                ll = np.hstack((ll, llt[:, 0:len(sl), j].reshape(llt.shape[0], -1)))
                j += 1

            y = np.array([i for sl in model.dset.batch_labels for i in sl])
        else:
            ll = as_np(probs)
            y = as_np(model.dset.batch_labels)

        if likelihoods is None:
            likelihoods = ll
            labels = y
        else:
            likelihoods = np.hstack((likelihoods, ll))
            labels = np.hstack((labels, y))

        logger.info('iter %d, cost: %f' % (it, cost))
        it += 1

    # TODO Split into multiple files for some datasets
    output_dir = pjoin(cfg.out_dir, 'likelihoods')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_likelihoods(likelihoods, labels, pjoin(output_dir, 'likelihoods.h5'))
