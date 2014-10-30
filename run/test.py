import os
from os.path import join as pjoin
import h5py
import numpy as np
import argparse
#from brown_corpus import BrownCorpus
from char_corpus import CharCorpus
#from nplm import NPLM, NPLMHyperparams
from nclm import NCLM, NCLMHyperparams
from optimizer import OptimizerHyperparams
from log_utils import get_logger
from run_utils import CfgStruct
from run_utils import load_config
from gpu_utils import gnumpy_setup
from ops import as_np

'''
Takes a trained model and writes the likelihoods
'''

logger = get_logger()
gnumpy_setup()

def write_likelihoods(likelihoods, out_file):
    f = h5py.File(out_file, 'w')
    dset = f.create_dataset('likelihoods', likelihoods.shape, dtype='float32')
    dset[...] = likelihoods
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', help='config file with run data for model to use')
    parser.add_argument('--params_file', help='optionally specify params file instead of using default params.pk in cfg_file directory')
    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    model_hps = NCLMHyperparams()
    opt_hps = OptimizerHyperparams()
    model_hps.set_from_dict(cfg)
    opt_hps.set_from_dict(cfg)
    cfg = CfgStruct(**cfg)

    # Load dataset
    dataset = CharCorpus(model_hps.context_size, model_hps.batch_size, subset='test')

    # Construct network
    model = NCLM(dataset, model_hps, opt_hps, train=False, opt='nag')
    # Load parameters
    if args.params_file:
        params_file = args.params_file
    else:
        params_file = pjoin(os.path.dirname(args.cfg_file), 'params.pk')
    with open(params_file, 'rb') as fin:
        model.from_file(fin)

    likelihoods = np.empty((model.likelihood_size, dataset.data.shape[1]), dtype=np.float32)
    it = 0
    while dataset.data_left():
        cost, probs = model.run(back=False)
        likelihoods[:, it*dataset.batch_size:(it+1)*dataset.batch_size] = as_np(probs)
        logger.info('iter %d, cost: %f' % (it, cost))
        it += 1

    # TODO Split into multiple files for some datasets
    output_dir = pjoin(cfg.out_dir, 'likelihoods')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_likelihoods(likelihoods, pjoin(output_dir, 'likelihoods.h5'))
