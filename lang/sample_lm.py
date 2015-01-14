import pickle
import os
from os.path import join as pjoin
import numpy as np
import argparse
#from brown_corpus import BrownCorpus
from preproc_char import CONTEXT
from ops import array, as_np
from dset_utils import one_hot
from dset_paths import CHAR_CORPUS_VOCAB_FILE
from model_utils import get_model_class_and_params
from optimizer import OptimizerHyperparams
from run_utils import CfgStruct
from run_utils import load_config
from train import MODEL_TYPE

'''
Sample text from NN-LM
'''


def sample_continuation(s, model, order, alpha=1.0):
    if 'rnn' in MODEL_TYPE:
        data = array(np.array([char_inds[w] for w in s[-1:]])).reshape(-1, 1)
    else:
        data = array(np.array([char_inds[w] for w in s[-order+1:]])).reshape(-1, 1)

    data = one_hot(data, model.hps.output_size)
    if MODEL_TYPE == 'brnn':
        data = data.reshape((data.shape[0], -1))
        model.T = 1
        model.bsize = 1
    if 'rnn' in MODEL_TYPE:
        _, probs = model.cost_and_grad(data, None, prev_h0=model.last_h)
        probs = np.squeeze(as_np(probs))
    else:
        data = data.reshape((-1, data.shape[2]))
        _, probs = model.cost_and_grad(data, None)
    probs = probs.ravel()

    # Higher alpha -> more and more like most likely sequence
    probs = probs ** alpha
    probs = probs / sum(probs)

    w = np.random.choice(range(model.hps.output_size), p=probs)
    char = chars[w]

    return char


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

    SAMPLES = 100
    SAMPLE_LENGTH = 100
    # PARAM
    ALPHA = 1.0
    # FIXME PARAM
    LM_ORDER = CONTEXT + 1

    with open(CHAR_CORPUS_VOCAB_FILE, 'rb') as fin:
        char_inds = pickle.load(fin)
    chars = dict((v, k) for k, v in char_inds.iteritems())

    # Construct network
    model = model_class(None, model_hps, opt_hps, train=False)
    # FIXME
    with open(pjoin(os.path.dirname(args.cfg_file), 'params_save_every.pk'), 'rb') as fin:
        model.from_file(fin)

    for j in range(SAMPLES):
        model.last_h = None
        if 'rnn' in MODEL_TYPE:
            sample_string = ['<null>'] * (LM_ORDER - 2) + ['<s>']
        else:
            sample_string = ['<s>']
        for k in range(SAMPLE_LENGTH):
            sample_string = sample_string +\
                [sample_continuation(sample_string, model, LM_ORDER, alpha=ALPHA)]
            # Don't sample after </s>, get gibberish
            if sample_string[-1] == '</s>':
                break

        #print ' '.join(sample_string[LM_ORDER - 2:])
        if 'rnn' in MODEL_TYPE:
            print ''.join(sample_string[LM_ORDER - 2:])
        else:
            print ''.join(sample_string)
