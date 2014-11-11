import os
from os.path import join as pjoin
import numpy as np
import argparse
#from brown_corpus import BrownCorpus
from char_corpus import CharCorpus, CONTEXT
from ops import array
from dset_utils import one_hot
from model_utils import get_model_class_and_params
from optimizer import OptimizerHyperparams
from run_utils import CfgStruct
from run_utils import load_config
from train import MODEL_TYPE

'''
Sample text from NN-LM
'''


def sample_continuation(s, model, order, alpha=1.0):
    # Higher alpha -> more and more like most likely sequence
    #data = array([model.dset.word_inds[w] for w in s[-order+1:]]).reshape(-1, 1)
    data = array(np.array([model.dset.char_inds[w] for w in s[-order+1:]])).reshape(-1, 1)
    data = one_hot(data, model.hps.output_size)
    data = data.reshape((-1, data.shape[2]))
    _, probs = model.cost_and_grad(data, None)
    #print probs.shape
    #probs = np.squeeze(as_np(probs))[:, -1]
    probs = probs.ravel()

    probs = probs ** alpha
    probs = probs / sum(probs)

    w = np.random.choice(range(model.hps.output_size), p=probs)
    #word = model.dset.words[w]
    char = model.dset.chars[w]
    #return word
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

    # Load dataset, just used for vocab
    # PARAM
    dataset = CharCorpus(LM_ORDER - 1, model_hps.batch_size, subset='dev')

    # Construct network
    model = model_class(dataset, model_hps, opt_hps, train=False, opt='nag')
    # FIXME
    with open(pjoin(os.path.dirname(args.cfg_file), 'params_save_every.pk'), 'rb') as fin:
        model.from_file(fin)

    for j in range(SAMPLES):
        sample_string = ['<null>'] * (LM_ORDER - 2) + ['<s>']
        for k in range(SAMPLE_LENGTH):
            sample_string = sample_string +\
                [sample_continuation(sample_string, model, LM_ORDER, alpha=ALPHA)]
            # Don't sample after </s>, get gibberish
            if sample_string[-1] == '</s>':
                break

        #print ' '.join(sample_string[LM_ORDER - 2:])
        print ''.join(sample_string[LM_ORDER - 2:])
