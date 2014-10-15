import os
from os.path import join as pjoin
import numpy as np
import argparse
#from brown_corpus import BrownCorpus
from char_corpus import CharCorpus
from ops import array, as_np
#from nplm import NPLM, NPLMHyperparams
from nclm import NCLM, NCLMHyperparams
from optimizer import OptimizerHyperparams
from run_utils import CfgStruct
from run_utils import load_config

'''
Sample text from NN-LM
'''


def sample_continuation(s, model, order, alpha=1.0):
    # Higher alpha -> more and more like most likely sequence
    #data = array([model.dset.word_inds[w] for w in s[-order+1:]]).reshape(-1, 1)
    data = array(np.array([model.dset.char_inds[w] for w in s[-order+1:]])).reshape(-1, 1)
    _, probs = model.cost_and_grad(data, None)
    probs = as_np(probs).ravel()

    probs = probs ** alpha
    probs = probs / sum(probs)

    w = np.random.choice(range(model.vocab_size), p=probs)
    #word = model.dset.words[w]
    char = model.dset.chars[w]
    #return word
    return char


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', help='config file with run data for model to use')
    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    #model_hps = NPLMHyperparams()
    model_hps = NCLMHyperparams()
    opt_hps = OptimizerHyperparams()
    model_hps.set_from_dict(cfg)
    opt_hps.set_from_dict(cfg)
    cfg = CfgStruct(**cfg)

    # Load dataset, just used for vocab
    #dataset = BrownCorpus(model_hps.context_size, model_hps.batch_size, subset='dev')
    dataset = CharCorpus(model_hps.context_size, model_hps.batch_size, subset='dev')

    # Construct network
    #model = NPLM(dataset, model_hps, opt_hps, train=False, opt='nag')
    model = NCLM(dataset, model_hps, opt_hps, train=False, opt='nag')
    with open(pjoin(os.path.dirname(args.cfg_file), 'params.pk'), 'rb') as fin:
        model.from_file(fin)

    SAMPLES = 100
    SAMPLE_LENGTH = 100
    # PARAM
    LM_ORDER = 12
    ALPHA = 1.0

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
