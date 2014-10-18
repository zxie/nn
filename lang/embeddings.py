import numpy as np
import os
import argparse
from os.path import join as pjoin
#from brown_corpus import BrownCorpus
from char_corpus import CharCorpus
#from nplm import NPLM, NPLMHyperparams
from nclm import NCLM, NCLMHyperparams
from optimizer import OptimizerHyperparams
from run_utils import CfgStruct
from run_utils import load_config
from sklearn.neighbors import KDTree
from ops import as_np

'''
Examine the embeddings learned by neural net language model
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', help='config file with run data for model to use')
    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    model_hps = NCLMHyperparams()
    opt_hps = OptimizerHyperparams()
    model_hps.set_from_dict(cfg)
    opt_hps.set_from_dict(cfg)
    cfg = CfgStruct(**cfg)

    # Load dataset
    #dataset = BrownCorpus(model_hps.context_size, model_hps.batch_size, subset='dev')
    dataset = CharCorpus(model_hps.context_size, model_hps.batch_size, subset='dev')

    # Construct network
    model = NCLM(dataset, model_hps, opt_hps, train=False, opt='nag')
    # Load parameters
    with open(pjoin(os.path.dirname(args.cfg_file), 'params.pk'), 'rb') as fin:
        model.from_file(fin)

    embeddings = as_np(model.params['C']).T
    # NOTE Normalizing
    embeddings = embeddings / np.sqrt(np.sum(np.square(embeddings), axis=1)).reshape((-1, 1))
    tree = KDTree(embeddings, leaf_size=30, metric='euclidean')
    #query = embeddings[model.dset.word_inds['king'], :]
    query = embeddings[model.dset.char_inds['e'], :]
    # PARAM
    dists, inds = tree.query(query, k=10)
    for dist, ind in zip(dists.ravel(), inds.ravel()):
        #print model.dset.words[ind], dist
        print model.dset.chars[ind], dist
