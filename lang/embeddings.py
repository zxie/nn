import numpy as np
import os
import argparse
from os.path import join as pjoin
from brown_corpus import BrownCorpus
from nplm_unraveled import NPLM, NPLMHyperparams
from optimizer import OptimizerHyperparams
from run_utils import CfgStruct
from run_utils import load_config
from sklearn.neighbors import KDTree
'''
Examine the embeddings learned by neural net language model
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', help='config file with run data for model to use')
    args = parser.parse_args()

    cfg = load_config(args.cfg_file)
    nplm_hps = NPLMHyperparams()
    opt_hps = OptimizerHyperparams()
    nplm_hps.set_from_dict(cfg)
    opt_hps.set_from_dict(cfg)
    cfg = CfgStruct(**cfg)

    # Load dataset
    dataset = BrownCorpus(nplm_hps.context_size, nplm_hps.batch_size, subset='dev')

    # Construct network
    model = NPLM(dataset, nplm_hps, opt_hps, train=False, opt='nag')
    # Load parameters
    with open(pjoin(os.path.dirname(args.cfg_file), 'params.pk'), 'rb') as fin:
        model.from_file(fin)

    embeddings = model.params['C'].as_numpy_array().T
    # NOTE Normalizing
    embeddings = embeddings / np.sqrt(np.sum(np.square(embeddings), axis=1)).reshape((-1, 1))
    tree = KDTree(embeddings, leaf_size=30, metric='euclidean')
    query = embeddings[model.dset.word_inds['king'], :]# - embeddings[model.dset.word_inds['man'], :]
    query = query / np.linalg.norm(query)
    # PARAM
    dists, inds = tree.query(query, k=10)
    for dist, ind in zip(dists.ravel(), inds.ravel()):
        print model.dset.words[ind], dist
