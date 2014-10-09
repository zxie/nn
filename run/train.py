import argparse
from os.path import join as pjoin
from dset import BrownCorpus
from nplm_unraveled import NPLM, NPLMHyperparams
from optimizer import OptimizerHyperparams
from param_utils import dump_to_json
from log_utils import get_logger

logger = get_logger()

if __name__ == '__main__':
    # TODO Be able to pass in different models into training script as well?

    nplm_hps = NPLMHyperparams({})
    opt_hps = OptimizerHyperparams()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('epochs', type=int, help='number of epochs to train')
    parser.add_argument('out_dir', help='output directory to write model files')
    nplm_hps.add_to_argparser(parser)
    opt_hps.add_to_argparser(parser)
    args = parser.parse_args()

    nplm_hps.set_from_args(args)
    opt_hps.set_from_args(args)
    dump_to_json([nplm_hps, opt_hps], pjoin(args.out_dir, 'cfg.json'))

    # Load dataset
    dataset = BrownCorpus(args.context_size, args.batch_size)

    # Construct network
    model = NPLM(dataset, nplm_hps, opt_hps, opt='nag')

    # Run training
    costs = list()
    for k in xrange(0, args.epochs):
        dataset.restart()
        it = 0
        while dataset.data_left():
            cost = model.run()
            costs.append(cost)
            logger.info('epoch %d, iter %d, obj=%f' % (k, it, cost))
            it += 1
        # Anneal
        model.opt.alpha /= args.anneal_factor
        # Save parameters
        with open(pjoin(args.out_dir, 'params_epoch{0:02}.pk'.format(k+1)), 'wb') as fout:
            model.to_file(fout)
