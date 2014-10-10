import os
import argparse
from os.path import join as pjoin
from dset import BrownCorpus
from nplm_unraveled import NPLM, NPLMHyperparams
from optimizer import OptimizerHyperparams
from log_utils import get_logger
from run_utils import dump_config, add_run_data

logger = get_logger()

if __name__ == '__main__':
    # TODO Be able to pass in different models into training script as well?

    nplm_hps = NPLMHyperparams()
    opt_hps = OptimizerHyperparams()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('epochs', type=int, help='number of epochs to train')
    parser.add_argument('out_dir', help='output directory to write model files')
    parser.add_argument('--cfg_file', help='cfg file for restarting run')
    nplm_hps.add_to_argparser(parser)
    opt_hps.add_to_argparser(parser)
    args = parser.parse_args()

    nplm_hps.set_from_args(args)
    opt_hps.set_from_args(args)
    cfg = args.__dict__.copy()
    add_run_data(cfg)
    dump_config(cfg, pjoin(args.out_dir, 'cfg.json'))

    # Load dataset
    dataset = BrownCorpus(args.context_size, args.batch_size)

    # Construct network
    model = NPLM(dataset, nplm_hps, opt_hps, opt='nag')

    # Run training
    for k in xrange(0, args.epochs):
        dataset.restart()
        it = 0
        while dataset.data_left():
            model.run()
            logger.info('epoch %d, iter %d, obj=%f, exp_obj=%f' % (k, it, model.opt.costs[-1], model.opt.expcosts[-1]))
            it += 1
        # Anneal
        model.opt.alpha /= args.anneal_factor
        # Save parameters
        params_file = pjoin(args.out_dir, 'params_epoch{0:02}.pk'.format(k+1))
        with open(params_file, 'wb') as fout:
            model.to_file(fout)
        # Symlink param file to latest
        sym_file = pjoin(args.out_dir, 'params.pk')
        if os.path.exists(sym_file):
            os.remove(sym_file)
        os.symlink(params_file, sym_file)
