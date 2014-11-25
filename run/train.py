import os
import argparse
from os.path import join as pjoin
#from brown_corpus import BrownCorpus
#from char_corpus import CharCorpus, CONTEXT
from char_stream import CharStream, CONTEXT
from model_utils import get_model_class_and_params
from optimizer import OptimizerHyperparams
from log_utils import get_logger
from run_utils import dump_config, add_run_data
from gpu_utils import gnumpy_setup

logger = get_logger()
gnumpy_setup()

# PARAM
SAVE_PARAMS_EVERY = 5000
MODEL_TYPE = 'nnjm'

def main():
    # TODO Be able to pass in different models into training script as well?

    model_class, model_hps = get_model_class_and_params(MODEL_TYPE)
    opt_hps = OptimizerHyperparams()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('epochs', type=int, help='number of epochs to train')
    parser.add_argument('out_dir', help='output directory to write model files')
    parser.add_argument('--cfg_file', help='cfg file for restarting run')
    model_hps.add_to_argparser(parser)
    opt_hps.add_to_argparser(parser)
    args = parser.parse_args()

    model_hps.set_from_args(args)
    opt_hps.set_from_args(args)
    cfg = args.__dict__.copy()
    if not cfg['cfg_file']:
        cfg['cfg_file'] = pjoin(args.out_dir, 'cfg.json')
    add_run_data(cfg)
    dump_config(cfg, cfg['cfg_file'])

    # Load dataset
    #dataset = BrownCorpus(args.context_size, args.batch_size)
    # FIXME Move context and data file to param...
    dataset = CharStream(CONTEXT, args.batch_size)

    # Construct network
    model = model_class(dataset, model_hps, opt_hps, opt='nag')

    # Run training
    for k in xrange(0, args.epochs):
        it = 0
        while dataset.data_left():
            model.run()
            if it % 100 == 0:
                logger.info('epoch %d, iter %d, obj=%f, exp_obj=%f' % (k, it, model.opt.costs[-1], model.opt.expcosts[-1]))
            it += 1
            if it % SAVE_PARAMS_EVERY == 0:
                params_file = pjoin(args.out_dir, 'params_save_every.pk')
                with open(params_file, 'wb') as fout:
                    model.to_file(fout)

        # Anneal
        model.opt.alpha /= args.anneal_factor

        # Save final parameters
        params_file = pjoin(args.out_dir, 'params_epoch{0:02}.pk'.format(k+1))
        with open(params_file, 'wb') as fout:
            model.to_file(fout)

        # Symlink param file to latest
        sym_file = pjoin(args.out_dir, 'params.pk')
        if os.path.exists(sym_file):
            os.remove(sym_file)
        os.symlink(params_file, sym_file)

        if k != args.epochs - 1:
            model.start_next_epoch()

if __name__ == '__main__':
    main()
