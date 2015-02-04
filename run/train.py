import os
import argparse
from os.path import join as pjoin
from char_stream import CharStream, CONTEXT
from utt_char_stream import UttCharStream
from bounce_vid import BounceVideo
from model_utils import get_model_class_and_params
from optimizer import OptimizerHyperparams
from log_utils import get_logger
from run_utils import dump_config, add_run_data
from gpu_utils import gnumpy_setup
import gnumpy as gnp
from run_utils import touch_file, TimeString
from exp_config import RUN_DIR

gnumpy_setup()
#gnp.track_memory_usage = True

# PARAM
SAVE_PARAMS_EVERY = 1000
#MODEL_TYPE = 'rnn'
MODEL_TYPE = 'brnn'
#MODEL_TYPE = 'dnn'

def main():
    # TODO Be able to pass in different models into training script as well?

    model_class, model_hps = get_model_class_and_params(MODEL_TYPE)
    opt_hps = OptimizerHyperparams()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('epochs', type=int, help='number of epochs to train')
    parser.add_argument('--opt', default='nag', help='optimizer to use', choices=['cm', 'nag'])
    parser.add_argument('--anneal_factor', type=float, default=2.0, help='annealing factor after each epoch')
    parser.add_argument('--cfg_file', help='cfg file for restarting run')
    parser.add_argument('--out_dir', default='', help='directory to output training data')
    model_hps.add_to_argparser(parser)
    opt_hps.add_to_argparser(parser)
    args = parser.parse_args()

    if not args.out_dir:
        out_dir = pjoin(RUN_DIR, str(TimeString()))
    else:
        out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    logger = get_logger(fpath=pjoin(out_dir, 'run.log'))

    model_hps.set_from_args(args)
    opt_hps.set_from_args(args)
    cfg = args.__dict__.copy()
    if not cfg['cfg_file']:
        cfg['cfg_file'] = pjoin(out_dir, 'cfg.json')
    add_run_data(cfg)
    dump_config(cfg, cfg['cfg_file'])

    # Load dataset
    #dataset = CharStream(CONTEXT, args.batch_size, step=1)
    #dataset = UttCharStream(args.batch_size)
    # FIXME PARAM
    dataset = BounceVideo(256, 128)

    # Construct network
    model = model_class(dataset, model_hps, opt_hps, opt=args.opt)

    # Run training
    for k in xrange(0, args.epochs):
        it = 0
        while dataset.data_left():
            model.run()

            if it % 1 == 0:
                logger.info('epoch %d, iter %d, obj=%f, exp_obj=%f, gnorm=%f, unorm/pnorm=%f' % (k, it, model.opt.costs[-1], model.opt.expcosts[-1], model.opt.grad_norm, model.opt.update_norm / model.opt.weight_norm))
                #gnp.memory_allocators()
                #print gnp.memory_in_use()
            it += 1
            if it % SAVE_PARAMS_EVERY == 0:
                params_file = pjoin(out_dir, 'params_save_every.pk')
                with open(params_file, 'wb') as fout:
                    model.to_file(fout)

        # Anneal
        model.opt.alpha /= args.anneal_factor

        # Save final parameters
        params_file = pjoin(out_dir, 'params_epoch{0:02}.pk'.format(k+1))
        with open(params_file, 'wb') as fout:
            model.to_file(fout)

        # Symlink param file to latest
        sym_file = pjoin(out_dir, 'params.pk')
        if os.path.exists(sym_file):
            os.remove(sym_file)
        os.symlink(params_file, sym_file)

        # Save epoch file to denote epoch completed
        epoch_file = pjoin(out_dir, 'epoch')
        with open(epoch_file, 'w') as fout:
            fout.write(str(k))

        if k != args.epochs - 1:
            model.start_next_epoch()

    # Once done write a sentinel file
    touch_file(pjoin(args.run_dir, 'sentinel'))

if __name__ == '__main__':
    main()
