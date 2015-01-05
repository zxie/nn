import os
import glob
import json
import argparse
import shutil
from subprocess import check_call
from time import strftime
from os.path import join as pjoin
from os.path import exists as fexists
from run_utils import TimeString, file_alive
from prettytable import PrettyTable
import joblib
from cluster_config import NUM_CPUS, DATASET, BROWSE_RUNS_KEYS
from run_utils import get_dev_acc, last_modified
from exp_config import EXP_ALIVE_SEC


def read_cfg(cfg_file, run_data):
    with open(cfg_file, 'r') as fin:
        cfg = json.load(fin)
    for key in cfg.keys():
        run_data[key] = cfg[key]


def process_run_dir(run_dir, figs=False):
    print run_dir
    run_data = dict()
    # Config file
    cfg_file = pjoin(run_dir, 'cfg.json')
    if not fexists(cfg_file):
        print 'No config file in %s' % run_dir
        return

    # Heartbeat file
    hb_file = pjoin(run_dir, 'params.pk')
    if not fexists(hb_file):
        print 'No heartbeat file in %s' % run_dir
        return

    # Epoch / complete

    # FIXME Shouldn't be necessary after current round of runs
    # where epoch written to file again
    acc_files = glob.glob(pjoin(run_dir, 'acc') + '/*.acc')
    acc_epochs = [int(os.path.splitext(os.path.basename(x))[0]) for x in acc_files]
    print 'acc_epochs:', acc_epochs
    if len(acc_epochs) == 0:
        last_acc_epoch = -1
    else:
        last_acc_epoch = max(acc_epochs)

    epoch_file = pjoin(run_dir, 'epoch')
    with open(epoch_file, 'r') as fin:
        try:
            epoch = int(fin.read().strip())
        except:
            epoch = -1
        # FIXME
        fixed_epoch = max(epoch, last_acc_epoch)
        run_data['epoch'] = fixed_epoch
    if fixed_epoch > epoch:
        with open(epoch_file, 'w') as fout:
            fout.write(str(fixed_epoch))

    # Read in dev accuracy if exists
    for k in range(epoch, -1, -1):
        dev_acc = get_dev_acc(run_dir, k)
        if dev_acc and dev_acc != -1:
            if 'dev_acc' in run_data:
                run_data['dev_acc'] = max(run_data['dev_acc'], dev_acc)
            else:
                run_data['dev_acc'] = dev_acc

    run_data['complete'] = os.path.exists(pjoin(run_dir, 'sentinel'))

    run_data['run'] = os.path.basename(run_dir)

    read_cfg(cfg_file, run_data)

    run_data['alive'] = file_alive(pjoin(run_dir, 'run.log'), max_dur_sec=EXP_ALIVE_SEC)

    if run_data['complete']:
        run_data['alive'] = "<span class='complete'>False</span>"
    elif run_data['alive']:
        run_data['alive'] = "<span class='alive'>true</span>"
    else:
        run_data['alive'] = "<span class='dead'>false</span>"

    # Parameters file

    if figs:
        plot_file = pjoin(run_dir, 'plot.png')
        cmd = 'python plot_results.py %s --out_file %s' % (run_dir, plot_file)
        # Check if heartbeat file has been modified after the plot image file
        if (not os.path.exists(plot_file)) or (last_modified(plot_file) < last_modified(hb_file)):
            print '%s modified, generating plot' % hb_file
            try:
                check_call(cmd, shell=True)
            except:
                pass
        if args.viewer_dir:
            plot_dir = pjoin(args.viewer_dir, 'plots')
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
            if os.path.exists(pjoin(run_dir, 'plot.png')):
                shutil.copyfile(pjoin(run_dir, 'plot.png'),
                        pjoin(plot_dir, '%s.png' % run_data['run']))

    return run_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', help='Directory containing runs')
    parser.add_argument('--viewer_dir', help='Write to webpage in this directory')
    parser.add_argument('--figs', action='store_true', help='Generate figures')
    args = parser.parse_args()

    # Get run directories

    run_dirs = list()
    for d in os.listdir(args.run_dir):
        folder = os.path.basename(d)
        if TimeString.match(folder) and not folder.endswith('bak'):
            print folder
            run_dirs.append(pjoin(args.run_dir, d))

    # Go through directories and read in summary statistics

    print run_dirs
    data = joblib.Parallel(n_jobs=NUM_CPUS)(joblib.delayed(process_run_dir)(run_dir, figs=args.figs) for run_dir in run_dirs)
    if None in data:
        ind = data.index(None)
        print 'Got None from process_run_dir, %s, trying again later...' % run_dirs[ind]
    while None in data:
        data.remove(None)

    # Output data in easily readable table

    keys = BROWSE_RUNS_KEYS

    if args.viewer_dir:
        for f in ['runs.html', 'viewer.css', 'viewer.js', 'jquery.tablesorter.min.js']:
            src = pjoin('viewer', f)
            dst = pjoin(args.viewer_dir, f)
            print 'Copying %s to %s' % (src, dst)
            shutil.copyfile(src, dst)
        json_data = {}
        json_data['keys'] = keys
        json_data['runs'] = data
        json_data['time'] = strftime('%Y-%m-%d %H:%M:%S')
        json_data['dset'] = DATASET
        json_data['figs'] = args.figs
        with open(pjoin(args.viewer_dir, 'data.json'), 'w') as fout:
            json.dump(json_data, fout)
    else:
        table = PrettyTable(keys)
        for j in range(len(data)):
            table.add_row([data[j][k] if k in data[j] else 'N/A' for k in keys])
        print table
