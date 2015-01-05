import time
from subprocess import check_call
from exp_config import RUN_DIR, VIEWER_DIR, BROWSE_RUNS_WAIT_SEC


while True:
    # TODO Evaluate models here
    print 'Generating runs page'
    check_call('python browse_runs.py %s --viewer_dir %s --figs' % (RUN_DIR, VIEWER_DIR), shell=True)
    time.sleep(BROWSE_RUNS_WAIT_SEC)
