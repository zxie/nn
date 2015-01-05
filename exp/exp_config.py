import os

EXP = 'ilm'

RUN_DIR = '/bak/%s' % EXP  # FIXME
if not os.path.exists(RUN_DIR):
    os.mkdir(RUN_DIR)
VIEWER_DIR = '~/nn/exp/viewer_instance'

EXP_ALIVE_SEC = 60*10
BROWSE_RUNS_WAIT_SEC = 600
