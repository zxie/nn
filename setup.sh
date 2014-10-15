NN_PATH=~/nn
export PYTHONPATH=$NN_PATH:$NN_PATH/opt:$NN_PATH/dsets:$NN_PATH/nets:$NN_PATH/gpu:$NN_PATH/run:$PYTHONPATH
export PYTHONPATH=~/libs/cudamat_old:$PYTHONPATH

# Throw error if gnumpy not using GPU
export GNUMPY_USE_GPU="no"
