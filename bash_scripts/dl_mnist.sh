#!/bin/bash

EXP_TYPE="mnist"
NOISE_TYPE="untargeted"
NUM_CLASSIFIERS=3
DATA_PATH='dl_experiments_data/mnist'
MODEL_PATH='mnist_dl_models'
MWU_ITERS=1
ALPHA=3.0
OPT_ITERS=10000
LR=.001
LOG_LEVEL="DEBUG"
PURPOSE='test'

CMD="python -m deep_learning_experiments -exp_type $EXP_TYPE -noise_type $NOISE_TYPE -num_classifiers $NUM_CLASSIFIERS
    -data_path $DATA_PATH -model_path $MODEL_PATH -mwu_iters $MWU_ITERS -alpha $ALPHA -opt_iters $OPT_ITERS
    -learning_rate $LR -log_level $LOG_LEVEL  -purpose $PURPOSE"

eval $CMD