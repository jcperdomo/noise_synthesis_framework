#!/bin/bash

EXP_TYPE="imagenet"
NOISE_TYPE="untargeted"
DATA_PATH='dl_experiments_data/imagenet'
MWU_ITERS=1
ALPHA=300.0
OPT_ITERS=5000
LR=.01
LOG_LEVEL="DEBUG"
PURPOSE='test'
HOLDOUT=0

CMD="python -m deep_learning_experiments -exp_type $EXP_TYPE -noise_type $NOISE_TYPE
    -data_path $DATA_PATH -mwu_iters $MWU_ITERS -alpha $ALPHA -opt_iters $OPT_ITERS
    -learning_rate $LR -log_level $LOG_LEVEL  -purpose $PURPOSE -holdout $HOLDOUT"

eval $CMD
