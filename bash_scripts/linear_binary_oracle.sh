#!/bin/bash
#
#SBATCH -t 0-04:00
#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --mem=100000
#SBATCH --mail-type=all
#SBATCH --mail-user=jperdomo@college.harvard.edu

EXP_TYPE="oracle"
NOISE_TYPE="untargeted"
NOISE_FUNC="grad_desc"
NUM_CLASSIFIERS=5
ITERS=1
ALPHA=.35
LOG_LEVEL="DEBUG"
MODEL_PATH='linear_models/binary'
DATA_PATH="linear_experiments_data/binary"
PURPOSE='cluster1'

CMD="python -m linear_experiments -exp_type $EXP_TYPE -noise_type $NOISE_TYPE -noise_func $NOISE_FUNC
    -num_classifiers $NUM_CLASSIFIERS -iters $ITERS -alpha $ALPHA -log_level $LOG_LEVEL -model_path $MODEL_PATH
    -data_path $DATA_PATH -purpose $PURPOSE"

eval $CMD