#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NODES=$3
PORT=${PORT:-29300}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NODES --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
