#!/bin/bash

source /fs/home/banaszewski/.bashrc
mamba activate libmtl
cd /fs/home/banaszewski/pool-banaszewski/LibMTL/examples/tdc

python -u main_pm6.py \
    --wandb \
    --model-backend GRIT \
    --weighting STCH \
    --arch HPS \
    --epochs 3 \
    --train-batch-size 256 \
    --loss-reduction sum \
    --lr 0.0002 \
    --optim adamw \
    --weight_decay 1e-3 \
    --model-encoder-channels 64 \
    --model-encoder-num-layers 6 \
    --model-encoder-dropout 0.1 \
    --model-encoder-pe-dim 10 \
    --model-decoder-channels 32 \
    --model-decoder-num-layers 3 \
    --model-decoder-dropout 0.1 \