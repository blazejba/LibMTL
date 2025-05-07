#!/bin/bash
source /fs/home/banaszewski/.bashrc
mamba activate libmtl
cd /fs/home/banaszewski/pool-banaszewski/LibMTL/examples/tdc

python main_pm6.py \
    --wandb \
    --weighting STCH \
    --save_path results/with_pretraining/ \
    --arch HPS \
    --epochs 5 \
    --train-batch-size 1024 \
    --lr-factor 0.9 \
    --loss-reduction sum \
    --lr 0.0002 \
    --model-encoder-channels 256 \
    --model-encoder-num-layers 5 \
    --model-encoder-dropout 0.3 \
    --model-decoder-channels 128 \
    --model-decoder-num-layers 3 \
    --model-decoder-dropout 0.3