#!/bin/bash
source /fs/home/banaszewski/.bashrc
mamba activate libmtl
cd /fs/home/banaszewski/pool-banaszewski/LibMTL/examples/tdc
# wandb agent BorgwardtLab/libmtl_tdc/nxz8urzb

python main_tdc.py \
    --wandb \
    --weighting FairGrad \
    --save_path results/ \
    --arch HPS \
    --epochs 250 \
    --train-batch-size 1024 \
    --lr-factor 0.9 \
    --loss-reduction sum \
    --lr 0.0002 \
    --FairGrad_alpha 0.5 \
    --model-encoder-channels 256 \
    --model-encoder-num-layers 5 \
    --model-encoder-dropout 0.3 \
    --model-decoder-channels 64 \
    --model-decoder-num-layers 1 \
    --model-decoder-dropout 0.3 \
