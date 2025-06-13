#!/bin/bash
#SBATCH --job-name=pm6_process
#SBATCH --array=1-20
#SBATCH --chdir=/fs/home/banaszewski/pool-banaszewski/LibMTL/examples/tdc
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=p.hpcl94c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=1-00:00:00

source /fs/home/banaszewski/.bashrc
mamba activate libmtl

# Determine which shard to process based on the array index (01‑20)
SHARD_ID=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
SHARD_PATH=/fs/home/banaszewski/pool-banaszewski/LibMTL/examples/tdc/data/pm6_raw/pm6_processed_${SHARD_ID}.parquet

python -u pm6_utils.py \
    --shard-path ${SHARD_PATH} \
    --cache-dir /fs/home/banaszewski/pool-banaszewski/LibMTL/examples/tdc/data/pm6_processed/ \
    --pe-dim 10 \
    --divide-ratio 20
