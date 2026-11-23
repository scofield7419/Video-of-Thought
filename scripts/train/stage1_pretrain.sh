#!/usr/bin/env bash
# MotionEpic stage 1: see configs/train/stage1_pretrain.yaml and docs/training.md.
# Usage: NUM_GPUS=8 bash scripts/train/stage1_pretrain.sh [extra --key value overrides]
set -euo pipefail
NUM_GPUS=${NUM_GPUS:-8}
DS_CONFIG=${DS_CONFIG:-configs/deepspeed/zero2.json}
torchrun --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT:-29500}" \
  -m vot.train.train \
  --config configs/train/stage1_pretrain.yaml \
  --deepspeed "${DS_CONFIG}" \
  "$@"
