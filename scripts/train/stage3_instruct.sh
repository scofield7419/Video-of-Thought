#!/usr/bin/env bash
# MotionEpic stage 3: see configs/train/stage3_instruct.yaml and docs/training.md.
# Usage: NUM_GPUS=8 bash scripts/train/stage3_instruct.sh [extra --key value overrides]
set -euo pipefail
NUM_GPUS=${NUM_GPUS:-8}
DS_CONFIG=${DS_CONFIG:-configs/deepspeed/zero2.json}
torchrun --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT:-29500}" \
  -m vot.train.train \
  --config configs/train/stage3_instruct.yaml \
  --deepspeed "${DS_CONFIG}" \
  "$@"
