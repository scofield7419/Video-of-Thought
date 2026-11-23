#!/usr/bin/env bash
# MotionEpic stage 2: see configs/train/stage2_grounding.yaml and docs/training.md.
# Usage: NUM_GPUS=8 bash scripts/train/stage2_grounding.sh [extra --key value overrides]
set -euo pipefail
NUM_GPUS=${NUM_GPUS:-8}
DS_CONFIG=${DS_CONFIG:-configs/deepspeed/zero2.json}
torchrun --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT:-29500}" \
  -m vot.train.train \
  --config configs/train/stage2_grounding.yaml \
  --deepspeed "${DS_CONFIG}" \
  "$@"
