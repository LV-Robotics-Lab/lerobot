#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATASET_REPO_ID="${DATASET_REPO_ID:-robotics-lv/aloha-mobile-dummy-lerobot-v3}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/mobile_aloha/pi05_lora}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

dataset_args=("--dataset.repo_id=${DATASET_REPO_ID}")
if [[ -n "${DATASET_ROOT}" ]]; then
  dataset_args+=("--dataset.root=${DATASET_ROOT}")
fi

cd "${REPO_ROOT}"
exec uv run lerobot-train \
  "${dataset_args[@]}" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=16 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=pi05_mobile_aloha_lora \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --save_freq=1000 \
  --wandb.enable="${WANDB_ENABLE}"
