#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET_REPO_ID="${DATASET_REPO_ID:-robotics-lv/aloha-mobile-dummy-lerobot-v3}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/mobile_aloha/diffusion_policy}"

STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_STEPS="${EVAL_STEPS:-5000}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

dataset_args=("--dataset.repo_id=${DATASET_REPO_ID}")
if [[ -n "${DATASET_ROOT}" ]]; then
  for required in meta/info.json meta/stats.json data/chunk-000/file-000.parquet; do
    if [[ ! -f "${DATASET_ROOT}/${required}" ]]; then
      echo "Dataset file missing: ${DATASET_ROOT}/${required}" >&2
      exit 1
    fi
  done
  dataset_args+=("--dataset.root=${DATASET_ROOT}")
fi

cd "${REPO_ROOT}"
exec uv run lerobot-train \
  "${dataset_args[@]}" \
  --dataset.video_backend=pyav \
  --dataset.return_uint8=true \
  --dataset.eval_split=0.1 \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --policy.n_obs_steps=2 \
  --policy.horizon=64 \
  --policy.n_action_steps=32 \
  --policy.resize_shape='[240,320]' \
  --policy.crop_ratio=0.9 \
  --policy.gradient_checkpointing=true \
  --policy.do_mask_loss_for_padding=true \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=aloha_mobile_diffusion \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --eval_steps="${EVAL_STEPS}" \
  --max_eval_samples=4096 \
  --save_freq="${SAVE_FREQ}" \
  --save_checkpoint=true \
  --env_eval_freq=0 \
  --wandb.enable="${WANDB_ENABLE}"
