#!/bin/bash
#SBATCH --job-name=sgmoe
#SBATCH --partition=pg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sgmoe-%j.out
#SBATCH --error=logs/sgmoe-%j.err

set -euo pipefail
# Run from your project directory on the Concordia SPEED cluster.
# Adjust the requested resources to match your account / partition limits.
source myvenv/bin/activate
mkdir -p logs
python train.py \
    --train_jsonl "${TRAIN_JSONL:-examples/train.jsonl}" \
    --image_root "${IMAGE_ROOT:-examples/data}" \
    --model_id "${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}" \
    --output_dir "${OUTPUT_DIR:-runs/sgmoe}" \
    --max_steps "${MAX_STEPS:-100}" \
    --last_n_layers 4 --shared_rank 4 --private_rank 4 --num_private 3 \
    --top_k 1 --init spectral --dtype bf16
