#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

DOWNLOAD_TOKENIZE="${DOWNLOAD_TOKENIZE:-1}"
for arg in "$@"; do
  case "$arg" in
    --download-tokenize=*) DOWNLOAD_TOKENIZE="${arg#*=}" ;;
    --download-tokenize) DOWNLOAD_TOKENIZE=1 ;;
    --skip-download-tokenize) DOWNLOAD_TOKENIZE=0 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

DATASET_DOWNLOAD_PID=""
if [ "$DOWNLOAD_TOKENIZE" = "1" ]; then
  # Download the first ~2B characters of pretraining dataset
  python -m nanochat.dataset -n 8
  # Download more shards in the background
  python -m nanochat.dataset -n 370 &
  DATASET_DOWNLOAD_PID=$!
  # Train + eval tokenizer
  python -m scripts.tok_train
  python -m scripts.tok_eval
fi

# # Download the first ~2B characters of pretraining dataset
# # each data shard is ~250M chars
# # so we download 2e9 / 250e6 = 8 data shards at this point
# # each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# # look at dev/repackage_data_reference.py for details on how this data was prepared
# python -m nanochat.dataset -n 8
# # Immediately also kick off downloading more shards in the background while tokenizer trains
# # Approximately 350 shards are needed for 10B tokens of data for pretraining.
# # The maximum total number of shards available in the entire dataset is 1822.
# python -m nanochat.dataset -n 370 &
# DATASET_DOWNLOAD_PID=$!
# # train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
# python -m scripts.tok_train
# # evaluate the tokenizer (report compression ratio etc.)
# python -m scripts.tok_eval

# # -----------------------------------------------------------------------------
# # Base model (pretraining)
# echo "Waiting for dataset download to complete..."
# wait $DATASET_DOWNLOAD_PID

# Baselines (single-change runs override these)
PT_BASE_DEPTH=20
PT_BASE_HEAD=128
PT_BASE_SEQ=2048
PT_BASE_WINDOW=SSSL
PT_BASE_WARMUP=0.02
PT_BASE_WARMDOWN=0.5
PT_BASE_FINAL_LR=0.02
PT_BASE_WEIGHT_DECAY=0.2
PT_BASE_MATRIX_LR=0.02
PT_BASE_EMBED_LR=0.3
PT_BASE_ADAM_B2=0.95

PT_BASE_NUM_ITERS=1500
NUM_TRAIN_SHARDS=128
NUM_VAL_SHARDS=16

SFT_BASE_EMBED_LR=0.24
SFT_BASE_UNEMBED_LR=0.0032
SFT_BASE_MATRIX_LR=0.016
SFT_BASE_WARMUP=0.08
SFT_BASE_WARMDOWN=0.6
SFT_BASE_FINAL_LR=0.02
SFT_BASE_INIT_LR=0.8
SFT_BASE_MMLU=4
SFT_BASE_GSM=4
SFT_BASE_DEVICE_BS=16


run_pt() {
  local name="$1"; shift
  torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --run="${name}" \
    --depth="${PT_BASE_DEPTH}" \
    --head-dim="${PT_BASE_HEAD}" \
    --max-seq-len="${PT_BASE_SEQ}" \
    --window-pattern="${PT_BASE_WINDOW}" \
    --warmup-ratio="${PT_BASE_WARMUP}" \
    --warmdown-ratio="${PT_BASE_WARMDOWN}" \
    --final-lr-frac="${PT_BASE_FINAL_LR}" \
    --weight-decay="${PT_BASE_WEIGHT_DECAY}" \
    --matrix-lr="${PT_BASE_MATRIX_LR}" \
    --embedding-lr="${PT_BASE_EMBED_LR}" \
    --adam-beta2="${PT_BASE_ADAM_B2}" \
    --fp8 \
    --num-iterations="${PT_BASE_NUM_ITERS}" \
    --num_train_shards="${NUM_TRAIN_SHARDS}" \
    --num_val_shards="${NUM_VAL_SHARDS}" \
    "$@"
}

run_sft() {
  local name="$1"; shift
  torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- \
    --run="${name}" \
    --embedding-lr="${SFT_BASE_EMBED_LR}" \
    --unembedding-lr="${SFT_BASE_UNEMBED_LR}" \
    --matrix-lr="${SFT_BASE_MATRIX_LR}" \
    --warmup-ratio="${SFT_BASE_WARMUP}" \
    --warmdown-ratio="${SFT_BASE_WARMDOWN}" \
    --final-lr-frac="${SFT_BASE_FINAL_LR}" \
    --init-lr-frac="${SFT_BASE_INIT_LR}" \
    --mmlu-epochs="${SFT_BASE_MMLU}" \
    --gsm8k-epochs="${SFT_BASE_GSM}" \
    --device-batch-size="${SFT_BASE_DEVICE_BS}" \
    "$@"
}

# Pretrain
run_pt pt_depth12 --depth=12
run_pt pt_depth16 --depth=16
run_pt pt_warmup_0 --warmup-ratio=0.0
run_pt pt_warmup_0p10 --warmup-ratio=0.10
run_pt pt_finallr_0 --final-lr-frac=0.0
run_pt pt_finallr_0p05 --final-lr-frac=0.05
run_pt pt_wd_0p1 --weight-decay=0.1
run_pt pt_wd_0p3 --weight-decay=0.3
run_pt pt_elr_0p2 --embedding-lr=0.2
run_pt pt_elr_0p4 --embedding-lr=0.4
run_pt pt_beta2_0p98 --adam-beta2=0.98
run_pt pt_num_train_shards_128 --num_train_shards=128
run_pt pt_num_train_shards_256 --num_train_shards=64


# SFT
run_sft sft_emb_lr_low --embedding-lr=0.18
run_sft sft_emb_lr_high --embedding-lr=0.42
run_sft sft_unemb_lr_low --unembedding-lr=0.0024
run_sft sft_unemb_lr_high --unembedding-lr=0.0056
run_sft sft_warmup_0p02 --warmup-ratio=0.02
run_sft sft_warmup_0p15 --warmup-ratio=0.15
run_sft sft_warmdown_0p35 --warmdown-ratio=0.35
run_sft sft_warmdown_0p75 --warmdown-ratio=0.75
run_sft sft_finallr_0p01 --final-lr-frac=0.01
run_sft sft_finallr_0p05 --final-lr-frac=0.05
run_sft sft_initlr_0p6 --init-lr-frac=0.6
run_sft sft_initlr_1p0 --init-lr-frac=1.0


################ ----------------------------------------------------------------------- ##############

# In case you want to change multiple parameters at once, comment the above and uncomment this and change whatever hparams
# you want to change. Check the number of nodes to 4/8 or whatever depending on the number of GPUs you have

#  -------------------------
# # Pretrain (50) — scripts.base_train
# # -------------------------
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_wu0p10 --depth=12 --warmup-ratio=0.10 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_wu0 --depth=16 --warmup-ratio=0.0 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_wu0p10 --depth=16 --warmup-ratio=0.10 --num-iterations=1500

# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0_fl0 --warmup-ratio=0.0 --final-lr-frac=0.0 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0_fl0p05 --warmup-ratio=0.0 --final-lr-frac=0.05 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0p10_fl0 --warmup-ratio=0.10 --final-lr-frac=0.0 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0p10_fl0p05 --warmup-ratio=0.10 --final-lr-frac=0.05 --num-iterations=1500


# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_fl0 --depth=12 --final-lr-frac=0.0 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_fl0p05 --depth=12 --final-lr-frac=0.05 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_fl0 --depth=16 --final-lr-frac=0.0 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_fl0p05 --depth=16 --final-lr-frac=0.05 --num-iterations=1500

# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_wd0p1 --depth=12 --weight-decay=0.1 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_wd0p3 --depth=12 --weight-decay=0.3 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_wd0p1 --depth=16 --weight-decay=0.1 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_wd0p3 --depth=16 --weight-decay=0.3 --num-iterations=1500

# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_elr0p2 --depth=12 --embedding-lr=0.2 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_elr0p4 --depth=12 --embedding-lr=0.4 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_elr0p2 --depth=16 --embedding-lr=0.2 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_elr0p4 --depth=16 --embedding-lr=0.4 --num-iterations=1500

# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_b20p90 --depth=12 --adam-beta2=0.90 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_b20p98 --depth=12 --adam-beta2=0.98 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_b20p90 --depth=16 --adam-beta2=0.90 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_b20p98 --depth=16 --adam-beta2=0.98 --num-iterations=1500

# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_sh64 --depth=12 --num_train_shards=64 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_sh256 --depth=12 --num_train_shards=256 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_sh64 --depth=16 --num_train_shards=64 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_sh256 --depth=16 --num_train_shards=256 --num-iterations=1500


# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0_wd0p1 --warmup-ratio=0.0 --weight-decay=0.1 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0_wd0p3 --warmup-ratio=0.0 --weight-decay=0.3 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0p10_wd0p1 --warmup-ratio=0.10 --weight-decay=0.1 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0p10_wd0p3 --warmup-ratio=0.10 --weight-decay=0.3 --num-iterations=1500

# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0_elr0p2 --warmup-ratio=0.0 --embedding-lr=0.2 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0_elr0p4 --warmup-ratio=0.0 --embedding-lr=0.4 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0p10_elr0p2 --warmup-ratio=0.10 --embedding-lr=0.2 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wu0p10_elr0p4 --warmup-ratio=0.10 --embedding-lr=0.4 --num-iterations=1500

# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_fl0_wd0p1 --final-lr-frac=0.0 --weight-decay=0.1 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_fl0_wd0p3 --final-lr-frac=0.0 --weight-decay=0.3 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_fl0p05_wd0p1 --final-lr-frac=0.05 --weight-decay=0.1 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_fl0p05_wd0p3 --final-lr-frac=0.05 --weight-decay=0.3 --num-iterations=1500

# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_fl0_elr0p2 --final-lr-frac=0.0 --embedding-lr=0.2 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_fl0_elr0p4 --final-lr-frac=0.0 --embedding-lr=0.4 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_fl0p05_elr0p2 --final-lr-frac=0.05 --embedding-lr=0.2 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_fl0p05_elr0p4 --final-lr-frac=0.05 --embedding-lr=0.4 --num-iterations=1500

# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wd0p1_elr0p2 --weight-decay=0.1 --embedding-lr=0.2 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wd0p1_elr0p4 --weight-decay=0.1 --embedding-lr=0.4 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wd0p3_elr0p2 --weight-decay=0.3 --embedding-lr=0.2 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_wd0p3_elr0p4 --weight-decay=0.3 --embedding-lr=0.4 --num-iterations=1500

# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d12_wu0_sh64 --depth=12 --warmup-ratio=0.0 --num_train_shards=64 --num-iterations=1500
# torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --run=pt_d16_fl0p05_elr0p4 --depth=16 --final-lr-frac=0.05 --embedding-lr=0.4 --num-iterations=1500


# # -------------------------
# # SFT (50) — scripts.chat_sft
# # -------------------------
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p18_um0p0024 --embedding-lr=0.18 --unembedding-lr=0.0024
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p18_um0p0056 --embedding-lr=0.18 --unembedding-lr=0.0056
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p42_um0p0024 --embedding-lr=0.42 --unembedding-lr=0.0024
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p42_um0p0056 --embedding-lr=0.42 --unembedding-lr=0.0056

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p18_wu0p02 --embedding-lr=0.18 --warmup-ratio=0.02
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p18_wu0p15 --embedding-lr=0.18 --warmup-ratio=0.15
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p42_wu0p02 --embedding-lr=0.42 --warmup-ratio=0.02
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p42_wu0p15 --embedding-lr=0.42 --warmup-ratio=0.15

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p18_wd0p35 --embedding-lr=0.18 --warmdown-ratio=0.35
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p18_wd0p75 --embedding-lr=0.18 --warmdown-ratio=0.75
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p42_wd0p35 --embedding-lr=0.42 --warmdown-ratio=0.35
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p42_wd0p75 --embedding-lr=0.42 --warmdown-ratio=0.75

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p18_fl0p01 --embedding-lr=0.18 --final-lr-frac=0.01
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p18_fl0p05 --embedding-lr=0.18 --final-lr-frac=0.05
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p42_fl0p01 --embedding-lr=0.42 --final-lr-frac=0.01
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p42_fl0p05 --embedding-lr=0.42 --final-lr-frac=0.05

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p18_il0p6 --embedding-lr=0.18 --init-lr-frac=0.6
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p18_il1p0 --embedding-lr=0.18 --init-lr-frac=1.0
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p42_il0p6 --embedding-lr=0.42 --init-lr-frac=0.6
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p42_il1p0 --embedding-lr=0.42 --init-lr-frac=1.0

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0024_wu0p02 --unembedding-lr=0.0024 --warmup-ratio=0.02
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0024_wu0p15 --unembedding-lr=0.0024 --warmup-ratio=0.15
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0056_wu0p02 --unembedding-lr=0.0056 --warmup-ratio=0.02
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0056_wu0p15 --unembedding-lr=0.0056 --warmup-ratio=0.15

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0024_wd0p35 --unembedding-lr=0.0024 --warmdown-ratio=0.35
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0024_wd0p75 --unembedding-lr=0.0024 --warmdown-ratio=0.75
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0056_wd0p35 --unembedding-lr=0.0056 --warmdown-ratio=0.35
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0056_wd0p75 --unembedding-lr=0.0056 --warmdown-ratio=0.75

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0024_fl0p01 --unembedding-lr=0.0024 --final-lr-frac=0.01
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0024_fl0p05 --unembedding-lr=0.0024 --final-lr-frac=0.05
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0056_fl0p01 --unembedding-lr=0.0056 --final-lr-frac=0.01
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_um0p0056_fl0p05 --unembedding-lr=0.0056 --final-lr-frac=0.05

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wu0p02_wd0p35 --warmup-ratio=0.02 --warmdown-ratio=0.35
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wu0p02_wd0p75 --warmup-ratio=0.02 --warmdown-ratio=0.75
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wu0p15_wd0p35 --warmup-ratio=0.15 --warmdown-ratio=0.35
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wu0p15_wd0p75 --warmup-ratio=0.15 --warmdown-ratio=0.75

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wu0p02_fl0p01 --warmup-ratio=0.02 --final-lr-frac=0.01
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wu0p02_fl0p05 --warmup-ratio=0.02 --final-lr-frac=0.05
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wu0p15_fl0p01 --warmup-ratio=0.15 --final-lr-frac=0.01
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wu0p15_fl0p05 --warmup-ratio=0.15 --final-lr-frac=0.05

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wd0p35_fl0p01 --warmdown-ratio=0.35 --final-lr-frac=0.01
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wd0p35_fl0p05 --warmdown-ratio=0.35 --final-lr-frac=0.05
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wd0p75_fl0p01 --warmdown-ratio=0.75 --final-lr-frac=0.01
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_wd0p75_fl0p05 --warmdown-ratio=0.75 --final-lr-frac=0.05

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_fl0p01_il0p6 --final-lr-frac=0.01 --init-lr-frac=0.6
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_fl0p01_il1p0 --final-lr-frac=0.01 --init-lr-frac=1.0
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_fl0p05_il0p6 --final-lr-frac=0.05 --init-lr-frac=0.6
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_fl0p05_il1p0 --final-lr-frac=0.05 --init-lr-frac=1.0

# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p18_um0p0024_wu0p02 --embedding-lr=0.18 --unembedding-lr=0.0024 --warmup-ratio=0.02
# torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- --run=sft_em0p42_wd0p75_fl0p05 --embedding-lr=0.42 --warmdown-ratio=0.75 --final-lr-frac=0.05