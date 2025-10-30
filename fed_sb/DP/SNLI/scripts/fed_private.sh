#!/bin/bash

# Disable wandb by default (use --wandb flag to enable)
export WANDB_MODE=disabled

CUDA_DEVICE=0

# Check if preprocessed data exists
if [ ! -d "DP/SNLI/data/processed_dataloaders" ]; then
    echo "Preprocessed data not found. Running with --dataset_not_processed flag..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python DP/SNLI/fed_trainer.py --lora_r 64 --epsilon 3 --dataset_not_processed
else
    echo "Using preprocessed data..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python DP/SNLI/fed_trainer.py --lora_r 64 --epsilon 3
fi
