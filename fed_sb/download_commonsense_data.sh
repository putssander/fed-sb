#!/bin/bash
# Download script for Commonsense Reasoning datasets

# Create directories
mkdir -p data/commonsense

echo "Downloading fine-tuning dataset..."
# Download fine-tuning data
wget -O data/commonsense/commonsense_170k.json \
  https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/ft-training_set/commonsense_170k.json

echo "Downloading evaluation datasets..."
# Base URL for evaluation datasets
BASE_URL="https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/dataset"

# List of evaluation datasets
datasets=(
  "boolq"
  "piqa"
  "social_i_qa"
  "hellaswag"
  "winogrande"
  "ARC-Challenge"
  "ARC-Easy"
  "openbookqa"
)

# Download each dataset
for dataset in "${datasets[@]}"; do
  echo "Downloading $dataset..."
  mkdir -p "data/commonsense/$dataset"
  
  # Download train, validation, and test files if they exist
  for split in train validation test; do
    wget -q -O "data/commonsense/$dataset/${split}.json" \
      "${BASE_URL}/${dataset}/${split}.json" 2>/dev/null || true
  done
done

echo "Download complete!"
echo "Files saved in data/commonsense/"
