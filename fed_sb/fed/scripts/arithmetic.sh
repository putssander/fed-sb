#!/bin/bash

# Parse command-line arguments
USE_WANDB=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --wandb)
            USE_WANDB=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--wandb]"
            exit 1
            ;;
    esac
done

# Configure wandb
if [ "$USE_WANDB" = false ]; then
    export WANDB_MODE=disabled
    echo "wandb is disabled. Use --wandb flag to enable."
else
    echo "wandb is enabled."
fi

# Define experiment configurations
declare -A EXPERIMENTS=(
    ["01-a"]="--lora_r 200 --method fed-sb"
)

# Configuration
BASE_MODEL="google/gemma-2-9b"
# BASE_MODEL="google/medgemma-4b-it" --> type `gemma3` but Transformers does not recognize this architecture.
BASE_MODEL="mistralai/Mistral-7B-v0.1"
GPU_ID=0
LORA_R=$(echo "${EXPERIMENTS[@]}" | grep -o 'lora_r [^ ]*' | cut -d' ' -f2 | sort -u)
echo "Unique lora_r values: $LORA_R"
BASE_DIR="experiments/instruction_tuning" 


if [ -d "$BASE_DIR" ]; then
  echo "Directory $BASE_DIR exists."
else
  echo "Directory $BASE_DIR does not exist."
fi

MODEL_NAME=$(basename "$BASE_MODEL")
LOG_DIR="experiment_logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to handle errors
handle_error() {
    local experiment=$1
    local stage=$2
    echo "[$(date)] Error occurred in experiment '$experiment' during stage: $stage" | tee -a "$LOG_DIR/error_log.txt"
    return 1
}

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local exp_args=$2
    local start_time=$(date +%s)
    
    echo "=== Starting Experiment: $exp_name ===" | tee -a "$LOG_DIR/${exp_name}_log.txt"
    echo "Configuration: $exp_args" | tee -a "$LOG_DIR/${exp_name}_log.txt"
    
    # Extract method from args
    local method=$(echo $exp_args | grep -o 'method [^ ]*' | cut -d' ' -f2)
    
    # Extract lora_r specific to this experiment
    local current_lora_r=$(echo $exp_args | grep -o 'lora_r [^ ]*' | cut -d' ' -f2)
    
    # Remove --method argument from exp_args
    exp_args=$(echo $exp_args | sed 's/--method [^ ]*//')
    
    # Training
    echo "[$(date)] Starting Training..." | tee -a "$LOG_DIR/${exp_name}_log.txt"
    CUDA_VISIBLE_DEVICES=$GPU_ID python fed/ARITHMETIC/train_it_fed.py \
        --model $BASE_MODEL \
        $exp_args \
        --data_path meta-math/MetaMathQA \
        --dataset_split "train[:20000]" \
        --dataset_field query response \
        --batch_size 1 \
        --eg_bs 3 \
        --scheduler cosine \
        --warmup_ratio 0.02 \
        --max_seq_length 512 \
        --seed 42 \
        --num_samples 50 \
        --agg_type $method \
        --device cuda 2>&1 | tee -a "$LOG_DIR/${exp_name}_log.txt" || handle_error "$exp_name" "Training"
    
    # Debug: List directories manually
    echo "==================="
    echo "Directories found:"
    ls -td "$BASE_DIR"/"$MODEL_NAME"/* 2>/dev/null
    echo "==================="

    # Find the run directory
    local RUN_DIR=$(ls -td "$BASE_DIR"/"$MODEL_NAME"/"$method"/* 2>/dev/null | head -1)
    # #! FIXING ABOVE PATH
    # local RUN_DIR=$(ls -td "$BASE_DIR"/"$MODEL_NAME"/* 2>/dev/null | head -1)
    echo "Run DIR: $RUN_DIR"

    if [ -z "$RUN_DIR" ]; then
        handle_error "$exp_name" "Finding run directory"
        exit 1
    fi
    
    # Paths
    local MERGED_MODEL_PATH="$RUN_DIR/merged_model"

    # Merging using the experiment-specific lora_r value
    echo "[$(date)] Starting Aggregation.." | tee -a "$LOG_DIR/${exp_name}_log.txt"
    CUDA_VISIBLE_DEVICES=$GPU_ID python fed/aggregator.py \
        --model_name "$BASE_MODEL" \
        --lora_r $current_lora_r \
        --agg_type $method \
        --max_seq_length 512 \
        --dir_path "$RUN_DIR" 2>&1 | tee -a "$LOG_DIR/${exp_name}_log.txt" || handle_error "$exp_name" "Merging"

    # Evaluation
    echo "[$(date)] Starting GSM8K Evaluation..." | tee -a "$LOG_DIR/${exp_name}_log.txt"
    CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/gsm8k_eval.py \
        --model "$MERGED_MODEL_PATH" \
        --data_file "data/math_eval/gsm8k_test.jsonl" \
        --batch_size 64 \
        --tensor_parallel_size 1 \
        --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG_DIR/${exp_name}_log.txt"
    
    # Evaluation
    echo "[$(date)] Starting MATH Evaluation..." | tee -a "$LOG_DIR/${exp_name}_log.txt"
    CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/MATH_eval.py \
        --model "$MERGED_MODEL_PATH" \
        --data_file "data/math_eval/MATH_test.jsonl" \
        --batch_size 64 \
        --tensor_parallel_size 1 \
        --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG_DIR/${exp_name}_log.txt"
    
    # Cleanup
    if [ -d "$MERGED_MODEL_PATH" ]; then
        rm -rf "$MERGED_MODEL_PATH"
    fi
    
    # Save experiment info
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    cat << EOF > "$RUN_DIR/experiment_info.txt"
Experiment: $exp_name
Run completed at: $(date)
Duration: $duration seconds
Base model: $BASE_MODEL
Configuration: $exp_args
GPU used: $GPU_ID
Merge script used: $MERGE_SCRIPT
EOF
    
    echo "=== Experiment $exp_name Complete ===" | tee -a "$LOG_DIR/${exp_name}_log.txt"
    echo "Results saved in: $RUN_DIR" | tee -a "$LOG_DIR/${exp_name}_log.txt"
    
    # Clear GPU memory between experiments
}

# Main execution
echo "=== Starting Experimental Pipeline ===" | tee -a "$LOG_DIR/main_log.txt"
echo "Total experiments to run: ${#EXPERIMENTS[@]}" | tee -a "$LOG_DIR/main_log.txt"

# Run all experiments
for exp_name in $(echo "${!EXPERIMENTS[@]}" | tr ' ' '\n' | sort -n); do
    run_experiment "$exp_name" "${EXPERIMENTS[$exp_name]}"
done

# Generate summary report
echo "=== Generating Summary Report ===" | tee -a "$LOG_DIR/main_log.txt"
{
    echo "Experiment Summary"
    echo "=================="
    echo "Generated at: $(date)"
    echo
    for exp_name in "${!EXPERIMENTS[@]}"; do
        echo "Experiment: $exp_name"
        echo "Configuration: ${EXPERIMENTS[$exp_name]}"
        echo "Log file: $LOG_DIR/${exp_name}_log.txt"
        echo "----------------------------------------"
    done
} > "$LOG_DIR/experiment_summary.txt"

echo "=== All Experiments Complete ===" | tee -a "$LOG_DIR/main_log.txt"
echo "Logs available in: $LOG_DIR"
echo "Summary report: $LOG_DIR/experiment_summary.txt"