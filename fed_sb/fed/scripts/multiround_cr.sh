#!/bin/bash

# Define experiment configurations
declare -A EXPERIMENTS=(
    ["01-a"]="--lora_r 200"
)

# Configuration
BASE_MODEL="meta-llama/Llama-3.2-3B"
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
    CUDA_VISIBLE_DEVICES=$GPU_ID python fed/CR/multiround_train_cr_fed.py \
        --model $BASE_MODEL \
        $exp_args \
        --eg_bs 10 \
        --scheduler linear \
        --warmup_ratio 0.02 \
        --max_seq_length 256 \
        --seed 42 \
        --num_samples 170 \
        --agg_type $method \
        --rounds 2 \
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
    
    
    # Merging
    echo "[$(date)] Starting Aggregation.." | tee -a "$LOG_DIR/${exp_name}_log.txt"
     CUDA_VISIBLE_DEVICES=$GPU_ID python fed/aggregator.py \
        --model_name "$BASE_MODEL" \
        --lora_r $current_lora_r \
        --agg_type $method \
        --max_seq_length 256 \
        --dir_path "$RUN_DIR" 2>&1 | tee -a "$LOG_DIR/${exp_name}_log.txt" || handle_error "$exp_name" "Merging"

    declare -a datasets=(
        "ARC-Challenge"
        "ARC-Easy"
        "boolq"
        "hellaswag"
        "openbookqa"
        "piqa"
        "social_i_qa"
        "winogrande"
    )

    # Loop through datasets and evaluate
    for dataset in "${datasets[@]}"; do
        echo "=== Evaluating on $dataset ==="
        
        
        CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/commonsense_eval.py \
            --model "$MERGED_MODEL_PATH" \
            --dataset "$dataset" \
            --data_file "data/commonsense/$dataset/test.json" \
            --batch_size 64 \
            --tensor_parallel_size 1 \
            --run_dir "$RUN_DIR"
    done
    
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
for exp_name in $(echo "${!EXPERIMENTS[@]}" | tr ' ' '\n' | sort); do
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