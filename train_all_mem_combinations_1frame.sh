#!/bin/bash

# Define the list of tasks
TASKS=(
    "NeedleReach-v0"
    # "NeedleReachMem-v0"
    "GauzeRetrieve-v0"
    # "GauzeRetrieveMem-v0"
    "NeedlePick-v0"
    # "NeedlePickMem-v0"
    "PegTransfer-v0"
    # "PegTransferMem-v0"
    "NeedleRegrasp-v0"
    # "NeedleRegraspMem-v0"
    "BiPegTransfer-v0"
    # "BiPegTransferMem-v0"
    "ECMReach-v0"
    # "ECMReachMem-v0"
    "MisOrient-v0"
    # "MisOrientMem-v0"
    "StaticTrack-v0"
    # "StaticTrackMem-v0"
    # "ActiveTrack-v0" # No Mem version exists
)

# Define the list of agents
AGENTS=(
    # "dex"
    "sacgru"
    "ddpggru"
    "ddpgbcgru"
    "colgru"
    "ampgru"
    "awacgru"
    "sqilgru"
    # "dexrnn2"
    "dexgru"
    # "dexlstm"
)

# Define the range of seeds
SEEDS=(1 2 3 4 5)

# Maximum number of concurrent jobs
MAX_CONCURRENT_JOBS=3

# Create necessary directories
mkdir -p logs
mkdir -p .checkpoints_mem_1d

# Function to check if a combination has been completed
is_completed() {
    local task=$1
    local agent=$2
    local seed=$3
    if [ -f ".checkpoints_mem_1d/${task}_${agent}_seed${seed}.done" ]; then
        return 0 # True
    else
        return 1 # False
    fi
}

# Function to mark a combination as completed
mark_completed() {
    local task=$1
    local agent=$2
    local seed=$3
    touch ".checkpoints_mem_1d/${task}_${agent}_seed${seed}.done"
}

# Function to run training
run_training() {
    local task=$1
    local agent=$2
    local seed=$3

    # Skip if this combination was already completed
    if is_completed "$task" "$agent" "$seed"; then
        echo "Skipping $task with $agent and seed $seed - already completed"
        return
    fi

    echo "Training $task with $agent and seed $seed..."

    # Create log file name
    log_file="logs/${task}_${agent}_seed${seed}.log"

    # # Run training
     python3 train.py \
         task="$task" \
         agent="$agent" \
         agent.timeframe=1 \
         seed="$seed" \
         2>&1 | tee "$log_file"

    # Check if training completed successfully
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Successfully completed training $task with $agent and seed $seed"
        mark_completed "$task" "$agent" "$seed"
    else
        echo "Failed training $task with $agent and seed $seed"
        # Optionally exit on failure
        # exit 1
    fi

    echo "----------------------------------------"
}

# Start training processes
for task in "${TASKS[@]}"; do
    for agent in "${AGENTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # Run training in background
            run_training "$task" "$agent" "$seed" &

            # Wait if we have reached MAX_CONCURRENT_JOBS
            while [ $(jobs -r | wc -l) -ge $MAX_CONCURRENT_JOBS ]; do
                sleep 1
            done
        done
    done
done

# Wait for all background jobs to finish
wait

echo "All training combinations completed!"

# Print summary of completed trainings
echo "Training Summary:"
echo "----------------------------------------"
total_combinations=$(( ${#TASKS[@]} * ${#AGENTS[@]} * ${#SEEDS[@]} ))
completed_count=$(ls .checkpoints_mem_1d/*.done 2>/dev/null | wc -l)
echo "Completed: $completed_count/$total_combinations combinations"

# List incomplete combinations if any exist
if [ "$completed_count" -lt "$total_combinations" ]; then
    echo "Incomplete combinations:"
    for task in "${TASKS[@]}"; do
        for agent in "${AGENTS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if ! is_completed "$task" "$agent" "$seed"; then
                    echo "- $task with $agent and seed $seed"
                fi
            done
        done
    done
fi
