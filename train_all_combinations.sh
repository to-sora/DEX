

# List of all tasks
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

# TASKS=(
#     # "NeedleReach-v0"
#     "NeedleReachMem-v0"
#     # "GauzeRetrieve-v0"
#     "GauzeRetrieveMem-v0"
#     # "NeedlePick-v0"
#     "NeedlePickMem-v0"
#     # "PegTransfer-v0"
#     "PegTransferMem-v0"
#     # "NeedleRegrasp-v0"
#     "NeedleRegraspMem-v0"
#     # "BiPegTransfer-v0"
#     "BiPegTransferMem-v0"
#     # "ECMReach-v0"
#     "ECMReachMem-v0"
#     # "MisOrient-v0"
#     "MisOrientMem-v0"
#     # "StaticTrack-v0"
#     "StaticTrackMem-v0"
#     # "ActiveTrack-v0" # No Mem version exists
# )
# List of all agents
AGENTS=(
    "dex"
    "sac"
    "ddpg"
    "ddpgbc"
    "col"
    "amp"
    "awac"
    "sqil"
    # "dexrnn2"
    # "dexgru"
    # "dexlstm"
)

# Maximum number of concurrent jobs
MAX_CONCURRENT_JOBS=4

# Create directories
mkdir -p logs
mkdir -p .checkpoints

# Function to check if a combination has been completed
is_completed() {
    local task=$1
    local agent=$2
    if [ -f ".checkpoints/${task}_${agent}.done" ]; then
        return 0 # True
    else
        return 1 # False
    fi
}

# Function to mark a combination as completed
mark_completed() {
    local task=$1
    local agent=$2
    touch ".checkpoints/${task}_${agent}.done"
}

# Function to run training
run_training() {
    local task=$1
    local agent=$2

    # Skip if this combination was already completed
    if is_completed "$task" "$agent"; then
        echo "Skipping $task with $agent - already completed"
        return
    fi

    echo "Training $task with $agent..."

    # Create log file name
    log_file="logs/${task}_${agent}.log"

    # Run training
    python -m train \
        agent=$agent \
        task=$task \
        use_wb=True \
        2>&1 | tee "$log_file"

    # Check if training completed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully completed training $task with $agent"
        mark_completed "$task" "$agent"
    else
        echo "Failed training $task with $agent"
        # Optionally exit on failure
        # exit 1
    fi

    echo "----------------------------------------"
}

# Start training processes
for task in "${TASKS[@]}"; do
    for agent in "${AGENTS[@]}"; do
        # Run training in background
        run_training "$task" "$agent" &

        # Wait if we have reached MAX_CONCURRENT_JOBS
        while [ $(jobs -r | wc -l) -ge $MAX_CONCURRENT_JOBS ]; do
            sleep 1
        done
    done
done

# Wait for all background jobs to finish
wait

echo "All training combinations completed!"

# Print summary of completed trainings
echo "Training Summary:"
echo "----------------------------------------"
total_combinations=$((${#TASKS[@]} * ${#AGENTS[@]}))
completed_count=$(ls .checkpoints/*.done 2>/dev/null | wc -l)
echo "Completed: $completed_count/$total_combinations combinations"

# List incomplete combinations if any exist
if [ $completed_count -lt $total_combinations ]; then
    echo "Incomplete combinations:"
    for task in "${TASKS[@]}"; do
        for agent in "${AGENTS[@]}"; do
            if ! is_completed "$task" "$agent"; then
                echo "- $task with $agent"
            fi
        done
    done
fi
