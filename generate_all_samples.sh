#!/bin/bash

# Create output directory structure
mkdir -p SurRoL/surrol/data/demo

# List of all tasks
TASKS=(
    "NeedleReach-v0"
    "NeedleReachMem-v0"
    "GauzeRetrieve-v0"
    "GauzeRetrieveMem-v0"
    "NeedlePick-v0"
    "NeedlePickMem-v0"
    "PegTransfer-v0"
    "PegTransferMem-v0"
    "NeedleRegrasp-v0"
    "NeedleRegraspMem-v0"
    "BiPegTransfer-v0"
    "BiPegTransferMem-v0"
    "ECMReach-v0"
    "ECMReachMem-v0"
    "MisOrient-v0"
    "MisOrientMem-v0"
    "StaticTrack-v0"
    "StaticTrackMem-v0"
    "ActiveTrack-v0" # No Mem version exists
)


# Generate samples for each task
for task in "${TASKS[@]}"; do
    echo "Generating samples for $task..."
    
    # Run sample generation using SurRoL's script
    python SurRoL/surrol/data/data_generation.py --env "$task"
    
    echo "Completed generating samples for $task"
    echo "----------------------------------------"
done

echo "All sample generation completed!"

# Optionally generate videos (uncomment if needed)
# echo "Generating videos for each task..."
# for task in "${TASKS[@]}"; do
#     echo "Generating video for $task..."
#     python SurRoL/surrol/data/data_generation.py --env "$task" --video
#     echo "Completed generating video for $task"
#     echo "----------------------------------------"
# done 