#!/bin/bash

# Enable nullglob to ensure that globs that don't match expand to nothing
shopt -s nullglob

# Base directory
base_dir="."

# Define colors
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print table header
printf "%-20s %-15s %-10s %-10s %-10s %-10s\n" "Task" "Agent" "Config" "Seed" "Train CSV" "Eval CSV"
printf "%-20s %-15s %-10s %-10s %-10s %-10s\n" "--------------------" "---------------" "----------" "----------" "----------" "----------"

# Iterate through all tasks
for task in "$base_dir"/*/; do
  # Check if it's a directory
  if [ -d "$task" ]; then
    task_name=$(basename "$task")
    
    # Iterate through all agents in the task
    for agent in "$task"*/; do
      if [ -d "$agent" ]; then
        agent_name=$(basename "$agent")
        
        # Iterate through all configs (e.g., d100)
        for config in "$agent"*/; do
          if [ -d "$config" ]; then
            config_name=$(basename "$config")
            
            # Iterate through all seeds (e.g., s1)
            for seed in "$config"*/; do
              if [ -d "$seed" ]; then
                seed_name=$(basename "$seed")
                
                # Check for train.csv and eval.csv
                train_csv_exists="No"
                eval_csv_exists="No"
                
                if [ -f "$seed/train.csv" ]; then
                  train_csv_exists="Yes"
                fi
                
                if [ -f "$seed/eval.csv" ]; then
                  eval_csv_exists="Yes"
                fi

                # Color "No" in red
                if [ "$train_csv_exists" = "No" ]; then
                  train_csv_exists="${RED}No${NC}"
                fi
                if [ "$eval_csv_exists" = "No" ]; then
                  eval_csv_exists="${RED}No${NC}"
                fi

                # Print the result in table format
                printf "%-20s %-15s %-10s %-10s %-10s %-10s\n" \
                  "$task_name" "$agent_name" "$config_name" "$seed_name" "$train_csv_exists" "$eval_csv_exists"
              fi
            done
          fi
        done
      fi
    done
  fi
done

# Disable nullglob to revert to default behavior
shopt -u nullglob
