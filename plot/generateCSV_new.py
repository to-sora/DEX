#!/usr/bin/env python3
import os
import sys
import pandas as pd

"""
This script automatically discovers tasks, agents, and seeds in the ../exp_local
directory (or a user-specified base_path) without relying on a config file.

It then creates two CSV files:
  1) pivot_max_reward.csv  - Mean (across seeds) of the maximum reward
  2) pivot_max_sr.csv      - Mean (across seeds) of the maximum success rate

Directory Assumptions:
  ../exp_local/
      ├── TaskA/
      │    ├── AgentA/
      │    │    └── .../s1/eval.csv
      │    │              /s2/eval.csv
      │    │              ...
      │    └── AgentB/
      │         └── ...
      └── TaskB/
           └── AgentA/
           │    └── ...
           └── AgentC/
                └── ...
            
Any subdirectory that directly contains "eval.csv" is considered a "seed" directory.

Warning:
  - If any (task, agent) combination has fewer than 5 discovered seeds, a warning is printed.

Usage:
  ./auto_pivot.py [BASE_PATH]

If BASE_PATH is omitted, defaults to "../exp_local".

Output:
  Creates pivot_max_reward.csv and pivot_max_sr.csv in the current directory.
  Columns: [Task, Agent1, Agent2, ...]
  Values: mean of the max reward (or SR) found in each agent-task across all discovered seeds.
Meo
"""

def main():
    # The user can provide a base_path as a command-line arg, default is '../exp_local'.
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "../exp_local"

    if not os.path.isdir(base_path):
        print(f"Error: base_path '{base_path}' does not exist or is not a directory.")
        sys.exit(1)

    column_reward = "episode_reward"
    column_sr = "episode_sr"
    eval_filename = "eval.csv"

    # Discover tasks (subdirectories of base_path)
    tasks = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]
    tasks.sort()

    # We'll build a data structure:
    # discovered[task][agent] = [list_of_seed_paths...]
    discovered = {}

    for task in tasks:
        task_path = os.path.join(base_path, task)
        # Find agents under the task
        if not os.path.isdir(task_path):
            continue

        agents = [
            d for d in os.listdir(task_path)
            if os.path.isdir(os.path.join(task_path, d))
        ]
        agents.sort()

        discovered[task] = {}

        for agent in agents:
            agent_path = os.path.join(task_path, agent)
            if not os.path.isdir(agent_path):
                continue

            # Recursively walk to find subdirectories containing eval.csv
            seeds_list = []
            for root, dirs, files in os.walk(agent_path):
                if eval_filename in files:
                    seeds_list.append(root)
            seeds_list.sort()

            # Store results
            discovered[task][agent] = seeds_list

    # Prepare pivot dictionaries:
    # We'll build them with row = tasks, columns = agents
    pivot_max_reward = {"Task": tasks}
    pivot_max_sr = {"Task": tasks}

    # Collect the union of all agents across all tasks for consistent pivot
    all_agents = set()
    for tsk in discovered:
        all_agents.update(discovered[tsk].keys())
    all_agents = sorted(list(all_agents))

    for agent in all_agents:
        pivot_max_reward[agent] = []
        pivot_max_sr[agent] = []

    # Now, fill pivot data
    for task in tasks:
        for agent in all_agents:
            seed_paths = discovered[task].get(agent, [])

            # If seed_paths < 5, print warning
            if len(seed_paths) < 5 and len(seed_paths) > 0:
                print(f"Warning: For (Task={task}, Agent={agent}), discovered only {len(seed_paths)} seeds (< 5). Meo")

            # We'll track the max reward and max SR for each seed
            max_rewards_for_seeds = []
            max_srs_for_seeds = []

            for seed_dir in seed_paths:
                csv_file = os.path.join(seed_dir, eval_filename)
                if not os.path.isfile(csv_file):
                    continue

                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    print(f"Error reading CSV file {csv_file}: {e}. Skipping...")
                    continue

                # Check reward
                if column_reward in df.columns:
                    rewards = pd.to_numeric(df[column_reward], errors='coerce').dropna()
                    if not rewards.empty:
                        max_rewards_for_seeds.append(rewards.max())

                # Check SR
                if column_sr in df.columns:
                    srs = pd.to_numeric(df[column_sr], errors='coerce').dropna()
                    if not srs.empty:
                        max_srs_for_seeds.append(srs.max())

            # Once we have all seeds' max reward, compute mean
            if max_rewards_for_seeds:
                mean_of_max_reward = sum(max_rewards_for_seeds) / len(max_rewards_for_seeds)
            else:
                mean_of_max_reward = "Nil"

            # Similarly for SR
            if max_srs_for_seeds:
                mean_of_max_sr = sum(max_srs_for_seeds) / len(max_srs_for_seeds)
            else:
                mean_of_max_sr = "Nil"

            pivot_max_reward[agent].append(mean_of_max_reward)
            pivot_max_sr[agent].append(mean_of_max_sr)

    # Convert pivot dict to DataFrame
    df_reward = pd.DataFrame(pivot_max_reward)
    df_sr = pd.DataFrame(pivot_max_sr)

    # Reorder columns so 'Task' is first, rest are agents
    columns_order = ["Task"] + all_agents
    df_reward = df_reward[columns_order]
    df_sr = df_sr[columns_order]

    # Write CSV files
    out_reward = "pivot_max_reward.csv"
    out_sr = "pivot_max_sr.csv"

    try:
        df_reward.to_csv(out_reward, index=False)
        df_sr.to_csv(out_sr, index=False)
        print(f"Generated '{out_reward}' and '{out_sr}'. Meo")
    except Exception as e:
        print(f"Error writing CSV outputs: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
