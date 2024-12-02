#!/usr/bin/env python3

import argparse
import json
import os
import sys
import pandas as pd
import csv

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pivot RL metrics from CSV files based on configuration.')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to the JSON configuration file.')
    return parser.parse_args()

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)

def compute_aggregate(series, mode):
    try:
        if mode == 'min':
            return series.min()
        elif mode == 'max':
            return series.max()
        elif mode == 'median':
            return series.median()
        elif mode == 'mean':
            return series.mean()
        elif mode == 'min_index':
            return series.idxmin()
        elif mode == 'max_index':
            return series.idxmax()
        else:
            return 'Nil'
    except:
        return 'Nil'

def main():
    args = parse_arguments()
    config = load_config(args.config)

    # Extract configuration with defaults
    agents = config.get('agents', [])
    tasks = config.get('tasks', [])
    mode = config.get('mode', 'train').lower()
    print_mode = config.get('print_mode', 'mean').lower()
    key_attribute = config.get('key_attribute', '')
    config_path = config.get('config_path', 'd100/s1')

    if not agents:
        print("No agents specified in the configuration.")
        sys.exit(1)
    if not tasks:
        print("No tasks specified in the configuration.")
        sys.exit(1)
    if mode not in ['train', 'eval']:
        print("Mode should be either 'train' or 'eval'.")
        sys.exit(1)
    if print_mode not in ['min', 'max', 'median', 'mean', 'min_index', 'max_index']:
        print("Print mode should be one of 'min', 'max', 'median', 'mean', 'min_index', or 'max_index'.")
        sys.exit(1)
    if not key_attribute:
        print("Key attribute not specified in the configuration.")
        sys.exit(1)

    # Initialize a dictionary to hold the pivot table data
    pivot_data = { 'Task': tasks }
    for agent in agents:
        pivot_data[agent] = []

    for task in tasks:
        for agent in agents:
            # Construct the path to the CSV file
            csv_file = os.path.join('../exp_local', task, agent, config_path, f"{mode}.csv")
            
            if not os.path.isfile(csv_file):
                aggregate_value = 'Nil'
                print(f"File not found: {csv_file}")
            else:
                try:
                    df = pd.read_csv(csv_file)
                    if key_attribute not in df.columns:
                        aggregate_value = 'Nil'
                        print(f"Column '{key_attribute}' not found in {csv_file}")
                    else:
                        # Drop NaN values in the key attribute and convert to numeric
                        series = pd.to_numeric(df[key_attribute], errors='coerce').dropna()
                        if series.empty:
                            aggregate_value = 'Nil'
                            print(f"No valid numeric data in column '{key_attribute}' for {csv_file}")
                        else:
                            aggregate = compute_aggregate(series, print_mode)
                            if pd.isna(aggregate):
                                aggregate_value = 'Nil'
                            else:
                                aggregate_value = aggregate
                    # If the mode is 'min_index' or 'max_index', ensure the index is included
                    if print_mode in ['min_index', 'max_index']:
                        aggregate_value = int(aggregate_value)  # Convert index to integer
                except Exception as e:
                    aggregate_value = 'Nil'
                    print(f"Error processing file {csv_file}: {e}")

            # Append the aggregate value to the corresponding agent's list
            pivot_data[agent].append(aggregate_value)

    # Create a DataFrame from the pivot data
    pivot_df = pd.DataFrame(pivot_data)

    # Define the output CSV filename with suffixes
    output_filename = f"{args.config}_{key_attribute}_{print_mode}_{mode}.csv"

    # Reorder columns to have 'Task' first
    columns_order = ['Task'] + agents
    pivot_df = pivot_df[columns_order]

    # Replace NaN with 'Nil' just in case
    pivot_df = pivot_df.fillna('Nil')

    # Save the pivot DataFrame to CSV
    try:
        pivot_df.to_csv(output_filename, index=False)
        print(f"Pivoted results have been written to {output_filename}")
    except Exception as e:
        print(f"Error writing to output file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
