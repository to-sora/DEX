#!/usr/bin/env python3

import argparse
import pandas as pd
import sys
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compute percentage or delta change between two pivoted RL metrics CSV files.')
    parser.add_argument('old_csv', type=str, help='Path to the old CSV file.')
    parser.add_argument('new_csv', type=str, help='Path to the new CSV file.')
    parser.add_argument('-o', '--output', type=str, default='change_output.csv',
                        help='Path to the output CSV file. Defaults to "change_output.csv".')
    parser.add_argument('-t', '--change_type', type=str, choices=['percentage', 'delta'], default='percentage',
                        help='Type of change to compute: "percentage" for percentage change or "delta" for absolute difference. Defaults to "percentage".')
    return parser.parse_args()

def load_csv(filepath):
    if not os.path.isfile(filepath):
        print(f"Error: File not found - {filepath}")
        sys.exit(1)
    try:
        df = pd.read_csv(filepath, dtype=str)
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)

def validate_dimensions(old_df, new_df):
    print(old_df.shape)
    print(new_df.shape)


    # Check number of rows
    if old_df.shape[0] != new_df.shape[0]:
        return False, 'rows'
    
    # Check number of columns
    if old_df.shape[1] != new_df.shape[1]:
        return False, 'columns'
    
    # Check 'Task' column
    if 'Task' not in old_df.columns or 'Task' not in new_df.columns:
        print("Error: Both CSV files must contain a 'Task' column.")
        sys.exit(1)
    
    # Check if 'Task' columns ma
    
    return True, None

def compute_change(old_df, new_df, change_type):
    # Initialize the result DataFrame with 'Task' column
    result_df = pd.DataFrame()
    result_df['Task'] = old_df['Task']
    
    # Iterate over agent columns
    agents = [col for col in old_df.columns if col != 'Task']
    for agent in agents:
        old_values = old_df[agent].astype(str).str.strip()
        new_values = new_df[agent].astype(str).str.strip()
        changes = []
        
        for old, new in zip(old_values, new_values):
            if old.upper() == 'NIL' or new.upper() == 'NIL':
                changes.append('Nil')
                continue
            try:
                old_num = float(old)
                new_num = float(new)
                if change_type == 'percentage':
                    if old_num == 0:
                        # Avoid division by zero
                        changes.append('Nil')
                    else:
                        change = ((new_num - old_num) / old_num) * 100
                        # Format to two decimal places
                        changes.append(f"{change:.2f}")
                elif change_type == 'delta':
                    change = new_num - old_num
                    # Format to two decimal places
                    changes.append(f"{change:.2f}")
                else:
                    # Unsupported change_type
                    changes.append('Nil')
            except ValueError:
                # If conversion to float fails
                changes.append('Nil')
        
        result_df[agent] = changes
    
    return result_df

def main():
    args = parse_arguments()
    
    # Load both CSV files
    old_df = load_csv(args.old_csv)
    new_df = load_csv(args.new_csv)
    
    # Validate dimensions and axes
    valid, changed_axis = validate_dimensions(old_df, new_df)
    if not valid:
        print(f"Error: The two CSV files have different {changed_axis}.")
        sys.exit(1)
    
    # Compute change based on the specified type
    result_df = compute_change(old_df, new_df, args.change_type)
    
    # Define the output CSV filename with suffixes
    # Example: episode_length_mean_percentage_change.csv or episode_length_mean_delta_change.csv
    old_filename = os.path.splitext(os.path.basename(args.old_csv))[0]
    new_filename = os.path.splitext(os.path.basename(args.new_csv))[0]
    
    # Extract key_attribute, print_mode, and mode from filenames if possible
    # Assuming filenames are in the format: key_attribute_print_mode_mode.csv
    def extract_parts(filename):
        parts = filename.split('_')
        if len(parts) < 3:
            return 'change', 'change', 'change'
        key_attribute = parts[0]
        print_mode = parts[1]
        mode = parts[2]
        return key_attribute, print_mode, mode
    
    key_attr_old, print_mode_old, mode_old = extract_parts(old_filename)
    key_attr_new, print_mode_new, mode_new = extract_parts(new_filename)
    
    # Ensure that key_attribute, print_mode, and mode are consistent
    if (key_attr_old != key_attr_new) or (print_mode_old != print_mode_new) or (mode_old != mode_new):
        suffix = args.change_type
    else:
        suffix = f"{args.change_type}_change"
    
    output_filename = f"{old_filename}_vs_{new_filename}_{suffix}.csv"
    
    # Alternatively, you can simply append the change type to the output name
    # For simplicity, let's use the user-specified output filename
    # output_filename = f"{key_attr_old}_{print_mode_old}_{mode_old}_{args.change_type}_change.csv"
    
    # Reorder columns to have 'Task' first
    columns_order = ['Task'] + [col for col in old_df.columns if col != 'Task']
    result_df = result_df[columns_order]
    
    # Replace NaN with 'Nil' just in case
    result_df = result_df.fillna('Nil')
    
    # Save the change DataFrame to CSV
    try:
        result_df.to_csv(args.output, index=False)
        print(f"Change results have been written to {args.output}")
    except Exception as e:
        print(f"Error writing to output file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
