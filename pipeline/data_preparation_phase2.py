"""
# Data Preparation Script for Phase 2
# Purpose: This script processes raw datasets from the 'Shape_Initial_shear' directory,
# including normalization, outlier removal, and splitting the data into training/validation datasets.
# This will ensure clean input for generative modeling in Phase 2.
"""

import os
import pandas as pd
import numpy as np

# Paths
base_path = "data/Shape_Initial_shear/"
output_path = "outputs/pipeline/phase2/"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)


# Helper function: Remove outliers
def remove_outliers(df, column, threshold=(-0.5, 0.5)):
    """Removes data points outside the specified threshold."""
    return df[(df[column] >= threshold[0]) & (df[column] <= threshold[1])]


# Processing pipeline
def process_files():
    seeds = [
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
    ]
    all_dataframes = []

    for seed in seeds:
        file_path = os.path.join(base_path, seed, "iteration_data.csv")

        if not os.path.exists(file_path):
            print(f"Skipping {seed}: iteration_data.csv not found.")
            continue

        # Load data
        df = pd.read_csv(file_path)

        # Normalize volume_fraction
        vf_min = df['volume_fractions'].min()
        vf_max = df['volume_fractions'].max()
        df['volume_fractions'] = (df['volume_fractions'] - vf_min) / (vf_max - vf_min)

        # Filter Poisson Ratios with predefined thresholds
        df = remove_outliers(df, 'poisson_ratios_v12')
        df = remove_outliers(df, 'poisson_ratios_v21')

        # Append data with seed info
        df["seed_name"] = seed
        all_dataframes.append(df)

    # Combine all datasets and save processed data
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.to_csv(os.path.join(output_path, "prepared_data.csv"), index=False)
    print(f"Data preparation complete. Output saved to {output_path}prepared_data.csv")


if __name__ == "__main__":
    process_files()