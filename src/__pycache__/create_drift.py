import pandas as pd
import numpy as np
import os

def generate_drift_data():
    # --- Configuration values are defined directly in this script ---
    # This removes the dependency on an external config.py file.
    raw_data_path = "data/raw/train.csv"
    output_path = "data/raw/simulated_drift_data.csv"
    # --- End of configuration ---

    try:
        if not os.path.exists(raw_data_path):
            print(f"Error: The source data file was not found at '{raw_data_path}'")
            print("Please ensure you have downloaded it from Kaggle and placed it correctly.")
            return

        df = pd.read_csv(raw_data_path)
        print("Original data loaded successfully.")

        # --- Introduce Artificial Drift ---

        # 1. Age Drift: Make a segment of the population older to simulate a change
        # in the demographic of passengers over time.
        mask_age = df['Pclass'] == 1
        df.loc[mask_age, 'Age'] = df.loc[mask_age, 'Age'] + np.random.uniform(10, 20, size=mask_age.sum())
        print("Introduced age drift for Pclass 1 passengers.")

        # 2. Embarked Drift: Change the distribution of embarkment points, simulating
        # a shift in popular travel routes.
        num_to_change = int(0.4 * len(df[df['Embarked'] == 'S']))
        s_indices = df[df['Embarked'] == 'S'].index
        indices_to_change = np.random.choice(s_indices, size=num_to_change, replace=False)
        df.loc[indices_to_change, 'Embarked'] = 'C'
        print("Introduced embarkment location drift (changed 40% of 'S' to 'C').")

        # --- Save the New Data ---
        df.to_csv(output_path, index=False)
        
        print(f"\nSuccessfully created simulated drift data at: {output_path}")
        print("You can now run the retraining workflow to test the drift detection mechanism.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    generate_drift_data()

