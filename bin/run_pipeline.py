#!/usr/bin/env python3
"""
Master execution script for the traffic prediction project.
Runs school holiday checks, data combination/preprocessing, 
model training, and evaluation using the provided sample dataset.
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.traffic_pipeline import (
    school_holiday_check,
    combine_dataframes,
    preprocess_train_data,
    evaluate_model
)

def main():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "sample_data.csv")
    
    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found at: {data_path}")
        print("Please ensure your sample CSV file is placed in the bin/ folder.")
        return
    
    print(f"Loading sample dataset from {data_path}...")
    df = pd.read_csv(data_path)

        
    script_dir2 = os.path.dirname(os.path.abspath(__file__))
    data_path2 = os.path.join(script_dir, "sample_data.csv")
    
    if not os.path.exists(data_path2):
        print(f"[Error] Dataset not found at: {data_path2}")
        print("Please ensure your sample CSV file is placed in the bin/ folder.")
        return
    
    ncdot_df = pd.read_csv(data_path2)
    
    # 3. Step: School Holiday Check
    print("Running school holiday feature engineering...")
    df_with_holidays = school_holiday_check(df)
    
    combined_df = combine_dataframes(df_with_holidays, ncdot_df)
    
    # 4. Step: Preprocess & Train XGBoost Model
    print("Preprocessing data and training the XGBoost model...")
    # (Assuming your preprocess_train_data function splits and returns the trained model,
    # or you can adjust it to return X_test and y_test for evaluation).
    # Let's run the preprocessing & training:
    model = preprocess_train_data(df_with_holidays)
    print("Model training complete!")
    
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()
