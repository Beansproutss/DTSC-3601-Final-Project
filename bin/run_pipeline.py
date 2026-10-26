#!/usr/bin/env python3
"""
Master execution script for the traffic prediction project.
Runs school holiday checks, data combination/preprocessing, 
model training, and evaluation using the provided sample dataset.
"""

import os
import sys
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create the filepath to find files in the src/ folder on the repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from the src/
from src.traffic_pipeline import (
    school_holiday_check,
    combine_dataframes,
    extract_ncdot_traffic_data,
    preprocess_train_data,
    evaluate_model,
    predict_custom_traffic
)

def main():

    # Preparing the file paths for the first dataset with the TomTom data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "sample_data.csv")

    # Handle any errors in the file path with a warning
    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found at: {data_path}")
        print("Please ensure your sample CSV file is placed in the bin/ folder.")
        return

    # Confirmation of the file loading into a dataframe
    print(f"Loading sample dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Preparing the file paths for the second dataset with the NCDOT data    
    script_dir2 = os.path.dirname(os.path.abspath(__file__))
    data_path2 = os.path.join(script_dir, "sample_ncdot_data.csv")
    
     # Handle any errors in the file path with a warning
    if not os.path.exists(data_path2):
        print(f"[Error] Dataset not found at: {data_path2}")
        print("Please ensure your sample CSV file is placed in the bin/ folder.")
        return
    
    # Confirmation of the file loading into a dataframe    
    print(f"Loading sample dataset from {data_path2}...")
    ncdot_df = pd.read_csv(data_path2)

    # Calls the function to clean the NCDOT dataframe
    print(f"Editing NCDOT Dataframe...")
    ncdot_df = extract_ncdot_traffic_data(ncdot_df)

    # Calls the function to determine if there are school holidays in the data
    print("Running school holiday feature engineering...")
    df_with_holidays = school_holiday_check(df)

    # Combining the two files into one dataframe 
    print("Combining the dataframes...")
    combined_df = combine_dataframes(df_with_holidays, ncdot_df)

    # Calls the function to train the model
    print("Preprocessing data and training the XGBoost model...")
    model, X_test, y_test = preprocess_train_data(df_with_holidays)
    print("Model training complete!")

    # Calls the function to evalute the trained model
    print("Evaluating your model!")
    print()
    print("-- Model Evaluation and Sample Predictions --")
    evaluate_model(model, X_test, y_test)

    # Calls the function to predict traffic for a new data point
    print()
    print("Predicting for a new data point (not in test data)...")
    print()
    predict_custom_traffic(model, 2, "2026-10-19 19:00:00")
    
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()
