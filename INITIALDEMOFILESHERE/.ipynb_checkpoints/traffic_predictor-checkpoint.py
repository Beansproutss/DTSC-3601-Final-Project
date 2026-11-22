#Importing necessary libraries
import pandas as pd
import boto3
import io
import gzip
import json
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import numpy as np

def school_holiday_check(traffic_df):
    """
    Checks if a given date falls within the defined school holiday periods (defined in the function for UNCC).

    Args:
        traffic_df (pd.DataFrame): A DataFrame containing traffic data with a 'timestamp' column to be compared to school holiday periods.

    Returns:
        traffic_df (pd.DataFrame): The input DataFrame with an additional 'school_holiday' column indicating whether each date falls within a school holiday period (1 for holiday, 0 for non-holiday).
    """
    # Make sure timestamps are datetimes before comparing them to the holiday dates
    traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"])
    
    # UNCC school holiday ranges as (start, end), inclusive; update these each semester
    school_holidays = [
        ("2026-10-12", "2026-10-13"), #Student recess
        ("2026-11-11", "2026-11-11"), #Veterans Day
        ("2026-11-25", "2026-11-28"),  # Thanksgiving
        ("2026-12-24", "2026-12-31")  # Winter break
    ]

    school_holidays = [
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in school_holidays
    ]

    # Flag = 1 if the row's date falls in any holiday range, otherwise 0 (missing dates get 0)
    traffic_df["school_holiday"] = traffic_df["timestamp"].apply(
        lambda date: int(pd.notna(date) and any(start <= date.normalize() <= end
                        for start, end in school_holidays
                    )))
    
    return traffic_df
    
def predict_custom_traffic(model, location_id, timestamp_str):
    """
    Takes a single location_id and timestamp string, automatically assigns the hardcoded free_flow_speed and AADT_2022, runs feature engineering, and predicts traffic speed and congestion level.
    
    Args:
        model: xgb.XGBRegressor: The trained XGBoost regression model.
        location_id (int): The location ID for which to predict traffic speed.
        timestamp_str (str): The timestamp string in the format 'YYYY-MM-DD HH:MM:SS' for which to predict traffic speed.
    Returns:
        tuple: A tuple containing the predicted traffic speed (float) and congestion level (str).
    """

    # Constants per location (free-flow speed in mph and AADT) so the user only supplies a place and time.
    # location_id comes from groupby().ngroup() in training, so these IDs only match if the training data is the same.
    LOCATION_PARAMS = {
    0: {"free_flow_speed": 27.0, "AADT_2022": 15500.0},
    1: {"free_flow_speed": 40.0, "AADT_2022": 42000.0},
    2: {"free_flow_speed": 31.0, "AADT_2022": 32500.0},
    3: {"free_flow_speed": 31.0, "AADT_2022": 26500.0},
    4: {"free_flow_speed": 31.0, "AADT_2022": 27500.0},
    5: {"free_flow_speed": 28.0, "AADT_2022": 18000.0},
    6: {"free_flow_speed": 55.0, "AADT_2022": 51500.0},
    7: {"free_flow_speed": 28.0, "AADT_2022": 32500.0},
    8: {"free_flow_speed": 25.0, "AADT_2022": 29500.0},
    9: {"free_flow_speed": 26.0, "AADT_2022": 47500.0},
    10: {"free_flow_speed": 29.0, "AADT_2022": 17000.0}
    }
    
    # 1. Validate or retrieve hardcoded parameters for the location
    if location_id not in LOCATION_PARAMS:
        raise ValueError(f"Unknown location_id: {location_id}. Choose between 0 and 10.")
    
    params = LOCATION_PARAMS[location_id]
    
    # 2. Build a single-row DataFrame for inference
    df = pd.DataFrame([{
        "location_id": location_id,
        "timestamp": timestamp_str,
        "free_flow_speed": params["free_flow_speed"],
        "AADT_2022": params["AADT_2022"]
    }])
    
    # 3. Parse timestamp and extract temporal features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    
    # 4. Flag school holidays (UNCC calendar)
    df = school_holiday_check(df)
    df["is_holiday"] = df["school_holiday"] # Approximation: training used federal holidays, but the model expects this column
    
    # 5. Select the features (same columns and order as training) and predict
    feature_cols = ["location_id", "day_of_week", "hour", "month", "school_holiday", "free_flow_speed", "is_holiday", "AADT_2022"]
    X_inference = df[feature_cols]
    
    predicted_speed = model.predict(X_inference)[0]
    
    # 6. Speed index = predicted / free-flow speed: >= 0.8 Low, >= 0.5 Medium, otherwise High traffic
    free_flow = params["free_flow_speed"]
    speed_index = predicted_speed / free_flow
    
    if speed_index >= 0.8:
        traffic_level = "Low Traffic"
    elif speed_index >= 0.5:
        traffic_level = "Medium Traffic"
    else:
        traffic_level = "High Traffic"
        
    print(f"\n--- Traffic Prediction Results ---")
    print(f"Location ID: {location_id}")
    print(f"Timestamp: {timestamp_str}")
    print(f"Free-Flow Speed: {free_flow} mph")
    print(f"AADT (Volume): {params['AADT_2022']}")
    print(f"Predicted Speed: {predicted_speed:.2f} mph")
    print(f"Speed Index: {speed_index:.2f}")
    print(f"Traffic Condition: {traffic_level}")
    
    return predicted_speed, traffic_level