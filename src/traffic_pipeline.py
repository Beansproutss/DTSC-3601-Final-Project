import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def school_holiday_check(traffic_df):
    """
    Checks if a given date falls within the defined school holiday periods (defined in the function for UNCC).

    Args:
        traffic_df (pd.DataFrame): A DataFrame containing traffic data with a 'timestamp' column to be compared to school holiday periods.

    Returns:
        traffic_df (pd.DataFrame): The input DataFrame with an additional 'school_holiday' column indicating whether each date falls within a school holiday period (1 for holiday, 0 for non-holiday).
    """
    # For days of the week, Monday = 0, Sunday = 6 
    # 0 is not a holiday, 1 is a holiday
    traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"])
    
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

    traffic_df["school_holiday"] = traffic_df["timestamp"].apply(
        lambda date: int(pd.notna(date) and any(start <= date.normalize() <= end
                        for start, end in school_holidays
                    )))
    return traffic_df

def extract_ncdot_traffic_data(ncdot_df):
    # Rebuilt to match indices 0 through 10 of ncdot_df perfectly
    ncdot_df = ncdot_df.copy()
    matched_latitudes = [
        35.287972,  # Row 0: EAST OF STETSON DR
        35.311361,  # Row 1: WEST OF I-485
        35.293111,  # Row 2: SOUTH OF UNIVERSITY POINTE BLVD
        35.309306,  # Row 3: EAST OF SUTHER RD
        35.306917,  # Row 4: WEST OF SUTHER RD
        35.313611,  # Row 5: EAST OF SR 5394 MALLARD CK
        35.311861,  # Row 6: WEST OF W 7TH ST 
        35.301861,  # Row 7: SOUTH OF US 29
        35.294889,  # Row 8: EAST OF SR 5316
        35.299639,  # Row 9: EAST OF SR 2939
        35.307778   # Row 10: NORTH OF NC 49
    ]

    matched_longitudes = [
        -80.759472, # Row 0
        -80.711222, # Row 1
        -80.749361, # Row 2
        -80.718056, # Row 3
        -80.722639, # Row 4
        -80.701944, # Row 5
        -80.744000, # Row 6 
        -80.752333, # Row 7
        -80.742111, # Row 8
        -80.745750, # Row 9
        -80.724583  # Row 10
    ]

    #loop through the matched coordinates to update df2_final
    for i in range(len(matched_latitudes)):
        ncdot_df.at[i, "matched_latitude"] = matched_latitudes[i]
        ncdot_df.at[i, "matched_longitude"] = matched_longitudes[i]

    return ncdot_df
    

def combine_dataframes(tomtom_df, ncdot_df):
    """
    Combines two DataFrames (tomtom_df and ncdot_df) by matching latitude and longitude values, and adds the AADT_2022 column from ncdot_df to tomtom_df.

    Args:
        tomtom_df (pd.DataFrame): The first DataFrame containing traffic data with latitude and longitude columns.
        ncdot_df (pd.DataFrame): The second DataFrame containing traffic data with matched_latitude, matched_longitude, and AADT_2022 columns.

    Returns:
        traffic_df (pd.DataFrame): A combined DataFrame with the AADT_2022 column added from ncdot_df to tomtom_df.
    """

    # Combine the two dataframes

    tomtom_df["latitude"] = pd.to_numeric(tomtom_df["latitude"])
    tomtom_df["longitude"] = pd.to_numeric(tomtom_df["longitude"])
    ncdot_df["matched_latitude"] = pd.to_numeric(ncdot_df["matched_latitude"])
    ncdot_df["matched_longitude"] = pd.to_numeric(ncdot_df["matched_longitude"])
    # Creating a round key for both dataframes based on rounded latitude and longitude

    ncdot_df["round_key"] = (
        ncdot_df["matched_latitude"].round(4).astype(str) + "_" + 
        ncdot_df["matched_longitude"].round(4).astype(str)
    )
    #Using AADT_2022 from ncdot_df to create a traffic map based on the round key
    traffic_map=ncdot_df.set_index("round_key")["AADT_2022"]

    tomtom_df["round_key"] = (
        tomtom_df["latitude"].round(4).astype(str) + "_" + 
        tomtom_df["longitude"].round(4).astype(str)
    )
    # Mapping the traffic data from the traffic map to the final dataframe based on the round key
    tomtom_df["AADT_2022"] = tomtom_df["round_key"].map(traffic_map)

    tomtom_df = tomtom_df.drop(columns=["round_key"])
    ncdot_df = ncdot_df.drop(columns=["round_key"])
    combined_traffic_df = tomtom_df.copy()  # Create a copy of tomtom_df to avoid modifying the original DataFrame

    return combined_traffic_df
  
def preprocess_train_data(combined_df):
    """
    Preprocesses the combined DataFrame by cleaning the data, creating new features, and preparing it for machine learning. Trains an XGBoost regression model on the preprocessed data.

    Args:
        combined_df (pd.DataFrame): The combined DataFrame containing all the data.

    Returns:
        model (xgb.XGBRegressor): The trained XGBoost regression model.
    """
    final_df = combined_df.copy()
    # Assigning a unique location ID to each combination of latitude and longitude
    final_df["location_id"] = final_df.groupby(["latitude", "longitude"]).ngroup()

    # Sorting the final dataframe by timestamp and resetting the index
    final_df = final_df.sort_values("timestamp").reset_index(drop=True)

    final_df["is_holiday"] = (final_df["holiday"] != "Not a federal holiday").astype(int)

    feature_cols = ["location_id", "day_of_week", "hour", "month", "free_flow_speed", "is_holiday", "AADT_2022"]
    target_col = "current_speed"

    valid_rows = final_df[feature_cols].notna().all(axis=1) & final_df[target_col].notna()
    df_clean = final_df.loc[valid_rows].copy()

    #Defining the features and target variable for the regression model
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    # Splitting the data into training and testing sets
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Training the XGBoost regression model
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of the trained model on the test set and prints various performance metrics.

    Args:
        model: xgb.XGBRegressor: The trained XGBoost regression model.
        X_test (pd.DataFrame): The test set features.
        y_test (pd.Series): The actual target values for the test set.
    Returns:
        None, it displays the evaluation metrics and predicted outputs for some inputs in the test set.
    """
    # Evaluating the model's performance on the test set
    y_pred = model.predict(X_test)

    # Shows the average absolute error between predicted and actual speeds
    mae = mean_absolute_error(y_test, y_pred) 
    # Shows the root mean squared error between predicted and actual speeds
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
    # Shows the proportion of variance in the target variable explained by the model
    r2 = r2_score(y_test, y_pred) 

    y_test_stable = np.where(y_test == 0, 1e-5, y_test)
    # Shows the mean absolute percentage error between predicted and actual speeds
    mape = np.mean(np.abs((y_test - y_pred) / y_test_stable)) * 100 

    print("Model Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.2f} mph")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mph")
    print(f"MAPE (Percentage Error): {mape:.2f}%")
    print(f"R² Score (Variance Explained): {r2:.4f}")

    # Displaying the predicted output speeds for each input in the test set
    for i in range(4):
        print(f"Input:\n{X_test.iloc[i]} \nOutput speed: {y_pred[i]} mph")
