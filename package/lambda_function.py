# %%
import requests
import json
import os
import urllib3
import boto3
import io
import gzip
import logging
import pandas as pd
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from io import StringIO

s3_client = boto3.client('s3')
s3_bucket = "s3-3601-final-project"
api_key = 'KFAr7dIvLb4Fw2ntylN7NFGLqRIJXYxc'

def lambda_handler(event, context):
    # Extract data from S3 bucket and prepare for processing

    # Defines the current timestamp
    today_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Defines the S3 bucket and key for the traffic data file
    s3_key = f"tomtom/traffic/{today_str}speeds1.csv"

    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

    data_list = []

    latitude = [35.307778, 35.313611, 35.311361, 35.309306, 35.306917, 35.293111, 35.294889, 35.299639, 35.287972, 35.301861, 35.311861] # fill in with the found values
    longitude = [-80.724583, -80.701944, -80.711222, -80.718056, -80.722639, -80.749361, -80.742111, -80.74575, -80.759472, -80.752333, -80.744]
    # Loop through the list of coordinates and fetch traffic data for each location
    for i in range(len(latitude)):

            params = {
                'key': api_key,
                'point': f"{latitude[i]},{longitude[i]}",
                'unit': "MPH",
                "openLr": "false"
            }

            response = requests.get(url, params)

            if response.status_code==200:
                data = response.json()
                flow_data = data.get("flowSegmentData", {})

                data_we_want = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    "latitude": latitude[i],
                    "longitude": longitude[i],
                    "current_speed": flow_data.get("currentSpeed"),
                    "free_flow_speed": flow_data.get("freeFlowSpeed"),
                    "current_delay": flow_data.get("currentDelay"),
                    "confidence": flow_data.get("confidence") # for flags of when we should somehow retrain?
                }

                data_list.append(data_we_want)
            else:
                print(f"Error in retrieving data: {response.status_code}")
                
    #making data_list into a DataFrame
    df = pd.DataFrame(data_list) 
    print(df.head())
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%Y-%m-%d_%H-%M-%S"
        )
        df["day_of_week"] = df["timestamp"].dt.day_name()


    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Create time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Calculate the speed ratio for each traffic record
    df["speed_ratio"] = (df["current_speed"] / df["free_flow_speed"])

    calendar = USFederalHolidayCalendar()
    #To use after getting traffic data from holidays
    holiday_names = calendar.holidays(
            start=df["timestamp"].min().normalize(),
            end=df["timestamp"].max().normalize(),
            return_name=True,
        )

    df["holiday"] = (
            #map timestamp to the correct holiday name
            df["timestamp"].dt.normalize()
            .map(holiday_names)
            .fillna("Not a federal holiday")
    )
    if not df.empty:
        holiday_names = calendar.holidays(
            start=df["timestamp"].min().normalize(),
            end=df["timestamp"].max().normalize(),
            return_name=True,
        )

        df["holiday"] = (
            df["timestamp"].dt.normalize()
            .map(holiday_names)
            .fillna("Not a federal holiday")
        )
    else:
        print("No traffic rows available.")

    final_df = df.copy()

    final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce")

    final_df["hour"] = final_df["timestamp"].dt.hour.astype("Int64")
    final_df["day_of_week"] = final_df["timestamp"].dt.dayofweek.astype("Int64")
    final_df["month"] = final_df["timestamp"].dt.month.astype("Int64")

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

    final_df["school_holiday"] = final_df["timestamp"].apply(
        lambda date: int(
            pd.notna(date)
            and any(
                start <= date.normalize() <= end
                for start, end in school_holidays
            )
        )
    )

    response2 = s3_client.get_object(
        Bucket=s3_bucket,
        Key="ncdot/traffic/aadt_stations_2026-09-27_15-13-28-302554.csv",
    )

    # Reading the decompressed CSV data into a DataFrame
    raw_bytes = response2["Body"].read()
    try:
        data_bytes = gzip.decompress(raw_bytes)
    except gzip.BadGzipFile:
        data_bytes = raw_bytes
    # Displaying the final DataFrame for the specific NCDOT traffic data file
    df2_final = pd.read_csv(io.BytesIO(data_bytes))

    # Edit the coordinates to match completely

    # Rebuilt to match indices 0 through 10 of df2_final perfectly
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
        df2_final.at[i, "matched_latitude"] = matched_latitudes[i]
        df2_final.at[i, "matched_longitude"] = matched_longitudes[i]

    # Combine the two dataframes

    final_df["latitude"] = pd.to_numeric(final_df["latitude"])
    final_df["longitude"] = pd.to_numeric(final_df["longitude"])
    df2_final["matched_latitude"] = pd.to_numeric(df2_final["matched_latitude"])
    df2_final["matched_longitude"] = pd.to_numeric(df2_final["matched_longitude"])
    # Creating a round key for both dataframes based on rounded latitude and longitude

    df2_final["round_key"] = (
        df2_final["matched_latitude"].round(4).astype(str) + "_" + 
        df2_final["matched_longitude"].round(4).astype(str)
    )
    #Using AADT_2022 from df2_final to create a traffic map based on the round key
    traffic_map=df2_final.set_index("round_key")["AADT_2022"]

    final_df["round_key"] = (
        final_df["latitude"].round(4).astype(str) + "_" + 
        final_df["longitude"].round(4).astype(str)
    )
    # Mapping the traffic data from the traffic map to the final dataframe based on the round key
    final_df["AADT_2022"] = final_df["round_key"].map(traffic_map)

    final_df = final_df.drop(columns=["round_key"])
    df2_final = df2_final.drop(columns=["round_key"])

    csv_buffer = StringIO()
    final_df.to_csv(csv_buffer, index=False)

    s3_client.put_object(
        Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue()
    )

    return {"statusCode": 200, "body": "Data pushed successfully!"}


