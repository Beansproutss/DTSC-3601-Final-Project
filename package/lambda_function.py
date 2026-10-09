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

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    s3_client.put_object(
        Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue()
    )
    
    return {"statusCode": 200, "body": "Data pushed successfully!"}


