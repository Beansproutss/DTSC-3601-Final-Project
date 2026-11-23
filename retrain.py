import gzip
import io
import os
import boto3
import numpy as np
import pandas as pd
import xgboost as xgb

s3 = boto3.client("s3")
sns = boto3.client("sns")

BUCKET = "s3-3601-final-project"
DATA_PREFIX = "tomtom/traffic/"
SNS_TOPIC_ARN = os.environ["arn:aws:sns:us-east-1:946033678820:ModelAccuracyAlert"]


def extract_clean_tomtom_traffic_data():
    paginator = s3.get_paginator("list_objects_v2")
    df_list = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix=DATA_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue

            raw_bytes = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            try:
                data_bytes = gzip.decompress(raw_bytes)
            except gzip.BadGzipFile:
                data_bytes = raw_bytes

            obj_df = pd.read_csv(io.BytesIO(data_bytes))
            if not obj_df.empty:
                df_list.append(obj_df)

    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)


def preprocess_train_data(combined_df: pd.DataFrame):
    final_df = combined_df.copy()
    final_df["location_id"] = final_df.groupby(
        ["latitude", "longitude"]
    ).ngroup()
    location_lookup = (
        final_df[["latitude", "longitude", "location_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    final_df = final_df.sort_values("timestamp").reset_index(drop=True)
    final_df["is_holiday"] = (
        final_df["holiday"] != "Not a federal holiday"
    ).astype(int)

    feature_cols = [
        "location_id",
        "day_of_week",
        "hour",
        "month",
        "school_holiday",
        "free_flow_speed",
        "is_holiday",
        "AADT_2022",
    ]
    target_col = "current_speed"

    valid_rows = (
        final_df[feature_cols].notna().all(axis=1)
        & final_df[target_col].notna()
    )
    df_clean = final_df.loc[valid_rows].copy()

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    return model, X_test, y_test, location_lookup


if __name__ == "__main__":
    combined_df = extract_clean_tomtom_traffic_data()
    model, X_test, y_test, location_lookup = preprocess_train_data(combined_df)

    preds = model.predict(X_test)
    mape = np.mean(np.abs((y_test - preds) / np.clip(y_test, 1e-5, None)))
    accuracy = float(1.0 - mape)
    print(f"Validation Accuracy: {accuracy:.2%}")

    if accuracy < 0.85:
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"⚠️ Model Retrain Alert: Accuracy dropped to {accuracy:.2%}",
            Message=(
                f"Retraining finished with {accuracy:.2%} accuracy (below 85%). "
                "The live model was NOT updated."
            ),
        )
    else:
        model.save_model("xgboost-model.json")
        location_lookup.to_csv("location_lookup.csv", index=False)
        s3.upload_file(
            "xgboost-model.json", BUCKET, "production/xgboost-model.json"
        )
        s3.upload_file(
            "location_lookup.csv", BUCKET, "production/location_lookup.csv"
        )
        print("Model updated in S3!")
