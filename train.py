import glob
import gzip
import io
import json
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


def load_local_traffic_data(input_dir: str) -> pd.DataFrame:
    """Replaces S3 pagination by reading files SageMaker downloaded to local disk."""
    df_list = []
    # Matches your exact key.endswith(".csv") restriction
    for filepath in glob.glob(
        os.path.join(input_dir, "**", "*.csv"), recursive=True
    ):
        with open(filepath, "rb") as f:
            raw_bytes = f.read()

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

    # Create location_id and save the lookup table for your live API Lambda
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
    input_dir = os.environ.get(
        "SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"
    )
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    combined_df = load_local_traffic_data(input_dir)
    model, X_test, y_test, location_lookup = preprocess_train_data(combined_df)

    # Evaluate accuracy (using 1 - MAPE for speed regression; adjust if you used R2)
    preds = model.predict(X_test)
    mape = np.mean(np.abs((y_test - preds) / np.clip(y_test, 1e-5, None)))
    accuracy = float(1.0 - mape)

    # Save model + location lookup (SageMaker zips /opt/ml/model into model.tar.gz)
    model.save_model(os.path.join(model_dir, "xgboost-model.json"))
    location_lookup.to_parquet(
        os.path.join(model_dir, "location_lookup.parquet"), index=False
    )

    # Save accuracy for the Step Functions evaluation Lambda
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": accuracy}, f)
