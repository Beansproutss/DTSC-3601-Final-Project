import csv
import json
import os
import boto3
import xgboost as xgb
from traffic_predictor import predict_custom_traffic


s3 = boto3.client("s3")
MODEL_BUCKET = os.environ["MODEL_BUCKET"]
MODEL_KEY = os.environ["MODEL_KEY"]
LOCAL_MODEL_PATH = "/tmp/model.json"
LOOKUP_KEY = "production/location_lookup.csv"

cached_model = None
cached_lookup = {}
last_modified = None

def _get_latest_model():
    global cached_model, cached_lookup, last_modified

    meta = s3.head_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
    s3_timestamp = meta["LastModified"]

    if cached_model is None or s3_timestamp != last_modified:
        s3.download_file(MODEL_BUCKET, MODEL_KEY, "/tmp/xgboost-model.json")
        s3.download_file(MODEL_BUCKET, LOOKUP_KEY, "/tmp/location_lookup.csv")

        model = xgb.XGBRegressor()
        model.load_model("/tmp/xgboost-model.json")

        lookup = {}
        with open("/tmp/location_lookup.csv", "r") as f:
            for row in csv.DictReader(f):
                lookup[(float(row["latitude"]), float(row["longitude"]))] =int(row["location_id"])
        cached_model = model
        cached_lookup = lookup
        last_modified = s3_timestamp
    return cached_model, cached_lookup

def lambda_handler(event, context):
    raw_body = event.get("body")
    if isinstance(raw_body, str):
        body = json.loads(raw_body)
    elif isinstance(raw_body, dict):
        body = raw_body
    else:
        body = {}

    dt_input = body.get("timestamp")
    location_id = body.get("location_id")

    if dt_input is None or location_id is None:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Missing 'timestamp' or 'location_id'."}),
        }

    try:
        location_id = int(location_id)
    except (TypeError, ValueError):
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "'location_id' must be an integer."}),
        }

    model, cached_lookup = _get_latest_model()

    predicted_speed, traffic_level = predict_custom_traffic(model, location_id, dt_input)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "predicted_speed": round(float(predicted_speed), 2),
            "traffic_level": traffic_level
        }),
    }

