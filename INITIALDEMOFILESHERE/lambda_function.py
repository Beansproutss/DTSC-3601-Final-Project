import json
import os
import boto3
import xgboost as xgb
from traffic_predictor import predict_custom_traffic


s3 = boto3.client("s3")
MODEL_BUCKET = os.environ["MODEL_BUCKET"]
MODEL_KEY = os.environ["MODEL_KEY"]
LOCAL_MODEL_PATH = "/tmp/model.json"
_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(LOCAL_MODEL_PATH):
            s3.download_file(MODEL_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)

        loaded = xgb.XGBRegressor()
        loaded.load_model(LOCAL_MODEL_PATH)
        _MODEL = loaded
    return _MODEL

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

    predicted_speed, traffic_level = predict_custom_traffic(_get_model(), location_id, dt_input)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "predicted_speed": round(float(predicted_speed), 2),
            "traffic_level": traffic_level
        }),
    }

