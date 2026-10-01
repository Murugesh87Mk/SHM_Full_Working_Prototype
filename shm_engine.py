from datetime import datetime, timezone
import cv2
import numpy as np
from ml_models import image_features, load_models


def analyze_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")
    model, labels, _ = load_models()
    features = image_features(image).reshape(1, -1)
    probs = model.predict_proba(features)[0]
    idx = int(np.argmax(probs))
    label = labels[idx]
    confidence = float(probs[idx]) * 100

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(np.mean(edges > 0))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rust_mask = ((hsv[:,:,0] < 30) & (hsv[:,:,1] > 75) & (hsv[:,:,2] > 50))
    dark_mask = gray < 75
    visual_score = min(100.0, edge_ratio * 180 + float(rust_mask.mean()) * 120 + float(dark_mask.mean()) * 50)

    if label == "No Damage":
        severity = "Low" if visual_score < 25 else "Moderate"
    elif visual_score >= 60:
        severity = "High"
    elif visual_score >= 30:
        severity = "Moderate"
    else:
        severity = "Low"

    return {
        "type": label,
        "confidence": round(confidence, 2),
        "severity": severity,
        "visual_score": round(visual_score, 2),
        "edge_ratio": round(edge_ratio * 100, 2),
        "rust_ratio": round(float(rust_mask.mean()) * 100, 2),
        "time": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
    }


def predict_risk(sensor, visual_score, age_days):
    _, _, risk_model = load_models()
    X = np.array([[sensor["strain"], sensor["vibration"], sensor["temperature"], visual_score, age_days]], dtype=float)
    risk = float(np.clip(risk_model.predict(X)[0], 0, 100))
    if risk >= 75:
        level = "Critical"
        action = "Immediate inspection and restricted operation are recommended."
    elif risk >= 50:
        level = "High"
        action = "Schedule a detailed structural inspection soon."
    elif risk >= 25:
        level = "Moderate"
        action = "Increase monitoring frequency and review trends."
    else:
        level = "Low"
        action = "Continue routine monitoring."
    return round(risk, 2), level, action


def sensor_snapshot(seed=None):
    rng = np.random.default_rng(seed)
    return {
        "strain": round(float(rng.uniform(0.05, 1.0)), 3),
        "vibration": round(float(rng.uniform(0.05, 0.9)), 3),
        "temperature": round(float(rng.uniform(18, 45)), 1),
    }
