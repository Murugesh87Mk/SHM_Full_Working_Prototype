"""Train lightweight local ML models used by the SHM prototype.

The image model is trained on procedurally generated visual patterns because the
project bundle does not contain a labelled structural-damage dataset. Replace
this training source with a real labelled dataset before making field claims.
"""
from pathlib import Path
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from config import MODEL_DIR

IMAGE_MODEL = MODEL_DIR / "damage_classifier.joblib"
RISK_MODEL = MODEL_DIR / "risk_model.joblib"
LABELS = ["No Damage", "Crack", "Rust", "Spalling"]


def image_features(img):
    img = cv2.resize(img, (128, 128))
    if img.ndim == 2:
        gray = img
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160)
    # Compact, interpretable visual descriptors.
    feat = [
        float(gray.mean()) / 255,
        float(gray.std()) / 255,
        float(edges.mean()) / 255,
        float(np.mean((hsv[:, :, 0] < 25) & (hsv[:, :, 1] > 70))) if hsv.size else 0,
        float(np.mean((hsv[:, :, 0] > 5) & (hsv[:, :, 0] < 30) & (hsv[:, :, 1] > 80))),
        float(np.mean(gray < 70)),
        float(np.mean(gray > 210)),
    ]
    # coarse grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    hist = hist / max(hist.sum(), 1)
    feat.extend(hist.tolist())
    return np.asarray(feat, dtype=np.float32)


def make_sample(label, rng):
    h = w = 128
    img = np.full((h, w, 3), rng.integers(145, 220), dtype=np.uint8)
    # texture
    noise = rng.normal(0, 10, (h, w, 1))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if label == "Crack":
        for _ in range(rng.integers(2, 6)):
            x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
            pts = [(x1, y1)]
            for _ in range(rng.integers(3, 8)):
                x1 = int(np.clip(x1 + rng.integers(-20, 21), 0, w-1))
                y1 = int(np.clip(y1 + rng.integers(8, 25), 0, h-1))
                pts.append((x1, y1))
            cv2.polylines(img, [np.array(pts)], False, (20, 20, 20), int(rng.integers(1, 3)))
    elif label == "Rust":
        for _ in range(rng.integers(4, 12)):
            center = (int(rng.integers(0,w)), int(rng.integers(0,h)))
            radius = int(rng.integers(5, 22))
            cv2.circle(img, center, radius, (30, 70, 150), -1)
    elif label == "Spalling":
        for _ in range(rng.integers(2, 7)):
            center = (int(rng.integers(10,w-10)), int(rng.integers(10,h-10)))
            axes = (int(rng.integers(7, 25)), int(rng.integers(5, 20)))
            cv2.ellipse(img, center, axes, rng.integers(0,180), 0, 360, (70,70,70), -1)
    return img


def train_damage_model(seed=42, per_class=300):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for idx, label in enumerate(LABELS):
        for _ in range(per_class):
            X.append(image_features(make_sample(label, rng)))
            y.append(idx)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    model.fit(np.vstack(X), np.asarray(y))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS}, IMAGE_MODEL)
    return model


def train_risk_model(seed=42, n=3000):
    rng = np.random.default_rng(seed)
    # Features: strain, vibration, temperature, current visual severity, age(days)
    strain = rng.uniform(0.05, 1.2, n)
    vibration = rng.uniform(0.05, 1.0, n)
    temp = rng.uniform(15, 50, n)
    visual = rng.uniform(0, 100, n)
    age = rng.uniform(0, 3650, n)
    X = np.column_stack([strain, vibration, temp, visual, age])
    risk = (strain / 1.2 * 30 + vibration * 22 + np.maximum(temp-25,0) * 0.7 + visual * 0.38 + age / 3650 * 12)
    risk += rng.normal(0, 3, n)
    risk = np.clip(risk, 0, 100)
    model = Pipeline([("scale", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=180, random_state=seed, min_samples_leaf=3))])
    model.fit(X, risk)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RISK_MODEL)
    return model


def ensure_models():
    if not IMAGE_MODEL.exists():
        train_damage_model()
    if not RISK_MODEL.exists():
        train_risk_model()


def load_models():
    ensure_models()
    bundle = joblib.load(IMAGE_MODEL)
    return bundle["model"], bundle["labels"], joblib.load(RISK_MODEL)
