from pathlib import Path
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename

from config import UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
from ml_models import ensure_models
from shm_engine import analyze_image, predict_risk, sensor_snapshot

app = Flask(__name__)
app.secret_key = "shm-prototype-local-secret"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    sensor = sensor_snapshot()
    return render_template("dashboard.html", sensor=sensor, damage=None, risk=None, error=None)


@app.post("/analyze")
def analyze():
    sensor = sensor_snapshot()
    damage = None
    risk = None
    error = None
    file = request.files.get("file")
    age_days = request.form.get("age_days", "365")
    try:
        age_days = max(0.0, min(float(age_days), 36500.0))
    except ValueError:
        age_days = 365.0
    if not file or not file.filename:
        error = "Please select an image."
    elif not allowed(file.filename):
        error = "Allowed image types: PNG, JPG, JPEG, WEBP."
    else:
        filename = secure_filename(file.filename)
        path = Path(app.config["UPLOAD_FOLDER"]) / filename
        file.save(path)
        try:
            damage = analyze_image(path)
            risk_value, risk_level, action = predict_risk(sensor, damage["visual_score"], age_days)
            risk = {"value": risk_value, "level": risk_level, "action": action, "age_days": age_days}
        except Exception as exc:
            error = f"Analysis failed: {exc}"
    return render_template("dashboard.html", sensor=sensor, damage=damage, risk=risk, error=error)


if __name__ == "__main__":
    ensure_models()
    app.run(host="127.0.0.1", port=5000, debug=False)
