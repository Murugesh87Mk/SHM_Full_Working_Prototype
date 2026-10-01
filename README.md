# AI-Based Structural Health Monitoring — Full Working Prototype

## What this version does
1. Upload a bridge/structure image through a Flask web dashboard.
2. Extract visual features with OpenCV.
3. Classify the image into **No Damage / Crack / Rust / Spalling** using a trained local ML classifier.
4. Calculate an image-based visual severity score.
5. Generate a sensor snapshot for strain, vibration and temperature.
6. Predict structural risk using a trained Random Forest regression model.
7. Convert risk into Low / Moderate / High / Critical levels and provide an action message.

## Important scientific limitation
The original project files did not contain a labelled real-world structural-damage dataset or actual sensor hardware. Therefore, the included models are trained on procedurally generated training data so the complete prototype is reproducible and runnable. This is suitable for a software demonstration, but **not evidence of field accuracy**. For a real deployment, train the image classifier on a labelled bridge/structure dataset and connect calibrated sensors.

## Run on Windows/macOS/Linux
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
python app.py
```
Open http://127.0.0.1:5000 in a browser.

The first run can also auto-create models if `python train.py` is skipped.

## Project structure
- `app.py` — Flask web application and upload route
- `shm_engine.py` — image analysis and risk inference
- `ml_models.py` — training and model loading
- `train.py` — reproducible model training
- `templates/dashboard.html` — dashboard UI
- `models/` — generated trained model files
- `static/uploads/` — uploaded inspection images

## Turning this into a field-ready system
- Replace generated image training samples with labelled crack/rust/spalling images.
- Add train/validation/test splits and report precision, recall, F1 and confusion matrix.
- Connect real strain gauges, accelerometers and temperature sensors through MQTT/REST/serial input.
- Store time-series observations in a database.
- Add authentication, HTTPS, audit logs and notification integration.
- Validate thresholds with structural engineers and site-specific standards.
