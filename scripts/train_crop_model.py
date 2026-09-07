import json
from pathlib import Path
import numpy as np
import xgboost as xgb
import joblib

ml_service_dir = Path(__file__).resolve().parent.parent / "ml-service"
models_dir = ml_service_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)
labels_path = models_dir / "crop_labels.json"

labels = json.loads(labels_path.read_text())

# Typical agronomic profiles for the 22 crops
# FEATURES: ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_PROFILES = {
    "rice":        {"N": (80, 100), "P": (35, 60),  "K": (35, 45),  "temperature": (20, 27), "humidity": (80, 90), "ph": (5.5, 7.0), "rainfall": (180, 300)},
    "maize":       {"N": (60, 100), "P": (35, 60),  "K": (15, 25),  "temperature": (18, 27), "humidity": (55, 75), "ph": (5.5, 7.0), "rainfall": (60, 110)},
    "chickpea":    {"N": (20, 60),  "P": (55, 80),  "K": (75, 85),  "temperature": (17, 22), "humidity": (14, 20), "ph": (6.0, 7.5), "rainfall": (65, 95)},
    "kidneybeans": {"N": (15, 40),  "P": (55, 80),  "K": (15, 25),  "temperature": (15, 25), "humidity": (18, 25), "ph": (5.5, 6.0), "rainfall": (60, 150)},
    "pigeonpeas":  {"N": (15, 40),  "P": (55, 80),  "K": (15, 25),  "temperature": (25, 35), "humidity": (45, 65), "ph": (5.5, 6.5), "rainfall": (90, 170)},
    "mothbeans":   {"N": (15, 40),  "P": (35, 60),  "K": (15, 25),  "temperature": (25, 32), "humidity": (40, 65), "ph": (6.0, 7.5), "rainfall": (30, 75)},
    "mungbean":    {"N": (15, 40),  "P": (35, 60),  "K": (15, 25),  "temperature": (27, 30), "humidity": (80, 90), "ph": (6.2, 7.2), "rainfall": (35, 60)},
    "blackgram":   {"N": (35, 60),  "P": (55, 80),  "K": (15, 25),  "temperature": (25, 35), "humidity": (60, 70), "ph": (6.5, 7.5), "rainfall": (60, 75)},
    "lentil":      {"N": (15, 40),  "P": (55, 80),  "K": (15, 25),  "temperature": (18, 30), "humidity": (60, 70), "ph": (6.0, 7.5), "rainfall": (35, 55)},
    "pomegranate": {"N": (15, 40),  "P": (10, 30),  "K": (35, 45),  "temperature": (18, 25), "humidity": (85, 95), "ph": (5.5, 7.2), "rainfall": (100, 115)},
    "banana":      {"N": (90, 120), "P": (70, 95),  "K": (45, 55),  "temperature": (25, 30), "humidity": (75, 85), "ph": (5.5, 6.5), "rainfall": (90, 120)},
    "mango":       {"N": (15, 40),  "P": (15, 35),  "K": (25, 35),  "temperature": (27, 36), "humidity": (45, 55), "ph": (4.5, 7.0), "rainfall": (85, 105)},
    "grapes":      {"N": (15, 40),  "P": (120, 145),"K": (195, 205),"temperature": (8, 42),  "humidity": (80, 85), "ph": (5.5, 6.5), "rainfall": (65, 75)},
    "watermelon":  {"N": (80, 120), "P": (5, 30),   "K": (45, 55),  "temperature": (24, 27), "humidity": (80, 90), "ph": (6.0, 7.0), "rainfall": (40, 60)},
    "muskmelon":   {"N": (80, 120), "P": (5, 30),   "K": (45, 55),  "temperature": (27, 30), "humidity": (90, 95), "ph": (6.0, 6.8), "rainfall": (20, 30)},
    "apple":       {"N": (15, 40),  "P": (120, 145),"K": (195, 205),"temperature": (21, 24), "humidity": (90, 95), "ph": (5.5, 6.5), "rainfall": (100, 130)},
    "orange":      {"N": (15, 40),  "P": (5, 30),   "K": (5, 15),   "temperature": (15, 35), "humidity": (90, 95), "ph": (6.0, 8.0), "rainfall": (100, 120)},
    "papaya":      {"N": (40, 60),  "P": (45, 70),  "K": (45, 55),  "temperature": (23, 44), "humidity": (90, 95), "ph": (6.5, 7.0), "rainfall": (40, 250)},
    "coconut":     {"N": (15, 40),  "P": (5, 30),   "K": (25, 35),  "temperature": (25, 29), "humidity": (90, 99), "ph": (5.5, 6.5), "rainfall": (130, 230)},
    "cotton":      {"N": (100, 140),"P": (35, 60),  "K": (15, 25),  "temperature": (22, 26), "humidity": (60, 85), "ph": (6.0, 8.0), "rainfall": (60, 100)},
    "jute":        {"N": (60, 100), "P": (35, 60),  "K": (35, 45),  "temperature": (23, 26), "humidity": (70, 90), "ph": (6.0, 7.5), "rainfall": (150, 200)},
    "coffee":      {"N": (80, 120), "P": (15, 40),  "K": (25, 35),  "temperature": (23, 28), "humidity": (50, 70), "ph": (6.0, 7.5), "rainfall": (115, 200)},
}

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

np.random.seed(42)
X_list = []
y_list = []

for idx, label in enumerate(labels):
    prof = CROP_PROFILES.get(label.lower(), {
        "N": (30, 80), "P": (20, 60), "K": (20, 60),
        "temperature": (20, 30), "humidity": (50, 80), "ph": (5.5, 7.5), "rainfall": (80, 180)
    })
    
    # Generate 150 synthetic samples per crop class with realistic variance
    n_samples = 150
    for _ in range(n_samples):
        sample = []
        for feat in FEATURES:
            low, high = prof[feat]
            val = np.random.uniform(low, high)
            sample.append(val)
        X_list.append(sample)
        y_list.append(idx)

X = np.array(X_list)
y = np.array(y_list)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective="multi:softprob",
    random_state=42
)
model.fit(X, y)

output_model_path = models_dir / "crop_model.joblib"
joblib.dump(model, str(output_model_path))
print(f"✅ Trained XGBoost Crop Model and saved to {output_model_path}")
