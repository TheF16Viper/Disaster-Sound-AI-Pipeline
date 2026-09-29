import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from extract_features import extract_features

X = []
y = []
dataset_folder = "advanced_dataset"

if not os.path.exists(dataset_folder):
    print(f"❌ Error: '{dataset_folder}' folder missing! Run generate_mock_audio.py first.")
    exit()

class_mapping = {
    "normal_rain": "🌤️ Normal Ambient Rain (Safe)",
    "rushing_flood": "🌊 Active Rushing Flood Hazard",
    "structural_seismic": "🏢 Structural Concrete Fracture Alert",
    "landslide_rumble": "🏔️ Landslide / Seismic Mudslide Activity",
    "wildfire_crackle": "🔥 Active Wildfire Proximity Alert"
}

print("🔄 Processing Multi-Class audio features from local folder...")
for folder_name, alert_label in class_mapping.items():
    folder_path = os.path.join(dataset_folder, folder_name)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            
            fingerprint = extract_features(file_path)
            if fingerprint is not None:
                # FIX: Check if this is your real downloaded file
                if "dragon" in filename:
                    print(f"🔥 Real file detected: '{filename}'. Applying Data Oversampling (x20)...")
                    # Duplicate its numerical footprint 20 times to balance out the 15 synthetic files!
                    for _ in range(20):
                        X.append(fingerprint)
                        y.append(alert_label)
                else:
                    X.append(fingerprint)
                    y.append(alert_label)

X = np.array(X)
y = np.array(y)

print(f"📊 Dataset Loaded: Balanced array containing {len(X)} matrices.")

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Optimizing multi-class Random Forest AI model layers...")
model = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_split=3, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"✅ Training Complete! Performance Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")

# Overwrite the old brain asset
model_filename = "disaster_model.joblib"
joblib.dump(model, model_filename)
print(f"💾 Exported balanced multi-class engine asset as '{model_filename}'")
