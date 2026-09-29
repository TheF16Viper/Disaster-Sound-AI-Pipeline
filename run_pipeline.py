import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from extract_features import extract_features 

# --- 1. LOAD THE LOCAL AUDIO FILES ---
X = []
y = []

dataset_folder = "mock_dataset"

if not os.path.exists(dataset_folder):
    print(f"❌ Error: '{dataset_folder}' directory not found. Please run generate_mock_audio.py first!")
    exit()

print("🔄 Reading audio files from local folder and extracting features...")

for filename in os.listdir(dataset_folder):
    if filename.endswith(".wav"):
        file_path = os.path.join(dataset_folder, filename)
        
        # Determine the label based on the file name structure
        if "rain_flood" in filename:
            label = "Heavy Rain / Flood Risk"
        elif "structural_crack" in filename:
            label = "Structural Cracking / Landslide Risk"
        else:
            continue
            
        # Extract features using your updated librosa function
        audio_fingerprint = extract_features(file_path)
        
        if audio_fingerprint is not None:
            X.append(audio_fingerprint)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"📊 Dataset parsing complete. Loaded {len(X)} audio samples from disk.")

# --- 2. TRAIN THE REGULARIZED MACHINE LEARNING MODEL ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training the regularized Random Forest AI engine on disk files...")
# Applying the anti-overfitting tree constraints
model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42)
model.fit(X_train, y_train)

# Calculate testing metrics
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"✅ Model Training Complete! File Pipeline Accuracy: {accuracy * 100:.2f}%")
print("\n📝 Detailed Performance Metrics for Disk Files:")
print(classification_report(y_test, predictions, zero_division=0))

# --- 3. TEST TRANSLATOR INPUT WITH PROBABILITIES ---
test_target_file = "mock_dataset/structural_crack_5.wav"

print(f"🕵️ Simulated Live Event: Testing the translator with file: '{test_target_file}'")
live_feature = extract_features(test_target_file).reshape(1, -1)
prediction = model.predict(live_feature)

# Calculate the prediction confidence
probabilities = model.predict_proba(live_feature)
confidence = np.max(probabilities) * 100

print("\n🚨 --- LIVE TRANSLATOR OUTPUT --- 🚨")
print(f"🔊 Sound Input: Analyzing disk signal matrices...")
print(f"🎯 AI Confidence Score: {confidence:.2f}%")
print(f"💬 Human Translation Alert: [ Warning: {prediction[0]} confirmed. Evacuate zone immediately. ]")
print("---------------------------------------\n")
