import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. GENERATING REALISTIC, MESSY TRAINING DATA ---
print("⏳ Generating complex, noisy synthetic data to prevent overfitting...")
X = []
y = []

# Baseline frequency vectors
for _ in range(100): # Increased sample count for better generalization
    # Class 0: Flood/Rain (Deep base frequencies + heavy random environment noise)
    base_rain = np.random.normal(loc=-15.0, scale=8.0, size=(20,))
    background_noise = np.random.normal(loc=0.0, scale=4.0, size=(20,)) # Overlapping mask
    X.append(base_rain + background_noise)
    y.append("Heavy Rain / Flood Risk")

for _ in range(100):
    # Class 1: Landslide/Cracking (Sharp peaks, but muffled by environmental noise)
    base_crack = np.random.normal(loc=-10.0, scale=9.0, size=(20,))
    background_noise = np.random.normal(loc=0.0, scale=4.0, size=(20,)) # Same noise mask to confuse the AI
    X.append(base_crack + background_noise)
    y.append("Structural Cracking / Landslide Risk")

X = np.array(X)
y = np.array(y)

# --- 2. THE MACHINE LEARNING PIPELINE WITH HYPERPARAMETER TUNING ---

# Split data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training a regularized Random Forest model...")
# Added max_depth=5 and min_samples_split=4 constraints to stop the trees from overfitting
model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42)
model.fit(X_train, y_train)

# --- 3. EVALUATING WITH COMPLEX METRICS ---
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"📊 Model Training Complete! New Realistic Accuracy: {accuracy * 100:.2f}%")
print("\n📝 Detailed Performance Metrics (Precision / Recall / F1-Score):")
print(classification_report(y_test, predictions))

# --- 4. REAL-TIME TRANSLATOR OUTPUT ---
def real_time_translator():
    random_index = np.random.randint(0, len(X_test))
    test_vector = X_test[random_index].reshape(1, -1)
    
    predicted_label = model.predict(test_vector)
    probabilities = model.predict_proba(test_vector)[0]
    confidence = max(probabilities) * 100
    
    print("🚨 --- LIVE TRANSLATOR OUTPUT --- 🚨")
    print(f"🔊 Signal Stream Status: Matrix processing complete.")
    print(f"🎯 AI Confidence Score: {confidence:.2f}%")
    print(f"💬 Human Translation Alert: [ {predicted_label[0]} detected. Take immediate shelter. ]")
    print("---------------------------------------\n")

real_time_translator()
