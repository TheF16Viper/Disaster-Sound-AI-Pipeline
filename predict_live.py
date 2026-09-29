import os
import numpy as np
import joblib
from extract_features import extract_features

def live_audio_alert_system(new_audio_path):
    model_filename = "disaster_model.joblib"
    
    # 1. Check if the saved model file exists
    if not os.path.exists(model_filename):
        print(f"❌ Error: Compiled brain file '{model_filename}' not found. Run train_and_save.py first!")
        return

    # 2. Instantly load the pre-trained brain asset
    print("🧠 Loading pre-trained AI architecture from disk asset...")
    loaded_model = joblib.load(model_filename)
    
    # 3. Process the new audio file using your librosa engine
    print(f"🔊 Processing live target stream: '{new_audio_path}'")
    fingerprint = extract_features(new_audio_path)
    
    if fingerprint is None:
        print("❌ Feature extraction failed. Signal corrupted.")
        return
        
    # Reshape the 1D array of 20 numbers into a 2D format required by the model (1 sample, 20 features)
    fingerprint_reshaped = fingerprint.reshape(1, -1)
    
    # 4. Generate the translation alert using the loaded brain
    predicted_label = loaded_model.predict(fingerprint_reshaped)[0]
    probabilities = loaded_model.predict_proba(fingerprint_reshaped)[0]
    
    # Find the confidence score percentage for the chosen prediction
    class_labels = loaded_model.classes_
    winning_index = np.where(class_labels == predicted_label)[0][0]
    confidence_score = probabilities[winning_index] * 100

    print("\n🚨 --- LIVE TRANSLATOR OUTPUT --- 🚨")
    print(f"🎯 AI Engine Confidence Score: {confidence_score:.2f}%")
    print(f"💬 Human Translation Alert: [ WARNING: {predicted_label} confirmed in this sector. ]")
    print("---------------------------------------\n")

if __name__ == "__main__":
    # Test it on a sample crackling audio file on your disk
    sample_file = "mock_dataset/structural_crack_3.wav"
    live_audio_alert_system(sample_file)
