import librosa
import numpy as np

def extract_features(file_path):
    """
    Loads an audio file and compresses its frequency spectrum 
    into a uniform 20-digit numerical array (MFCC vector).
    """
    try:
        # FIX: Removed the resample_type argument to support modern Librosa versions
        audio, sample_rate = librosa.load(file_path, sr=22050)
        
        # Extract 20 MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        
        # Take the average across the time duration to get a single row of 20 numbers
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Feature extraction module updated and ready!")
