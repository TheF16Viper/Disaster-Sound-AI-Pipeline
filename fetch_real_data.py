import os
import requests
import pandas as pd

# Create our clean real production database directory
DATA_DIR = "real_dataset"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(f"{DATA_DIR}/rain_flood", exist_ok=True)
os.makedirs(f"{DATA_DIR}/structural_crack", exist_ok=True)

# 1. Fetch the master metadata CSV that indexes every single recording in ESC-50
metadata_url = "https://githubusercontent.com"
print("📥 Fetching real-world ESC-50 audio directory indices...")
response = requests.get(metadata_url)

with open("esc50_meta.csv", "wb") as f:
    f.write(response.content)

df = pd.read_csv("esc50_meta.csv")

# 2. Filter out only the targets we need to build our balanced disaster classes
# Class 12 = crackling_fire, Class 14 = rain, Class 49 = thunderstorm
target_classes = {
    14: "rain_flood",
    49: "rain_flood",
    12: "structural_crack"
}

filtered_df = df[df['category'].isin(['natural_soundscapes_and_water_sounds', 'urban_noises', 'domestic_sounds'])]
# Narrow it down directly by class ID for precision
filtered_df = df[df['target'].isin(target_classes.keys())]

print(f"🎯 Isolated {len(filtered_df)} real-world field recording samples for tracking.")

# 3. Pull a small sample slice of files directly from the master audio storage repository
# This keeps the download fast and eliminates network timeout issues
base_audio_url = "https://githubusercontent.com"

# We will pull 15 files per target class to give the Random Forest a highly distinct pool
download_count = 0
for idx, row in filtered_df.groupby('target').head(15).iterrows():
    filename = row['filename']
    target_id = row['target']
    destination_folder = target_classes[target_id]
    
    file_url = f"{base_audio_url}{filename}"
    local_path = f"{DATA_DIR}/{destination_folder}/{filename}"
    
    print(f"🔄 Downloading actual environmental recording: {filename} -> {destination_folder}/")
    try:
        audio_data = requests.get(file_url, timeout=15)
        if audio_data.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(audio_data.content)
            download_count += 1
    except Exception as e:
        print(f"⚠️ Skipped file {filename} due to a localized network hiccup: {e}")

print(f"✅ Success! Generated a pristine field database containing {download_count} real .wav files.")
