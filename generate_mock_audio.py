import numpy as np
import scipy.io.wavfile as wav
import os

def create_advanced_multiclass_dataset():
    DATA_DIR = "advanced_dataset"
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 5 specialized disaster/environmental folders
    categories = [
        "normal_rain",          # High-frequency continuous shimmer (hiss)
        "rushing_flood",        # High-volume roaring mid-to-low frequencies
        "structural_seismic",   # Abrupt, explosive isolated shock cracks (short)
        "landslide_rumble",     # Long, sustained, violent low infrasonic rumbles
        "wildfire_crackle"      # Continuous, random micro-popping crackles
    ]
    
    for cat in categories:
        os.makedirs(os.path.join(DATA_DIR, cat), exist_ok=True)

    sample_rate = 22050
    duration = 4.0  # 4-second sound clips
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    print("🔊 Synthesizing advanced Multi-Class audio data locally...")

    for i in range(1, 16):  # 15 sample clips per class
        # 1. NORMAL RAIN: High-frequency Gaussian noise (shimmering hiss/patter)
        rain_wave = np.random.normal(0, 0.08, len(t))
        wav.write(f"{DATA_DIR}/normal_rain/rain_{i}.wav", sample_rate, rain_wave.astype(np.float32))

        # 2. RUSHING FLOOD: Deep water textures (Heavy low hum + moving mid frequencies)
        flood_wave = 0.5 * np.sin(2 * np.pi * 150 * t) + np.random.normal(0, 0.25, len(t))
        wav.write(f"{DATA_DIR}/rushing_flood/flood_{i}.wav", sample_rate, flood_wave.astype(np.float32))

        # 3. STRUCTURAL SEISMIC: Instantaneous sudden concrete crack (dies out in 0.5 seconds)
        envelope_short = np.exp(-t * 12)  # Sharp decay
        crack_wave = (0.7 * np.sin(2 * np.pi * 1500 * t)) * envelope_short + np.random.normal(0, 0.02, len(t))
        wav.write(f"{DATA_DIR}/structural_seismic/seismic_{i}.wav", sample_rate, crack_wave.astype(np.float32))

        # 4. LANDSLIDE RUMBLE: Sustained earthquake grind (Massive low-frequency rumbles < 90Hz)
        rumble_wave = 0.6 * np.sin(2 * np.pi * 55 * t) + 0.4 * np.sin(2 * np.pi * 85 * t) + np.random.normal(0, 0.1, len(t))
        wav.write(f"{DATA_DIR}/landslide_rumble/landslide_{i}.wav", sample_rate, rumble_wave.astype(np.float32))

        # 5. WILDFIRE CRACKLE: Continuous background crackling pops
        fire_wave = np.random.normal(0, 0.03, len(t))
        # Add random popping spikes throughout the track
        pops = np.random.choice([0, 1], size=len(t), p=[0.999, 0.001]) * np.random.normal(0, 0.6, len(t))
        fire_wave += pops
        wav.write(f"{DATA_DIR}/wildfire_crackle/fire_{i}.wav", sample_rate, fire_wave.astype(np.float32))

    print("✅ Multi-Class database successfully written to the 'advanced_dataset/' folder.")

if __name__ == "__main__":
    create_advanced_multiclass_dataset()
