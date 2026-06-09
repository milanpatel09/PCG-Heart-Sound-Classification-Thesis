import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.preprocessing import (
    load_signal, 
    schmidt_spike_removal, 
    butterworth_bandpass, 
    segment_signal, 
    normalize_signal
)

# CONFIGURATION 
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
TARGET_SR = 1000
WINDOW_SECONDS = 5
# Folders usually found in PhysioNet 2016
FOLDERS = ['validation']
# FOLDERS = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']

def process_dataset():
    # Ensure output directory exists
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    all_segments = []
    all_labels = []
    
    print("--- Stage 1: Preprocessing Started ---")
    
    for folder in FOLDERS:
        folder_path = os.path.join(RAW_DATA_PATH, folder)
        
        # Check if folder exists (in case you only downloaded subset)
        if not os.path.exists(folder_path):
            print(f"Skipping {folder} (folder not found in data/raw)")
            continue
            
        # Load the REFERENCE.csv which contains labels
        # CSV Format: filename, label (1=Abnormal, -1=Normal)
        ref_path = os.path.join(folder_path, 'REFERENCE.csv')
        try:
            df = pd.read_csv(ref_path, names=['filename', 'label'])
        except FileNotFoundError:
            print(f"Skipping {folder} (REFERENCE.csv not found)")
            continue
            
        print(f"Processing {folder} ({len(df)} files)...")
        
        # Iterate over every file in the CSV
        for _, row in tqdm(df.iterrows(), total=len(df)):
            filename = row['filename']
            raw_label = row['label']
            
            # --- LABEL MAPPING ---
            # PhysioNet uses: -1 (Abnormal), 1 (Normal)
            # We convert to: 1 (Abnormal/Positive), 0 (Normal/Negative)
            binary_label = 1 if raw_label == 1 else 0
            
            file_path = os.path.join(folder_path, f"{filename}.wav")
            
            # 1. Load
            signal, sr = load_signal(file_path, target_sr=TARGET_SR)
            if signal is None: continue
            
            # 2. Spike Removal (Cleaning friction)
            signal = schmidt_spike_removal(signal, sr)
            
            # 3. Bandpass Filter (Focus on heart sounds)
            signal = butterworth_bandpass(signal, fs=sr)
            
            # 4. Normalize (Z-score)
            signal = normalize_signal(signal)
            
            # 5. Segment (Cut into 5s chunks)
            segments = segment_signal(signal, sr, window_size=WINDOW_SECONDS)
            
            # Append segments and duplicate the label for each segment
            for seg in segments:
                all_segments.append(seg)
                all_labels.append(binary_label)

    # SAVING
    if len(all_segments) == 0:
        print("ERROR: No data processed! Check your 'data/raw' folder structure.")
        return

    # Convert list to huge numpy array
    X = np.array(all_segments)
    y = np.array(all_labels)
    
    print(f"\nProcessing Complete.")
    print(f"Total Processed Segments: {X.shape[0]}")
    print(f"Data Shape: {X.shape} (Segments, Timepoints)")
    
    # Save to disk
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_data.npy'), X)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'y_data.npy'), y)
    print(f"Saved 'X_data.npy' and 'y_data.npy' to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    process_dataset()