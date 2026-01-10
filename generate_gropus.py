import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.preprocessing import load_signal, segment_signal

# --- CONFIGURATION (Must match Stage 1) ---
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
TARGET_SR = 1000
WINDOW_SECONDS = 5
FOLDERS = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']

def generate_groups():
    print("--- Rebuilding Patient Index (Groups) ---")
    
    # 1. Load the existing labels to verify the count later
    y_path = os.path.join(PROCESSED_DATA_PATH, 'y_data.npy')
    if not os.path.exists(y_path):
        print("Error: y_data.npy not found. Run Stage 1 first.")
        return
    
    # We need to match this number EXACTLY
    target_count = len(np.load(y_path))
    print(f"Target Total Samples (from y_data): {target_count}")
    
    all_groups = []
    
    # 2. Replay the exact loop from Stage 1
    for folder in FOLDERS:
        folder_path = os.path.join(RAW_DATA_PATH, folder)
        if not os.path.exists(folder_path):
            continue
            
        ref_path = os.path.join(folder_path, 'REFERENCE.csv')
        try:
            # Pandas reads CSVs deterministically (row 1, row 2, row 3...)
            df = pd.read_csv(ref_path, names=['filename', 'label'])
        except FileNotFoundError:
            continue
            
        print(f"Scanning {folder}...")
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            filename = row['filename']
            file_path = os.path.join(folder_path, f"{filename}.wav")
            
            # Load signal to get exact length (same logic as Stage 1)
            signal, sr = load_signal(file_path, target_sr=TARGET_SR)
            
            # If Stage 1 failed to load this file, Stage 1 skipped it.
            # We must skip it too to keep alignment.
            if signal is None:
                continue
                
            # Calculate how many segments Stage 1 created for this file
            segments = segment_signal(signal, sr, window_size=WINDOW_SECONDS)
            num_segments = len(segments)
            
            # Add the Patient ID for EACH segment
            # If file 'a0001' produced 3 segments, we add ['a0001', 'a0001', 'a0001']
            for _ in range(num_segments):
                all_groups.append(filename)
                
    # 3. Final Verification
    all_groups = np.array(all_groups)
    print(f"\nGenerated Groups Shape: {all_groups.shape}")
    
    if len(all_groups) == target_count:
        print("✅ SUCCESS: Group count matches y_data count perfectly!")
        save_path = os.path.join(PROCESSED_DATA_PATH, 'groups.npy')
        np.save(save_path, all_groups)
        print(f"Saved to {save_path}")
    else:
        print(f"❌ CRITICAL MISMATCH: Calculated {len(all_groups)} groups, but y_data has {target_count}.")
        print("Possible causes: Data changed, Stage 1 settings changed, or a file load error occurred differently.")

if __name__ == "__main__":
    generate_groups()