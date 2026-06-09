import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import gc

# Import your existing classes
from src.dataset import PCGDataset
from src.models import AudioResNet

# ==========================================
# 1. CONFIGURATION
# ==========================================
FEATURES = ['melspec', 'mfcc', 'cwt', 'spectrogram', 'scattering']
ARCH = 'resnet18'

DATA_PATH = 'data/processed'
FEATURES_PATH = 'data/features'
MODELS_DIR = 'models_checkpoints'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load True Labels and Patient Groups
    label_path = os.path.join(DATA_PATH, 'y_data.npy')
    groups_path = os.path.join(DATA_PATH, 'groups.npy')
    
    y_data = np.load(label_path)
    groups = np.load(groups_path)
    total_samples = len(y_data)
    
    print(f"Loaded {total_samples} segments.")

    # 2. Create Unique Segment Names (e.g., 'a0001_seg0', 'a0001_seg1')
    segment_names = []
    current_group = ""
    seg_idx = 0
    
    for g in groups:
        if g != current_group:
            current_group = g
            seg_idx = 0
        segment_names.append(f"{g}_seg{seg_idx}")
        seg_idx += 1

    # Initialize a master matrix to hold all predictions (24450 rows, 5 columns)
    oof_predictions = np.zeros((total_samples, len(FEATURES)), dtype=np.float32)

    # 3. Setup the EXACT SAME SPLITTER used in Stage 3
    # This guarantees the Validation indices match your training runs perfectly
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. Extract Out-of-Fold (OOF) Predictions
    for feature_idx, feature_name in enumerate(FEATURES):
        print(f"\n{'='*50}")
        print(f" Extracting Unbiased Scores for: {feature_name.upper()}")
        print(f"{'='*50}")
        
        batch_size = 8 if feature_name == 'cwt' else 32
        
        feat_path = os.path.join(FEATURES_PATH, feature_name, f"{feature_name}.npy")
        if not os.path.exists(feat_path):
             feat_path = os.path.join(FEATURES_PATH, f"{feature_name}.npy")
             
        # Load dataset
        dataset = PCGDataset(feat_path, label_path)
        
        # Loop through the 5 Folds
        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(dataset.X, y_data, groups)):
            current_fold = fold_idx + 1
            
            # CRITICAL: We only load the Validation Subset, and shuffle MUST be False
            val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
            
            # Load the specific model trained on the OTHER 4 folds
            model = AudioResNet(architecture=ARCH).to(device)
            weight_path = os.path.join(MODELS_DIR, ARCH, feature_name, f"fold_{current_fold}", "best_model.pth")
            
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"Missing weight file: {weight_path}. Did you rename the best epoch to 'best_model.pth'?")
                
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()
            
            fold_probs = []
            
            # Predict the Validation Set
            with torch.no_grad():
                for inputs, _ in tqdm(val_loader, desc=f"Predicting Fold {current_fold}", leave=False):
                    inputs = inputs.to(device)
                    
                    logits = model(inputs)
                    probs = F.softmax(logits, dim=1)
                    
                    # Grab probability for Class 1 (Abnormal)
                    abnormal_probs = probs[:, 1].cpu().numpy()
                    fold_probs.extend(abnormal_probs)
            
            # Inject the predictions directly into the correct rows of our master matrix
            oof_predictions[val_idx, feature_idx] = fold_probs
            
            # Memory Management
            del model
            torch.cuda.empty_cache()
            
        # Clean up the dataset from RAM before loading the next feature
        del dataset
        gc.collect()

    # 5. Assemble the final CSV
    csv_data = {
        "Segment_Name": segment_names
    }
    
    # Add the 5 feature columns
    for feature_idx, feature_name in enumerate(FEATURES):
        csv_data[f"{feature_name}_score"] = oof_predictions[:, feature_idx]
        
    # Add the True Target column
    csv_data["Target"] = y_data
    
    df = pd.DataFrame(csv_data)
    
    # 6. Save the CSV
    out_path = os.path.join(DATA_PATH, "meta_features_unbiased.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\n✅ SUCCESS! Unbiased CSV generated.")
    print(f"Saved to: {out_path}")
    print("\nPreview of the final data:")
    print(df.head(10))

if __name__ == "__main__":
    main()