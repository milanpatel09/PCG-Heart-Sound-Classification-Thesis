import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import gc

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

# ==========================================
# 2. MAIN EXTRACTION LOGIC
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load labels and groups to set up the exact same OOF splits
    label_path = os.path.join(DATA_PATH, 'y_data.npy')
    groups_path = os.path.join(DATA_PATH, 'groups.npy')
    
    y_data = np.load(label_path)
    groups = np.load(groups_path)
    total_samples = len(y_data)
    
    # Initialize the Meta-Feature Matrix (X_meta) and Labels (y_meta)
    X_meta = np.zeros((total_samples, len(FEATURES)), dtype=np.float32)
    y_meta = np.zeros((total_samples,), dtype=np.int64)
    
    # Use the EXACT same split logic as Stage 3 to prevent leakage
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    for feature_idx, feature_name in enumerate(FEATURES):
        print(f"\n{'='*50}")
        print(f" Extracting OOF Confidence Scores for: {feature_name.upper()}")
        print(f"{'='*50}")
        
        # Determine batch size (CWT is huge, needs smaller batch)
        batch_size = 8 if feature_name == 'cwt' else 32
        
        feat_path = os.path.join(FEATURES_PATH, feature_name, f"{feature_name}.npy")
        if not os.path.exists(feat_path):
             feat_path = os.path.join(FEATURES_PATH, f"{feature_name}.npy")
             
        dataset = PCGDataset(feat_path, label_path)
        
        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(dataset.X, y_data, groups)):
            current_fold = fold_idx + 1
            
            # --- CRITICAL: shuffle=False ---
            # Keeps the rows in X_meta perfectly aligned with the correct patients
            val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
            
            # Look for the user's manually selected best_model.pth
            model = AudioResNet(architecture=ARCH).to(device)
            weights_path = os.path.join(MODELS_DIR, ARCH, feature_name, f"fold_{current_fold}", "best_model.pth")
            
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Missing: {weights_path}. Did you rename your best epoch to 'best_model.pth'?")
                
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()
            
            fold_probs = []
            fold_labels = []
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Predicting Fold {current_fold}", leave=False):
                    inputs = inputs.to(device)
                    
                    # Forward pass -> Softmax -> Grab Class 1 (Abnormal) probability
                    logits = model(inputs)
                    probs = F.softmax(logits, dim=1)
                    abnormal_probs = probs[:, 1].cpu().numpy()
                    
                    fold_probs.extend(abnormal_probs)
                    fold_labels.extend(labels.numpy())
            
            # Safely place the predictions into the correct rows of our master matrix
            X_meta[val_idx, feature_idx] = fold_probs
            
            # Save the true labels during the first feature loop to guarantee alignment
            if feature_idx == 0:
                y_meta[val_idx] = fold_labels
                
            del model
            torch.cuda.empty_cache()
            
        del dataset
        gc.collect()

    # ==========================================
    # 3. SAVE THE META-DATASET FOR STAGE 5
    # ==========================================
    print(f"\n{'='*50}")
    print(" EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"Final X_meta shape: {X_meta.shape}")
    print(f"Final y_meta shape: {y_meta.shape}")
    
    x_save_path = os.path.join(DATA_PATH, 'X_meta.npy')
    y_save_path = os.path.join(DATA_PATH, 'y_meta.npy')
    
    np.save(x_save_path, X_meta)
    np.save(y_save_path, y_meta)
    
    print(f"Saved High-Level Features to: {x_save_path}")
    print(f"Saved Meta Labels to:         {y_save_path}")

if __name__ == "__main__":
    main()