import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score

# Import your custom modules
from src.dataset import PCGDataset
from src.models import AudioResNet

# --- CONFIGURATION ---
DATA_PATH = 'data/processed'
FEATURES_PATH = 'data/features'
MODELS_DIR = 'models_checkpoints'
GROUPS_PATH = os.path.join(DATA_PATH, 'groups.npy')
LABEL_PATH = os.path.join(DATA_PATH, 'y_data.npy')

ARCHITECTURES = ['resnet18']
FEATURES = ['spectrogram', 'melspec', 'mfcc', 'cwt', 'scattering']  
FOLDS = [1, 2, 3, 4, 5]
BATCH_SIZE = 32

def evaluate(model, loader, device):
    """Evaluation logic matching your Stage 3 script."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    sens = recall_score(all_labels, all_preds, pos_label=1)
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    mss = (sens + spec) / 2
    
    return acc, sens, spec, f1, mss

def get_val_loader(feature_name, fold_num, batch_size):
    """Reconstructs the EXACT validation set for a specific fold."""
    feat_path = os.path.join(FEATURES_PATH, feature_name, f"{feature_name}.npy")
    if not os.path.exists(feat_path):
        return None
        
    full_dataset = PCGDataset(feat_path, LABEL_PATH)
    groups = np.load(GROUPS_PATH)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    val_idx = None
    current_fold = 1
    for _, v_idx in sgkf.split(full_dataset.X, full_dataset.y, groups=groups):
        if current_fold == fold_num:
            val_idx = v_idx
            break
        current_fold += 1
        
    return DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}\n")
    
    for arch in ARCHITECTURES:
        for feature in FEATURES:
            print(f"--- Evaluating {arch.upper()} with {feature.upper()} ---")
            
            fold_metrics = {'acc': [], 'sens': [], 'spec': [], 'f1': [], 'mss': []}
            valid_folds = 0
            
            # Adjust batch size for CWT to prevent CUDA Out of Memory errors
            current_batch_size = 8 if 'cwt' in feature.lower() else BATCH_SIZE

            for fold in FOLDS:
                fold_dir = os.path.join(MODELS_DIR, arch, feature, f"fold_{fold}")
                ckpt_path = os.path.join(fold_dir, "best_model.pth")
                
                if not os.path.exists(ckpt_path):
                    continue
                    
                val_loader = get_val_loader(feature, fold, current_batch_size)
                if val_loader is None:
                    continue
                
                valid_folds += 1
                model = AudioResNet(architecture=arch).to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                
                acc, sens, spec, f1, mss = evaluate(model, val_loader, device)
                
                print(f"  Fold {fold} -> MSS: {mss:.4f} | Acc: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f} | F1: {f1:.4f}")
                
                fold_metrics['acc'].append(acc)
                fold_metrics['sens'].append(sens)
                fold_metrics['spec'].append(spec)
                fold_metrics['f1'].append(f1)
                fold_metrics['mss'].append(mss)
            
            if valid_folds > 0:
                print(f"\n  Final Results for {arch.upper()} + {feature.upper()} ({valid_folds}/5 Folds):")
                print(f"  Accuracy : {np.mean(fold_metrics['acc']):.4f} ± {np.std(fold_metrics['acc']):.4f}")
                print(f"  Sens     : {np.mean(fold_metrics['sens']):.4f} ± {np.std(fold_metrics['sens']):.4f}")
                print(f"  Spec     : {np.mean(fold_metrics['spec']):.4f} ± {np.std(fold_metrics['spec']):.4f}")
                print(f"  F1-Score : {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
                print(f"  MSS      : {np.mean(fold_metrics['mss']):.4f} ± {np.std(fold_metrics['mss']):.4f}\n")
                print("-" * 60)
            else:
                print(f"  [!] No best_model.pth found for {arch.upper()} + {feature.upper()}.\n")

if __name__ == "__main__":
    main()