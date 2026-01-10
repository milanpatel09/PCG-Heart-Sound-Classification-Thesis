import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split  # <--- CHANGED: Random Split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from tqdm import tqdm

from src.dataset import PCGDataset
from src.models import AudioResNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CONFIGURATION
DATA_PATH = 'data/processed'
FEATURES_PATH = 'data/features'
MODELS_DIR = 'models_checkpoints_leaky' # <--- Changed folder name to avoid mixing
GROUPS_PATH = os.path.join(DATA_PATH, 'groups.npy') 

import sys
# Auto-detect CWT to adjust batch size
is_cwt = '--feature' in sys.argv and 'cwt' in sys.argv
BATCH_SIZE = 8 if is_cwt else 32
EPOCHS = 25 
LEARNING_RATE = 1e-4

def get_class_weights(y_data):
    class_counts = np.bincount(y_data)
    total = len(y_data)
    weights = total / (2 * class_counts)
    return torch.tensor(weights, dtype=torch.float32)

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    sens = recall_score(all_labels, all_preds, pos_label=1)
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    mean_sens_spec = (sens + spec) / 2
    return running_loss/len(loader), acc, sens, spec, f1, mean_sens_spec

def main():
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, required=True)
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'])
    args = parser.parse_args()
    
    feature_name = args.feature.upper()
    SAVE_DIR = os.path.join(MODELS_DIR, args.arch, args.feature)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"\n=== Training {args.arch.upper()} on {feature_name} (LEAKY SPLIT) ===")
    print(f"Saving checkpoints to: {SAVE_DIR}")
    print("WARNING: This run allows Data Leakage (Patient mixing).")
    
    feat_path = os.path.join(FEATURES_PATH, args.feature, f"{args.feature}.npy")
    label_path = os.path.join(DATA_PATH, 'y_data.npy')
    
    full_dataset = PCGDataset(feat_path, label_path)
    # Groups are loaded just to prove leakage exists later
    groups = np.load(GROUPS_PATH)
    
    # --- CHANGED: Leaky Random Stratified Split ---
    # This ignores groups and splits randomly across all segments.
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=full_dataset.y, # Maintains 3:1 ratio
        shuffle=True, 
        random_state=42
    )
    
    print(f"Train Samples: {len(train_idx)} | Val Samples: {len(val_idx)}")
    
    # --- LEAKAGE CHECK (Will Fail on purpose) ---
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    overlap = len(train_groups.intersection(val_groups))
    
    if overlap > 0:
        print(f"⚠️  LEAKAGE CONFIRMED: {overlap} patients are in BOTH Train and Val.")
        print("    (This allows the model to 'cheat' by recognizing background noise.)")
    else:
        print("✅ No Leakage (This is unlikely with random split).")
        
    # Check Class Balance
    y_train = full_dataset.y[train_idx]
    y_val = full_dataset.y[val_idx]
    print(f"Train Abn Ratio: {y_train.sum()/len(y_train):.2%}")
    print(f"Val Abn Ratio:   {y_val.sum()/len(y_val):.2%}")

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioResNet(architecture=args.arch).to(device)
    
    weights = get_class_weights(full_dataset.y).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"{'Epoch':<5} | {'Loss':<8} | {'V.Acc':<7} | {'Sens':<7} | {'Spec':<7} | {'F1':<7} | {'M.S&S':<7}")
    print("-" * 75)

    for epoch in range(EPOCHS):
        model.train()
        train_running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
        
        avg_train_loss = train_running_loss / len(train_loader)
        v_loss, acc, sens, spec, f1, mss = evaluate(model, val_loader, criterion, device)
        print(f"{epoch+1:<5} | {avg_train_loss:<8.4f} | {acc:<7.4f} | {sens:<7.4f} | {spec:<7.4f} | {f1:<7.4f} | {mss:<7.4f}")
        
        # Save per epoch
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

    print(f"\nTraining Complete. Saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()