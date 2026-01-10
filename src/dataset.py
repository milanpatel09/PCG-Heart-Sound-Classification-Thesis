import torch
from torch.utils.data import Dataset 
import numpy as np

class PCGDataset(Dataset):
    def __init__(self, feature_path, label_path):
        """
        Loads features and labels.
        Applies Z-Score Normalization (Standardization) per sample.
        """
        self.y = np.load(label_path)
        # mmap_mode='r' keeps memory usage low for large files (CWT)
        self.X = np.load(feature_path, mmap_mode='r')
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # 1. Load data
        data = np.array(self.X[idx]) 
        label = self.y[idx]
        
        # 2. Z-Score Normalization (Standardization)
        # Formula: (x - mean) / (std + epsilon)
        mean = data.mean()
        std = data.std()
        
        # 1e-6 prevents division by zero if a file is purely silent
        data_norm = (data - mean) / (std + 1e-6)
            
        # 3. Add Channel dimension: (1, H, W)
        data_tensor = torch.tensor(data_norm).float().unsqueeze(0)
        
        return data_tensor, torch.tensor(label).long()