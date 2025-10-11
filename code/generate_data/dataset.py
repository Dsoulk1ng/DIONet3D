# data/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from .transforms import ToTensor

class IRDropDatasetMultiDie(Dataset):
    def __init__(self, root_dir, design_ids):
        
        self.root_dir = root_dir
        self.design_ids = design_ids
        self.feature_dir = os.path.join(root_dir, 'feature')
        self.label_dir = os.path.join(root_dir, 'label')

        
    def __len__(self):
        return len(self.design_ids)
    
    
    def __getitem__(self, idx):

        design_id = self.design_ids[idx]

        top_feat = np.load(os.path.join(self.feature_dir, f'top_die_{design_id}.npy'))
        bottom_feat = np.load(os.path.join(self.feature_dir, f'bottom_die_{design_id}.npy'))
        top_label = np.load(os.path.join(self.label_dir, f'top_die_{design_id}.npy'))
        top_label = np.expand_dims(top_label, axis=0)
        bottom_label = np.load(os.path.join(self.label_dir, f'bottom_die_{design_id}.npy'))
        bottom_label = np.expand_dims(bottom_label, axis=0)
        
        top_feat = torch.tensor(top_feat, dtype=torch.float32)
        bottom_feat = torch.tensor(bottom_feat, dtype=torch.float32)
        top_label = torch.tensor(top_label, dtype=torch.float32)
        bottom_label = torch.tensor(bottom_label, dtype=torch.float32)

        return top_feat, bottom_feat, top_label, bottom_label
