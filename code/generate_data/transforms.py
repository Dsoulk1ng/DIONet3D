import torch
import numpy as np

class ToTensor:
    def __call__(self, global_feat, d3_feat):
        global_feat = torch.tensor(global_feat, dtype=torch.float32)
        d3_feat = torch.tensor(d3_feat, dtype=torch.float32)
        
        global_feat = (global_feat - global_feat.mean()) / (global_feat.std() + 1e-5)
        d3_feat = (d3_feat - d3_feat.mean()) / (d3_feat.std() + 1e-5)
        
        return global_feat, d3_feat
