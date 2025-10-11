import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_data import ToTensor, IRDropDatasetMultiDie
from model import IRDropNetMultiDie
from config import get_config
from tqdm import tqdm
import random
import os
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pytorch_msssim import ssim


def compute_advanced_metrics(pred: torch.Tensor, label: torch.Tensor):
    pred_np = pred.squeeze().cpu().numpy()
    label_np = label.squeeze().cpu().numpy()

    pred_flat = pred_np.flatten()
    label_flat = label_np.flatten()

    mae = mean_absolute_error(label_flat, pred_flat)
    nmae = mae / (np.mean(label_flat) + 1e-8)
    r2 = r2_score(label_flat, pred_flat)

    mse = np.mean((label_flat - pred_flat) ** 2)
    max_val = np.max(label_np)
    psnr = 20 * np.log10(max_val + 1e-8) - 10 * np.log10(mse + 1e-8)

    pred_ssim = torch.tensor(pred_np).unsqueeze(0).unsqueeze(0).clamp(0, 1)
    label_ssim = torch.tensor(label_np).unsqueeze(0).unsqueeze(0).clamp(0, 1)
    ssim_score = ssim(pred_ssim, label_ssim, data_range=1.0, size_average=True).item()

    pear, _ = pearsonr(label_flat, pred_flat)
    spea, _ = spearmanr(label_flat, pred_flat)
    kend, _ = kendalltau(label_flat, pred_flat)
    cc = np.corrcoef(label_flat.ravel(), pred_flat.ravel())[0, 1]
    return {
        "NMAE": nmae,
        "R2": r2,
        "PSNR": psnr,
        "SSIM": ssim_score,
        "Pearson": pear,
        "Spearman": spea,
        "Kendall": kend,
        "CC": cc
    }

def visualize_pred(pred, label, title, save_dir):
    vmin=0.7
    vmax=0.9
    levels=10

    boundaries = np.linspace(vmin, vmax, levels)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=256)
    
    
    base_cmap = plt.get_cmap('jet')
    new_colors = base_cmap(np.linspace(0,1,256))
    new_colors[0] = np.array([0, 0, 0, 1])
    black_cmap = mcolors.ListedColormap(new_colors)

    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    B = pred.shape[0]
    os.makedirs(save_dir, exist_ok=True)
    for i in range(B):
        
        pred_i = np.array(pred[i, 0])
        label_i = np.array(label[i, 0])

        epsilon = 1e-4
        pred_i[pred_i < 0.5] = vmin - epsilon
        label_i[label_i < 0.5] = vmin - epsilon


        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        img1 = plt.imshow(pred_i, cmap=black_cmap, norm=norm)
        cbar = plt.colorbar(img1, ticks=boundaries)
        cbar.ax.set_yticklabels([f'{v:.3f}' for v in boundaries])
        plt.title(f"{title} - Prediction")
        plt.axis('off')

        plt.subplot(1,2,2)
        img2 = plt.imshow(label_i, cmap=black_cmap, norm=norm)
        cbar = plt.colorbar(img2, ticks=boundaries)
        cbar.ax.set_yticklabels([f'{v:.3f}' for v in boundaries])
        plt.title(f"{title} - Ground Truth")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{title}_sample_{i}.png"))
        plt.close()


def test():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ids = sorted(os.listdir(os.path.join(config.test_dir, 'feature')))
    test_ids = [f.split('_die_')[1].replace('.npy', '') for f in test_ids]
    test_ids = list(set(test_ids))

    test_loader = DataLoader(IRDropDatasetMultiDie(config.test_dir, test_ids), batch_size=1, shuffle=False)

    model = IRDropNetMultiDie().to(device)
    state_dict = torch.load('checkpoints_v3/best_model_all.pth', map_location=device)
        
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    all_metrics_top, all_metrics_bot = [], []

    with torch.no_grad():
        for idx, (top_feat, bot_feat, top_label, bot_label) in enumerate(tqdm(test_loader, desc="Testing")):
            top_feat, bot_feat, top_label, bot_label = [x.to(device) for x in [top_feat, bot_feat, top_label, bot_label]]
            
            pred_top, pred_bot = model(top_feat, bot_feat)
            
            metrics_top = compute_advanced_metrics(pred_top, top_label)
            metrics_bot = compute_advanced_metrics(pred_bot, bot_label)
            all_metrics_top.append(metrics_top)
            all_metrics_bot.append(metrics_bot)

            visualize_pred(pred_top, top_label, title=f"Top_Die_{idx}", save_dir="pred_results/top_die")
            visualize_pred(pred_bot, bot_label, title=f"Bottom_Die_{idx}", save_dir="pred_results/bottom_die")

    def aggregate(metrics_list):
        return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}

    avg_top = aggregate(all_metrics_top)
    avg_bot = aggregate(all_metrics_bot)

    print("\n=== Top Die Metrics ===")
    for k, v in avg_top.items():
        print(f"{k}: {v:.5f}")

    print("\n=== Bottom Die Metrics ===")
    for k, v in avg_bot.items():
        print(f"{k}: {v:.5f}")

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")



if __name__ == "__main__":
    test()
