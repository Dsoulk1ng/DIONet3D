import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_data import ToTensor, IRDropDatasetMultiDie
from model import IRDropNetMultiDie, Hybrid_loss_v2
from config import get_config
import os
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import torch.nn as nn




def linear_warmup(epoch, warmup_epochs, base_value, target_value):
    if epoch >= warmup_epochs:
        return target_value
    return base_value + (target_value - base_value) * (epoch / warmup_epochs)

def get_design_ids(feature_dir):
    files = os.listdir(feature_dir)
    return sorted(set(f.split('_die_')[1].replace('.npy', '') for f in files))



def train():
    
    config = get_config()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    design_ids = get_design_ids(os.path.join(config.data_dir, 'feature'))
    
    # 数据转换
    transform = ToTensor()
    kf = KFold(n_splits = config.n_splits, shuffle = True, random_state = 42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(design_ids)):

        train_losses = []
        val_losses = []

        print(f"\n--- Fold {fold + 1}/{config.n_splits} ---")

        train_id = [design_ids[i] for i in train_idx]
        val_id = [design_ids[i] for i in val_idx]

        train_loader = DataLoader(IRDropDatasetMultiDie(config.data_dir, train_id), batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(IRDropDatasetMultiDie(config.data_dir, val_id), batch_size=config.batch_size, shuffle=False)

        model = IRDropNetMultiDie().to(device)
        #model = IRDropNetMultiDie()
        #model = nn.DataParallel(model, device_ids=[0, 1])
        #model = model.cuda(device=0)

        
        criterion = Hybrid_loss_v2().to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=True)

        #optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)


        best_val_loss = float('inf')
        best_epoch = 0
        early_stop_patience = 30
        early_stop_counter = 0
        history = []                                            # 用来收集所有 epoch 的指标

        for epoch in range(config.epochs):
            
            model.train()
            train_loss = 0.0
            train_metrics = {'mse_top': 0, 'mse_bot': 0, 'edge': 0, 'ssim': 0}
            
            # Warm-up 动态调权
            #criterion.beta = linear_warmup(epoch, 20, 0.0, config.beta)
            #criterion.gamma = linear_warmup(epoch, 20, 0.0, config.gamma)
            #criterion.lambda_var = linear_warmup(epoch, 20, 0.0, config.lambda_var)

            for top_feat, bot_feat, top_label, bot_label in train_loader:
                
                top_feat = top_feat.to(device)
                bot_feat = bot_feat.to(device)
                
                top_label = top_label.to(device)
                bot_label = bot_label.to(device)

                optimizer.zero_grad()

                pred_top, pred_bot = model(top_feat, bot_feat)
                mask_density_top = top_feat[:, 3:4, :, :]  # shape: [B, 1, H, W]
                mask_density_bot = bot_feat[:, 3:4, :, :]
                
                loss, parts = criterion(pred_top, pred_bot, top_label, bot_label, mask_density_top, mask_density_bot, epoch)
                
                loss.backward()
                
                optimizer.step()

                train_loss += loss.item()
                for k in train_metrics:
                    train_metrics[k] += parts[k]
                
            model.eval()
            val_loss = 0.0
            val_metrics = {'mse_top': 0, 'mse_bot': 0, 'edge': 0, 'ssim': 0}

            with torch.no_grad():
                for top_feat, bot_feat, top_label, bot_label in val_loader:
                    top_feat = top_feat.to(device)
                    bot_feat = bot_feat.to(device)
                    
                    top_label = top_label.to(device)
                    bot_label = bot_label.to(device)

                    pred_top, pred_bot = model(top_feat, bot_feat)
                    mask_density_top = top_feat[:, 3:4, :, :]  # shape: [B, 1, H, W]
                    mask_density_bot = bot_feat[:, 3:4, :, :]
                    loss, parts = criterion(pred_top, pred_bot, top_label, bot_label, mask_density_top, mask_density_bot, epoch)

                    val_loss += loss.item()
                    for k in val_metrics:
                        val_metrics[k] += parts[k]
            
            avg_val_loss = val_loss/len(val_loader)
            scheduler.step(avg_val_loss)
            fold_results.append(avg_val_loss)    
                  
            # -------------- after one epoch -----------------
            epoch_log = {
                "epoch":        epoch + 1,
                "train_total":  train_loss / len(train_loader),
                "val_total":    val_loss   / len(val_loader),
                "train_mse_top":  train_metrics['mse_top']  / len(train_loader),
                "train_mse_bot":  train_metrics['mse_bot']  / len(train_loader),
                "train_edge":     train_metrics['edge']     / len(train_loader),
                "train_ssim":      train_metrics['ssim']/ len(train_loader),
                "val_mse_top":    val_metrics['mse_top']    / len(val_loader),
                "val_mse_bot":    val_metrics['mse_bot']    / len(val_loader),
                "val_edge":       val_metrics['edge']       / len(val_loader),
                "val_ssim":        val_metrics['ssim']  / len(val_loader),
            }
            history.append(epoch_log)               
            
            print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f} | "
                        f"Train (MSE_Top: {train_metrics['mse_top']:.4f}, MSE_Bot: {train_metrics['mse_bot']:.4f}, Edge_loss: {train_metrics['edge']:.4f}, SSIM_loss: {train_metrics['ssim']:.4f}) | "
                        f"Val (MSE_Top: {val_metrics['mse_top']:.4f}, MSE_Bot: {val_metrics['mse_bot']:.4f}, Edge_loss: {val_metrics['edge']:.4f}, SSIM_loss: {val_metrics['ssim']:.4f})")
            
            if (epoch+1) % 50 == 0:
                torch.save(model.state_dict(), f'checkpoints_v4/irdrop_fold_{fold+1}_epoch_{epoch+1}.pth')
                plt.figure(figsize=(10,6))
                plt.plot(train_losses, label='Train Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Training and Validation Loss (Fold {fold+1})')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'loss_curve/loss_curve_fold_{fold+1}_epoch_{epoch+1}.png')
                plt.close()
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'checkpoints_v4/best_model_fold_{fold+1}_epoch_{epoch+1}.pth')
                print(f"Best Model saved for fold {fold+1} at epoch {epoch+1}")

            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            if epoch > 600:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= early_stop_patience:
                    print(f"🛑 Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1} with Val Loss: {best_val_loss:.6f}")
                    break
            
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints_v4/irdrop_fold_{fold+1}.pth')
            print(f"Model saved for fold {fold+1}")

        plt.figure(figsize=(10,6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss (Fold {fold+1})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'loss_curve/loss_curve_fold_{fold+1}.png')
        plt.close()

        df = pd.DataFrame(history)
        csv_path  = f"logs/fold_{fold+1}.csv"

        df.to_csv(csv_path, index=False)

        print(f"✔️  loss 日志已保存: {csv_path}")


    print(f"All Fold Validation Results: {fold_results}")
    print(f"Mean Validation Loss: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

            

if __name__ == "__main__":
    train()
