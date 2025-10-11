import os
import numpy as np
import matplotlib.pyplot as plt

def process_npy_file(file_path, output_dir, prefix=''):
    data = np.load(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    print(f"Processing {base_name}: shape = {data.shape}")

    os.makedirs(output_dir, exist_ok=True)

    if data.ndim == 2:
        save_txt_path = os.path.join(output_dir, f"{base_name}.txt")
        save_img_path = os.path.join(output_dir, f"{base_name}.png")
        np.savetxt(save_txt_path, data, fmt="%.4f")
        plt.imshow(data, cmap='jet', interpolation='nearest', vmin=0.7, vmax=0.9)
        plt.colorbar()
        plt.title(base_name)
        plt.savefig(save_img_path)
        plt.close()

    elif data.ndim == 3:
        C, H, W = data.shape
        for c in range(C):
            channel_data = data[c]
            save_txt_path = os.path.join(output_dir, f"{prefix}{base_name}_ch{c}.txt")
            save_img_path = os.path.join(output_dir, f"{prefix}{base_name}_ch{c}.png")
            np.savetxt(save_txt_path, channel_data, fmt="%.4f")
            plt.imshow(channel_data, cmap='jet', interpolation='nearest', vmin=0, vmax=1.0)
            plt.colorbar()
            plt.savefig(save_img_path)
            plt.close()
    else:
        print(f"Unsupported data shape: {data.shape}")

def process_all(root_dir, subfolder, prefix=''):
    npy_dir = os.path.join(root_dir, subfolder)
    output_dir = os.path.join(root_dir, f"{subfolder}_vis")

    for file in os.listdir(npy_dir):
        if file.endswith(".npy"):
            npy_path = os.path.join(npy_dir, file)
            process_npy_file(npy_path, output_dir, prefix)

if __name__ == '__main__':
    ROOT = './processed/training_set'  
    process_all(ROOT, 'feature', prefix='feat_')
    #process_all(ROOT, 'label', prefix='label_')
