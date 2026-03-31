import os
from typing import Union, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DeepmimoO1Dataset(Dataset):
    def __init__(self, root_dir, mode='train', max_samples=None):  # 1. 增加 max_samples 参数
        """
        max_samples: (int, optional) 限制加载的最大样本数。如果不填，则加载所有。
        """
        self.path = os.path.join(root_dir, mode)
        len_path = os.path.join(self.path, 'data_len.npy')

        # --- 第一步：获取物理文件的真实总数 ---
        if os.path.exists(len_path):
            real_len = int(np.load(len_path))
        else:
            files = [f for f in os.listdir(self.path) if f.startswith('data_') and f.endswith('.npy')]
            real_len = len(files)
            if real_len == 0:
                print(f"Warning: No data found in {self.path}")

        # --- 第二步：根据 max_samples 截断长度 ---
        if max_samples is not None:
            # 这里的 min 是为了防止你设置的 1000 超过了实际只有 500 个文件的情况
            self.len = min(real_len, max_samples)
        else:
            self.len = real_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # 加载 data_i.npy
        file_path = os.path.join(self.path, f'data_{index}.npy')
        # 数据通常是 [32, 32, 32, 2] (batch/seq, ant, car, complex)
        data = np.load(file_path)
        return torch.from_numpy(data).float()


class DICHASUSDataset(Dataset):
    def __init__(self, root_dir, dataset_name='train_nearest32_seq5000.h5', max_samples=None, load_to_memory=True):
        """
        DICHASUS 邻域数据集加载类。

        Args:
            root_dir (str): 数据集所在的文件夹路径。
            dataset_name (str): H5 文件名。
            max_samples (int, optional): 限制加载的最大样本数。
                                         由于 FPS 的特性，取前 N 个样本依然保持空间均匀分布。
            load_to_memory (bool): 是否将所有数据加载到内存。
                                   True: 初始化慢，训练快 (推荐内存足够时使用)。
                                   False: 初始化快，训练时有磁盘 IO 开销。
        """
        self.file_path = os.path.join(root_dir, dataset_name)
        self.load_to_memory = load_to_memory

        # 检查文件是否存在
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")

        # 1. 获取数据集的总长度 (不读取具体数据)
        with h5py.File(self.file_path, 'r') as f:
            total_samples = f['data'].shape[0]

        # 2. 确定有效样本数
        if max_samples is not None and max_samples < total_samples:
            self.num_samples = max_samples
            print(f"[Dataset] Truncating dataset from {total_samples} to {self.num_samples} (FPS compatible).")
        else:
            self.num_samples = total_samples

        # 3. 数据加载逻辑
        self.data = None
        self.loc = None
        self.snr = None
        self.h5_file = None  # 用于 load_to_memory=False 时的懒加载

        if self.load_to_memory:
            print(f"[Dataset] Loading {self.num_samples} samples into memory...")
            with h5py.File(self.file_path, 'r') as f:
                # 直接切片读取，利用 HDF5 的切片特性
                # 读取为 Numpy 数组，后续转 Tensor
                self.data = f['data'][:self.num_samples]  # [N, K, 32, 32] Complex
                self.loc = f['loc'][:self.num_samples]  # [N, K, 3]
                self.snr = f['snr'][:self.num_samples]  # [N, K, 32]
            print("[Dataset] Loaded successfully.")
        else:
            print(f"[Dataset] Lazy loading mode enabled. Data will be read from disk on-the-fly.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        返回:
            sample (dict): {
                'csi': torch.complex64, [K, 32(ant), 32(car)]
                'loc': torch.float32,   [K, 3]
                'snr': torch.float32    [K, 32(ant)]
            }
        """
        if self.load_to_memory:
            # --- 内存模式 ---
            # 直接从内存数组获取
            csi_data = self.data[idx]
            loc_data = self.loc[idx]
            snr_data = self.snr[idx]
        else:
            # --- 硬盘模式 (Lazy Loading) ---
            # 注意：不能在 __init__ 打开文件并一直这就着，因为 DataLoader 多进程会 crash
            # 必须在每个 worker 内部打开文件
            if self.h5_file is None:
                self.h5_file = h5py.File(self.file_path, 'r')

            csi_data = self.h5_file['data'][idx]
            loc_data = self.h5_file['loc'][idx]
            snr_data = self.h5_file['snr'][idx]

        # --- 转换为 PyTorch Tensor ---
        csi_tensor = torch.from_numpy(csi_data)
        csi_tensor = torch.view_as_real(csi_tensor).float()
        loc_tensor = torch.from_numpy(loc_data)
        snr_tensor = torch.from_numpy(snr_data)

        #TODO: 序列数据集是中心点在最前面，后续的点逐渐变远

        # --- 在 K 维度进行一致性打乱 ---
        # 1. 获取序列长度 K
        K = csi_tensor.shape[0]
        # 2. 生成 0 到 K-1 的随机排列索引
        shuffled_indices = torch.randperm(K)
        # 3. 使用相同的随机索引对三个 Tensor 的第 0 维进行切片
        csi_tensor = csi_tensor[shuffled_indices]
        loc_tensor = loc_tensor[shuffled_indices]
        snr_tensor = snr_tensor[shuffled_indices]

        return {
            'csi': csi_tensor,  # [K, 32, 32, 2]
            'loc': loc_tensor,  # [K, 3]
            'snr': snr_tensor   # [K, 32]
        }
        # return csi_tensor

    def close(self):
        """关闭 HDF5 文件句柄 (如果是 lazy loading 模式)"""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None




