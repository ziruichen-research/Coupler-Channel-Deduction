import os
from typing import Union, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DeepmimoO1Dataset(Dataset):
    """DeepMIMO dataset backed by per-sample `.npy` files."""

    def __init__(self, root_dir, mode='train', max_samples=None):
        """
        Args:
            root_dir: Dataset root directory.
            mode: Subdirectory name (e.g., 'train', 'specifictest_mobile').
            max_samples: Optional cap on loaded sample count.
        """
        self.path = os.path.join(root_dir, mode)
        len_path = os.path.join(self.path, 'data_len.npy')

        # Get physical sample count from metadata file or by scanning files.
        if os.path.exists(len_path):
            real_len = int(np.load(len_path))
        else:
            files = [f for f in os.listdir(self.path) if f.startswith('data_') and f.endswith('.npy')]
            real_len = len(files)
            if real_len == 0:
                print(f"Warning: No data found in {self.path}")

        self.len = min(real_len, max_samples) if max_samples is not None else real_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        file_path = os.path.join(self.path, f'data_{index}.npy')
        # 数据通常是 [32, 32, 32, 2] (batch/seq, ant, car, complex)
        data = np.load(file_path)
        return torch.from_numpy(data).float()


class DICHASUSDataset(Dataset):
    """DICHASUS HDF5 dataset with optional eager or lazy loading."""

    def __init__(self, root_dir, dataset_name='train_nearest32_seq5000.h5', max_samples=None, load_to_memory=True):
        """
        Args:
            root_dir: Directory that contains the HDF5 file.
            dataset_name: HDF5 file name.
            max_samples: Optional cap on sample count.
            load_to_memory: If True, preload arrays into RAM.
        """
        self.file_path = os.path.join(root_dir, dataset_name)
        self.load_to_memory = load_to_memory

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")

        with h5py.File(self.file_path, 'r') as f:
            total_samples = f['data'].shape[0]

        if max_samples is not None and max_samples < total_samples:
            self.num_samples = max_samples
            print(f"[Dataset] Truncating dataset from {total_samples} to {self.num_samples}.")
        else:
            self.num_samples = total_samples

        self.data = None
        self.loc = None
        self.snr = None
        # File handle is opened lazily per worker in __getitem__.
        self.h5_file = None

        if self.load_to_memory:
            print(f"[Dataset] Loading {self.num_samples} samples into memory...")
            with h5py.File(self.file_path, 'r') as f:
                self.data = f['data'][:self.num_samples]
                self.loc = f['loc'][:self.num_samples]
                self.snr = f['snr'][:self.num_samples]
            print("[Dataset] Loaded successfully.")
        else:
            print("[Dataset] Lazy loading mode enabled.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                csi: [K, Ant, Car, 2]
                loc: [K, 3]
                snr: [K, Ant]
        """
        if self.load_to_memory:
            csi_data = self.data[idx]
            loc_data = self.loc[idx]
            snr_data = self.snr[idx]
        else:
            # Avoid sharing an opened HDF5 handle across DataLoader workers.
            if self.h5_file is None:
                self.h5_file = h5py.File(self.file_path, 'r')
            csi_data = self.h5_file['data'][idx]
            loc_data = self.h5_file['loc'][idx]
            snr_data = self.h5_file['snr'][idx]

        csi_tensor = torch.from_numpy(csi_data)
        csi_tensor = torch.view_as_real(csi_tensor).float()
        loc_tensor = torch.from_numpy(loc_data)
        snr_tensor = torch.from_numpy(snr_data)

        # Shuffle along sequence dimension K with a shared permutation.
        K = csi_tensor.shape[0]
        shuffled_indices = torch.randperm(K)
        csi_tensor = csi_tensor[shuffled_indices]
        loc_tensor = loc_tensor[shuffled_indices]
        snr_tensor = snr_tensor[shuffled_indices]

        return {
            'csi': csi_tensor,
            'loc': loc_tensor,
            'snr': snr_tensor,
        }

    def close(self):
        """Close HDF5 handle in lazy-loading mode."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
