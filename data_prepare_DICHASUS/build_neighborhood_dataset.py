"""Build neighborhood-style HDF5 datasets from point-wise channel data.

This script selects center points (FPS or random), finds k-nearest neighbors
for each center, and writes neighborhood tensors in batched HDF5 format.
"""

import os
import h5py
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt


def visualize_fps_result(all_points, center_indices, title="FPS Sampling Result"):
    """Visualize all points and selected FPS centers on the XY plane.

    Args:
        all_points (np.ndarray): Point coordinates with shape [N, 2] or [N, 3].
        center_indices (np.ndarray): Indices of selected FPS centers, shape [M].
        title (str): Figure title.

    Returns:
        None
    """
    # Use XY projection for a top-down coverage view.
    pts_x = all_points[:, 0]
    pts_y = all_points[:, 1]

    centers_x = all_points[center_indices, 0]
    centers_y = all_points[center_indices, 1]

    plt.figure(figsize=(10, 8), dpi=100)

    # Background points for density perception.
    plt.scatter(pts_x, pts_y, c='lightgray', s=1, alpha=0.5, label='Original Data')

    # Foreground centers selected by FPS.
    plt.scatter(centers_x, centers_y, c='red', s=15, marker='o', label='FPS Centers')

    plt.title(f"{title}\n(Total: {len(all_points)}, Selected: {len(center_indices)})")
    plt.xlabel("Position X (m)")
    plt.ylabel("Position Y (m)")
    plt.legend(loc='upper right')
    plt.axis('equal')  # Keep physical scale unchanged.
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


def farthest_point_sampling(points, n_samples):
    """CPU FPS: select spatially spread center indices from point coordinates.

    Args:
        points (np.ndarray): Input coordinates with shape [N, D].
        n_samples (int): Number of centers to sample.

    Returns:
        np.ndarray: Sampled center indices with shape [n_samples].
    """
    N, D = points.shape
    centroids_indices = np.zeros(n_samples, dtype=np.int64)

    # Track each point's shortest distance to the selected center set.
    distance = np.full(N, np.inf)
    farthest_index = np.random.randint(0, N)

    print(f"Starting FPS to select {n_samples} centers from {N} points...")
    for i in tqdm(range(n_samples), desc="FPS Sampling"):
        centroids_indices[i] = farthest_index
        centroid = points[farthest_index]

        dist_to_current = np.sum((points - centroid) ** 2, axis=1)
        distance = np.minimum(distance, dist_to_current)

        # Next center is the point farthest from current center set.
        farthest_index = np.argmax(distance)

    return centroids_indices


def farthest_point_sampling_gpu(points, n_samples, device='cuda'):
    """GPU FPS implemented with PyTorch for large datasets.

    Args:
        points (np.ndarray): Input coordinates with shape [N, D].
        n_samples (int): Number of centers to sample.
        device (str): Torch device string, for example `'cuda'` or `'cuda:1'`.

    Returns:
        np.ndarray: Sampled center indices with shape [n_samples].

    """
    assert n_samples <= points.shape[0], "n_samples must be less than or equal to the number of points."

    # Move coordinates to GPU (float32 for speed).
    points_tensor = torch.from_numpy(points).float().to(device)
    N, D = points_tensor.shape

    centroids_indices = torch.zeros(n_samples, dtype=torch.long, device=device)

    # distances[i] = shortest squared distance to selected centers.
    distances = torch.full((N,), float('inf'), device=device)

    farthest_index = torch.randint(0, N, (1,), dtype=torch.long, device=device).item()

    for i in tqdm(range(n_samples), desc="FPS Sampling"):
        centroids_indices[i] = farthest_index
        centroid = points_tensor[farthest_index].unsqueeze(0)

        # Squared Euclidean distance to the newly selected center.
        dist_to_current = torch.sum((points_tensor - centroid) ** 2, dim=1)
        distances = torch.minimum(distances, dist_to_current)
        farthest_index = torch.argmax(distances).item()

    # Return CPU numpy indices for downstream HDF5 indexing.
    return centroids_indices.cpu().numpy()


def create_neighborhood_dataset(
        src_h5_path,
        dst_h5_path,
        num_neighborhoods=5000,
        k_neighbors=32,
        mode='fps',
        device='cuda',
        batch_size=1000
):
    """
    Create a neighborhood dataset where each sample is a KNN local patch.
    """
    print(f"Loading source data from {src_h5_path}...")
    with h5py.File(src_h5_path, 'r') as f_src:
        data_all = f_src['data'][:]
        loc_all = f_src['loc'][:]
        snr_all = f_src['snr'][:]

    total_points = loc_all.shape[0]

    # Step 1: select center points.
    if mode == 'fps':
        print(f"Running FPS on {device.upper()}...")
        # FPS is run on XY coordinates to preserve planar coverage.
        center_indices = farthest_point_sampling_gpu(loc_all[:, :2], num_neighborhoods, device=device)

    else:
        print("Using Random Sampling for centers...")
        center_indices = np.random.choice(total_points, num_neighborhoods, replace=False)

    print(f"Selected {len(center_indices)} center points.")

    # Optional sanity-check visualization.
    # print("Visualizing sampling result...")
    # visualize_fps_result(loc_all, center_indices, title=f"Sampling Mode: {mode.upper()}")

    # Step 2: find KNN neighborhoods on CPU.
    print(f"Finding {k_neighbors} nearest neighbors for each center...")
    knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree')
    knn.fit(loc_all[:, :2])

    distances, neighbor_indices_matrix = knn.kneighbors(loc_all[center_indices, :2])

    # Step 3: materialize and save the neighborhood dataset.
    print(f"Constructing neighborhood dataset: {num_neighborhoods} samples, each size {k_neighbors}...")

    os.makedirs(os.path.dirname(dst_h5_path), exist_ok=True)

    _, n_ant, n_sub = data_all.shape
    _, n_loc_dim = loc_all.shape

    os.makedirs(os.path.dirname(dst_h5_path), exist_ok=True)

    with h5py.File(dst_h5_path, 'w') as f_dst:
        # Preallocate destination datasets to reduce peak memory usage.
        dset_data = f_dst.create_dataset(
            'data',
            shape=(num_neighborhoods, k_neighbors, n_ant, n_sub),
            dtype='complex64',
            chunks=(1, 1, n_ant, n_sub)
        )

        dset_loc = f_dst.create_dataset(
            'loc',
            shape=(num_neighborhoods, k_neighbors, n_loc_dim),
            dtype='float32',
            chunks=(1, 1, n_loc_dim)
        )

        dset_snr = f_dst.create_dataset(
            'snr',
            shape=(num_neighborhoods, k_neighbors, n_ant),
            dtype='float32'
        )

        # Index datasets are small and can be written directly.
        f_dst.create_dataset('center_indices', data=center_indices)
        f_dst.create_dataset('neighbor_indices', data=neighbor_indices_matrix)

        # Batch write to avoid loading all neighborhoods at once.
        total_batches = (num_neighborhoods + batch_size - 1) // batch_size

        for i in tqdm(range(0, num_neighborhoods, batch_size), total=total_batches, desc="Writing H5 Batches"):
            start_idx = i
            end_idx = min(i + batch_size, num_neighborhoods)

            batch_indices = neighbor_indices_matrix[start_idx:end_idx]

            batch_data_val = data_all[batch_indices]
            batch_loc_val = loc_all[batch_indices]
            batch_snr_val = snr_all[batch_indices]

            dset_data[start_idx:end_idx] = batch_data_val
            dset_loc[start_idx:end_idx] = batch_loc_val
            dset_snr[start_idx:end_idx] = batch_snr_val

    print(f"Saved neighborhood dataset to {dst_h5_path}")


if __name__ == '__main__':
    # Example paths.
    base_dir = '../data/DICHASUS/h5_CD/0152-0157_DirectSample512to32'

    # Build train neighborhoods.
    create_neighborhood_dataset(
        src_h5_path=os.path.join(base_dir, 'train.h5'),
        dst_h5_path=os.path.join(base_dir, 'train_nearest32_seq36000.h5'),
        num_neighborhoods=36000,  # 想要的样本数
        k_neighbors=32,  # 每个样本包含的点数
        mode='fps',  # 推荐使用 fps
        device='cuda:0'
    )

    # Build test neighborhoods.
    create_neighborhood_dataset(
        src_h5_path=os.path.join(base_dir, 'test.h5'),
        dst_h5_path=os.path.join(base_dir, 'test_nearest32_seq36000.h5'),
        num_neighborhoods=36000,  # 想要的样本数
        k_neighbors=32,  # 每个样本包含的点数
        mode='fps',  # 推荐使用 fps
        device='cuda:0'
    )




