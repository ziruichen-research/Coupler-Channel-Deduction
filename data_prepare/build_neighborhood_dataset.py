import os
import h5py
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt


def visualize_fps_result(all_points, center_indices, title="FPS Sampling Result"):
    """
    可视化原始数据分布和 FPS 采样后的中心点分布。

    Args:
        all_points (np.ndarray): 所有原始点的坐标 [N, 2] 或 [N, 3]。
        center_indices (np.ndarray): FPS 选出的中心点索引 [M]。
        title (str): 图表标题。
    """
    # 提取坐标 (只取前两维 X, Y)
    # 即使是 3D 数据，通常俯视图 (X-Y) 最能反映分布
    pts_x = all_points[:, 0]
    pts_y = all_points[:, 1]

    centers_x = all_points[center_indices, 0]
    centers_y = all_points[center_indices, 1]

    plt.figure(figsize=(10, 8), dpi=100)

    # 1. 画出所有原始点 (背景，用灰色/蓝色，高透明度，小点)
    # alpha=0.1 非常重要，可以看清密度，避免视觉遮挡
    plt.scatter(pts_x, pts_y, c='lightgray', s=1, alpha=0.5, label='Original Data')

    # 2. 画出 FPS 选中的中心点 (前景，用红色，大点)
    plt.scatter(centers_x, centers_y, c='red', s=15, marker='o', label='FPS Centers')

    # 3. 设置图表属性
    plt.title(f"{title}\n(Total: {len(all_points)}, Selected: {len(center_indices)})")
    plt.xlabel("Position X (m)")
    plt.ylabel("Position Y (m)")
    plt.legend(loc='upper right')
    plt.axis('equal')  # 关键！保证物理比例尺一致
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


def farthest_point_sampling(points, n_samples):
    """
    使用最远点采样 (FPS) 从 points 中选出 n_samples 个均匀分布的关键点索引。

    Args:
        points (np.ndarray): [N, 2] 或 [N, 3] 的坐标数组。
        n_samples (int): 需要采样的点数。

    Returns:
        centroids_indices (np.ndarray): 选出的点的索引，长度为 n_samples。
    """
    N, D = points.shape
    centroids_indices = np.zeros(n_samples, dtype=np.int64)

    # 1. 随机选择第一个点
    distance = np.full(N, np.inf)  # 初始化到已选集合的距离为无穷大
    farthest_index = np.random.randint(0, N)

    print(f"Starting FPS to select {n_samples} centers from {N} points...")
    for i in tqdm(range(n_samples), desc="FPS Sampling"):
        centroids_indices[i] = farthest_index
        centroid = points[farthest_index]

        # 2. 更新所有点到当前已选点集的最小距离
        # 计算所有点到当前新选中点的距离
        dist_to_current = np.sum((points - centroid) ** 2, axis=1)
        # 更新 distance：取 min(原有距离, 到新点的距离)
        distance = np.minimum(distance, dist_to_current)

        # 3. 选取距离最大的点作为下一个中心
        farthest_index = np.argmax(distance)

    return centroids_indices


def farthest_point_sampling_gpu(points, n_samples, device='cuda'):
    """
    使用 PyTorch 在 GPU 上加速最远点采样 (FPS)。

    Args:
        points (np.ndarray): [N, D] 的坐标数组 (Numpy)。
        n_samples (int): 需要采样的点数。
        device (str): 计算设备，默认 'cuda'。

    Returns:
        centroids_indices (np.ndarray): 选出的点的索引 (Numpy)，长度为 n_samples。
    """
    assert n_samples <= points.shape[0], "n_samples must be less than or equal to the number of points."

    # 1. 数据转移到 GPU
    # 确保输入是 float32 以提高速度，Points shape: [N, D]
    points_tensor = torch.from_numpy(points).float().to(device)
    N, D = points_tensor.shape

    # 存储结果的索引
    centroids_indices = torch.zeros(n_samples, dtype=torch.long, device=device)

    # 初始化距离为无穷大
    # distance[i] 表示第 i 个点到当前已选点集的最短距离
    distances = torch.full((N,), float('inf'), device=device)

    # 随机选择第一个点
    farthest_index = torch.randint(0, N, (1,), dtype=torch.long, device=device).item()

    # 2. 迭代选择
    # 使用 torch.arange 生成索引，用于 mask 操作（可选，这里直接用 full index 也可以）

    for i in tqdm(range(n_samples), desc="FPS Sampling"):
        # 记录当前选中的点索引
        centroids_indices[i] = farthest_index

        # 取出当前选中的点的坐标 [1, D]
        centroid = points_tensor[farthest_index].unsqueeze(0)

        # 计算所有点到当前选中点的欧氏距离平方 [N]
        # (x-y)^2 = x^2 + y^2 - 2xy，或者直接用广播 (points - centroid)^2
        # 广播减法在 N 很大时可能会占用显存，但 N=50000 显存完全够用
        dist_to_current = torch.sum((points_tensor - centroid) ** 2, dim=1)

        # 更新距离表：取 min(原有距离, 到新点的距离)
        distances = torch.minimum(distances, dist_to_current)

        # 选取距离最大的点作为下一个中心
        farthest_index = torch.argmax(distances).item()

    # 3. 转回 CPU Numpy
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
    print(f"Loading source data from {src_h5_path}...")
    with h5py.File(src_h5_path, 'r') as f_src:
        # 加载所有数据
        data_all = f_src['data'][:]
        loc_all = f_src['loc'][:]
        snr_all = f_src['snr'][:]

    total_points = loc_all.shape[0]

    # --- 第一步：选择 Center Points ---
    if mode == 'fps':
        print(f"Running FPS on {device.upper()}...")
        # 调用 GPU 加速版 FPS
        # 注意：只传入 X, Y 坐标 (loc_all[:, :2])
        center_indices = farthest_point_sampling_gpu(loc_all[:, :2], num_neighborhoods, device=device)

    else:
        print("Using Random Sampling for centers...")
        center_indices = np.random.choice(total_points, num_neighborhoods, replace=False)

    print(f"Selected {len(center_indices)} center points.")

    # =========== [新增] 可视化验证步骤 ===========
    # 为了防止阻塞流程，可以只在特定条件下显示，或者保存图片
    # print("Visualizing sampling result...")
    # visualize_fps_result(loc_all, center_indices, title=f"Sampling Mode: {mode.upper()}")

    # --- 第二步：KNN 查找邻域 (保持不变，CPU处理) ---
    print(f"Finding {k_neighbors} nearest neighbors for each center...")
    knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree')
    knn.fit(loc_all[:, :2])

    # 找到所有中心点的 K 个最近邻
    # distances: [num_neighborhoods, k]
    # neighbor_indices: [num_neighborhoods, k] (这就是我们要的索引矩阵)
    distances, neighbor_indices_matrix = knn.kneighbors(loc_all[center_indices, :2])

    # --- 第三步：构建并保存数据集 ---
    print(f"Constructing neighborhood dataset: {num_neighborhoods} samples, each size {k_neighbors}...")

    os.makedirs(os.path.dirname(dst_h5_path), exist_ok=True)

    # 获取数据形状
    # data_all shape: [N_total, 32, 32]
    # target shape: [num_neighborhoods, k_neighbors, 32, 32]
    _, n_ant, n_sub = data_all.shape
    _, n_loc_dim = loc_all.shape

    os.makedirs(os.path.dirname(dst_h5_path), exist_ok=True)

    with h5py.File(dst_h5_path, 'w') as f_dst:
        # 1. 预先创建 Dataset (不赋值，只定义形状和类型)
        # 这时硬盘上会分配空间，但内存占用极小

        # Data Dataset
        dset_data = f_dst.create_dataset(
            'data',
            shape=(num_neighborhoods, k_neighbors, n_ant, n_sub),
            dtype='complex64',
            chunks=(1, 1, n_ant, n_sub)  # 保持主要数据维度的连续性
        )

        # Loc Dataset
        dset_loc = f_dst.create_dataset(
            'loc',
            shape=(num_neighborhoods, k_neighbors, n_loc_dim),
            dtype='float32',
            chunks=(1, 1, n_loc_dim)
        )

        # SNR Dataset
        dset_snr = f_dst.create_dataset(
            'snr',
            shape=(num_neighborhoods, k_neighbors, n_ant),
            dtype='float32'
        )

        # 索引数据比较小，可以直接写入
        f_dst.create_dataset('center_indices', data=center_indices)
        f_dst.create_dataset('neighbor_indices', data=neighbor_indices_matrix)

        # 2. 分批次写入 (Batch Processing)
        # 即使 N=50000，分批次每次处理 1000 个，内存占用也就 500MB 左右
        total_batches = (num_neighborhoods + batch_size - 1) // batch_size

        for i in tqdm(range(0, num_neighborhoods, batch_size), total=total_batches, desc="Writing H5 Batches"):
            start_idx = i
            end_idx = min(i + batch_size, num_neighborhoods)

            # 获取当前批次的邻居索引 [Batch, K]
            batch_indices = neighbor_indices_matrix[start_idx:end_idx]

            # 利用 Fancy Indexing 提取数据 (此时只提取一小部分到内存)
            # data_all 在内存中，读取速度非常快
            batch_data_val = data_all[batch_indices]  # Shape: [Batch, K, 32, 32]
            batch_loc_val = loc_all[batch_indices]  # Shape: [Batch, K, 3]
            batch_snr_val = snr_all[batch_indices]  # Shape: [Batch, K, 32]

            # 写入 HDF5 对应的切片位置
            dset_data[start_idx:end_idx] = batch_data_val
            dset_loc[start_idx:end_idx] = batch_loc_val
            dset_snr[start_idx:end_idx] = batch_snr_val

    print(f"Saved neighborhood dataset to {dst_h5_path}")


if __name__ == '__main__':
    # 配置路径
    # base_dir = '../data_ssd/DICHASUS/h5_CD/ca0x_DirectSample512to32'
    base_dir = '../data_ssd/DICHASUS/h5_CD/0152-0157_DirectSample512to32_v2'

    # 处理 Train 集
    create_neighborhood_dataset(
        src_h5_path=os.path.join(base_dir, 'train.h5'),
        dst_h5_path=os.path.join(base_dir, 'train_nearest32_seq36000.h5'),
        num_neighborhoods=36000,  # 想要的样本数
        k_neighbors=32,  # 每个样本包含的点数
        mode='fps',  # 推荐使用 fps
        device='cuda:1'
    )

    # 处理 Test 集 (如果需要的话，通常测试集也可以做成 Neighborhood 形式)
    create_neighborhood_dataset(
        src_h5_path=os.path.join(base_dir, 'test.h5'),
        dst_h5_path=os.path.join(base_dir, 'test_nearest32_seq36000.h5'),
        num_neighborhoods=36000,  # 想要的样本数
        k_neighbors=32,  # 每个样本包含的点数
        mode='fps',  # 推荐使用 fps
        device='cuda:1'
    )




