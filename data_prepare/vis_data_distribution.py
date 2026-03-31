import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import h5py

plot_config = {
    "font.family": 'Times New Roman',
    "font.size": 14
}
rcParams.update(plot_config)


def visualize_spatial_snr(loc, snr, sample_ratio=1.0, vmin=None, vmax=None,
                          is_save=True, save_path=None, fig_name=None):
    """
    可视化数据的空间分布 (XY) 和平均 SNR (Color)。
    该函数只负责绘图逻辑，不负责文件读取。

    Args:
        loc (np.ndarray): 位置数据，形状为 [N, 2] 或 [N, 3]。
        snr (np.ndarray): 信噪比数据，形状可以是 [N, 32] (自动计算平均值) 或 [N,] (直接使用)。
        sample_ratio (float): 采样比例 (0.0 - 1.0)，例如 0.1 表示随机抽取 10% 的点。
        title (str, optional): 图表自定义标题。
    """
    # 1. 基础检查
    num_samples = loc.shape[0]
    if snr.shape[0] != num_samples:
        raise ValueError(f"Loc shape {loc.shape} and SNR shape {snr.shape} mismatch in length.")

    print(f"Visualizing data with {num_samples} samples available.")

    # 2. 处理 SNR 数据
    # 如果是多天线 [N, 32]，计算平均值；如果是 [N,] 则直接使用
    if snr.ndim == 2:
        print("Detected multi-antenna SNR, calculating mean across antennas...")
        avg_snr = np.mean(snr, axis=1)
    else:
        avg_snr = snr

    # 3. 降采样 (Downsampling)
    if sample_ratio < 1.0:
        sample_size = int(num_samples * sample_ratio)
        print(f"Downsampling to {sample_size} samples ({sample_ratio * 100:.1f}%)...")
        # 生成随机索引
        indices = np.random.choice(num_samples, sample_size, replace=False)
        loc_sampled = loc[indices]
        snr_sampled = avg_snr[indices]
    else:
        loc_sampled = loc
        snr_sampled = avg_snr

    # 4. 绘图
    plt.figure(figsize=(8, 6))

    # Scatter plot
    # x: loc[:, 0], y: loc[:, 1]
    # c: snr_sampled (颜色映射)
    sc = plt.scatter(loc_sampled[:, 0], loc_sampled[:, 1],
                     c=snr_sampled, cmap='viridis',
                     s=6, alpha=0.7, vmin=vmin, vmax=vmax)

    # 添加颜色条
    cbar = plt.colorbar(sc)
    cbar.set_label('Average SNR (dB)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # 装饰
    # plot_title = title if title else f'Spatial Distribution of SNR\n(Sampled {len(loc_sampled)} points)'
    # plt.title(plot_title)
    plt.xlabel('X (m)', fontsize=16)
    plt.ylabel('Y (m)', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')  # 关键：保证物理长宽比一致
    plt.tight_layout()

    if is_save:
        assert save_path is not None and fig_name is not None
        png_path = os.path.join(save_path, f'{fig_name}.png')
        pdf_path = os.path.join(save_path, f'{fig_name}.pdf')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
        print(f"Saved figure as PNG: {png_path}")
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.0)
        print(f"Saved figure as PDF: {pdf_path}")

    plt.show()


def load_data(file_path):
    """
    [新增] 通用数据加载函数，支持 .npz 和 .h5
    """
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)

    if ext == '.npz':
        # --- NPZ 读取逻辑 ---
        with np.load(file_path) as data_dict:
            loc = data_dict['loc']
            snr = data_dict['snr']
            return loc, snr

    elif ext == '.h5':
        # --- H5 读取逻辑 ---
        with h5py.File(file_path, 'r') as f:
            # 注意：h5py 读取出来的数据在文件关闭后可能会失效
            # 所以这里用 [:] 将数据完全加载到内存中
            loc = f['loc'][:]
            snr = f['snr'][:]
            return loc, snr
    else:
        raise ValueError(f"Unsupported file format: {ext}")


if __name__ == '__main__':
    # 配置路径：这里指向你刚刚生成的 train.h5
    file_path = '../data_ssd/DICHASUS/h5_CD/0152-0157_DirectSample512to32_v2/test.h5'
    save_path = '../results/DICHASUS-015x/'
    fig_name = 'snr_distribution'

    # 1. 可视化 Train Set
    loc_data, snr_data = load_data(file_path)
    visualize_spatial_snr(loc_data, snr_data, sample_ratio=0.2,
                          vmin=0, vmax=25, is_save=True, save_path=save_path, fig_name=fig_name)
