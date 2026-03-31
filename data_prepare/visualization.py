import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange


def visualize_upa_phase(data, subcarrier_idx=16):
    """
    可视化指定子载波上的 UPA 天线相位分布
    """
    # 1. 提取指定子载波的数据 [32]
    # data 的形状是 (Antenna, Subcarrier)
    ant_response = data[:, subcarrier_idx]

    # 2. 相对相位校准 (参考 REFTXOffsetCalibration)
    # 这一步至关重要。绝对相位是随机的，只有相对于参考天线的相位差才有物理意义。
    # 这里选取重排后的第 0 号天线（即 UPA 左上角的天线）作为参考。
    # 乘以共轭等于减去参考天线的相位。
    phase_relative = ant_response * np.conj(ant_response[0])

    # 3. 使用 einops 重排成 UPA 结构
    # 你的 assignments 是按行展平的 (Row-Major)，即先铺满第一行，再铺第二行...
    # assignments matrix 是 4 行 8 列
    # 所以 pattern 是 '(h w) -> h w'，其中 h=4 (y轴), w=8 (x轴)
    phase_grid = rearrange(phase_relative, '(h w) -> h w', h=4, w=8)

    # 4. 可视化
    plt.figure(figsize=(10, 5))
    # np.angle 计算相位 (弧度)
    plt.imshow(np.angle(phase_grid), cmap='twilight', interpolation='nearest')
    plt.colorbar(label='Phase [rad]')
    plt.title(f'UPA Phase Distribution (Subcarrier {subcarrier_idx})')
    plt.xlabel('Antenna Position X (Column)')
    plt.ylabel('Antenna Position Y (Row)')

    # 确保 y 轴方向符合直觉（通常矩阵索引 0 在顶部，imshow 默认也是如此，无需反转）
    plt.show()


def plot_space_frequency_spectrum(data, title="Space-Frequency Magnitude Spectrum",
                                  xlabel="Subcarrier Index (Frequency)",
                                  ylabel="Antenna Index (Space)",
                                  use_db=False):
    """
    绘制空频域谱（幅值）。

    参数:
        data (np.ndarray or torch.Tensor): 输入数据，形状应为 (32, 32) 或 (Num_Antennas, Num_Carriers)。
                                           如果是复数，会自动计算幅值。
        title (str): 图表的标题。
        xlabel (str): X轴标签，通常是频率/载波维。
        ylabel (str): Y轴标签，通常是空间/天线维。
        use_db (bool): 是否使用分贝(dB)刻度显示 (20*log10(abs(x)))。默认为 False (线性幅值)。
    """

    # 1. 数据类型转换：如果是 PyTorch Tensor，转换为 Numpy
    # if isinstance(data, torch.Tensor):
    #     data = data.detach().cpu().numpy()

    # 2. 维度检查：确保是 2D 矩阵
    if data.ndim != 2:
        raise ValueError(f"Input data must be a 2D matrix, but got shape {data.shape}")

    # 3. 计算幅值
    # 如果是复数，np.abs 计算模；如果是实数，取绝对值
    magnitude = np.abs(data)

    # 4. (可选) 转换为 dB 刻度，便于观察动态范围大的信号
    if use_db:
        # 加上一个微小值 eps 防止 log(0)
        magnitude = 20 * np.log10(magnitude + 1e-9)
        if "dB" not in title:
            title += " (dB)"

    # 5. 绘图
    plt.figure(figsize=(8, 6))

    # 使用 imshow 绘制热力图
    # origin='lower' 让 (0,0) 点在左下角（符合通常坐标系习惯），
    # 如果习惯矩阵索引顺序（(0,0)在左上角），可以去掉 origin='lower' 或改为 'upper'
    im = plt.imshow(magnitude, aspect='auto', cmap='jet', origin='upper')

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('Magnitude (dB)' if use_db else 'Magnitude')

    # 设置标题和标签
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # 设置刻度（可选，确保整数刻度）
    # plt.xticks(np.arange(0, data.shape[1], 4)) # 每隔4个画一个刻度
    # plt.yticks(np.arange(0, data.shape[0], 4))

    plt.tight_layout()
    plt.show()


def visualize_spatial_snr(loc, snr, sample_ratio=1.0, title=None):
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
    plt.figure(figsize=(10, 8))

    # Scatter plot
    # x: loc[:, 0], y: loc[:, 1]
    # c: snr_sampled (颜色映射)
    sc = plt.scatter(loc_sampled[:, 0], loc_sampled[:, 1],
                     c=snr_sampled, cmap='viridis',
                     s=3, alpha=0.7)

    # 添加颜色条
    cbar = plt.colorbar(sc)
    cbar.set_label('Average SNR')

    # 装饰
    plot_title = title if title else f'Spatial Distribution of SNR\n(Sampled {len(loc_sampled)} points)'
    plt.title(plot_title)
    plt.xlabel('Position X (m)')
    plt.ylabel('Position Y (m)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')  # 关键：保证物理长宽比一致

    plt.tight_layout()
    plt.show()


def visualize_trajectory_over_time(loc, time, title="Agent Trajectory with Time", connect_points=False):
    """
    可视化不同时间的坐标位置，以显示轨迹。
    会自动根据 time 对 loc 进行排序，以还原真实的运动顺序。

    Args:
        loc (np.ndarray): 位置坐标，形状 [N, 2] 或 [N, 3]。
        time (np.ndarray): 时间戳，形状 [N,] 或 [N, 1]。
        title (str): 图表标题。
        connect_points (bool): 是否用细线将点连接起来（适合观察连续性，但如果数据有中断可能会产生跨越长线）。
    """
    # 1. 数据预处理
    # 确保 time 是 1D 数组
    time = np.ravel(time)

    # 关键步骤：根据时间进行排序
    # 因为之前做数据集的时候可能 shuffle 过，不排序的话轨迹是乱的
    sort_idx = np.argsort(time)
    loc_sorted = loc[sort_idx]
    time_sorted = time[sort_idx]

    # 提取 X, Y
    x = loc_sorted[:, 0]
    y = loc_sorted[:, 1]

    # 2. 绘图
    plt.figure(figsize=(10, 8))

    # (可选) 绘制连接线，使用浅灰色，表示路径
    if connect_points:
        plt.plot(x, y, c='gray', alpha=0.3, linewidth=1, linestyle='--')

    # 绘制散点，颜色代表时间
    # cmap='plasma' 或 'viridis' 对时间序列非常友好
    sc = plt.scatter(x, y, c=time_sorted, cmap='plasma', s=5, alpha=0.8)

    # 3. 标记起点和终点
    plt.scatter(x[0], y[0], c='green', s=150, marker='*', edgecolors='black', label='Start', zorder=10)
    plt.scatter(x[-1], y[-1], c='red', s=150, marker='X', edgecolors='black', label='End', zorder=10)

    # 4. 添加颜色条
    # 将时间戳格式化一下，如果是很大的 unix timestamp，colorbar 会显示科学计数法
    # 这里我们显示为 "Relative Time (s)"，即从第0秒开始
    relative_time = time_sorted - time_sorted[0]
    # 如果你想看相对时间，可以用这个：
    # sc.set_array(relative_time)

    cbar = plt.colorbar(sc)
    cbar.set_label('Time (Timestamp)')

    # 5. 装饰
    plt.title(f"{title}\n(Total Points: {len(loc)})")
    plt.xlabel("Position X (m)")
    plt.ylabel("Position Y (m)")
    plt.axis('equal')  # 保证物理比例一致
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

