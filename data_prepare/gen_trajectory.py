import os
import h5py
import numpy as np
from scipy.spatial import KDTree
import random
from tqdm import tqdm
from collections import deque

import matplotlib.pyplot as plt


def visualize_source_distribution(test_h5_path):
    """
    可视化整个测试集的空间分布。
    """
    print(f"Visualizing source data distribution from: {test_h5_path} ...")

    if not os.path.exists(test_h5_path):
        print(f"[Error] File not found: {test_h5_path}")
        return

    with h5py.File(test_h5_path, 'r') as f:
        # 读取位置数据
        locs = f['loc'][:]  # [N, 3] (假设是 x, y, z)

    num_samples = locs.shape[0]
    print(f"Loaded {num_samples} samples.")

    # 提取 X, Y
    x = locs[:, 0]
    y = locs[:, 1]

    # 计算一些统计信息，帮助理解场景尺寸
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    print(f"Scene Bounds:")
    print(f"  X: [{x_min:.3f}, {x_max:.3f}] (Span: {x_max - x_min:.3f}m)")
    print(f"  Y: [{y_min:.3f}, {y_max:.3f}] (Span: {y_max - y_min:.3f}m)")

    # 开始绘图
    plt.figure(figsize=(10, 10), dpi=100)

    # 绘制所有点
    # s=1: 点的大小设为1，避免重叠太严重
    # alpha=0.2: 透明度设为0.2，这样重叠的地方颜色会更深，能看出密度分布
    plt.scatter(x, y, s=1, c='blue', alpha=0.1, label='Source Data Points')

    # 设置标题和标签
    plt.title(f'Source Test Dataset Distribution\nN={num_samples}, Area: ~{x_max - x_min:.1f}x{y_max - y_min:.1f}m')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')

    # 保持比例一致 (非常重要，否则正方形会被拉成矩形)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 保存图片
    # save_path = test_h5_path.replace('.h5', '_distribution.png')
    # plt.savefig(save_path)
    # print(f"Distribution plot saved to: {save_path}")

    # 如果在支持界面的环境，可以取消注释下面这行
    plt.show()
    # plt.close()


def visualize_trajectory(h5_path):
    """
    读取生成的 H5 文件并可视化 XY 平面轨迹。
    颜色随时间变化 (Viridis Colormap: 紫色=起点 -> 黄色=终点)
    """
    print(f"Visualizing trajectory from {h5_path}...")

    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return

    with h5py.File(h5_path, 'r') as f:
        locs = f['loc'][:]  # [N, 3]

    num_points = locs.shape[0]

    # 提取 X 和 Y 坐标 (假设是在 XY 平面上移动)
    x = locs[:, 0]
    y = locs[:, 1]

    # 创建颜色映射 (根据索引/时间)
    times = np.arange(num_points)

    plt.figure(figsize=(10, 8), dpi=100)

    # 1. 画轨迹连线 (表现连续性)
    plt.plot(x, y, c='gray', alpha=0.3, linewidth=1, label='Path Trace')

    # 2. 画散点 (表现采样密度和时间顺序)
    # c=times 让颜色随时间变化
    sc = plt.scatter(x, y, c=times, cmap='viridis', s=15, alpha=0.9, label='Trajectory Points')

    # 3. 标记起点和终点
    plt.scatter(x[0], y[0], c='red', s=150, marker='*', edgecolors='black', zorder=10, label='Start')
    plt.scatter(x[-1], y[-1], c='blue', s=120, marker='X', edgecolors='black', zorder=10, label='End')

    # 装饰图表
    plt.colorbar(sc, label='Time Step (Index)')
    plt.title(f'Trajectory Visualization (N={num_points})\nColor represents time progression')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')  # 保持 XY 比例一致，避免图形变形

    # 保存或显示
    save_path = h5_path.replace('.h5', '_plot.png')
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()


def generate_gap_aware_trajectory(
        test_h5_path,
        output_path,
        num_points=2000,
        target_step_dist=0.04,  # 稍微增大基础步长到 4cm，适应线内移动
        gap_search_radius=0.3,  # [新增] 换道搜索半径：30cm (足够跨越大部分间隙)
        turn_sigma_degrees=60,  # [调整] 增大到 60 度，允许更灵活的转向
        taboo_len=200,  # [调整] 增加记忆长度，防止在换道时反复横跳
        seed=None
):
    """
    针对条纹状（Raster）数据优化的轨迹生成器。
    具备“换道”能力，可以在扫描线之间跳跃。
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    print(f"Loading metadata from {test_h5_path}...")
    if not os.path.exists(test_h5_path):
        raise FileNotFoundError(f"File not found: {test_h5_path}")

    with h5py.File(test_h5_path, 'r') as f:
        all_locs = f['loc'][:]

    total_samples = all_locs.shape[0]
    print(f"Data loaded. Points: {total_samples}")

    print("Building KDTree...")
    tree = KDTree(all_locs)

    # 基础移动参数
    # 正常步长范围: [2cm, 6cm]
    min_step = 0.02
    max_step = target_step_dist + 0.02

    sigma_rad = np.deg2rad(turn_sigma_degrees)

    print(f"Generating Gap-Aware Trajectory:")
    print(f"  - Normal Step: {min_step:.3f}m ~ {max_step:.3f}m")
    print(f"  - Gap Jump Radius: up to {gap_search_radius:.2f}m")

    recent_indices = deque(maxlen=taboo_len)
    trajectory_indices = []

    # --- 初始化 (寻找非孤岛起点) ---
    valid_start_found = False
    while not valid_start_found:
        p0_idx = np.random.randint(total_samples)
        curr_loc = all_locs[p0_idx]
        # 检查周围是否有足够邻居
        dists, idxs = tree.query(curr_loc, k=20)
        # 只要最近的几个邻居里有稍微远一点的（不是重叠点），就算有效
        if dists[-1] > 0.01:
            valid_start_found = True

    trajectory_indices.append(p0_idx)
    recent_indices.append(p0_idx)

    # 初始化第二个点 (找最近的有效点)
    dists, idxs = tree.query(all_locs[p0_idx], k=50)
    p1_idx = p0_idx
    for idx in idxs:
        if idx != p0_idx:
            p1_idx = idx
            break

    trajectory_indices.append(p1_idx)
    recent_indices.append(p1_idx)

    # --- 主循环 ---
    curr_idx = p1_idx
    prev_idx = p0_idx

    pbar = tqdm(total=num_points)
    pbar.update(2)

    while len(trajectory_indices) < num_points:
        curr_loc = all_locs[curr_idx]
        prev_loc = all_locs[prev_idx]

        # 1. 计算当前速度方向
        vec_velocity = curr_loc - prev_loc
        norm_velocity = np.linalg.norm(vec_velocity)
        if norm_velocity < 1e-6: norm_velocity = 1.0

        # --- 阶段 A: 标准搜索 (沿线走) ---
        # 搜索半径较小，保证轨迹平滑
        candidate_indices = tree.query_ball_point(curr_loc, r=max_step)

        valid_candidates = []
        for idx in candidate_indices:
            if idx in recent_indices: continue

            vec_cand = all_locs[idx] - curr_loc
            dist = np.linalg.norm(vec_cand)
            if dist >= min_step:
                valid_candidates.append((idx, vec_cand, dist))

        # --- 阶段 B: 换道搜索 (Gap Jumping) ---
        # 如果标准搜索找不到点（比如到了线的尽头），或者剩下的点都在身后（被惯性权重过滤掉）
        # 我们就开启“上帝视角”，搜索更大的范围

        is_gap_jump = False
        if not valid_candidates:
            # print(f" [Info] End of line at step {len(trajectory_indices)}. Attempting gap jump...")
            is_gap_jump = True

            # 扩大搜索半径到 gap_search_radius (e.g., 0.3m)
            wide_indices = tree.query_ball_point(curr_loc, r=gap_search_radius)

            for idx in wide_indices:
                if idx in recent_indices: continue
                if idx == curr_idx: continue

                vec_cand = all_locs[idx] - curr_loc
                dist = np.linalg.norm(vec_cand)

                # 换道时，允许距离稍大，但也得大于最小步长
                if dist >= min_step:
                    valid_candidates.append((idx, vec_cand, dist))

        # --- 阶段 C: 强力救场 (KNN Fallback) ---
        # 如果连大范围搜索都全是禁忌点（把自己围死了）
        if not valid_candidates:
            # print(" [Warn] Stuck. Using Aggressive KNN Fallback...")
            # 搜索最近的 500 个点 (足以跨越几米长的密集轨迹线)
            dists, idxs = tree.query(curr_loc, k=500)

            for i, idx in enumerate(idxs):
                if idx == curr_idx: continue
                if idx not in recent_indices:
                    # 只要不是太远 (比如超过 1米)，都可以作为救命点
                    if dists[i] < 1.0:
                        vec_cand = all_locs[idx] - curr_loc
                        valid_candidates.append((idx, vec_cand, dists[i]))

            # 如果还是不行，说明真的无路可走，清空部分记忆
            if not valid_candidates:
                # print(" [Alert] Clearing memory...")
                recent_indices.clear()
                # 重新让当前点入列，防止立刻原地不动
                recent_indices.append(curr_idx)
                # 这里不 break，下一轮循环会自动重新搜索

        if not valid_candidates:
            print(f"\n[Stop] Terminated at {len(trajectory_indices)}.")
            break

        # 2. 概率权重计算
        weights = []
        final_candidates = []

        for idx, vec_cand, dist in valid_candidates:
            # 计算角度
            dot_val = np.dot(vec_velocity, vec_cand)
            cos_theta = dot_val / (norm_velocity * dist)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_rad = np.arccos(cos_theta)

            # 基础权重: 高斯分布
            # 如果是 Gap Jump (换道)，我们放宽对角度的惩罚
            # 因为换道通常需要 90度 甚至 180度 转弯
            current_sigma = sigma_rad
            if is_gap_jump or dist > max_step:
                current_sigma = np.deg2rad(90)  # 换道时允许大转弯

            w = np.exp(- (angle_rad ** 2) / (2 * current_sigma ** 2))

            # 距离惩罚/奖励
            # 我们希望优先选近的点（保持平滑），但在换道模式下，我们必须选远的点
            # 所以这里不做强硬的距离惩罚，主要靠 KDTree 的半径来控制

            weights.append(w)
            final_candidates.append(idx)

        # 3. 采样
        weights = np.array(weights)
        if np.sum(weights) == 0:
            # 如果所有候选点权重都极小（比如都在正后方），则均匀采样，强行掉头
            probs = None
        else:
            probs = weights / np.sum(weights)

        next_idx = np.random.choice(final_candidates, p=probs)

        trajectory_indices.append(next_idx)
        recent_indices.append(next_idx)
        prev_idx = curr_idx
        curr_idx = next_idx
        pbar.update(1)

    pbar.close()

    # --- 保存 ---
    actual_len = len(trajectory_indices)
    print(f"Trajectory finished. Length: {actual_len}/{num_points}")
    print(f"Saving to {output_path}...")

    with h5py.File(test_h5_path, 'r') as f_src, h5py.File(output_path, 'w') as f_dst:
        for k, v in f_src.attrs.items():
            f_dst.attrs[k] = v
        indices_np = np.array(trajectory_indices)
        src_data = f_src['data'][:]
        src_loc = f_src['loc'][:]
        src_snr = f_src['snr'][:]

        f_dst.create_dataset('data', data=src_data[indices_np], dtype='complex64')
        f_dst.create_dataset('loc', data=src_loc[indices_np], dtype='float32')
        f_dst.create_dataset('snr', data=src_snr[indices_np], dtype='float32')
        f_dst.create_dataset('time', data=np.arange(actual_len, dtype='int32'))
        f_dst.create_dataset('original_indices', data=indices_np, dtype='int64')

    print("Done.")


def generate_global_guided_trajectory(
        test_h5_path,
        output_path,
        num_points=2000,
        target_step_dist=0.04,  # 4cm 步长
        gap_search_radius=0.4,  # 40cm 换道半径 (足够跨越条纹间隙)
        turn_sigma_degrees=60,
        taboo_len=150,
        waypoints_count=5,  # [新参数] 轨迹将依次经过5个大区域
        attraction_strength=0.6,  # [新参数] 0~1, 目标引力强度。越大越直奔目标，越小越随机。
        seed=None
):
    """
    全局引导轨迹生成器。
    通过设立全局锚点 (Waypoints)，强制轨迹“跑遍”整个地图，避免陷入局部。
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    print(f"Loading metadata from {test_h5_path}...")
    if not os.path.exists(test_h5_path):
        raise FileNotFoundError(f"File not found.")

    with h5py.File(test_h5_path, 'r') as f:
        all_locs = f['loc'][:]

    total_samples = all_locs.shape[0]
    tree = KDTree(all_locs)

    # 1. 自动生成全局路书 (Waypoints)
    # 策略：随机选择几个离得比较远的点作为途经点
    print(f"Planning {waypoints_count} global waypoints...")
    waypoints_indices = []

    # 起点：随机
    current_wp_idx = np.random.randint(total_samples)
    waypoints_indices.append(current_wp_idx)

    for _ in range(waypoints_count):
        # 随机采样 100 个点，选一个离当前锚点最远的，作为下一个锚点
        # 这保证了轨迹会大幅度跨越空间
        candidates = np.random.choice(total_samples, 100, replace=False)
        curr_loc = all_locs[current_wp_idx]
        dists = np.linalg.norm(all_locs[candidates] - curr_loc, axis=1)
        best_cand = candidates[np.argmax(dists)]
        waypoints_indices.append(best_cand)
        current_wp_idx = best_cand

    # 将路书转换为队列
    waypoint_queue = deque(waypoints_indices[1:])  # 去掉起点，剩下的作为目标
    current_target_idx = waypoint_queue.popleft()
    current_target_loc = all_locs[current_target_idx]

    print(f"Generating Trajectory with Global Guidance...")

    # 基础参数
    min_step = 0.02
    max_step = target_step_dist + 0.02
    sigma_rad = np.deg2rad(turn_sigma_degrees)

    recent_indices = deque(maxlen=taboo_len)
    trajectory_indices = [waypoints_indices[0]]  # 放入起点
    recent_indices.append(waypoints_indices[0])

    curr_idx = waypoints_indices[0]

    # 初始化一个虚拟的前一点，给一个初始速度方向指向第一个目标
    start_loc = all_locs[curr_idx]
    initial_dir = current_target_loc - start_loc
    initial_dir /= (np.linalg.norm(initial_dir) + 1e-6)
    prev_loc = start_loc - initial_dir * target_step_dist

    pbar = tqdm(total=num_points)
    pbar.update(1)

    while len(trajectory_indices) < num_points:
        curr_loc = all_locs[curr_idx]

        # --- 0. 目标管理 ---
        # 检查是否已经接近当前目标 (比如进入 0.5米 范围内)
        dist_to_target = np.linalg.norm(current_target_loc - curr_loc)

        # 如果接近目标，或者当前目标“太难到达”（卡太久），切换下一个目标
        if dist_to_target < 0.5:
            if waypoint_queue:
                current_target_idx = waypoint_queue.popleft()
                current_target_loc = all_locs[current_target_idx]
                # print(f" [Nav] Reached waypoint. Next target dist: {np.linalg.norm(current_target_loc - curr_loc):.1f}m")
            else:
                # 没目标了，重新随机一个远处的
                new_cand = np.random.randint(total_samples)
                current_target_idx = new_cand
                current_target_loc = all_locs[new_cand]
                # print(f" [Nav] All waypoints done. Randomizing new target.")

        # --- 1. 矢量计算 ---
        # 惯性向量 (Momentum)
        vec_momentum = curr_loc - prev_loc
        norm_mom = np.linalg.norm(vec_momentum)
        if norm_mom < 1e-6: vec_momentum = np.random.randn(3); norm_mom = 1.0
        vec_momentum /= norm_mom

        # 目标向量 (Attraction)
        vec_target = current_target_loc - curr_loc
        norm_target = np.linalg.norm(vec_target)
        if norm_target < 1e-6:
            vec_target = vec_momentum  # 已到达
        else:
            vec_target /= norm_target

        # 混合期望方向：惯性 + 目标引力
        # alpha 控制我们多大程度上顺从惯性，多大程度上屈服于目标
        # attraction_strength = 0.6 意味着主要受目标指引
        vec_desired = (1 - attraction_strength) * vec_momentum + attraction_strength * vec_target
        vec_desired /= np.linalg.norm(vec_desired)

        # --- 2. 搜索候选点 ---
        # A. 标准搜索
        candidate_indices = tree.query_ball_point(curr_loc, r=max_step)

        valid_candidates = []
        for idx in candidate_indices:
            if idx in recent_indices: continue
            dist = np.linalg.norm(all_locs[idx] - curr_loc)
            if dist >= min_step:
                valid_candidates.append((idx, dist))

        # B. 换道搜索 (Gap Jump)
        # 只有当 1.没找到点 或者 2.想要去的方向和能去的方向不一致时 (比如想去上方，但周围全是水平点)
        need_gap_jump = False
        if not valid_candidates:
            need_gap_jump = True
        else:
            # 检查：现有的候选点是否能帮我靠近目标？
            # 简单的检查方法：计算所有候选向量与目标向量的平均点积
            # 如果都很小，说明我们在沿着扫描线走，但目标在侧面 -> 需要换道
            best_alignment = -1
            for idx, dist in valid_candidates:
                v_c = (all_locs[idx] - curr_loc) / dist
                align = np.dot(v_c, vec_target)
                best_alignment = max(best_alignment, align)

            if best_alignment < 0.3:  # 夹角大于 70度，说明没法靠近目标
                need_gap_jump = True

        if need_gap_jump:
            # 扩大搜索范围
            wide_indices = tree.query_ball_point(curr_loc, r=gap_search_radius)
            for idx in wide_indices:
                if idx in recent_indices: continue
                dist = np.linalg.norm(all_locs[idx] - curr_loc)
                if dist >= min_step:
                    # 对于换道点，我们要严格筛选：必须在这个Gap Jump能帮我们靠近目标时才选
                    v_gap = (all_locs[idx] - curr_loc) / dist
                    if np.dot(v_gap, vec_target) > 0.5:  # 必须在目标扇面内 (60度)
                        valid_candidates.append((idx, dist))

        # C. 绝境求生 (Fallback)
        if not valid_candidates:
            dists, idxs = tree.query(curr_loc, k=500)
            for i, idx in enumerate(idxs):
                if idx == curr_idx: continue
                if idx not in recent_indices:
                    if dists[i] < 0.8:  # 只要不太远
                        valid_candidates.append((idx, dists[i]))

            if not valid_candidates:
                # 清空记忆，允许掉头
                recent_indices.clear()
                recent_indices.append(curr_idx)
                continue

        # --- 3. 概率权重打分 ---
        weights = []
        final_cands = []

        for idx, dist in valid_candidates:
            vec_cand = all_locs[idx] - curr_loc
            vec_cand /= dist

            # 核心打分逻辑：
            # 看这个候选方向，和我们“期望方向(vec_desired)”的夹角
            dot_val = np.dot(vec_desired, vec_cand)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            angle = np.arccos(dot_val)

            # 同样使用高斯分布，角度越小分越高
            # 注意：如果是 Gap Jump (距离较远)，我们放宽 Sigma，允许大动作
            cur_sigma = sigma_rad
            if dist > max_step: cur_sigma = np.deg2rad(90)

            w = np.exp(- (angle ** 2) / (2 * cur_sigma ** 2))

            weights.append(w)
            final_cands.append(idx)

        weights = np.array(weights)
        if np.sum(weights) == 0:
            probs = None
        else:
            probs = weights / np.sum(weights)

        next_idx = np.random.choice(final_cands, p=probs)

        # 更新状态
        trajectory_indices.append(next_idx)
        recent_indices.append(next_idx)
        prev_loc = curr_loc
        curr_idx = next_idx
        pbar.update(1)

    pbar.close()

    # --- 保存 ---
    print(f"Saving {len(trajectory_indices)} points to {output_path}...")
    with h5py.File(test_h5_path, 'r') as f_src, h5py.File(output_path, 'w') as f_dst:
        for k, v in f_src.attrs.items():
            f_dst.attrs[k] = v
        indices_np = np.array(trajectory_indices)
        src_data = f_src['data'][:]
        src_loc = f_src['loc'][:]
        src_snr = f_src['snr'][:]
        f_dst.create_dataset('data', data=src_data[indices_np], dtype='complex64')
        f_dst.create_dataset('loc', data=src_loc[indices_np], dtype='float32')
        f_dst.create_dataset('snr', data=src_snr[indices_np], dtype='float32')
        f_dst.create_dataset('time', data=np.arange(len(trajectory_indices), dtype='int32'))
        f_dst.create_dataset('original_indices', data=indices_np, dtype='int64')

    print("Done.")


if __name__ == "__main__":
    base_dir = '../data_ssd/DICHASUS/h5_CD/0152-0157_DirectSample512to32'
    test_file = os.path.join(base_dir, 'test.h5')
    output_file = os.path.join(base_dir, 'trajectory_dense_2k.h5')

    # 运行可视化
    visualize_source_distribution(test_file)

    if os.path.exists(test_file):
        generate_global_guided_trajectory(
            test_h5_path=test_file,
            output_path=output_file,
            num_points=2000,

            target_step_dist=0.04,  # 4cm 步长
            gap_search_radius=0.4,  # 40cm 跨越半径

            # [关键参数]
            waypoints_count=8,  # 设定8个途经点，让它多跑跑
            attraction_strength=0.7,  # 较强的目标引力，强制它离开舒适区
            taboo_len=100,

            seed=2026
        )

    # 2. 运行可视化
    visualize_trajectory(output_file)