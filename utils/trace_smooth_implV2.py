import os
import h5py
import numpy as np
import torch
import hdf5storage
from einops import rearrange
from tqdm import tqdm
from utils.metrics import Nmse, Corr, Nmse_ant_snr_weighted


# --- 辅助函数 ---
def channel_index(x_index, y_index, path, high=181):
    index = x_index * high + y_index
    mat_path = os.path.join(path, 'mat', 'DeepMIMO_dataset_' + str(index) + '.mat')
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Mat file not found: {mat_path}")
    channel_data_mat = hdf5storage.loadmat(mat_path)
    ans = channel_data_mat['DeepMIMO_dataset']
    return ans


# ==============================================================================
# Core Inference Logic (显存优化版)
# ==============================================================================
def core_trace_smooth_inference(model, truth_channel_list, past_len, conf, device,
                                snr_list=None):
    """
    核心推理逻辑（显存优化版）：
    数据驻留在 CPU，仅在推理时将必要的 Window 移动到 GPU。

    Args:
        model: PyTorch 模型
        truth_channel_list: list of tensors (CPU), shape [(1, Ant, Car, 2), ...]
        past_len: 历史长度
        conf: 配置字典
        device: 计算设备
        snr_list: list of tensors (CPU), shape [(1, Ant), ...]

    Returns:
        dict: 包含所有评估指标的字典
    """
    input_ant_size = conf['input_ant_size']
    input_car_size = conf['input_car_size']
    ant_size = conf['num_ant']
    car_size = conf['num_car']

    step_ant = ant_size // input_ant_size
    step_car = car_size // input_car_size

    # deduction_channel_list 存储 CPU 上的 Tensor，防止显存随时间线性增长
    deduction_channel_list = []

    # 结果容器
    metrics = {
        'nmse_list_ar': [],
        'nmse_snrW_list_ar': [],
        'corr_list_ar': [],
        'nmse_list_ideal': [],
        'nmse_snrW_list_ideal': [],
        'corr_list_ideal': []
    }

    len_trace = len(truth_channel_list)
    model.eval()

    print(f"[CoreInference] Running inference on {len_trace} frames, past_len={past_len}...")
    pbar = tqdm(range(len_trace))
    for i in pbar:
        # 1. 获取当前时刻真值 (从 CPU 加载到 GPU)
        # 注意：这里我们尽量延迟 .to(device) 的时机，或者用完即丢
        pbar_dict = {}
        h_now_gt_cpu = truth_channel_list[i].cpu()
        h_now_gt_gpu = h_now_gt_cpu.to(device)
        h_now_gt = h_now_gt_gpu.cpu().detach().numpy()

        # 准备当前时刻的部分观测输入 (GPU)
        h_now_input = h_now_gt_gpu[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                                   0:step_car * (input_car_size - 1) + 1:step_car, :]
        # 获取当前 batch size 和其他维度，用于构造空 tensor
        B, Ant, Car, C = h_now_gt_gpu.shape

        if i < past_len:
            # Warm-up 阶段：直接将真值存入推演历史 (存 CPU)
            deduction_channel_list.append(h_now_gt_cpu)
            continue  # 跳过推理，直接进入下一帧

        # -------------------------------------------------------
        # === Part 1: 自回归 (Auto-Regressive / Error Propagation) ===
        # -------------------------------------------------------
        # 准备 Past (来自推演历史 deduction_channel_list，它们在 CPU)
        # 切片取出最近 past_len 帧 -> Stack -> 移动到 GPU
        h_past_list = deduction_channel_list[i - past_len:i]
        if len(h_past_list) == 0:
            # 如果没有历史数据（past_len=0 或 刚开始），构造 (B, 0, ...) 的空 tensor
            h_past_input = torch.zeros((B, 0, Ant, Car, C), device=device)
        else:
            # 正常 Stack
            h_past_input = torch.stack(h_past_list, dim=0)  # (Len, B, Ant, Car, C)
            h_past_input = rearrange(h_past_input, 'len b ant car c -> b len ant car c')
            h_past_input = h_past_input.to(device)

        # 推理
        with torch.no_grad():
            h_now_output_ar = model(h_now_input, h_past_input)

        # 更新推演历史 (关键优化：将生成结果移回 CPU 存储，防止 OOM)
        deduction_channel_list.append(h_now_output_ar.detach().cpu())

        # 计算 AR 指标 (使用 GPU 上的 Tensor 计算，内部转 numpy)
        h_now_pred_ar = h_now_output_ar.cpu().detach().numpy()
        nmse_ar = Nmse(h_now_pred_ar, h_now_gt)
        rou_ar = Corr(h_now_pred_ar, h_now_gt)
        metrics['nmse_list_ar'].append(np.mean(nmse_ar))
        metrics['corr_list_ar'].append(np.mean(rou_ar))
        pbar_dict['AR'] = f"{metrics['nmse_list_ar'][-1]:.4f}"

        if snr_list is not None:
            snr_now = snr_list[i].cpu().numpy()  # [1, ant]
            nmse_snrW_ar = Nmse_ant_snr_weighted(h_now_pred_ar, h_now_gt, snr_now)
            metrics['nmse_snrW_list_ar'].append(np.mean(nmse_snrW_ar))
            pbar_dict['AR-snrW'] = f"{metrics['nmse_snrW_list_ar'][-1]:.4f}"

        # -------------------------------------------------------
        # === Part 2: 理想情况 (Ideal / No Error Propagation) ===
        # -------------------------------------------------------
        # 准备 Past (来自真值 truth_channel_list，它们在 CPU)
        h_past_list_truth = truth_channel_list[i - past_len:i]
        if len(h_past_list_truth) == 0:
            h_past_input_truth = torch.zeros((B, 0, Ant, Car, C), device=device)
        else:
            h_past_input_truth = torch.stack(h_past_list_truth, dim=0)
            h_past_input_truth = rearrange(h_past_input_truth, 'len b ant car c -> b len ant car c')
            h_past_input_truth = h_past_input_truth.to(device)

        # 推理
        with torch.no_grad():
            h_now_output_ideal = model(h_now_input, h_past_input_truth)

        # 计算 Ideal 指标
        h_now_pred_ideal = h_now_output_ideal.cpu().detach().numpy()
        nmse_ideal = Nmse(h_now_pred_ideal, h_now_gt)
        rou_ideal = Corr(h_now_pred_ideal, h_now_gt)
        metrics['nmse_list_ideal'].append(np.mean(nmse_ideal))
        metrics['corr_list_ideal'].append(np.mean(rou_ideal))
        pbar_dict['Ideal'] = f"{metrics['nmse_list_ideal'][-1]:.4f}"

        if snr_list is not None:
            snr_now = snr_list[i].cpu().numpy()  # [1, ant]
            nmse_snrW_ideal = Nmse_ant_snr_weighted(h_now_pred_ideal, h_now_gt, snr_now)
            metrics['nmse_snrW_list_ideal'].append(np.mean(nmse_snrW_ideal))
            pbar_dict['Ideal-snrW'] = f"{metrics['nmse_snrW_list_ideal'][-1]:.4f}"

        if len(metrics['nmse_list_ar']) > 0 and i % 10 == 0:
            pbar.set_postfix(pbar_dict)

    # 打印简报
    if len(metrics['nmse_list_ar']) > 0:
        print(f"Average AR NMSE: {np.mean(metrics['nmse_list_ar']):.5f}")
        print(f"Average Ideal NMSE: {np.mean(metrics['nmse_list_ideal']):.5f}")
        if snr_list is not None:
            print(f"Average AR-snrW NMSE: {np.mean(metrics['nmse_snrW_list_ar']):.5f}")
            print(f"Average Ideal-snrW NMSE: {np.mean(metrics['nmse_snrW_list_ideal']):.5f}")

    return metrics


# ==============================================================================
# DeepMIMO Implementation (Wrapper)
# ==============================================================================
def run_trace_smooth_deepmimoO1(model, past_len, conf, device, save_dir):
    path = conf['data_dirs'][0]

    # --- 加载 Trace ---
    x_trace = np.load(os.path.join(path, 'x_trace_smooth.npy'))
    y_trace = np.load(os.path.join(path, 'y_trace_smooth.npy'))
    len_trace = x_trace.shape[0]

    # --- 加载数据并构建 Truth List (保持在 CPU) ---
    truth_channel_list = []
    const_h = conf.get('const_h', 10000.0)
    print("[DeepMIMO] Loading ground truth from .mat files...")

    for i in range(len_trace):
        reverse_x_index = x_trace[i]
        y_index = y_trace[i]
        x_index = 899 - reverse_x_index

        channel_data = channel_index(x_index, y_index, path)
        channel_data = torch.tensor(channel_data)
        channel_data = rearrange(channel_data, '(b ant) car c -> b ant car c', b=1)

        # 保持在 CPU 上，DeepMIMO 特有的 Scaling
        channel_data = channel_data.float() * const_h
        truth_channel_list.append(channel_data)

    # --- 执行核心逻辑 ---
    metrics = core_trace_smooth_inference(model, truth_channel_list, past_len, conf, device)

    # --- 保存 ---
    h5_path = os.path.join(save_dir, f'trace_smooth_metrics_deepmimo_pastlen{past_len}.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('nmse_list_ar', data=np.array(metrics['nmse_list_ar']))
        f.create_dataset('nmse_list_ideal', data=np.array(metrics['nmse_list_ideal']))
        f.create_dataset('corr_list_ar', data=np.array(metrics['corr_list_ar']))
        f.create_dataset('corr_list_ideal', data=np.array(metrics['corr_list_ideal']))
        f.attrs['past_len'] = past_len
        f.attrs['description'] = 'DICHASUS trace smooth results with trajectory'

    print(f"[DeepMIMO] Metrics saved to {h5_path}")


# ==============================================================================
# DICHASUS Implementation (New)
# ==============================================================================
def run_trace_smooth_DICHASUS(model, past_len, conf, device, save_dir):
    """
    针对 DICHASUS 数据集的 Trace Smooth 测试。

    Args:
        dataset_path (str): .h5 数据集文件的完整路径 (e.g., .../data_ca01.h5)
        part_name (str): 子数据集名称 (e.g., 'dichasus-ca01_part4')
    """
    dataset_path = conf['dichasus']['trace_src_dataset']
    part_name = conf['dichasus']['trace_part_name']
    print(f"[DICHASUS] Opening dataset: {dataset_path}")
    print(f"[DICHASUS] Target Part: {part_name}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # --- 1. 读取数据 (HDF5) ---
    with h5py.File(dataset_path, 'r') as f:
        # 1.1 解析 Part 索引，attribute 是 "part_names" (逗号分隔字符串)
        if 'part_names' in f.attrs:
            part_names_str = f.attrs['part_names']
            if isinstance(part_names_str, bytes):
                part_names_str = part_names_str.decode('utf-8')
            part_list = part_names_str.split(',')
        else:
            # 只有1个part的情况，或者没有存名字，则假设全部都是
            print("[DICHASUS] Warning: 'part_names' attribute not found. Assuming single part or using all data.")
            part_list = [part_name]  # Hack to pass the check below if logic matches

        try:
            target_idx = part_list.index(part_name)
        except ValueError:
            raise ValueError(f"Part '{part_name}' not found in dataset attributes: {part_list}")

        # 1.2 获取 Mask
        if 'source_indices' in f:
            source_indices = f['source_indices'][:]
            mask = (source_indices == target_idx)
            print(f"[DICHASUS] Found {np.sum(mask)} samples for part {part_name}.")
        else:
            print("[DICHASUS] 'source_indices' not found. Loading ALL data.")
            mask = slice(None)  # select all

        # 1.3 加载数据 (按 Mask)
        # 注意：这里我们一次性加载到内存，假设显存/内存足够。
        # DICHASUS data shape: [N, 32, 32] Complex64
        data_complex = f['data'][()][mask]
        loc_data = f['loc'][()][mask]
        time_data = f['time'][()][mask]
        snr_data = f['snr'][()][mask]

    # --- 2. 数据排序 (Time Sorted) ---
    # 虽然 DICHASUS H5 生成时通常已排序，但这里为了鲁棒性再次排序
    sort_indices = np.argsort(time_data)

    data_complex = data_complex[sort_indices]
    loc_data = loc_data[sort_indices]
    time_data = time_data[sort_indices]
    snr_data = snr_data[sort_indices]

    print("[DICHASUS] Data loaded and sorted by timestamp.")

    # --- 3. 格式转换 (Complex -> Tensor CPU) ---
    # [修改] 转换为 PyTorch Tensor 但不移动到 GPU
    data_tensor = torch.from_numpy(data_complex)
    data_real_imag = torch.view_as_real(data_tensor).float()   # View as Real: [N, 32, 32] -> [N, 32, 32, 2]
    data_real_imag = data_real_imag.unsqueeze(1)   # 增加 Batch 维度: [N, 1, 32, 32, 2]
    snr_tensor = torch.from_numpy(snr_data)   # [N, ant]
    snr_data = snr_tensor.unsqueeze(1)   # 增加 Batch 维度: [N, 1, ant]

    # 转换为 List 以适配 core_trace_smooth_inference 接口
    # (虽然可以直接传 tensor 并在 kernel 里切片，但为了兼容 DeepMIMO 的 list 结构，这里 split)
    truth_channel_list = [t for t in data_real_imag]
    snr_list = [t for t in snr_data]

    # --- 4. 执行核心逻辑 ---
    metrics = core_trace_smooth_inference(model, truth_channel_list, past_len, conf, device,
                                          snr_list=snr_list)

    # --- 5. 保存结果 (含轨迹信息) ---
    h5_filename = f'trace_smooth_metrics_{part_name}_pastlen{past_len}.h5'
    h5_save_path = os.path.join(save_dir, h5_filename)

    print(f"[DICHASUS] Saving metrics and trajectory to {h5_save_path}...")
    with h5py.File(h5_save_path, 'w') as f:
        # 性能指标
        f.create_dataset('nmse_list_ar', data=np.array(metrics['nmse_list_ar']))
        f.create_dataset('nmse_list_ideal', data=np.array(metrics['nmse_list_ideal']))
        f.create_dataset('nmse_snrW_list_ar', data=np.array(metrics['nmse_snrW_list_ar']))
        f.create_dataset('nmse_snrW_list_ideal', data=np.array(metrics['nmse_snrW_list_ideal']))
        f.create_dataset('corr_list_ar', data=np.array(metrics['corr_list_ar']))
        f.create_dataset('corr_list_ideal', data=np.array(metrics['corr_list_ideal']))
        # 轨迹与时间信息 (用于后续绘图)
        # 注意：这些数据的长度等于 len_trace，但指标数据的长度等于 len_trace - past_len
        # 画图时通常需要对齐
        f.create_dataset('time', data=time_data)
        f.create_dataset('loc', data=loc_data)
        # 元数据
        f.attrs['part_name'] = part_name
        f.attrs['past_len'] = past_len
        f.attrs['mean_nmse_list_ar'] = float(np.mean(metrics['nmse_list_ar']))
        f.attrs['mean_nmse_list_ideal'] = float(np.mean(metrics['nmse_list_ideal']))
        f.attrs['mean_nmse_snrW_list_ar'] = float(np.mean(metrics['nmse_snrW_list_ar']))
        f.attrs['mean_nmse_snrW_list_ideal'] = float(np.mean(metrics['nmse_snrW_list_ideal']))
        f.attrs['description'] = 'DICHASUS trace smooth results with trajectory'

    print("[DICHASUS] Done.")