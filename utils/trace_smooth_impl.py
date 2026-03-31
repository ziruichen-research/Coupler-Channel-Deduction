import os

import h5py
import numpy as np
import torch
import hdf5storage
from einops import rearrange
from tqdm import tqdm

from utils.metrics import Nmse, Corr


# --- 辅助函数 (保持原样，仅增加 path 参数) ---
def channel_index(x_index, y_index, path, high=181):
    # 从mat文件返回信道数据,注意，这里的x,y是plot的x轴和y轴，与deepmimo的x轴和y轴正好转置
    index = x_index * high + y_index
    # 注意：这里需要确保你的数据目录下有 mat 文件夹
    mat_path = os.path.join(path, 'mat', 'DeepMIMO_dataset_' + str(index) + '.mat')
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Mat file not found: {mat_path}")
    channel_data_mat = hdf5storage.loadmat(mat_path)
    ans = channel_data_mat['DeepMIMO_dataset']
    return ans


def run_trace_smooth_deepmimoO1(model, past_len, conf, device, save_dir):
    """
    封装 test_trace_smooth.py 的核心逻辑
    """
    # --- 1. 参数设置 (对应原代码的开头配置) ---
    # 原代码: path='/home/...'
    path = conf['data_dirs'][0]

    # 原代码: input_ant_size=4 ...
    input_ant_size = conf['input_ant_size']
    input_car_size = conf['input_car_size']
    ant_size = conf['num_ant']
    car_size = conf['num_car']

    step_ant = ant_size // input_ant_size
    step_car = car_size // input_car_size

    # 原代码: past_len=8 (虽然原代码是在循环前定义的，这里提出来)
    # past_len = 8

    # --- 2. 加载 Trace 坐标 (对应原代码的 np.load) ---
    # 原代码: x_trace=np.load('x_trace_smooth.npy')
    try:
        x_trace = np.load(os.path.join(path, 'x_trace_smooth.npy'))
        y_trace = np.load(os.path.join(path, 'y_trace_smooth.npy'))
    except FileNotFoundError:
        print(f"[TraceSmooth] Error: x_trace_smooth.npy or y_trace_smooth.npy not found in {path}")
        return

    len_trace = x_trace.shape[0]
    print(f"[TraceSmooth] Trace length: {len_trace}")

    # --- 3. 准备 Truth Channel List (对应原代码的第一个 for 循环) ---
    truth_channel_list = []
    print("[TraceSmooth] Loading ground truth from .mat files...")

    for i in range(len_trace):
        reverse_x_index = x_trace[i]
        y_index = y_trace[i]
        x_index = 899 - reverse_x_index

        # 调用辅助函数
        channel_data = channel_index(x_index, y_index, path)

        channel_data = torch.tensor(channel_data)  # numpy转tensor
        channel_data = rearrange(channel_data, '(b ant) car c -> b ant car c', b=1)  # 加上个batch

        # 原代码: channel_data=channel_data.cuda().float()
        channel_data = channel_data.to(device).float()
        channel_data = channel_data * 10000
        truth_channel_list.append(channel_data)

    # --- 4. 核心推理循环 (对应原代码的第二个 for 循环) ---
    # 变量初始化保持一致
    deduction_channel_list = []
    nmse_list = []
    rou_list = []
    nmse_list_truth = []
    rou_list_truth = []

    model.eval()
    print("[TraceSmooth] Running inference loop...")

    for i in tqdm(range(len_trace)):
        if (i < past_len):
            deduction_channel_list.append(truth_channel_list[i])
            # 补充：为了保持列表长度一致，前几帧也填入0或占位符，方便绘图
            # 原代码这里其实没有 append nmse，导致 nmse_list 长度比 len_trace 短
            # 为了完全复刻，这里不 append，保持原样
        else:
            # === Part 1: 自回归 (Error Propagation) ===
            h_now = truth_channel_list[i]
            h_now_input = h_now[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                                0:step_car * (input_car_size - 1) + 1:step_car, :]

            # 准备 Past (来自 deduction_channel_list)
            h_past_list = deduction_channel_list[i - past_len:i]
            h_past_input = torch.stack(h_past_list, dim=0)
            h_past_input = rearrange(h_past_input, 'len b ant car c -> b len ant car c')

            # 推理
            h_now_output = model(h_now_input, h_past_input)

            # 计算指标
            nmse = Nmse(h_now_output.cpu().detach().numpy(), h_now.cpu().detach().numpy())
            rou = Corr(h_now_output.cpu().detach().numpy(), h_now.cpu().detach().numpy())

            # 更新列表
            deduction_channel_list.append(h_now_output)
            nmse_list.append(np.mean(nmse))  # Nmse 返回的是 batch 向量，这里取 mean 变成标量
            rou_list.append(np.mean(rou))

            # # 打印日志
            # if ((i - past_len) % 100 == 0):
            #     print(f"Step {i}: AR NMSE {np.mean(nmse):.5f}, Corr {np.mean(rou):.5f}")

            # === Part 2: 理想情况 (Ideal / Truth) ===
            # 准备 Now (同上)
            h_now = truth_channel_list[i]  # 其实没变
            h_now_input = h_now[:, 0:step_ant * (input_ant_size - 1) + 1:step_ant,
                                0:step_car * (input_car_size - 1) + 1:step_car, :]

            # 准备 Past (来自 truth_channel_list)
            h_past_list = truth_channel_list[i - past_len:i]
            h_past_input = torch.stack(h_past_list, dim=0)
            h_past_input = rearrange(h_past_input, 'len b ant car c -> b len ant car c')

            # 推理
            h_now_output = model(h_now_input, h_past_input)

            # 计算指标
            nmse = Nmse(h_now_output.cpu().detach().numpy(), h_now.cpu().detach().numpy())
            rou = Corr(h_now_output.cpu().detach().numpy(), h_now.cpu().detach().numpy())

            nmse_list_truth.append(np.mean(nmse))
            rou_list_truth.append(np.mean(rou))

    # --- 5. 结果保存 (替换原代码的 matplotlib 画图部分) ---
    # 原代码计算了均值并打印，我们也做
    if len(nmse_list) > 0:
        print(f"Average AR NMSE: {sum(nmse_list) / len(nmse_list)}")
        print(f"Average Ideal NMSE: {sum(nmse_list_truth) / len(nmse_list)}")

    # # 保存数据供后续绘图脚本使用
    # # 保存为 numpy 数组
    # np.save(os.path.join(save_dir, 'nmse_list_ar.npy'), np.array(nmse_list))
    # np.save(os.path.join(save_dir, 'nmse_list_ideal.npy'), np.array(nmse_list_truth))
    # # 也可以保存 rou_list
    # np.save(os.path.join(save_dir, 'corr_list_ar.npy'), np.array(rou_list))
    # np.save(os.path.join(save_dir, 'corr_list_ideal.npy'), np.array(rou_list_truth))
    #
    # print(f"[TraceSmooth] Data saved to {save_dir}")

    # 设置保存路径
    h5_filename = 'trace_smooth_metrics.h5'
    h5_path = os.path.join(save_dir, h5_filename)

    print(f"[TraceSmooth] Saving metrics to {h5_path}...")

    try:
        # 使用 'w' 模式打开，如果文件存在则覆盖
        with h5py.File(h5_path, 'w') as f:
            # 创建数据集 (Datasets)
            # 必须将 list 转换为 numpy array 才能存入 HDF5
            f.create_dataset('nmse_list_ar', data=np.array(nmse_list))
            f.create_dataset('nmse_list_ideal', data=np.array(nmse_list_truth))
            f.create_dataset('corr_list_ar', data=np.array(rou_list))
            f.create_dataset('corr_list_ideal', data=np.array(rou_list_truth))

            # (可选) 你可以添加一些属性来描述文件
            f.attrs['description'] = 'Results from run_trace_smooth_deepmimoO1'
            f.attrs['trace_length'] = len_trace

        print(f"[TraceSmooth] Data successfully saved to HDF5.")

    except Exception as e:
        print(f"[TraceSmooth] Failed to save HDF5 file: {e}")