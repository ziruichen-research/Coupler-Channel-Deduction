import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.get_logger().setLevel('ERROR')

import json
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm
import h5py
from data_prepare.visualization import visualize_trajectory_over_time

# https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-ca0x/


def downsampling_DICHASUS_dataset(raw_dataset_paths, offset_path, assignments_matrix,
                                  save_path, filename,
                                  effective_bandwidth=1.0,
                                  downsampling_factor=32,
                                  downsampling_method='fir',
                                  is_ant_power_norm=False):
    # 提取所有 Part 的名称，用于后续保存 Metadata
    # 例如 ['.../dichasus-ca01_part1.tfrecords', '.../part2...'] -> ['dichasus-ca01_part1', 'dichasus-ca01_part2']
    part_names = [os.path.splitext(os.path.basename(p))[0] for p in raw_dataset_paths]
    source_name_str = ",".join(part_names)
    print(f"Global processing plan: {source_name_str}")
    print(f"Config: Bandwidth={effective_bandwidth}, Method={downsampling_method.upper()}, Factor={downsampling_factor}")

    # 1. 预处理定义 (这部分对所有 Part 通用)
    feature_description = {
        'csi': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'gt-interp-age-tachy': tf.io.FixedLenFeature([], tf.float32, default_value=0),
        'pos-tachy': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'snr': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'time': tf.io.FixedLenFeature([], tf.float32, default_value=0),
    }

    def record_parse_function(proto):
        record = tf.io.parse_single_example(proto, feature_description)
        csi = tf.ensure_shape(tf.io.parse_tensor(record['csi'], out_type=tf.float32), (32, 1024, 2))
        csi = tf.signal.fftshift(csi, axes=1)
        csi = tf.complex(csi[:, :, 0], csi[:, :, 1])
        pos = tf.ensure_shape(tf.io.parse_tensor(record['pos-tachy'], out_type=tf.float64), (3))
        pos = tf.cast(pos, tf.float32)
        snr = tf.io.parse_tensor(record['snr'], out_type=tf.float32)
        snr = tf.ensure_shape(snr, (32,))
        time = tf.ensure_shape(record["time"], ())
        return csi, pos, snr, time

    # 读取校准参数
    with open(offset_path, 'r') as file:
        offsets = json.load(file)

    def apply_calibration(csi, pos, snr, time):
        sto_offset = tf.tensordot(tf.constant(offsets['sto']), 2 * np.pi * tf.range(tf.shape(csi)[1], dtype=np.float32) / tf.cast(tf.shape(csi)[1], np.float32), axes=0)
        cpo_offset = tf.tensordot(tf.constant(offsets['cpo']), tf.ones(tf.shape(csi)[1], dtype=np.float32), axes=0)
        csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))
        return csi, pos, snr, time

    # 定义天线重排索引
    assignments = np.ravel(assignments_matrix)
    ftype = 'fir'

    # 用于收集最终合并后的数据列表
    final_data_chunks = []
    final_loc_chunks = []
    final_snr_chunks = []
    final_time_chunks = []
    final_source_idx_chunks = []

    # 对 dataset_paths 进行循环，逐个 Part 处理
    for part_idx, single_path in enumerate(raw_dataset_paths):
        current_part_name = part_names[part_idx]
        print(f"\n>>> Processing Part {part_idx + 1}/{len(raw_dataset_paths)}: {current_part_name}")

        # 针对当前文件创建 dataset
        raw_dataset = tf.data.TFRecordDataset([single_path]).map(record_parse_function)
        raw_dataset = raw_dataset.map(apply_calibration)

        # 临时列表，用于存储当前 Part 的数据
        part_data_list = []
        part_loc_list = []
        part_snr_list = []
        part_time_list = []

        # 处理当前 Part 的循环
        for i, (csi, pos, snr, time) in enumerate(tqdm(raw_dataset, desc=f"Parsing {current_part_name}")):
            # if i >= 25: break

            data = csi.numpy().astype(np.complex64)
            loc = pos.numpy().astype(np.float32)
            snr = snr.numpy().astype(np.float32)
            time = time.numpy().astype(np.float32)

            # --- [新增步骤] 截取有效频带 (Effective Bandwidth) ---
            if effective_bandwidth < 1.0:
                num_sc = data.shape[1]  # 通常是 1024
                num_effective = int(num_sc * effective_bandwidth)
                start_idx = (num_sc - num_effective) // 2
                # 截取中心部分
                data = data[:, start_idx: start_idx + num_effective]

            # B. 天线重排
            data = data[assignments, :]
            snr = snr[assignments]

            # C. 功率校准
            global_power = np.mean(np.abs(data) ** 2)
            ant_power = np.mean(np.abs(data) ** 2, axis=1, keepdims=True)
            data_power_calibrated = (data / np.sqrt(ant_power)) * np.sqrt(global_power)

            # A. 降采样
            data_input = data_power_calibrated if is_ant_power_norm else data
            if downsampling_method == 'fir':
                # 方式 1: FIR 滤波器抗混叠 + 抽取 (推荐，保留低频信息)
                data_real = sp.signal.decimate(data_input.real, downsampling_factor, axis=1, zero_phase=True, ftype=ftype)
                data_imag = sp.signal.decimate(data_input.imag, downsampling_factor, axis=1, zero_phase=True, ftype=ftype)
                data_save = data_real + 1j * data_imag
            elif downsampling_method == 'direct':
                # 方式 2: 直接切片等间隔采样 (简单，对应子载波 0, 32, 64...)
                # data_input shape: [32, 1024] -> [32, 32]
                data_save = data_input[:, ::downsampling_factor]
            else:
                raise ValueError(f"Unknown downsampling method: {downsampling_method}. Use 'fir' or 'direct'.")

            part_data_list.append(data_save)
            part_loc_list.append(loc)
            part_snr_list.append(snr)
            part_time_list.append(time)

        # 当前 Part 读取完毕，堆叠为 numpy 数组
        part_data = np.stack(part_data_list, axis=0)
        part_loc = np.stack(part_loc_list, axis=0)
        part_snr = np.stack(part_snr_list, axis=0)
        part_time = np.stack(part_time_list, axis=0)

        # [关键修改]：在 Part 内部按时间排序
        print(f"Sorting {current_part_name} by timestamp...")
        sort_idx = np.argsort(part_time)

        part_data = part_data[sort_idx]
        part_loc = part_loc[sort_idx]
        part_snr = part_snr[sort_idx]
        part_time = part_time[sort_idx]
        part_source_idx = np.full((part_data.shape[0],), part_idx, dtype=np.int8)

        # 将排序好的当前 Part 数据加入最终列表
        final_data_chunks.append(part_data)
        final_loc_chunks.append(part_loc)
        final_snr_chunks.append(part_snr)
        final_time_chunks.append(part_time)
        final_source_idx_chunks.append(part_source_idx)

    # 循环结束后，将所有 Part 拼接在一起 (Concatenate)
    print("\nConcatenating all parts...")
    data = np.concatenate(final_data_chunks, axis=0)
    loc = np.concatenate(final_loc_chunks, axis=0)
    snr = np.concatenate(final_snr_chunks, axis=0)
    time = np.concatenate(final_time_chunks, axis=0)
    source_indices = np.concatenate(final_source_idx_chunks, axis=0)

    print(f"All processing done. Final Shape: {data.shape}")

    # --- 保存为 HDF5 ---
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)

    print(f"Saving to HDF5: {filepath} ...")
    with h5py.File(filepath, 'w') as f:
        # [关键修改] 动态计算 Chunk 大小
        # 使用实际的数据维度 data.shape[2] 作为 chunk 的最后一维，而不是写死 32
        chunk_shape = (1, 32, data.shape[2])
        f.create_dataset('data', data=data, chunks=chunk_shape)

        f.create_dataset('loc', data=loc)
        f.create_dataset('snr', data=snr)
        f.create_dataset('time', data=time)
        f.create_dataset('source_indices', data=source_indices)

        f.attrs['part_names'] = source_name_str
        f.attrs['downsampling_method'] = downsampling_method
        f.attrs['effective_bandwidth'] = effective_bandwidth # 记录参数
        f.attrs['downsampling_factor'] = downsampling_factor

    print(f'Saved sorted dataset to {filepath}')
    print(f'Source composition: {source_name_str}')

    # 简单的轨迹可视化（验证拼接和排序是否正确）
    # visualize_trajectory_over_time(loc, time, title=f"Trajectory: {source_name_str}")


if __name__ == '__main__':
    # # ca0x
    # is_ant_power_norm = False
    # effective_bandwidth = 0.5
    # downsampling_factor = 16
    # downsampling_method = 'direct'
    # raw_dataset_paths = [f'../data_ssd/DICHASUS/src/ca0x/dichasus-ca01_part{k + 1}.tfrecords' for k in range(4)]
    # offset_path = '../data_ssd/DICHASUS/src/ca0x/reftx-offsets-dichasus-ca0x.json'
    # assignments_matrix = [
    #     [0, 13, 31, 29, 3, 7, 1, 12],
    #     [30, 26, 21, 25, 24, 8, 22, 15],
    #     [28, 5, 10, 14, 6, 2, 16, 18],
    #     [19, 4, 23, 17, 20, 11, 9, 27]
    # ]
    # save_path = '../data_ssd/DICHASUS/src_h5/ca0x_DirectSample512to32'
    # filename = f'data_ca01_ant_norm.h5' if is_ant_power_norm else f'data_ca01.h5'

    # 015x
    is_ant_power_norm = False
    effective_bandwidth = 0.5
    downsampling_factor = 16
    downsampling_method = 'direct'
    # idx_list = ['0152', '0153', '0154', '0155', '0157']   # 没有 0158
    idx_list = ['0158']  # 只有 0158
    raw_dataset_paths = [f'../data_ssd/DICHASUS/src/015x/dichasus-{idx}.tfrecords' for idx in idx_list]
    offset_path = '../data_ssd/DICHASUS/src/015x/reftx-offsets-dichasus-015x.json'
    assignments_matrix = [
        [28, 5, 10, 14, 6, 2, 16, 18],
        [19, 4, 23, 17, 20, 11, 9, 27],
        [31, 29, 0, 13, 1, 12, 3, 7],
        [30, 26, 21, 25, 22, 15, 24, 8]

    ]
    save_path = '../data_ssd/DICHASUS/src_h5/0152-0157_DirectSample512to32_v2'
    # filename = f'data_0152-0157_ant_norm.h5' if is_ant_power_norm else f'data_0152-0157.h5'
    filename = f'data_0158_ant_norm.h5' if is_ant_power_norm else f'data_0158.h5'

    downsampling_DICHASUS_dataset(raw_dataset_paths, offset_path, assignments_matrix,
                                  save_path, filename,
                                  effective_bandwidth=effective_bandwidth,
                                  downsampling_factor=downsampling_factor,
                                  downsampling_method=downsampling_method,
                                  is_ant_power_norm=is_ant_power_norm)



