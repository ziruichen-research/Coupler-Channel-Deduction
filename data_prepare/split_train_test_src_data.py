import os
import h5py
import numpy as np
from tqdm import tqdm


def create_split_dataset(
        source_h5_path,
        output_dir,
        num_train=50000,
        num_test=50000,
        seed=42
):
    """
    加载全量 H5 数据，打乱后分割为 Train/Test HDF5 文件。
    会自动保留原始文件中的 time, source_indices 和 attributes。

    Args:
        source_h5_path (str): 源 .h5 文件路径
        output_dir (str): 输出 H5 文件的目录
        num_train (int): 训练集样本数量
        num_test (int): 测试集样本数量
        seed (int): 随机种子，保证可复现性
    """

    # 1. 加载原始数据
    print(f"Loading source data from: {source_h5_path} ...")
    if not os.path.exists(source_h5_path):
        raise FileNotFoundError(f"Source file not found: {source_h5_path}")

    with h5py.File(source_h5_path, 'r') as f_src:
        # 将数据加载到内存中以加快 Shuffle 速度
        # 假设内存足够 (对于 ~11w 样本，占用约 1GB RAM，通常没问题)
        print("Reading data into memory...")
        raw_data = f_src['data'][:]  # [N, 32, 32]
        raw_loc = f_src['loc'][:]    # [N, 3]
        raw_snr = f_src['snr'][:]    # [N, 32]

        # [修改] 为了后续排序，这里必须确保 time 和 source_indices 存在
        raw_time = f_src['time'][:]
        raw_source_indices = f_src['source_indices'][:]

        # 读取全局属性 (如 part_names)
        src_attrs = dict(f_src.attrs)

    total_samples = raw_data.shape[0]
    print(f"Total samples loaded: {total_samples}")

    # 2. 检查数据量是否足够
    if num_train + num_test > total_samples:
        raise ValueError(
            f"Requested samples (Train:{num_train} + Test:{num_test} = {num_train + num_test}) "
            f"exceed total available samples ({total_samples})!"
        )

    # 3. 生成随机索引并打乱
    print(f"Shuffling data with seed {seed}...")
    np.random.seed(seed)
    # 生成一个全量的随机排列索引
    permuted_indices = np.random.permutation(total_samples)

    # 4. 定义分割区间
    # 训练集：取前 num_train 个
    train_indices = permuted_indices[:num_train]
    # 测试集：取紧接着的 num_test 个
    test_indices = permuted_indices[num_train: num_train + num_test]

    # 剩余的数据（如果有的话）会被丢弃或保留备用
    print(f"Split Summary:")
    print(f"  - Train samples: {len(train_indices)}")
    print(f"  - Test samples:  {len(test_indices)}")
    print(f"  - Unused samples: {total_samples - (num_train + num_test)}")

    # 5. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 6. 定义保存函数
    def save_to_h5(indices, split_name):
        filename = f"{split_name}.h5"
        filepath = os.path.join(output_dir, filename)

        # --- [修改] 排序逻辑开始 ---
        print(f"Sorting {split_name} indices by Source -> Time before saving...")

        # 提取当前 split 选中的样本对应的 source 和 time
        # indices 是随机打乱的，我们现在要根据它们对应的值找回物理顺序
        current_sources = raw_source_indices[indices]
        current_times = raw_time[indices]

        # 使用 lexsort 进行联合排序，np.lexsort((secondary_key, primary_key))
        # 我们希望先按 source_indices (来源) 排序，如果来源相同，再按 time (时间戳) 排序
        sort_order = np.lexsort((current_times, current_sources))

        # 使用排序后的顺序重新排列 indices
        sorted_indices = indices[sort_order]
        # --- [修改] 排序逻辑结束 ---

        print(f"Writing {split_name} dataset to {filepath} ...")

        with h5py.File(filepath, 'w') as f:
            # --- 传递原始属性 (Attributes) ---
            for k, v in src_attrs.items():
                f.attrs[k] = v

            # --- 保存 CSI Data ---
            f.create_dataset(
                'data',
                data=raw_data[sorted_indices],
                dtype='complex64',
                # compression='gzip',
                # compression_opts=4  # 压缩级别 0-9
                chunks=(1, 32, 32)
            )
            # --- 保存 Location ---
            f.create_dataset('loc', data=raw_loc[sorted_indices], dtype='float32')
            # --- 保存 SNR ---
            f.create_dataset('snr', data=raw_snr[sorted_indices], dtype='float32')
            # --- 保存 Time ---
            f.create_dataset('time', data=raw_time[sorted_indices], dtype='float32')
            # --- 保存 Source Indices (对应哪个轨迹 part) ---
            f.create_dataset('source_indices', data=raw_source_indices[sorted_indices], dtype='int8')
            # --- 保存原始索引，便于追溯 ---
            f.create_dataset('original_indices', data=sorted_indices, dtype='int64')

        print(f"Finished writing {split_name}.")

    # 7. 执行保存
    save_to_h5(train_indices, 'train')
    save_to_h5(test_indices, 'test')

    print("\nAll done!")


if __name__ == '__main__':
    # 配置参数
    IS_ANT_POWER_NORM = False  # 对应你上一步生成的文件名
    # 根据之前的脚本逻辑确定文件名
    # input_filename = 'data_ca01_ant_norm.npz' if IS_ANT_POWER_NORM else 'data_ca01.npz'
    # source_path = os.path.join('../data_ssd/DICHASUS/src_npz/ca0x', input_filename)

    # source_path = '../data_ssd/DICHASUS/src_h5/ca0x_DirectSample512to32/data_ca01.h5'
    source_path = '../data_ssd/DICHASUS/src_h5/0152-0157_DirectSample512to32/data_0152-0157.h5'

    # 输出目录
    # output_base_dir = '../data_ssd/DICHASUS/h5_CD/ca0x_DirectSample512to32'
    output_base_dir = '../data_ssd/DICHASUS/h5_CD/0152-0157_DirectSample512to32_v2'

    # 如果做了归一化，建议分开存，以免混淆
    if IS_ANT_POWER_NORM:
        output_base_dir += '_norm'

    # 检查文件是否存在，避免路径错误
    if not os.path.exists(source_path):
        print(f"[Warning] Source path not found: {source_path}")
        print("Please check the input filename and directory.")
    else:
        create_split_dataset(
            source_h5_path=source_path,
            output_dir=output_base_dir,
            num_train=36000,
            num_test=36000,
            seed=2026  # 设置一个固定的种子
        )




