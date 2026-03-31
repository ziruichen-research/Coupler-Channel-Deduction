import numpy as np
import torch
import torch.nn.functional as F


#TODO: 再仔细检查评价指标写得对不对

def Nmse(pred, target):
    """
    计算样本级 NMSE (Normalized Mean Square Error)
    Input/Output Shape: [Batch, ...] (任意维度，只要第0维是Batch)
    Return: [Batch] (每个样本的NMSE)
    """
    # 保持 Batch 维度 (axis 0)，将其余维度展平
    b = target.shape[0]
    target_flat = target.reshape(b, -1)
    pred_flat = pred.reshape(b, -1)

    # 计算误差范数平方: ||x - \hat{x}||^2
    # axis=1 表示对每个样本的所有特征求范数
    error_sq = np.linalg.norm(target_flat - pred_flat, axis=1) ** 2

    # 计算真实值范数平方: ||x||^2
    power_sq = np.linalg.norm(target_flat, axis=1) ** 2

    # 避免分母为0
    return error_sq / (power_sq + 1e-10)


def Nmse_ant_snr_weighted(pred, target, snrdb, clip_min=None, clip_max=None, **kwargs):
    """
    计算基于天线SNR加权的样本级 NMSE

    参数:
    - pred: [Batch, Antenna, Freq, 2]
    - target: [Batch, Antenna, Freq, 2]
    - snrdb: [Batch, Antenna]

    返回:
    - [Batch] (每个样本的加权NMSE)
    """
    # 1. 限制 SNR 范围并转换为线性权重
    snrdb_clipped = snrdb
    if clip_min is not None or clip_max is not None:
        snrdb_clipped = np.clip(snrdb, a_min=clip_min, a_max=clip_max)
    snr_linear = 10.0 ** (snrdb_clipped / 10.0)

    # 计算权重，在天线维度 (axis=2) 上归一化，使得每根天线的平均权重为 1
    # keepdims=True 保持维度为 [Batch, 1] 方便广播
    mean_snr = np.mean(snr_linear, axis=-1, keepdims=True)
    weights = snr_linear / (mean_snr + 1e-10)

    # 2. 维度对齐 (Broadcasting)
    # 将 weights 从 [Batch, Antenna] 扩展为 [Batch, Antenna, 1, 1]
    weights = np.expand_dims(np.expand_dims(weights, axis=-1), axis=-1)

    # 3. 计算加权误差能量 (Numerator)
    # (target - pred)**2 会逐元素计算实部和虚部的平方差
    error_sq = (target - pred) ** 2
    weighted_error_sq = error_sq * weights
    # 对除 Batch 外的所有维度求和: axis=(1, 2, 3)
    sum_weighted_error = np.sum(weighted_error_sq, axis=(1, 2, 3))

    # 4. 计算加权信号能量 (Denominator)
    # 同样地，分母也需要加权，以保持 NMSE 定义的正确性
    power_sq = target ** 2
    weighted_power_sq = power_sq * weights
    sum_weighted_power = np.sum(weighted_power_sq, axis=(1, 2, 3))

    # 5. 计算最终的加权 NMSE
    return sum_weighted_error / (sum_weighted_power + 1e-10)


def Corr(pred, target):
    """
    计算样本级相关系数 (Correlation Coefficient)
    Input/Output Shape: [Batch, Ant, Car, 2]
    Return: [Batch] (每个样本的相关系数)
    """
    # 确保输入是 [Batch, Ant, Car, 2]
    # 如果输入已经是复数形式，逻辑需要调整；这里假设是实部虚部最后一维分离的形式

    # 1. 转换为复数: [Batch, Ant, Car]
    input_c = target[..., 0] + 1j * target[..., 1]
    output_c = pred[..., 0] + 1j * pred[..., 1]

    # Formula: (1/K) * sum_k [ |a_k^H * b_k| / (||a_k|| * ||b_k||) ]
    # 其中 k 是载波(Car)索引, a_k, b_k 是天线(Ant)向量

    # 2. 计算分子: |a^H * b|
    # 对 Ant 维度 (axis=1) 做内积: sum(input * conj(output))
    # 结果 shape: [Batch, Car]
    inner_prod = np.sum(input_c * np.conj(output_c), axis=1)
    numerator = np.abs(inner_prod)

    # 3. 计算分母: ||a|| * ||b||
    # 对 Ant 维度求范数
    norm_in = np.linalg.norm(input_c, axis=1)  # [Batch, Car]
    norm_out = np.linalg.norm(output_c, axis=1)  # [Batch, Car]
    denominator = norm_in * norm_out + 1e-10

    # 4. 计算每个载波的相关系数
    corr_per_car = numerator / denominator  # [Batch, Car]

    # 5. 对载波取平均，得到每个样本的相关系数
    return np.mean(corr_per_car, axis=1)  # [Batch]


def mse_weighted_by_ant_snr(pred, target, snrdb,
                            clip_min=None, clip_max=None,
                            use_softmax=False, temperature=1.0):
    """
    计算基于天线SNR加权的MSE损失函数。

    参数:
    - pred: 预测的当前CSI，维度 [batch, antenna, freq, 2]
    - target: 真实的当前CSI，维度 [batch, antenna, freq, 2]
    - snrdb: 当前CSI的各天线的信噪比 (dB)，维度 [batch, antenna]
    - clip_min, clip_max: 对SNR(dB)的截断范围，防止异常值导致梯度爆炸
    - use_softmax: 是否使用 Softmax 进行非线性加权映射
    - temperature: Softmax 的温度系数，T越大权重分布越均匀，T越小越集中于高SNR天线
    """
    # 1. 异常值截断 (Clipping)
    # 对 dB 域的 SNR 进行截断，防止转换到线性域时数值溢出
    snrdb_clipped = snrdb
    if clip_min is not None or clip_max is not None:
        snrdb_clipped = torch.clamp(snrdb, min=clip_min, max=clip_max)

    # 2. 将 SNR 从 dB 转换为线性尺度 (Linear Scale)
    # 公式: SNR_linear = 10 ^ (SNR_dB / 10)
    snr_linear = 10.0 ** (snrdb_clipped / 10.0)

    # 获取天线数量，用于后续权重归一化
    num_antennas = snrdb.shape[-1]

    # 3. 计算权重 (Weights)
    if use_softmax:
        # 使用 Softmax 时，直接在 Linear 域除以温度系数。
        # 注意：为了数值稳定，可以减去最大值（PyTorch的softmax底层已处理，但除以T要小心）
        weights = F.softmax(snr_linear / temperature, dim=-1)
        # 乘以天线数，使权重均值为 1 (原本Softmax后总和为1)
        weights = weights * num_antennas
    else:
        # 线性加权：除以该Batch/Seq下所有天线的平均线性SNR，使权重均值为 1
        # 加上 1e-8 防止除以 0 的情况（比如所有天线SNR都极低）
        mean_snr = torch.mean(snr_linear, dim=-1, keepdim=True)
        weights = snr_linear / (mean_snr + 1e-8)

    # 4. 维度对齐 (Broadcasting)
    # 当前 weights 维度: [batch, seq, antenna]
    # 目标 pred 维度: [batch, seq, antenna, freq, 2]
    # 在最后增加两个维度，使其变成 [batch, seq, antenna, 1, 1] 以便逐元素相乘
    weights = weights.unsqueeze(-1).unsqueeze(-1)

    # 5. 计算加权 MSE
    # (pred - target) ** 2 会同时计算实部和虚部的平方误差
    squared_error = (pred - target) ** 2

    # 应用权重：高 SNR 天线的误差会被放大，低 SNR 的误差会被缩小
    weighted_squared_error = squared_error * weights

    # 6. 求所有维度的平均值得到最终 Loss
    loss = torch.mean(weighted_squared_error)

    return loss
