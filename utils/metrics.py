import numpy as np
import torch
import torch.nn.functional as F


def Nmse(pred, target):
    """Compute sample-wise NMSE for arrays shaped as [batch, ...]."""
    b = target.shape[0]
    target_flat = target.reshape(b, -1)
    pred_flat = pred.reshape(b, -1)

    error_sq = np.linalg.norm(target_flat - pred_flat, axis=1) ** 2
    power_sq = np.linalg.norm(target_flat, axis=1) ** 2
    return error_sq / (power_sq + 1e-10)


def Nmse_ant_snr_weighted(pred, target, snrdb, clip_min=None, clip_max=None, **kwargs):
    """
    Compute sample-wise NMSE with antenna-level SNR weighting.

    Args:
        pred: [B, Ant, Freq, 2]
        target: [B, Ant, Freq, 2]
        snrdb: [B, Ant]
    Returns:
        [B]
    """
    snrdb_clipped = snrdb
    if clip_min is not None or clip_max is not None:
        snrdb_clipped = np.clip(snrdb, a_min=clip_min, a_max=clip_max)
    snr_linear = 10.0 ** (snrdb_clipped / 10.0)

    # Normalize per-sample antenna weights to mean 1.
    mean_snr = np.mean(snr_linear, axis=-1, keepdims=True)
    weights = snr_linear / (mean_snr + 1e-10)
    weights = np.expand_dims(np.expand_dims(weights, axis=-1), axis=-1)

    weighted_error_sq = (target - pred) ** 2 * weights
    sum_weighted_error = np.sum(weighted_error_sq, axis=(1, 2, 3))

    weighted_power_sq = target ** 2 * weights
    sum_weighted_power = np.sum(weighted_power_sq, axis=(1, 2, 3))

    return sum_weighted_error / (sum_weighted_power + 1e-10)


def Corr(pred, target):
    """Compute sample-wise complex correlation for [B, Ant, Car, 2] tensors."""
    input_c = target[..., 0] + 1j * target[..., 1]
    output_c = pred[..., 0] + 1j * pred[..., 1]

    # Per-carrier |a^H b| / (||a|| ||b||), then average over carriers.
    inner_prod = np.sum(input_c * np.conj(output_c), axis=1)
    numerator = np.abs(inner_prod)

    norm_in = np.linalg.norm(input_c, axis=1)
    norm_out = np.linalg.norm(output_c, axis=1)
    denominator = norm_in * norm_out + 1e-10

    corr_per_car = numerator / denominator
    return np.mean(corr_per_car, axis=1)


def mse_weighted_by_ant_snr(pred, target, snrdb,
                            clip_min=None, clip_max=None,
                            use_softmax=False, temperature=1.0):
    """
    Compute antenna-SNR-weighted MSE loss.

    Args:
        pred: [B, Seq, Ant, Freq, 2]
        target: [B, Seq, Ant, Freq, 2]
        snrdb: [B, Seq, Ant] in dB
        clip_min, clip_max: Optional SNR clipping range in dB
        use_softmax: Use softmax weighting instead of linear normalization
        temperature: Softmax temperature
    """
    snrdb_clipped = snrdb
    if clip_min is not None or clip_max is not None:
        snrdb_clipped = torch.clamp(snrdb, min=clip_min, max=clip_max)

    snr_linear = 10.0 ** (snrdb_clipped / 10.0)
    num_antennas = snrdb.shape[-1]

    if use_softmax:
        weights = F.softmax(snr_linear / temperature, dim=-1)
        # Keep average weight around 1 (softmax sums to 1).
        weights = weights * num_antennas
    else:
        mean_snr = torch.mean(snr_linear, dim=-1, keepdim=True)
        weights = snr_linear / (mean_snr + 1e-8)

    # Align to [B, Seq, Ant, Freq, 2].
    weights = weights.unsqueeze(-1).unsqueeze(-1)
    weighted_squared_error = (pred - target) ** 2 * weights
    loss = torch.mean(weighted_squared_error)

    return loss
