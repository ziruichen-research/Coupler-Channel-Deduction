"""Visualization helpers for channel and trajectory inspection.

This module provides compact plotting utilities used during DICHASUS
preprocessing and sanity checks.
"""

import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange


def visualize_upa_phase(data, subcarrier_idx=16):
    """Visualize relative UPA phase distribution on one subcarrier.

    Args:
        data (np.ndarray): Complex channel matrix with shape [num_ant, num_subcarrier].
        subcarrier_idx (int): Subcarrier index to visualize.

    Returns:
        None
    """
    ant_response = data[:, subcarrier_idx]

    # Relative phase is physically meaningful; absolute phase is not.
    # Use antenna 0 as phase reference.
    phase_relative = ant_response * np.conj(ant_response[0])

    # Reshape flattened antenna order back to a 4x8 UPA grid.
    phase_grid = rearrange(phase_relative, '(h w) -> h w', h=4, w=8)

    plt.figure(figsize=(10, 5))
    plt.imshow(np.angle(phase_grid), cmap='twilight', interpolation='nearest')
    plt.colorbar(label='Phase [rad]')
    plt.title(f'UPA Phase Distribution (Subcarrier {subcarrier_idx})')
    plt.xlabel('Antenna Position X (Column)')
    plt.ylabel('Antenna Position Y (Row)')

    plt.show()


def plot_space_frequency_spectrum(data, title="Space-Frequency Magnitude Spectrum",
                                  xlabel="Subcarrier Index (Frequency)",
                                  ylabel="Antenna Index (Space)",
                                  use_db=False):
    """Plot magnitude spectrum over antenna (space) and subcarrier (frequency).

    Args:
        data (np.ndarray): Input matrix with shape [num_ant, num_subcarrier].
        title (str): Figure title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        use_db (bool): If True, display magnitude in dB.

    Returns:
        None

    Raises:
        ValueError: If `data` is not a 2D array.
    """

    # Optional torch->numpy conversion (kept as reference).
    # if isinstance(data, torch.Tensor):
    #     data = data.detach().cpu().numpy()

    if data.ndim != 2:
        raise ValueError(f"Input data must be a 2D matrix, but got shape {data.shape}")

    magnitude = np.abs(data)

    # dB view is useful for large dynamic range.
    if use_db:
        magnitude = 20 * np.log10(magnitude + 1e-9)
        if "dB" not in title:
            title += " (dB)"

    plt.figure(figsize=(8, 6))

    # Use origin='upper' to match matrix-style indexing.
    im = plt.imshow(magnitude, aspect='auto', cmap='jet', origin='upper')

    cbar = plt.colorbar(im)
    cbar.set_label('Magnitude (dB)' if use_db else 'Magnitude')

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # plt.xticks(np.arange(0, data.shape[1], 4))
    # plt.yticks(np.arange(0, data.shape[0], 4))

    plt.tight_layout()
    plt.show()


def visualize_spatial_snr(loc, snr, sample_ratio=1.0, title=None):
    """Visualize XY distribution colored by average SNR.

    Args:
        loc (np.ndarray): Locations with shape [N, 2] or [N, 3].
        snr (np.ndarray): SNR values with shape [N] or [N, num_ant].
        sample_ratio (float): Fraction of points to draw in (0, 1].
        title (str | None): Optional figure title.

    Returns:
        None

    Raises:
        ValueError: If `loc` and `snr` lengths do not match.
    """
    num_samples = loc.shape[0]
    if snr.shape[0] != num_samples:
        raise ValueError(f"Loc shape {loc.shape} and SNR shape {snr.shape} mismatch in length.")

    print(f"Visualizing data with {num_samples} samples available.")

    # Average SNR over antennas when needed.
    if snr.ndim == 2:
        print("Detected multi-antenna SNR, calculating mean across antennas...")
        avg_snr = np.mean(snr, axis=1)
    else:
        avg_snr = snr

    # Optional random downsampling.
    if sample_ratio < 1.0:
        sample_size = int(num_samples * sample_ratio)
        print(f"Downsampling to {sample_size} samples ({sample_ratio * 100:.1f}%)...")
        indices = np.random.choice(num_samples, sample_size, replace=False)
        loc_sampled = loc[indices]
        snr_sampled = avg_snr[indices]
    else:
        loc_sampled = loc
        snr_sampled = avg_snr

    plt.figure(figsize=(10, 8))

    sc = plt.scatter(loc_sampled[:, 0], loc_sampled[:, 1],
                     c=snr_sampled, cmap='viridis',
                     s=3, alpha=0.7)

    cbar = plt.colorbar(sc)
    cbar.set_label('Average SNR')

    plot_title = title if title else f'Spatial Distribution of SNR\n(Sampled {len(loc_sampled)} points)'
    plt.title(plot_title)
    plt.xlabel('Position X (m)')
    plt.ylabel('Position Y (m)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


def visualize_trajectory_over_time(loc, time, title="Agent Trajectory with Time", connect_points=False):
    """Visualize a trajectory with time-coded color after time sorting.

    Args:
        loc (np.ndarray): Locations with shape [N, 2] or [N, 3].
        time (np.ndarray): Timestamps with shape [N] or [N, 1].
        title (str): Figure title.
        connect_points (bool): If True, draw a path line between points.

    Returns:
        None
    """
    time = np.ravel(time)

    # Sort by timestamp to restore trajectory order after any shuffle.
    sort_idx = np.argsort(time)
    loc_sorted = loc[sort_idx]
    time_sorted = time[sort_idx]

    x = loc_sorted[:, 0]
    y = loc_sorted[:, 1]

    plt.figure(figsize=(10, 8))

    # Optional path line for continuity.
    if connect_points:
        plt.plot(x, y, c='gray', alpha=0.3, linewidth=1, linestyle='--')

    sc = plt.scatter(x, y, c=time_sorted, cmap='plasma', s=5, alpha=0.8)

    # Mark trajectory start/end.
    plt.scatter(x[0], y[0], c='green', s=150, marker='*', edgecolors='black', label='Start', zorder=10)
    plt.scatter(x[-1], y[-1], c='red', s=150, marker='X', edgecolors='black', label='End', zorder=10)

    # Relative time is kept for optional custom colorbar mapping.
    relative_time = time_sorted - time_sorted[0]
    # sc.set_array(relative_time)

    cbar = plt.colorbar(sc)
    cbar.set_label('Time (Timestamp)')

    plt.title(f"{title}\n(Total Points: {len(loc)})")
    plt.xlabel("Position X (m)")
    plt.ylabel("Position Y (m)")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

