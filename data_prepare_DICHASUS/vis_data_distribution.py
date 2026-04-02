"""Visualize spatial SNR distribution from NPZ/HDF5 datasets.

This script is a lightweight inspection utility for plotting XY location points
colored by per-sample average SNR.
"""

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
    """Plot XY positions with color mapped to average SNR.

    Args:
        loc (np.ndarray): Locations with shape [N, 2] or [N, 3].
        snr (np.ndarray): SNR values with shape [N] or [N, num_ant].
        sample_ratio (float): Fraction of points to draw in (0, 1].
        vmin (float | None): Minimum colormap value.
        vmax (float | None): Maximum colormap value.
        is_save (bool): If True, save the figure to disk.
        save_path (str | None): Output directory for figure files.
        fig_name (str | None): Figure name without extension.

    Returns:
        None

    Raises:
        ValueError: If `loc` and `snr` lengths do not match.
    """
    num_samples = loc.shape[0]
    if snr.shape[0] != num_samples:
        raise ValueError(f"Loc shape {loc.shape} and SNR shape {snr.shape} mismatch in length.")

    print(f"Visualizing data with {num_samples} samples available.")

    # Average SNR over antennas when input shape is [N, num_ant].
    if snr.ndim == 2:
        print("Detected multi-antenna SNR, calculating mean across antennas...")
        avg_snr = np.mean(snr, axis=1)
    else:
        avg_snr = snr

    # Optional random subsampling for faster plotting.
    if sample_ratio < 1.0:
        sample_size = int(num_samples * sample_ratio)
        print(f"Downsampling to {sample_size} samples ({sample_ratio * 100:.1f}%)...")
        indices = np.random.choice(num_samples, sample_size, replace=False)
        loc_sampled = loc[indices]
        snr_sampled = avg_snr[indices]
    else:
        loc_sampled = loc
        snr_sampled = avg_snr

    plt.figure(figsize=(8, 6))

    sc = plt.scatter(loc_sampled[:, 0], loc_sampled[:, 1],
                     c=snr_sampled, cmap='viridis',
                     s=6, alpha=0.7, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(sc)
    cbar.set_label('Average SNR (dB)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.xlabel('X (m)', fontsize=16)
    plt.ylabel('Y (m)', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
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
    """Load `loc` and `snr` from either `.npz` or `.h5` files.

    Args:
        file_path (str): Input dataset file path.

    Returns:
        tuple[np.ndarray, np.ndarray]: `(loc, snr)` arrays.

    Raises:
        FileNotFoundError: If `file_path` does not exist.
        ValueError: If file extension is not supported.
    """
    print(f"Loading data from {file_path}...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)

    if ext == '.npz':
        with np.load(file_path) as data_dict:
            loc = data_dict['loc']
            snr = data_dict['snr']
            return loc, snr

    elif ext == '.h5':
        with h5py.File(file_path, 'r') as f:
            # Use [:] to materialize arrays before file close.
            loc = f['loc'][:]
            snr = f['snr'][:]
            return loc, snr
    else:
        raise ValueError(f"Unsupported file format: {ext}")


if __name__ == '__main__':
    # Example usage.
    file_path = '../data/DICHASUS/h5_CD/0152-0157_DirectSample512to32/test.h5'
    save_path = '../results/DICHASUS-015x/'
    fig_name = 'snr_distribution'

    # Plot and save spatial SNR map.
    loc_data, snr_data = load_data(file_path)
    visualize_spatial_snr(loc_data, snr_data, sample_ratio=0.2,
                          vmin=0, vmax=25, is_save=True, save_path=save_path, fig_name=fig_name)
