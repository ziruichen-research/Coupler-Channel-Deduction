"""Split a source HDF5 dataset into train/test subsets.

This script shuffles sample indices, creates train/test splits, restores
within-source temporal order (source_indices -> time), and saves split HDF5
files while preserving source metadata.
"""

import os
import h5py
import numpy as np


def create_split_dataset(
        source_h5_path,
        output_dir,
        num_train=50000,
        num_test=50000,
        seed=42
):
    """Create train/test HDF5 files from a single source HDF5 file.

    Args:
        source_h5_path (str): Path to the source HDF5 file.
        output_dir (str): Directory to save `train.h5` and `test.h5`.
        num_train (int): Number of training samples.
        num_test (int): Number of test samples.
        seed (int): Random seed for reproducible shuffling.

    Returns:
        None

    Raises:
        FileNotFoundError: If `source_h5_path` does not exist.
        ValueError: If requested split size exceeds available samples.
    """

    # Load all arrays into memory for fast index-based slicing.
    print(f"Loading source data from: {source_h5_path} ...")
    if not os.path.exists(source_h5_path):
        raise FileNotFoundError(f"Source file not found: {source_h5_path}")

    with h5py.File(source_h5_path, 'r') as f_src:
        print("Reading data into memory...")
        raw_data = f_src['data'][:]  # [N, 32, 32]
        raw_loc = f_src['loc'][:]    # [N, 3]
        raw_snr = f_src['snr'][:]    # [N, 32]

        # Required for restoring physical order after random split.
        raw_time = f_src['time'][:]
        raw_source_indices = f_src['source_indices'][:]

        # Preserve global attributes (e.g., part names, downsampling config).
        src_attrs = dict(f_src.attrs)

    total_samples = raw_data.shape[0]
    print(f"Total samples loaded: {total_samples}")

    # Validate split sizes.
    if num_train + num_test > total_samples:
        raise ValueError(
            f"Requested samples (Train:{num_train} + Test:{num_test} = {num_train + num_test}) "
            f"exceed total available samples ({total_samples})!"
        )

    # Shuffle sample indices with a fixed seed.
    print(f"Shuffling data with seed {seed}...")
    np.random.seed(seed)
    permuted_indices = np.random.permutation(total_samples)

    # Split index ranges.
    train_indices = permuted_indices[:num_train]
    test_indices = permuted_indices[num_train: num_train + num_test]

    print(f"Split Summary:")
    print(f"  - Train samples: {len(train_indices)}")
    print(f"  - Test samples:  {len(test_indices)}")
    print(f"  - Unused samples: {total_samples - (num_train + num_test)}")

    # Create output directory.
    os.makedirs(output_dir, exist_ok=True)

    # Save helper for a single split.
    def save_to_h5(indices, split_name):
        filename = f"{split_name}.h5"
        filepath = os.path.join(output_dir, filename)

        print(f"Sorting {split_name} indices by Source -> Time before saving...")

        current_sources = raw_source_indices[indices]
        current_times = raw_time[indices]

        # Restore physical order: primary key = source_indices, secondary = time.
        sort_order = np.lexsort((current_times, current_sources))
        sorted_indices = indices[sort_order]

        print(f"Writing {split_name} dataset to {filepath} ...")

        with h5py.File(filepath, 'w') as f:
            for k, v in src_attrs.items():
                f.attrs[k] = v

            f.create_dataset(
                'data',
                data=raw_data[sorted_indices],
                dtype='complex64',
                # compression='gzip',
                # compression_opts=4
                chunks=(1, 32, 32)
            )
            f.create_dataset('loc', data=raw_loc[sorted_indices], dtype='float32')
            f.create_dataset('snr', data=raw_snr[sorted_indices], dtype='float32')
            f.create_dataset('time', data=raw_time[sorted_indices], dtype='float32')
            f.create_dataset('source_indices', data=raw_source_indices[sorted_indices], dtype='int8')
            # Keep original positions in the source dataset for traceability.
            f.create_dataset('original_indices', data=sorted_indices, dtype='int64')

        print(f"Finished writing {split_name}.")

    # Save both splits.
    save_to_h5(train_indices, 'train')
    save_to_h5(test_indices, 'test')

    print("\nAll done!")


if __name__ == '__main__':
    # Example config.
    IS_ANT_POWER_NORM = False

    source_path = '../data/DICHASUS/src_h5/0152-0157_DirectSample512to32_v2/data_0152-0157.h5'

    # Output directory.
    output_base_dir = '../data/DICHASUS/h5_CD/0152-0157_DirectSample512to32'

    # Separate normalized and non-normalized outputs.
    if IS_ANT_POWER_NORM:
        output_base_dir += '_norm'

    # Basic path check before processing.
    if not os.path.exists(source_path):
        print(f"[Warning] Source path not found: {source_path}")
        print("Please check the input filename and directory.")
    else:
        create_split_dataset(
            source_h5_path=source_path,
            output_dir=output_base_dir,
            num_train=36000,
            num_test=36000,
            seed=2026
        )




