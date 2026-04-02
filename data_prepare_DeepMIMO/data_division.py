"""Prepare DeepMIMO train/test samples by spatial box sampling.

Workflow:
1. Split the full scene into non-overlapping `stride x stride` boxes.
2. Randomly choose `train_box` and `test_box` boxes.
3. For each selected box, sample `choice_in_box` points from the box and
   load the corresponding `DeepMIMO_dataset_<index>.mat` files.
4. Save each sampled set as one `.npy` file under `train/` or `test/`.
"""

import os
import random

import hdf5storage
import numpy as np
from tqdm import tqdm


path = '../data/DeepMIMO/O1/32antx32car_3p5GHz_40MHz_R501-1400_V1_23-9-25'

np.random.seed(1)
random.seed(1)

x_length = 181
y_length = 900
stride = 6

# Number of output samples for each split.
train_box = 3000
test_box = 1000

# Number of points sampled inside each box.
choice_in_box = 32


def point_index_from_box(
    x_box_index: int,
    y_box_index: int,
    offset_x: int,
    offset_y: int,
    x_size: int,
    box_stride: int,
) -> int:
    """Convert box index + in-box offset to global DeepMIMO point index.

    Args:
        x_box_index (int): Box index on x-axis.
        y_box_index (int): Box index on y-axis.
        offset_x (int): Offset inside the box on x-axis.
        offset_y (int): Offset inside the box on y-axis.
        x_size (int): Full scene width in points.
        box_stride (int): Box size (same as stride).

    Returns:
        int: Global point index used in `DeepMIMO_dataset_<index>.mat`.
    """
    base_index = y_box_index * x_size * box_stride + x_box_index * box_stride
    return base_index + offset_y * x_size + offset_x


def build_split(
    split_name: str,
    split_box_count: int,
    box_indices: np.ndarray,
    start_offset: int,
    x_box_count: int,
) -> None:
    """Generate one split (`train` or `test`) and save sampled `.npy` files.

    Args:
        split_name (str): Output split directory name.
        split_box_count (int): Number of boxes to process for this split.
        box_indices (np.ndarray): Shuffled global box indices.
        start_offset (int): Start index inside `box_indices` for this split.
        x_box_count (int): Number of boxes along x-axis.

    Returns:
        None
    """
    split_dir = os.path.join(path, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for i in tqdm(range(split_box_count), desc=f'Building {split_name}'):
        box_id = box_indices[start_offset + i]
        x_box_index = box_id % x_box_count
        y_box_index = box_id // x_box_count

        in_box_positions = list(range(stride * stride))
        sampled_positions = random.sample(in_box_positions, choice_in_box)

        data_list = []
        for position in sampled_positions:
            offset_x = position % stride
            offset_y = position // stride

            point_index = point_index_from_box(
                x_box_index=x_box_index,
                y_box_index=y_box_index,
                offset_x=offset_x,
                offset_y=offset_y,
                x_size=x_length,
                box_stride=stride,
            )

            mat_path = os.path.join(path, 'mat', f'DeepMIMO_dataset_{point_index}.mat')
            data_mat = hdf5storage.loadmat(mat_path)
            data_list.append(data_mat['DeepMIMO_dataset'])

        # Shuffle order within each sampled set to avoid fixed local ordering.
        random.shuffle(data_list)
        np.save(os.path.join(split_dir, f'data_{i}.npy'), np.array(data_list))

    np.save(os.path.join(split_dir, 'data_len.npy'), split_box_count)


def main() -> None:
    """Entry point for DeepMIMO train/test split generation."""
    x_box = x_length // stride
    y_box = y_length // stride
    total_box = x_box * y_box
    print('total box:', total_box)

    box_index = np.arange(total_box)
    np.random.shuffle(box_index)

    build_split(
        split_name='train',
        split_box_count=train_box,
        box_indices=box_index,
        start_offset=0,
        x_box_count=x_box,
    )
    build_split(
        split_name='test',
        split_box_count=test_box,
        box_indices=box_index,
        start_offset=train_box,
        x_box_count=x_box,
    )


if __name__ == '__main__':
    main()
